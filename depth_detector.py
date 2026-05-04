"""
Squat depth detector using MediaPipe BlazePose (Tasks API, mediapipe >= 0.10).

IPF standard: hip crease must pass below the top of the knee cap.
In 2D side-profile (y increases downward): hip_y > knee_y = depth achieved.

Note: MediaPipe landmark 23/24 (hip joint center) approximates the hip crease.
The anatomical crease is a few cm anterior; this approximation is acceptable for
clear depth calls. Borderline calls may need manual review.

The pose model (~5MB) is downloaded automatically on first run to models/.
"""

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from params import (
    MIN_DEPTH_FRAMES,
    SMOOTHING_WINDOW,
    CLOSE_THRESHOLD,
)

# Model auto-download
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# MediaPipe BlazePose landmark indices (same across all API versions)
LEFT_HIP      = 23
RIGHT_HIP     = 24
LEFT_KNEE     = 25
RIGHT_KNEE    = 26
LEFT_WRIST    = 15
RIGHT_WRIST   = 16
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HEEL     = 29
RIGHT_HEEL    = 30

# Joint registry — drives both _infer_one_frame and pipeline draw-smoothing.
# Each entry: {"idx": int, "vis": bool, "z": bool}
#   idx: MediaPipe landmark index
#   vis: whether the landmark includes a visibility score (3rd tuple element)
#   z:   whether the landmark includes a z-depth value (4th tuple element)
JOINTS = {
    "left_hip":       {"idx": LEFT_HIP,       "vis": True,  "z": True},
    "right_hip":      {"idx": RIGHT_HIP,      "vis": True,  "z": True},
    "left_knee":      {"idx": LEFT_KNEE,       "vis": True,  "z": False},
    "right_knee":     {"idx": RIGHT_KNEE,      "vis": True,  "z": False},
    "left_shoulder":  {"idx": LEFT_SHOULDER,   "vis": False, "z": False},
    "right_shoulder": {"idx": RIGHT_SHOULDER,  "vis": False, "z": False},
    "left_wrist":     {"idx": LEFT_WRIST,      "vis": False, "z": False},
    "right_wrist":    {"idx": RIGHT_WRIST,     "vis": False, "z": False},
    "left_heel":      {"idx": LEFT_HEEL,       "vis": False, "z": False},
    "right_heel":     {"idx": RIGHT_HEEL,      "vis": False, "z": False},
}


def _ensure_model():
    if not MODEL_PATH.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        print(f"Downloading MediaPipe pose model (~5MB) to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def analyze_video(path: str) -> dict:
    """
    Analyze a squat video and return a depth judgment.

    Returns a dict with:
        result: "pass" | "fail" | "borderline" | "indeterminate"
        bottom_frame: int (frame index of detected squat bottom)
        hip_y: float (hip y-coordinate at bottom, pixels)
        knee_y: float (knee y-coordinate at bottom, pixels)
        side: "left" | "right"
        max_consecutive_depth_frames: int
        error: str (only present when result is "indeterminate")
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"result": "indeterminate", "error": f"Cannot open video: {path}"}
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # prevent double-rotate on macOS

    _ensure_model()
    rotation = _get_rotation(cap)
    frames_data = _extract_landmarks(cap, rotation)
    cap.release()

    valid_frames = [(i, f) for i, f in enumerate(frames_data) if f is not None]
    if not valid_frames:
        return {"result": "indeterminate", "error": "No pose detected in video"}

    side = _select_side(valid_frames)
    if side is None:
        return {
            "result": "indeterminate",
            "error": "Landmarks not visible enough — ensure side-profile camera angle",
        }

    hip_key = f"{side}_hip"
    knee_key = f"{side}_knee"

    hip_ys = [f[hip_key][1] for _, f in valid_frames]
    frame_indices = [i for i, _ in valid_frames]
    smooth_hip_ys = _rolling_average(hip_ys, SMOOTHING_WINDOW)

    bottom_local = _find_bottom_frame(smooth_hip_ys)
    if bottom_local is None:
        return {
            "result": "indeterminate",
            "error": "Could not detect squat bottom — ensure full descent is visible",
        }

    bottom_global = frame_indices[bottom_local]
    bottom_data = valid_frames[bottom_local][1]
    hip_y = bottom_data[hip_key][1]
    knee_y = bottom_data[knee_key][1]
    frame_height = bottom_data["height"]

    # Check depth across the entire video — any MIN_DEPTH_FRAMES consecutive
    # frames where hip_y > knee_y counts as a pass, regardless of where the
    # bottom was detected.
    all_depth_flags = [f[hip_key][1] > f[knee_key][1] for _, f in valid_frames]
    max_consec = _max_consecutive_true(all_depth_flags)

    # Borderline: use the bottom frame's proximity for close calls
    close = abs(hip_y - knee_y) < (frame_height * CLOSE_THRESHOLD)

    # Also check borderline across the whole video (min gap ever seen)
    min_gap = min(abs(f[hip_key][1] - f[knee_key][1]) for _, f in valid_frames)
    close = close or min_gap < (frame_height * CLOSE_THRESHOLD)

    if max_consec >= MIN_DEPTH_FRAMES:
        result = "pass"
    elif close:
        result = "borderline"
    else:
        result = "fail"

    depth_flags = all_depth_flags

    return {
        "result": result,
        "bottom_frame": bottom_global,
        "hip_y": hip_y,
        "knee_y": knee_y,
        "side": side,
        "depth_flags": depth_flags,
        "max_consecutive_depth_frames": max_consec,
    }


# --- Internal helpers ---

def _get_rotation(cap: cv2.VideoCapture) -> int:
    """Read rotation metadata from video container (handles iPhone MOV files)."""
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    # CAP_PROP_ORIENTATION_META returns 0 if unavailable or not rotated
    return rotation


def _rotate_frame(frame, angle: int):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _make_detector():
    """
    Create and return a MediaPipe PoseLandmarker configured for VIDEO mode.

    The caller is responsible for the lifecycle — use as a context manager:
        with _make_detector() as detector:
            ...

    VIDEO mode requires strictly monotonically increasing timestamps and a
    single persistent detector instance across all frames in a video.
    """
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def _infer_one_frame(frame, detector, frame_idx: int, fps: float) -> dict | None:
    """
    Run MediaPipe BlazePose inference on a single already-decoded BGR frame.

    Single-frame equivalent of _extract_landmarks for use in the single-pass
    pipeline loop.  The detector must be the same instance across all frames
    in the video — VIDEO mode is stateful.

    Args:
        frame:     BGR numpy array, already rotated to display orientation.
        detector:  Open PoseLandmarker from _make_detector().
        frame_idx: 0-based frame index; must be strictly monotonically increasing.
        fps:       Video frame rate; used to compute timestamp_ms.

    Returns:
        fdata dict matching the _extract_landmarks schema, or None if no pose
        was detected in this frame.
    """
    h, w = frame.shape[:2]
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
    timestamp_ms = int(frame_idx * 1000 / fps)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks[0]
    fdata = {"frame_idx": frame_idx, "width": w, "height": h}
    for name, meta in JOINTS.items():
        p = lm[meta["idx"]]
        entry = (p.x * w, p.y * h)
        if meta["vis"]:
            entry += (p.visibility,)
        if meta["z"]:
            entry += (p.z,)
        fdata[name] = entry
    return fdata


def _extract_landmarks(cap: cv2.VideoCapture, rotation: int) -> list:
    """Extract per-frame landmark data using the MediaPipe Tasks API (mediapipe >= 0.10)."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames_data = []
    with _make_detector() as detector:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = _rotate_frame(frame, rotation)
            frames_data.append(_infer_one_frame(frame, detector, frame_idx, fps))
            frame_idx += 1

    return frames_data


def _select_side(valid_frames: list) -> str | None:
    """
    Pick which side's landmarks to use by selecting the hip closer to the camera.

    Uses the z-coordinate from the first valid frame (negative = closer to camera).
    Camera-to-subject geometry is static, so a single frame snapshot is sufficient.
    Returns None when both hips are equidistant or no valid frame is available.
    """
    _, first = valid_frames[0]
    left_z  = first["left_hip"][3]
    right_z = first["right_hip"][3]
    if left_z == right_z:
        return None
    return "left" if left_z < right_z else "right"


def _rolling_average(values: list, window: int) -> list:
    """Centered rolling average to smooth jitter in landmark positions."""
    arr = np.array(values, dtype=float)
    result = np.convolve(arr, np.ones(window) / window, mode="same")
    # Fix edge effects: use raw values at edges where convolution window is incomplete
    half = window // 2
    result[:half] = arr[:half]
    result[-half:] = arr[-half:]
    return result.tolist()


def _find_bottom_frame(smooth_hip_ys: list) -> int | None:
    """
    Find the squat bottom as the deepest hip position within the rep.

    Each segment starts and ends at a standing position (peak of the height
    signal), so the true bottom is simply the global maximum of hip Y
    (lowest physical position) within the segment.  We constrain the search
    to the middle 80% of the segment to avoid standing-position noise at
    the edges triggering a false bottom near the start or end.
    """
    n = len(smooth_hip_ys)
    if n < 3:
        return None

    margin = max(1, int(n * 0.10))
    search = smooth_hip_ys[margin: n - margin]
    if not search:
        return int(np.argmax(smooth_hip_ys))

    return margin + int(np.argmax(search))


def _max_consecutive_true(flags: list) -> int:
    max_count = count = 0
    for f in flags:
        count = count + 1 if f else 0
        max_count = max(max_count, count)
    return max_count
