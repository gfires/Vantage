"""
pose.py — MediaPipe BlazePose pose extraction layer.

Handles model lifecycle, per-frame landmark inference, video rotation, and
side selection.  All squat-specific logic (depth classification, rep
segmentation, metrics) lives elsewhere.

The pose model (~9MB) is downloaded automatically on first run to models/.
"""

import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "pose_landmarker_full.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def _ensure_model() -> None:
    if not MODEL_PATH.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        print(f"Downloading MediaPipe pose model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


# ── Joint registry ────────────────────────────────────────────────────────────
# Drives both _infer_one_frame and pipeline draw-smoothing.
# Each entry: {"idx": int, "vis": bool, "z": bool}
#   idx: MediaPipe BlazePose landmark index (consistent across API versions)
#   vis: include visibility score as 3rd tuple element
#   z:   include z-depth as 4th tuple element

JOINTS = {
    "left_hip":       {"idx": 23, "vis": True,  "z": True},
    "right_hip":      {"idx": 24, "vis": True,  "z": True},
    "left_knee":      {"idx": 25, "vis": True,  "z": False},
    "right_knee":     {"idx": 26, "vis": True,  "z": False},
    "left_shoulder":  {"idx": 11, "vis": False, "z": False},
    "right_shoulder": {"idx": 12, "vis": False, "z": False},
    "left_wrist":     {"idx": 15, "vis": True, "z": False},
    "right_wrist":    {"idx": 16, "vis": True, "z": False},
    "left_heel":      {"idx": 29, "vis": True, "z": False},
    "right_heel":     {"idx": 30, "vis": True, "z": False},
}


# ── Detector lifecycle ────────────────────────────────────────────────────────

def _make_detector():
    """
    Create a MediaPipe PoseLandmarker in VIDEO mode.

    Use as a context manager — VIDEO mode is stateful and requires a single
    persistent instance with strictly monotonically increasing timestamps:

        with _make_detector() as detector:
            fdata = _infer_one_frame(frame, detector, frame_idx, fps)
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


# ── Per-frame inference ───────────────────────────────────────────────────────

def _infer_one_frame(frame, detector, frame_idx: int, fps: float) -> dict | None:
    """
    Run BlazePose inference on a single BGR frame.

    Args:
        frame:     BGR numpy array, already rotated to display orientation.
        detector:  Open PoseLandmarker from _make_detector().
        frame_idx: 0-based index; must increase monotonically across calls.
        fps:       Video frame rate, used to compute timestamp_ms.

    Returns:
        fdata dict with keys: frame_idx, width, height, and one entry per
        JOINTS key.  Each joint value is a tuple (x_px, y_px[, visibility][, z]).
        Returns None if no pose detected.
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
    fdata: dict = {"frame_idx": frame_idx, "width": w, "height": h}
    for name, meta in JOINTS.items():
        p     = lm[meta["idx"]]
        entry = (p.x * w, p.y * h)
        if meta["vis"]:
            entry += (p.visibility,)
        if meta["z"]:
            entry += (p.z,)
        fdata[name] = entry
    return fdata


# ── Video utilities ───────────────────────────────────────────────────────────

def _get_rotation(cap: cv2.VideoCapture) -> int:
    """
    Read rotation angle from video container metadata.

    iPhone MOV files embed a rotation tag that OpenCV exposes via
    CAP_PROP_ORIENTATION_META.  Combined with disabling
    CAP_PROP_ORIENTATION_AUTO (which would double-rotate on macOS),
    this gives the correct display orientation.
    """
    return int(cap.get(cv2.CAP_PROP_ORIENTATION_META))


def _rotate_frame(frame, angle: int):
    """Apply a rotation angle (degrees, clockwise) to a BGR frame."""
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ── Camera calibration ────────────────────────────────────────────────────────

@dataclass
class CameraCalibration:
    roll_deg: float = 0.0     # vertical tilt (Hough upright); positive = top leans right
    azimuth_deg: float = 0.0  # φ from vertical; 0° = pure side profile, 90° = facing camera
                               # correction factor = sin(φ); at φ≈90° (side-on) sin≈1 → minimal change


def _landmark_vec(fdata: dict | None, left_key: str, right_key: str, vis_thresh: float = 0.5):
    """
    Extract (left_pt, right_pt) pixel tuple from fdata if both landmarks exceed vis_thresh.
    Returns None if fdata is None or either landmark is below threshold.
    """
    if fdata is None:
        return None
    lm = fdata[left_key]
    rm = fdata[right_key]
    if lm[2] < vis_thresh or rm[2] < vis_thresh:
        return None
    return (int(lm[0]), int(lm[1])), (int(rm[0]), int(rm[1]))


def _azimuth_from_fdata(fdata: dict | None):
    """
    Compute azimuth vector: heels first, wrists as fallback.
    Returns (source_label, (left_pt, right_pt)) or (None, None).
    """
    heel_vec = _landmark_vec(fdata, "left_heel", "right_heel")
    if heel_vec is not None:
        return "heels", heel_vec
    wrist_vec = _landmark_vec(fdata, "left_wrist", "right_wrist")
    if wrist_vec is not None:
        return "wrists", wrist_vec
    return None, None


def azimuth_deg_from_fdata(fdata: dict | None) -> float | None:
    """
    Compute φ (degrees from side profile) from fdata heel/wrist landmarks.

    Convention:
        φ = 0°  → heel vector vertical (degenerate) → pure side profile → sin(φ)=0, no correction
        φ = 90° → heel vector horizontal            → facing camera     → sin(φ)=1, full correction

    Returns None if no suitable landmarks are visible.
    """
    label, vec = _azimuth_from_fdata(fdata)
    if vec is None:
        return None
    left_pt, right_pt = vec
    dx = right_pt[0] - left_pt[0]
    dy = right_pt[1] - left_pt[1]
    return math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-6)))


def estimated_markers(
    fdata: dict,
    side: str,
    cal: "CameraCalibration | None" = None,
) -> tuple[float, float, float, float]:
    """
    Estimate anatomical hip-crease and knee-top positions from joint landmarks.

    Both offsets are computed in azimuth-decompressed (true) space so that unit
    vectors are geometrically correct, then recompressed before adding to the
    anchor joint.

    Hip crease = hip + a*(torso_unit)*torso_len + b*(femur_unit)*femur_len
                      + c*sin(hip_flex)*(sagittal)
      torso = hip→shoulder, femur = hip→knee, hip_flex = included angle at hip.

    Knee top   = knee + a*(tibial_unit)*tibial_len + b*(femoral_unit)*femoral_len
                      + g*sin(knee_flex)*(sagittal)
      tibial = heel→knee, femoral = hip→knee, knee_flex = included angle at knee.

    The sagittal axis is the roll-corrected horizontal, oriented anteriorly using
    the sign of (knee_x - heel_x).

    Returns:
        (hc_y, kt_y, hc_x, kt_x) in full-resolution pixel coordinates.
        hc_y > kt_y in screen coords means hip crease is below knee top (depth).
    """
    from params import (
        HC_TORSO_COEFF, HC_FEMUR_COEFF, HC_SAGITTAL_COEFF,
        KT_TIBIAL_COEFF, KT_FEMORAL_COEFF, KT_SAGITTAL_COEFF,
    )

    heel     = fdata[f"{side}_heel"]
    knee     = fdata[f"{side}_knee"]
    shoulder = fdata[f"{side}_shoulder"]
    hip      = fdata[f"{side}_hip"]

    sin_az   = math.sin(math.radians(cal.azimuth_deg)) if cal is not None else 0.0
    use_az   = sin_az > 1e-6

    def _dx(x):
        return x / sin_az if use_az else x

    def _rx(x):
        return x * sin_az if use_az else x

    def _decomp(pt):
        return (_dx(pt[0]), pt[1])

    shoulder_d = _decomp(shoulder)
    hip_d      = _decomp(hip)
    knee_d     = _decomp(knee)
    heel_d     = _decomp(heel)

    def _unit(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        mag = math.hypot(dx, dy)
        return (dx / mag, dy / mag) if mag > 1e-6 else (0.0, 0.0)

    def _dist(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    # Sagittal axis: roll-corrected horizontal, oriented anteriorly
    roll_rad = math.radians(cal.roll_deg if cal is not None else 0.0)
    sagittal_sign = 1.0 if (knee[0] - heel[0]) >= 0 else -1.0
    sag_x = math.cos(roll_rad) * sagittal_sign
    sag_y = math.sin(roll_rad) * sagittal_sign

    # ── Hip crease ────────────────────────────────────────────────────────────
    torso_unit  = _unit(hip_d, shoulder_d)   # hip→shoulder
    femur_unit  = _unit(hip_d, knee_d)       # hip→knee
    torso_len   = _dist(hip_d, shoulder_d)
    femur_len   = _dist(hip_d, knee_d)

    dot_hip     = torso_unit[0]*femur_unit[0] + torso_unit[1]*femur_unit[1]
    hip_flex    = math.acos(max(-1.0, min(1.0, dot_hip)))

    hc_off_x = (HC_TORSO_COEFF    * torso_unit[0] * torso_len
              + HC_FEMUR_COEFF    * femur_unit[0] * femur_len
              + HC_SAGITTAL_COEFF * math.sin(hip_flex) * sag_x)
    hc_off_y = (HC_TORSO_COEFF    * torso_unit[1] * torso_len
              + HC_FEMUR_COEFF    * femur_unit[1] * femur_len
              + HC_SAGITTAL_COEFF * math.sin(hip_flex) * sag_y)

    hc_x = hip[0] + _rx(hc_off_x)
    hc_y = hip[1] + hc_off_y

    # ── Knee top ──────────────────────────────────────────────────────────────
    tibial_unit  = _unit(heel_d, knee_d)     # heel→knee
    femoral_unit = _unit(hip_d, knee_d)      # hip→knee
    tibial_len   = _dist(heel_d, knee_d)
    femoral_len  = femur_len                 # same as femur computed above

    # Knee flexion = angle at knee between (knee→heel) and (knee→hip)
    dot_knee  = (-tibial_unit[0])*(-femoral_unit[0]) + (-tibial_unit[1])*(-femoral_unit[1])
    knee_flex = math.acos(max(-1.0, min(1.0, dot_knee)))

    kt_off_x = (KT_TIBIAL_COEFF   * tibial_unit[0]  * tibial_len
              + KT_FEMORAL_COEFF  * femoral_unit[0] * femoral_len
              + KT_SAGITTAL_COEFF * math.sin(knee_flex) * sag_x)
    kt_off_y = (KT_TIBIAL_COEFF   * tibial_unit[1]  * tibial_len
              + KT_FEMORAL_COEFF  * femoral_unit[1] * femoral_len
              + KT_SAGITTAL_COEFF * math.sin(knee_flex) * sag_y)

    kt_x = knee[0] + _rx(kt_off_x)
    kt_y = knee[1] + kt_off_y

    return hc_y, kt_y, hc_x, kt_x


def _max_consecutive_true(flags: list[bool]) -> int:
    """Return the length of the longest run of True values in flags."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


# ── Side selection ────────────────────────────────────────────────────────────

def _select_side(valid_frames: list) -> str | None:
    """
    Choose left or right landmarks by picking the hip closer to the camera.

    Uses the z-coordinate of the first valid frame (negative = closer).
    Camera geometry is static so a single-frame snapshot is sufficient.
    Returns None if both hips are equidistant (exact tie, rare).
    """
    _, first = valid_frames[0]
    left_z  = first["left_hip"][3]
    right_z = first["right_hip"][3]
    if left_z == right_z:
        return None
    return "left" if left_z < right_z else "right"
