"""
scripts/calibrate.py — Rack upright tilt detection diagnostic.

Runs on the first 5 frames of a squat video.  For each frame:
  - Detects the longest near-vertical line (rack upright) via Gaussian blur + Canny + HoughLinesP
  - Computes the angle that line makes against true pixel-vertical (signed degrees)
  - Annotates the frame with the detected upright (green), reference vertical (white), and angle text

Outputs:
  - Terminal: per-frame angles + median
  - File: <video_stem>_calibration.jpg (5 annotated frames stacked vertically)

Usage:
    python scripts/calibrate.py path/to/video.MOV
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from params import (
    CAL_PROBE_FRAMES,
    CAL_BLUR_KERNEL,
    CAL_CANNY_LOW,
    CAL_CANNY_HIGH,
    CAL_HOUGH_THRESHOLD,
    CAL_HOUGH_MIN_LENGTH,
    CAL_HOUGH_MAX_GAP,
    CAL_UPRIGHT_TOL_DEG,
    VISIBILITY_THRESHOLD,
)


def detect_upright_tilt(frames: list) -> float | None:
    """
    Compute the median rack-upright tilt across a list of frames.

    Args:
        frames: BGR frames (full-resolution, already rotated for device orientation).

    Returns:
        Median tilt in degrees (positive = top leans right of pixel-vertical), or
        None if no upright was detected in any frame.
    """
    angles = [_detect_upright(f)[0] for f in frames]
    valid  = [a for a in angles if a is not None]
    return float(np.median(valid)) if valid else None


def detect_heel_azimuth(fdata_list: list) -> dict | None:
    """
    Estimate camera azimuth from the heel-to-heel vector across a list of fdata dicts.

    The vector from left_heel to right_heel is the lateral axis (along the bar)
    projected onto the image plane. Its angle from image-horizontal is the azimuth
    offset: 0° means the camera is dead side-on, 90° means front/rear.

    Args:
        fdata_list: list of per-frame landmark dicts (or None) from _infer_one_frame.
                    Heels must have visibility enabled (vis=True in JOINTS).

    Returns:
        dict with keys:
            azimuth_deg   float  angle of heel vector from image-horizontal (0=side-on, 90=front)
            heel_vec_deg  float  raw image-plane angle of left→right heel vector (degrees)
            left_heel     (x, y) averaged left heel pixel position
            right_heel    (x, y) averaged right heel pixel position
            n_frames      int    number of frames that contributed
        or None if insufficient confident heel detections.
    """
    left_xs, left_ys, right_xs, right_ys = [], [], [], []

    for fdata in fdata_list:
        if fdata is None:
            continue
        lh = fdata.get("left_heel")
        rh = fdata.get("right_heel")
        if lh is None or rh is None:
            continue
        # Heels now have visibility as 3rd element (index 2)
        lvis = lh[2] if len(lh) > 2 else 0.0
        rvis = rh[2] if len(rh) > 2 else 0.0
        if lvis < VISIBILITY_THRESHOLD or rvis < VISIBILITY_THRESHOLD:
            continue
        left_xs.append(lh[0]);  left_ys.append(lh[1])
        right_xs.append(rh[0]); right_ys.append(rh[1])

    if not left_xs:
        return None

    lx = sum(left_xs)  / len(left_xs)
    ly = sum(left_ys)  / len(left_ys)
    rx = sum(right_xs) / len(right_xs)
    ry = sum(right_ys) / len(right_ys)

    dx = rx - lx
    dy = ry - ly

    # Angle of left→right vector from image-horizontal
    heel_vec_deg = math.degrees(math.atan2(-dy, dx))  # negate dy: screen y is flipped

    # Azimuth = how far the camera is from pure side-on.
    # Side-on: heels overlap (vector ~ zero length, azimuth undefined).
    # Front-on: heel vector is horizontal, heel_vec_deg ~ 0°.
    # The azimuth is the complement: how horizontal the heel vector is.
    azimuth_deg = 90.0 - abs(heel_vec_deg)

    return {
        "azimuth_deg":  round(azimuth_deg,  1),
        "heel_vec_deg": round(heel_vec_deg,  1),
        "left_heel":    (lx, ly),
        "right_heel":   (rx, ry),
        "n_frames":     len(left_xs),
    }


def _get_rotation(cap: cv2.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_ORIENTATION_META))


def _rotate_frame(frame, angle: int):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _detect_upright(frame_full: np.ndarray) -> tuple[float | None, tuple | None]:
    """
    Detect the dominant near-vertical line (rack upright) in a frame.

    Runs on a half-res copy for speed; angle is scale-invariant.

    Returns:
        (tilt_deg, (x1, y1, x2, y2)) in full-res coordinates, or (None, None).
        tilt_deg: signed degrees from pixel-vertical. Positive = top leans right.
    """
    H, W = frame_full.shape[:2]
    small = cv2.resize(frame_full, (W // 2, H // 2))

    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, CAL_BLUR_KERNEL, 0)
    edges = cv2.Canny(blur, CAL_CANNY_LOW, CAL_CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=CAL_HOUGH_THRESHOLD,
        minLineLength=CAL_HOUGH_MIN_LENGTH,
        maxLineGap=CAL_HOUGH_MAX_GAP,
    )

    if lines is None:
        return None, None

    best_line = None
    best_len  = 0.0

    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        # Angle from vertical: atan2(|dx|, |dy|) → 0° = perfectly vertical
        angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy)))
        if angle_from_vertical > CAL_UPRIGHT_TOL_DEG:
            continue
        if length > best_len:
            best_len  = length
            best_line = (x1, y1, x2, y2, dx, dy)

    if best_line is None:
        return None, None

    x1, y1, x2, y2, dx, dy = best_line

    # Signed tilt: positive = top of upright leans right (clockwise)
    # dy is negative when line goes upward (y2 < y1 means top of segment is at bottom in image)
    # We want the angle of the upright relative to the downward direction.
    tilt_deg = math.degrees(math.atan2(dx, abs(dy)))

    # Scale coords back to full-res
    full = (x1 * 2, y1 * 2, x2 * 2, y2 * 2)
    return tilt_deg, full


def _extend_to_frame_height(x1, y1, x2, y2, H: int) -> tuple[int, int, int, int]:
    """Extrapolate a line segment to span y=0 .. y=H."""
    if y1 == y2:
        return x1, 0, x2, H
    slope = (x2 - x1) / (y2 - y1)  # dx/dy
    top_x    = int(x1 + slope * (0 - y1))
    bottom_x = int(x1 + slope * (H - y1))
    return top_x, 0, bottom_x, H


def _annotate_frame(
    frame: np.ndarray,
    frame_idx: int,
    tilt_deg: float | None,
    line_full,
    heel_info: dict | None = None,
) -> np.ndarray:
    """Draw upright line, reference vertical, heel vector, and angle text onto a copy of frame."""
    out = frame.copy()
    H, W = out.shape[:2]

    if tilt_deg is not None and line_full is not None:
        x1, y1, x2, y2 = line_full
        ex1, ey1, ex2, ey2 = _extend_to_frame_height(x1, y1, x2, y2, H)

        # Detected upright — bright green, extended full height
        cv2.line(out, (ex1, ey1), (ex2, ey2), (0, 220, 0), 2, cv2.LINE_AA)

        # Reference vertical — white, through the midpoint x of the detected line
        mid_x = (ex1 + ex2) // 2
        cv2.line(out, (mid_x, 0), (mid_x, H), (220, 220, 220), 1, cv2.LINE_AA)

        label = f"Frame {frame_idx}  upright: {tilt_deg:+.1f}° from vertical"
        color = (0, 220, 0)
    else:
        label = f"Frame {frame_idx}  no upright detected"
        color = (0, 60, 220)

    # Heel vector overlay — cyan dot per heel, line between them, azimuth label
    if heel_info is not None:
        lh = (int(heel_info["left_heel"][0]),  int(heel_info["left_heel"][1]))
        rh = (int(heel_info["right_heel"][0]), int(heel_info["right_heel"][1]))
        cv2.circle(out, lh, 8, (200, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(out, rh, 8, (200, 200, 0), -1, cv2.LINE_AA)
        cv2.line(out, lh, rh, (200, 200, 0), 2, cv2.LINE_AA)
        az  = heel_info["azimuth_deg"]
        hvd = heel_info["heel_vec_deg"]
        az_label = f"azimuth ~{az:.1f}°  heel vec {hvd:+.1f}°"
        cv2.putText(out, az_label, (22, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2, cv2.LINE_AA)

    # Text background for readability
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness  = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(out, (16, 16), (16 + tw + 12, 16 + th + baseline + 10), (0, 0, 0), -1)
    cv2.putText(out, label, (22, 16 + th + 4), font, font_scale, color, thickness, cv2.LINE_AA)

    return out


def run(video_path: str) -> None:
    from depth_detector import _ensure_model, _get_rotation as _dr_get_rotation, _make_detector, _infer_one_frame, _rotate_frame as _dr_rotate_frame

    path = Path(video_path)
    cap  = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Error: cannot open {path}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rotation = _get_rotation(cap)

    _ensure_model()

    angles:           list[float | None] = []
    fdata_list:       list               = []
    raw_frames:       list[np.ndarray]   = []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    with _make_detector() as detector:
        for i in range(CAL_PROBE_FRAMES):
            ret, raw = cap.read()
            if not ret:
                print(f"Frame {i}: video ended early")
                break
            raw = _rotate_frame(raw, rotation)
            raw_frames.append(raw)
            tilt_deg, _ = _detect_upright(raw)
            angles.append(tilt_deg)
            fdata = _infer_one_frame(raw, detector, i, fps)
            fdata_list.append(fdata)

    cap.release()

    # Compute heel azimuth from averaged fdata across probe frames
    heel_info = detect_heel_azimuth(fdata_list)

    # Annotate frames — pass heel_info (averaged, not per-frame) to every frame
    annotated_frames = [
        _annotate_frame(raw_frames[i], i, angles[i],
                        _detect_upright(raw_frames[i])[1], heel_info)
        for i in range(len(raw_frames))
    ]

    # ── Terminal output ───────────────────────────────────────────────────────
    print()
    for i, a in enumerate(angles):
        if a is not None:
            print(f"  Frame {i}: upright {a:+.1f}°")
        else:
            print(f"  Frame {i}: no upright detected")

    valid = [a for a in angles if a is not None]
    if valid:
        median = float(np.median(valid))
        print(f"\n  Median tilt: {median:+.1f}°")
    else:
        print("\n  Median tilt: n/a (no uprights detected in any frame)")

    print()
    if heel_info is not None:
        print(f"  Heel vector: {heel_info['heel_vec_deg']:+.1f}° from image-horizontal  ({heel_info['n_frames']} frames)")
        print(f"  Azimuth:     ~{heel_info['azimuth_deg']:.1f}°  (0=side-on, 90=front/rear)")
    else:
        print("  Heel vector: n/a (heels not detected with sufficient confidence)")
    print()

    # ── Write output image ────────────────────────────────────────────────────
    if not annotated_frames:
        print("No frames to write.", file=sys.stderr)
        sys.exit(1)

    stacked   = cv2.vconcat(annotated_frames)
    out_path  = path.parent / (path.stem + "_calibration.jpg")
    cv2.imwrite(str(out_path), stacked, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Saved: {out_path}")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/calibrate.py path/to/video.MOV", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])
