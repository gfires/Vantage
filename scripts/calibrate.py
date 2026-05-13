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
import pose as _pose

from params import (
    CAL_PROBE_FRAMES,
    CAL_BLUR_KERNEL,
    CAL_CANNY_LOW,
    CAL_CANNY_HIGH,
    CAL_HOUGH_THRESHOLD,
    CAL_HOUGH_MIN_LENGTH,
    CAL_HOUGH_MAX_GAP,
    CAL_UPRIGHT_TOL_DEG,
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


def _landmark_vec(fdata: dict | None, left_key: str, right_key: str, vis_thresh: float = 0.5):
    """
    Extract a (left_pt, right_pt) pixel tuple from fdata if both landmarks exceed vis_thresh.
    Requires vis=True for both joints in JOINTS (visibility at tuple index 2).
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
    heel_vec  = _landmark_vec(fdata, "left_heel",  "right_heel")
    if heel_vec is not None:
        return "heels", heel_vec
    wrist_vec = _landmark_vec(fdata, "left_wrist", "right_wrist")
    if wrist_vec is not None:
        return "wrists", wrist_vec
    return None, None


def _draw_axis(out: np.ndarray, cx: int, cy: int, dx: float, dy: float,
               length: int, color: tuple, label: str) -> None:
    """Draw a centered axis arrow of given pixel half-length from (cx, cy) along (dx, dy)."""
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return
    ux, uy = dx / norm, dy / norm
    p1 = (int(cx - ux * length), int(cy - uy * length))
    p2 = (int(cx + ux * length), int(cy + uy * length))
    cv2.line(out, p1, p2, color, 3, cv2.LINE_AA)
    cv2.arrowedLine(out, p1, p2, color, 3, cv2.LINE_AA, tipLength=0.15)
    cv2.putText(out, label, (p2[0] + 6, p2[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def _annotate_frame(
    frame: np.ndarray,
    frame_idx: int,
    tilt_deg: float | None,
    line_full,
    azimuth_label: str | None,
    azimuth_vec,
) -> np.ndarray:
    """
    Draw all detected calibration data onto a copy of frame:
      - Green: vertical axis (rack upright + reference line)
      - Cyan:  azimuth axis (heels or wrists), centered on landmark midpoint
      - Orange: sagittal axis (perpendicular to azimuth in image plane)
    """
    out = frame.copy()
    H, W = out.shape[:2]

    # ── Vertical axis (rack upright) ─────────────────────────────────────────
    GREEN = (0, 220, 0)
    if tilt_deg is not None and line_full is not None:
        x1, y1, x2, y2 = line_full
        ex1, ey1, ex2, ey2 = _extend_to_frame_height(x1, y1, x2, y2, H)
        cv2.line(out, (ex1, ey1), (ex2, ey2), GREEN, 2, cv2.LINE_AA)
        mid_x = (ex1 + ex2) // 2
        cv2.line(out, (mid_x, 0), (mid_x, H), (220, 220, 220), 1, cv2.LINE_AA)
        upright_text  = f"Frame {frame_idx}  vertical: {tilt_deg:+.1f}° from pixel-vertical"
        upright_color = GREEN
    else:
        upright_text  = f"Frame {frame_idx}  vertical: not detected"
        upright_color = (0, 60, 220)

    # ── Azimuth + sagittal axes ───────────────────────────────────────────────
    CYAN   = (255, 255,   0)
    ORANGE = (0,   165, 255)
    AXIS_LEN = min(W, H) // 6

    az_text = "azimuth: n/a  (sagittal: n/a)"
    if azimuth_vec is not None:
        lp, rp = azimuth_vec
        cx = (lp[0] + rp[0]) // 2
        cy = (lp[1] + rp[1]) // 2

        # Draw the raw landmark points
        cv2.circle(out, lp, 7, CYAN, -1)
        cv2.circle(out, rp, 7, CYAN, -1)

        # Azimuth direction: left→right along landmark pair
        adx, ady = rp[0] - lp[0], rp[1] - lp[1]
        az_deg   = math.degrees(math.atan2(ady, adx))
        _draw_axis(out, cx, cy, adx, ady, AXIS_LEN, CYAN, "az")

        # Sagittal: perpendicular to azimuth in the image plane (rotate 90° CCW)
        sdx, sdy = -ady, adx
        sag_deg  = math.degrees(math.atan2(sdy, sdx))
        _draw_axis(out, cx, cy, sdx, sdy, AXIS_LEN, ORANGE, "sag")

        az_text = f"azimuth ({azimuth_label}): {az_deg:+.1f}°  sagittal: {sag_deg:+.1f}°"

    # ── Text overlay ──────────────────────────────────────────────────────────
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    rows = [upright_text, az_text]
    max_w  = max(cv2.getTextSize(r, font, scale, thick)[0][0] for r in rows)
    _, th  = cv2.getTextSize("X", font, scale, thick)[0]
    base   = cv2.getTextSize("X", font, scale, thick)[1]
    line_h = th + base + 6
    box_h  = line_h * len(rows) + 10
    cv2.rectangle(out, (16, 16), (16 + max_w + 12, 16 + box_h), (0, 0, 0), -1)
    cv2.putText(out, upright_text, (22, 16 + th + 4),          font, scale, upright_color, thick, cv2.LINE_AA)
    cv2.putText(out, az_text,      (22, 16 + th + 4 + line_h), font, scale, CYAN,          thick, cv2.LINE_AA)

    return out


def run(video_path: str) -> None:
    path = Path(video_path)
    cap  = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Error: cannot open {path}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rotation = _get_rotation(cap)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0

    angles:           list[float | None] = []
    annotated_frames: list[np.ndarray]   = []

    _pose._ensure_model()
    with _pose._make_detector() as detector:
        for i in range(CAL_PROBE_FRAMES):
            ret, raw = cap.read()
            if not ret:
                print(f"Frame {i}: video ended early")
                break
            raw = _rotate_frame(raw, rotation)
            tilt_deg, line_full = _detect_upright(raw)
            angles.append(tilt_deg)

            fdata = _pose._infer_one_frame(raw, detector, i, fps)
            az_label, az_vec = _azimuth_from_fdata(fdata)

            annotated_frames.append(
                _annotate_frame(raw, i, tilt_deg, line_full, az_label, az_vec)
            )

    cap.release()

    # ── Terminal output ───────────────────────────────────────────────────────
    print()
    for i, a in enumerate(angles):
        if a is not None:
            print(f"  Frame {i}: {a:+.1f}°")
        else:
            print(f"  Frame {i}: no upright detected")

    valid = [a for a in angles if a is not None]
    if valid:
        median = float(np.median(valid))
        print(f"\n  Median tilt: {median:+.1f}°")
    else:
        print("\n  Median tilt: n/a (no uprights detected in any frame)")
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
