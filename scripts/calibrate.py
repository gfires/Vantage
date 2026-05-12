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

import cv2
import numpy as np

# ── Tunable constants ─────────────────────────────────────────────────────────
PROBE_FRAMES       = 5
BLUR_KERNEL        = (5, 5)
CANNY_LOW          = 50
CANNY_HIGH         = 150
HOUGH_THRESHOLD    = 40       # accumulator votes (half-res, so halved from full-res 80)
HOUGH_MIN_LENGTH   = 40       # px at half-res
HOUGH_MAX_GAP      = 10       # px at half-res
UPRIGHT_TOL_DEG    = 20       # lines within this many degrees of vertical are candidates


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
    blur  = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP,
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
        if angle_from_vertical > UPRIGHT_TOL_DEG:
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
) -> np.ndarray:
    """Draw upright line, reference vertical, and angle text onto a copy of frame."""
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

    # Text background for readability
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness  = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(out, (16, 16), (16 + tw + 12, 16 + th + baseline + 10), (0, 0, 0), -1)
    cv2.putText(out, label, (22, 16 + th + 4), font, font_scale, color, thickness, cv2.LINE_AA)

    return out


def run(video_path: str) -> None:
    path = Path(video_path)
    cap  = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Error: cannot open {path}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rotation = _get_rotation(cap)

    angles:           list[float | None] = []
    annotated_frames: list[np.ndarray]   = []

    for i in range(PROBE_FRAMES):
        ret, raw = cap.read()
        if not ret:
            print(f"Frame {i}: video ended early")
            break
        raw = _rotate_frame(raw, rotation)
        tilt_deg, line_full = _detect_upright(raw)
        angles.append(tilt_deg)
        annotated_frames.append(_annotate_frame(raw, i, tilt_deg, line_full))

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
