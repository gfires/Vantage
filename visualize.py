"""
visualize.py — Squat depth overlay tool.

Renders an annotated MP4 with:
  - Skeleton overlay (shoulder, hip, knee, wrist) for the detected side
  - Hip→knee segment color-coded by depth state (white / yellow / green)
  - Magenta ring on the bottom frame's hip joint
  - HUD text: DEPTH ✓ / NO DEPTH + PASS/FAIL/BORDERLINE badge
  - Spatial hip trail (last 60 frames, fading dots, green when depth active)
  - Scrolling graph HUD (bottom-left): hip_y vs knee_y over last 90 frames

Usage:
    python visualize.py path/to/video.MOV
    → writes path/to/video_annotated.mp4
"""

import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from depth_detector import (
    _ensure_model,
    _get_rotation,
    _rotate_frame,
    _extract_landmarks,
    _select_side,
    _rolling_average,
    _find_bottom_frame,
    CLOSE_THRESHOLD,
    DEPTH_WINDOW,
    MIN_DEPTH_FRAMES,
    SMOOTHING_WINDOW,
)

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
WHITE   = (255, 255, 255)
GREEN   = (80, 230, 0)
YELLOW  = (0, 220, 255)
MAGENTA = (255, 0, 255)
RED     = (60, 60, 220)
DARK    = (20, 20, 20)
GRAY    = (160, 160, 160)

TRAIL_LEN    = 60   # frames of hip position history to show
GRAPH_FRAMES = 90   # frames shown in scrolling graph

# ── Output toggles ────────────────────────────────────────────────────────────
SAVE_VIDEO  = True   # write annotated MP4 alongside input, then auto-open it
SHOW_LIVE   = True   # display frames in a cv2 window as they are rendered

# ── Estimated anatomical marker tuning ───────────────────────────────────────
# Knee-top marker: heel→knee vector, extended this fraction *past* the knee.
# 0.10 = 10% of heel-to-knee distance above the knee joint center.
KNEE_TOP_OVERSHOOT   = 0.18

# Hip-crease marker: shoulder→hip vector, this fraction of the way from shoulder.
# 0.90 = 90% along shoulder→hip (i.e. 10% above the hip joint center).
HIP_CREASE_FRAC      = 0.88


# ── Main entry ────────────────────────────────────────────────────────────────

def main():
    # Usage: visualize.py <video> [--side left|right]
    args = sys.argv[1:]
    force_side = None
    if "--side" in args:
        idx = args.index("--side")
        force_side = args[idx + 1].lower()
        args = args[:idx] + args[idx + 2:]
    video_path = args[0] if args else "tests/videos/valid_1.MOV"

    path = Path(video_path)
    output_path = path.parent / (path.stem + "_annotated.mp4")

    print(f"\nInput:  {path}")
    print(f"Output: {output_path}")
    if force_side:
        print(f"Side:   {force_side} (forced)")

    _ensure_model()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open {path}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # prevent double-rotate on macOS

    rotation = _get_rotation(cap)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print("Pass 1: extracting pose landmarks...")
    analysis = _analyze(cap, rotation, force_side)
    cap.release()

    if analysis is None:
        print("ERROR: Could not detect pose or squat bottom. Check camera angle.")
        sys.exit(1)

    frames_data, side, bottom_global, result, smooth_hip_ys, knee_ys, valid_frame_indices = analysis

    print(f"  Side: {side} | Bottom frame: {bottom_global} | Result: {result.upper()}")
    print("Pass 2: rendering annotated video...")

    cap2 = cv2.VideoCapture(str(path))
    cap2.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # prevent double-rotate on macOS
    _render(cap2, rotation, fps, frames_data, side, bottom_global, result,
            smooth_hip_ys, knee_ys, valid_frame_indices, str(output_path))
    cap2.release()

    print("\nDone.")


# ── Pass 1: analysis ──────────────────────────────────────────────────────────

def _analyze(cap, rotation, force_side=None):
    """
    Run landmark extraction + depth judgment.

    Returns:
        (frames_data, side, bottom_global, result,
         smooth_hip_ys, knee_ys, valid_frame_indices)
        or None on failure.
    """
    frames_data = _extract_landmarks(cap, rotation)

    valid_frames = [(i, f) for i, f in enumerate(frames_data) if f is not None]
    if not valid_frames:
        return None

    if force_side in ("left", "right"):
        side = force_side
    else:
        side = _select_side(valid_frames)
    if side is None:
        return None

    hip_key  = f"{side}_hip"
    knee_key = f"{side}_knee"

    valid_frame_indices = [i for i, _ in valid_frames]
    hip_ys  = [f[hip_key][1]  for _, f in valid_frames]
    knee_ys = [f[knee_key][1] for _, f in valid_frames]
    smooth_hip_ys = _rolling_average(hip_ys, SMOOTHING_WINDOW)

    bottom_local = _find_bottom_frame(smooth_hip_ys)
    if bottom_local is None:
        return None

    bottom_global = valid_frame_indices[bottom_local]
    bottom_data   = valid_frames[bottom_local][1]
    hip_y_bottom  = bottom_data[hip_key][1]
    knee_y_bottom = bottom_data[knee_key][1]
    frame_height  = bottom_data["height"]

    # Use estimated anatomical markers for depth judgment in the visualizer
    def _marker_depth_flag(f):
        (hc_y, kt_y), _ = _estimated_marker_ys(f, side)
        return hc_y > kt_y

    all_depth_flags = [_marker_depth_flag(f) for _, f in valid_frames]
    max_consec = _max_consecutive_true(all_depth_flags)

    # Borderline: closest the estimated markers got to each other
    def _marker_gap(f):
        (hc_y, kt_y), _ = _estimated_marker_ys(f, side)
        return abs(hc_y - kt_y)

    min_gap = min(_marker_gap(f) for _, f in valid_frames)
    close = min_gap < (frame_height * CLOSE_THRESHOLD)

    if max_consec >= MIN_DEPTH_FRAMES:
        result = "pass"
    elif close:
        result = "borderline"
    else:
        result = "fail"

    return frames_data, side, bottom_global, result, smooth_hip_ys, knee_ys, valid_frame_indices


# ── Pass 2: rendering ─────────────────────────────────────────────────────────

def _render(cap, rotation, fps, frames_data, side, bottom_global, result,
            smooth_hip_ys, knee_ys, valid_frame_indices, output_path):
    """
    Re-read the video frame-by-frame, draw all overlays.
    Writes annotated MP4 if SAVE_VIDEO is True.
    Shows a live cv2 window if SHOW_LIVE is True.
    Auto-opens the saved file after render if SAVE_VIDEO is True.
    """
    # Read one frame to get dimensions
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first = cap.read()
    if not ret:
        print("ERROR: Could not read first frame for dimensions.")
        return
    first = _rotate_frame(first, rotation)
    h, w = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    delay_ms = max(1, int(1000 / fps))  # for SHOW_LIVE playback pacing

    hip_key = f"{side}_hip"

    # Build global→local index map for graph lookups
    global_to_local = {g: l for l, g in enumerate(valid_frame_indices)}

    trail = deque(maxlen=TRAIL_LEN)          # each entry: ((x, y), depth_active)
    depth_history = deque(maxlen=TRAIL_LEN)  # parallel bool list

    total = len(frames_data)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = _rotate_frame(frame, rotation)
        fdata = frames_data[frame_idx] if frame_idx < len(frames_data) else None

        if fdata is not None:
            hip_x  = fdata[hip_key][0]
            hip_y  = fdata[hip_key][1]
            frame_h = fdata["height"]

            (hc_y, kt_y), _ = _estimated_marker_ys(fdata, side)
            depth_active = hc_y > kt_y
            near_depth   = abs(hc_y - kt_y) < (frame_h * CLOSE_THRESHOLD)
            is_bottom    = (frame_idx == bottom_global)

            trail.append(((int(hip_x), int(hip_y)), depth_active))

            # ── Draw overlays onto an overlay copy for alpha blending ──
            overlay = frame.copy()

            _draw_trail(overlay, trail)
            _draw_skeleton(overlay, fdata, side, depth_active, near_depth, is_bottom)

            # blend overlay with original
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            local_idx = global_to_local.get(frame_idx)
            _draw_graph(frame, smooth_hip_ys, knee_ys, local_idx, w, h)
            _draw_hud_text(frame, depth_active, near_depth, is_bottom, result, frame_idx, bottom_global)
        else:
            # No pose detected — write raw frame, still draw graph if possible
            _draw_graph(frame, smooth_hip_ys, knee_ys, None, w, h)

        if SAVE_VIDEO:
            out.write(frame)

        if SHOW_LIVE:
            cv2.imshow("Squat Depth", frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break

        frame_idx += 1

        if not SHOW_LIVE and frame_idx % 30 == 0:
            print(f"  frame {frame_idx}/{total}", end="\r", flush=True)

    if not SHOW_LIVE:
        print(f"  frame {frame_idx}/{total}")

    if SAVE_VIDEO:
        out.release()
        print(f"  Saved: {output_path}")
        subprocess.run(["open", output_path])

    if SHOW_LIVE:
        cv2.destroyAllWindows()


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_skeleton(frame, fdata, side, depth_active, near_depth, is_bottom):
    """Draw joint connections and circles for the selected side."""
    hip      = _pt(fdata[f"{side}_hip"])
    knee     = _pt(fdata[f"{side}_knee"])
    shoulder = _pt(fdata[f"{side}_shoulder"])
    wrist    = _pt(fdata[f"{side}_wrist"])
    heel     = _pt(fdata[f"{side}_heel"])

    (hc_y, kt_y), (hc_x, kt_x) = _estimated_marker_ys(fdata, side)
    hip_crease = (int(hc_x), int(hc_y))
    knee_top   = (int(kt_x), int(kt_y))

    # Depth-state color for the hip→knee segment
    if depth_active:
        seg_color = GREEN
    elif near_depth:
        seg_color = YELLOW
    else:
        seg_color = WHITE

    # Torso line (shoulder → hip)
    cv2.line(frame, shoulder, hip, GRAY, 2, cv2.LINE_AA)

    # Shin line (knee → heel)
    cv2.line(frame, knee, heel, GRAY, 2, cv2.LINE_AA)

    # Critical segment: hip → knee
    cv2.line(frame, hip, knee, seg_color, 3, cv2.LINE_AA)

    # Joint circles
    for pt, color, radius in [
        (shoulder, GRAY,      6),
        (knee,     seg_color, 7),
        (heel,     GRAY,      5),
        (wrist,    GRAY,      5),
    ]:
        cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)

    # Hip circle (larger, color-coded)
    cv2.circle(frame, hip, 8, seg_color, -1, cv2.LINE_AA)

    # Bottom frame: magenta ring around hip
    if is_bottom:
        cv2.circle(frame, hip, 18, MAGENTA, 3, cv2.LINE_AA)

    # ── Estimated anatomical markers ─────────────────────────────────────────
    MARKER_COLOR = (60, 60, 60)   # dark dot, easy to distinguish from joints

    # Line between the two markers — color-coded same as depth state
    cv2.line(frame, hip_crease, knee_top, seg_color, 2, cv2.LINE_AA)

    cv2.circle(frame, knee_top,   6, MARKER_COLOR, -1, cv2.LINE_AA)
    cv2.circle(frame, knee_top,   6, WHITE,         1, cv2.LINE_AA)
    cv2.circle(frame, hip_crease, 6, MARKER_COLOR, -1, cv2.LINE_AA)
    cv2.circle(frame, hip_crease, 6, WHITE,         1, cv2.LINE_AA)


def _draw_hud_text(frame, depth_active, near_depth, is_bottom, result, frame_idx, bottom_global):
    """Top-left depth state text + top-right result badge."""
    h, w = frame.shape[:2]

    # ── Depth state (top-left) ──
    if depth_active:
        label = "DEPTH  +"
        color = GREEN
    elif near_depth:
        label = "BORDERLINE"
        color = YELLOW
    else:
        label = "NO DEPTH"
        color = WHITE

    _draw_label(frame, label, (16, 40), color, scale=1.0, thickness=2)

    if is_bottom:
        _draw_label(frame, "BOTTOM", (16, 78), MAGENTA, scale=0.7, thickness=2)

    # ── Overall result (top-right), shown after bottom is known ──
    if frame_idx >= bottom_global:
        result_color = {
            "pass":        GREEN,
            "fail":        RED,
            "borderline":  YELLOW,
            "indeterminate": GRAY,
        }.get(result, WHITE)
        _draw_label(frame, result.upper(), (w - 160, 40), result_color, scale=1.0, thickness=2)


def _draw_trail(frame, trail):
    """Draw fading hip position trail. Older dots are smaller and dimmer."""
    n = len(trail)
    for i, ((x, y), depth_active) in enumerate(trail):
        age_ratio = i / max(n - 1, 1)       # 0 = oldest, 1 = newest
        radius = max(2, int(2 + age_ratio * 6))
        alpha  = 0.15 + 0.85 * age_ratio    # fade older dots

        color = GREEN if depth_active else WHITE
        # Dim color by alpha (blend toward black)
        dimmed = tuple(int(c * alpha) for c in color)

        cv2.circle(frame, (x, y), radius, dimmed, -1, cv2.LINE_AA)


def _draw_graph(frame, smooth_hip_ys, knee_ys, current_local, frame_w, frame_h):
    """
    Bottom-left scrolling line chart: hip_y (white) vs knee_y (yellow).
    Green fill between curves when hip_y > knee_y.
    """
    PAD_L, PAD_B = 12, 12
    GW, GH = 300, 100   # panel width, height
    x0 = PAD_L
    y0 = frame_h - PAD_B - GH

    # Semi-transparent dark background
    bg = frame.copy()
    cv2.rectangle(bg, (x0, y0), (x0 + GW, y0 + GH), DARK, -1)
    cv2.addWeighted(bg, 0.6, frame, 0.4, 0, frame)

    if current_local is None or len(smooth_hip_ys) < 2:
        return

    # Window of frames to display
    end_local   = current_local + 1
    start_local = max(0, end_local - GRAPH_FRAMES)
    hip_slice   = smooth_hip_ys[start_local:end_local]
    knee_slice  = knee_ys[start_local:end_local]

    if len(hip_slice) < 2:
        return

    all_vals = hip_slice + knee_slice
    y_min, y_max = min(all_vals), max(all_vals)
    y_range = max(y_max - y_min, 1.0)

    def to_px(val):
        norm = (val - y_min) / y_range           # 0=top of data, 1=bottom
        # In image coords, larger y = lower on screen; squat goes down = larger y
        # We want "deeper" (larger hip_y) to appear lower in the graph panel too
        gy = y0 + int(norm * (GH - 4)) + 2
        return gy

    n = len(hip_slice)
    def to_gx(i):
        return x0 + 2 + int(i / max(n - 1, 1) * (GW - 4))

    # Build point lists
    hip_pts  = [(to_gx(i), to_px(v)) for i, v in enumerate(hip_slice)]
    knee_pts = [(to_gx(i), to_px(v)) for i, v in enumerate(knee_slice)]

    # Green fill where hip_y > knee_y (depth active)
    for i in range(n - 1):
        h1, h2 = hip_pts[i][1],  hip_pts[i + 1][1]
        k1, k2 = knee_pts[i][1], knee_pts[i + 1][1]
        if hip_slice[i] > knee_slice[i] or hip_slice[i + 1] > knee_slice[i + 1]:
            pts = np.array([
                hip_pts[i], hip_pts[i + 1],
                knee_pts[i + 1], knee_pts[i],
            ], dtype=np.int32)
            fill_layer = frame.copy()
            cv2.fillPoly(fill_layer, [pts], (0, 100, 30))
            cv2.addWeighted(fill_layer, 0.4, frame, 0.6, 0, frame)

    # Draw knee_y (yellow dashed)
    for i in range(n - 1):
        if i % 4 < 2:  # simple dash: draw every other segment
            cv2.line(frame, knee_pts[i], knee_pts[i + 1], YELLOW, 1, cv2.LINE_AA)

    # Draw hip_y (white solid)
    for i in range(n - 1):
        cv2.line(frame, hip_pts[i], hip_pts[i + 1], WHITE, 1, cv2.LINE_AA)

    # Vertical cursor at current frame (right edge)
    cx = to_gx(n - 1)
    cv2.line(frame, (cx, y0 + 2), (cx, y0 + GH - 2), GREEN, 1, cv2.LINE_AA)

    # Panel border
    cv2.rectangle(frame, (x0, y0), (x0 + GW, y0 + GH), GRAY, 1)


# ── Utility ───────────────────────────────────────────────────────────────────

def _pt(landmark) -> tuple:
    """Convert (x, y, ...) landmark to integer (x, y) point."""
    return (int(landmark[0]), int(landmark[1]))


def _estimated_marker_ys(fdata, side) -> tuple:
    """
    Return (hip_crease_y, knee_top_y) as floats using the tuned offsets.
    Used for both depth detection logic and drawing.
    """
    heel     = fdata[f"{side}_heel"]
    knee     = fdata[f"{side}_knee"]
    shoulder = fdata[f"{side}_shoulder"]
    hip      = fdata[f"{side}_hip"]

    # knee_top: heel→knee direction, overshot by KNEE_TOP_OVERSHOOT
    kt_y = heel[1] + (knee[1] - heel[1]) * (1.0 + KNEE_TOP_OVERSHOOT)
    kt_x = heel[0] + (knee[0] - heel[0]) * (1.0 + KNEE_TOP_OVERSHOOT)

    # hip_crease: HIP_CREASE_FRAC of the way from shoulder to hip
    hc_y = shoulder[1] + (hip[1] - shoulder[1]) * HIP_CREASE_FRAC
    hc_x = shoulder[0] + (hip[0] - shoulder[0]) * HIP_CREASE_FRAC

    return (hc_y, kt_y), (hc_x, kt_x)


def _draw_label(frame, text, origin, color, scale=0.8, thickness=2):
    """Draw text with a dark semi-transparent background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    pad = 4
    bg = frame.copy()
    cv2.rectangle(bg, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), DARK, -1)
    cv2.addWeighted(bg, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _max_consecutive_true(flags):
    max_count = count = 0
    for f in flags:
        count = count + 1 if f else 0
        max_count = max(max_count, count)
    return max_count


if __name__ == "__main__":
    main()
