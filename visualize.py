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
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

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

# ── Drawing smoothing ─────────────────────────────────────────────────────────
DRAW_SMOOTHING = 3    # rolling average window for skeleton/marker drawing coords only
                      # does NOT affect depth classification — display only

# ── Rep segmentation tuning ───────────────────────────────────────────────────
REP_SMOOTHING  = 15   # smoothing window for hip-crease height signal
MIN_REP_FRAMES = 15   # min frames between standing peaks (also min segment length)
# A segment is only counted as a rep if the hip-crease dropped by at least this
# fraction of frame height from the standing peak. Rejects hip-hinge warmup
# movements that never approach squat depth.
MIN_DESCENT_THRESHOLD = 0.10  # 10% of frame height


# ── Main entry ────────────────────────────────────────────────────────────────

def main():
    # Usage: visualize.py <video> [--side left|right]
    args = sys.argv[1:]
    force_side = None
    if "--side" in args:
        idx = args.index("--side")
        force_side = args[idx + 1].lower()
        args = args[:idx] + args[idx + 2:]
    video_path = args[0] if args else "tests/raw_videos/valid_1.MOV"

    path = Path(video_path)
    annotated_dir = path.parent.parent / "annotated_videos"
    annotated_dir.mkdir(exist_ok=True)
    output_path = annotated_dir / (path.stem + "_annotated.mp4")

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

    frames_data, draw_frames, side, reps, smooth_hip_ys, knee_ys, valid_frame_indices = analysis

    for i, rep in enumerate(reps, 1):
        print(f"  Rep {i}: {rep['result'].upper():12} (bottom frame {rep['bottom_global']})")
    print("Pass 2: rendering annotated video...")

    cap2 = cv2.VideoCapture(str(path))
    cap2.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # prevent double-rotate on macOS
    _render(cap2, rotation, fps, frames_data, draw_frames, side, reps,
            smooth_hip_ys, knee_ys, valid_frame_indices, str(output_path))
    cap2.release()

    print("\nDone.")


# ── Pass 1: analysis ──────────────────────────────────────────────────────────

def _segment_reps(valid_frames, side):
    """
    Segment valid_frames into individual reps using hip-crease Y relative to heel Y.

    Signal: (heel_y - hip_crease_y) — positive and large when standing tall,
    small (near zero or negative) at squat bottom. We find peaks of this signal
    (standing positions) using prominence + distance constraints, then use those
    peaks as rep boundaries.

    A state-machine confirmation pass rejects segments that never actually
    descended: the signal must drop by at least MIN_DESCENT_THRESHOLD from the
    peak into a true valley before the rep boundary is accepted.

    Returns list of (start_local, end_local) index pairs into valid_frames.
    Falls back to a single segment covering the whole video if no boundaries found.
    """
    # Build the height signal: heel_y - hip_crease_y
    # (larger = standing taller; smaller = deeper squat)
    height_signal = []
    for _, f in valid_frames:
        heel_y = f[f"{side}_heel"][1]
        (hc_y, _), _ = _estimated_marker_ys(f, side)
        height_signal.append(heel_y - hc_y)

    smooth = np.array(_rolling_average(height_signal, REP_SMOOTHING))
    n = len(smooth)

    if n < MIN_REP_FRAMES:
        return [(0, n)]

    frame_height = valid_frames[0][1]["height"]

    # Peaks of the height signal = standing positions = rep boundaries.
    # prominence: must rise at least 8% of frame height from surrounding valley
    # distance: peaks must be at least MIN_REP_FRAMES apart
    prominence_threshold = frame_height * 0.08
    peaks, _ = find_peaks(
        smooth,
        prominence=prominence_threshold,
        distance=MIN_REP_FRAMES,
    )

    # Wrap detected peaks with sentinel boundaries at start and end so that
    # the first rep (before the first standing peak) and last rep (after the
    # last standing peak) are not dropped.
    if len(peaks) >= 1:
        boundary_locals = [0] + list(peaks) + [n]
    else:
        # No peaks at all — fall back to whole video
        return [(0, n)]

    # State machine confirmation: for each candidate boundary pair, verify the
    # signal actually descended by at least MIN_DESCENT_THRESHOLD somewhere
    # inside the segment. Uses the segment's own max as the standing reference
    # so edge segments (which start at sentinel 0, not a real peak) are judged
    # fairly.
    descent_threshold = frame_height * MIN_DESCENT_THRESHOLD
    segments = []
    for start, end in zip(boundary_locals, boundary_locals[1:]):
        if end - start < MIN_REP_FRAMES:
            continue
        seg = smooth[start:end]
        peak_val = float(seg.max())
        valley_val = float(seg.min())
        if (peak_val - valley_val) >= descent_threshold:
            segments.append((start, end))

    if not segments:
        segments = [(0, n)]

    return segments


def _classify_segment(seg_frames, side, frame_height):
    """
    Run depth classification on a single rep's frames.
    Returns dict: {result, bottom_global, start_global, end_global}
    """
    if not seg_frames:
        return None

    hip_key  = f"{side}_hip"
    knee_key = f"{side}_knee"

    indices  = [i for i, _ in seg_frames]
    hip_ys   = [f[hip_key][1] for _, f in seg_frames]
    smooth_h = _rolling_average(hip_ys, SMOOTHING_WINDOW)

    bottom_local = _find_bottom_frame(smooth_h)
    if bottom_local is None:
        return None
    bottom_global = indices[bottom_local]

    def _depth_flag(f):
        (hc_y, kt_y), _ = _estimated_marker_ys(f, side)
        return hc_y > kt_y

    depth_flags = [_depth_flag(f) for _, f in seg_frames]
    max_consec  = _max_consecutive_true(depth_flags)

    gaps = [abs(_estimated_marker_ys(f, side)[0][0] - _estimated_marker_ys(f, side)[0][1])
            for _, f in seg_frames]
    min_gap = min(gaps)

    # Reject segments that never approached parallel — not a real rep
    if min_gap > frame_height * MIN_DESCENT_THRESHOLD:
        return None

    close = min_gap < (frame_height * CLOSE_THRESHOLD)

    if max_consec >= MIN_DEPTH_FRAMES:
        result = "pass"
    elif close:
        result = "borderline"
    else:
        result = "fail"

    return {
        "result":        result,
        "bottom_global": bottom_global,
        "start_global":  indices[0],
        "end_global":    indices[-1],
    }


def _smooth_landmarks_for_drawing(frames_data):
    """
    Return a parallel list to frames_data with x/y coordinates smoothed by
    DRAW_SMOOTHING frames for display only. None entries are preserved.
    Visibility scores are copied from the original (not smoothed).
    Classification logic always uses frames_data, never draw_frames.
    """
    joints_xy = [
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_shoulder", "right_shoulder", "left_heel", "right_heel",
        "left_wrist", "right_wrist",
    ]
    # Joints that carry a visibility third element
    joints_vis = {"left_hip", "right_hip", "left_knee", "right_knee"}

    # Build per-joint x/y arrays (None frames contribute NaN so edges stay clean)
    coords = {}
    for j in joints_xy:
        xs = [f[j][0] if f is not None else float("nan") for f in frames_data]
        ys = [f[j][1] if f is not None else float("nan") for f in frames_data]
        coords[j] = (
            _rolling_average(xs, DRAW_SMOOTHING),
            _rolling_average(ys, DRAW_SMOOTHING),
        )

    draw_frames = []
    for i, f in enumerate(frames_data):
        if f is None:
            draw_frames.append(None)
            continue
        df = dict(f)  # shallow copy — width/height/frame_idx unchanged
        for j in joints_xy:
            sx, sy = coords[j][0][i], coords[j][1][i]
            if j in joints_vis:
                df[j] = (sx, sy, f[j][2])   # preserve visibility
            else:
                df[j] = (sx, sy)
        draw_frames.append(df)

    return draw_frames


def _analyze(cap, rotation, force_side=None):
    """
    Run landmark extraction + per-rep depth judgment.

    Returns:
        (frames_data, draw_frames, side, reps, smooth_hip_ys, knee_ys, valid_frame_indices)
        draw_frames: smoothed copy of frames_data for skeleton/marker rendering only.
        reps: list of {result, bottom_global, start_global, end_global}
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
    knee_key = f"{side}_knee"  # used for knee_ys below

    valid_frame_indices = [i for i, _ in valid_frames]
    hip_ys  = [f[hip_key][1]  for _, f in valid_frames]
    knee_ys = [f[knee_key][1] for _, f in valid_frames]
    smooth_hip_ys = _rolling_average(hip_ys, SMOOTHING_WINDOW)

    frame_height = valid_frames[0][1]["height"]

    segments = _segment_reps(valid_frames, side)
    reps = []
    for start_l, end_l in segments:
        rep = _classify_segment(valid_frames[start_l:end_l], side, frame_height)
        if rep is not None:
            reps.append(rep)

    if not reps:
        return None

    draw_frames = _smooth_landmarks_for_drawing(frames_data)

    return frames_data, draw_frames, side, reps, smooth_hip_ys, knee_ys, valid_frame_indices


# ── Pass 2: rendering ─────────────────────────────────────────────────────────

def _render(cap, rotation, fps, frames_data, draw_frames, side, reps,
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

    # Build global→local index map for graph lookups
    global_to_local = {g: l for l, g in enumerate(valid_frame_indices)}

    # Precompute set of bottom frames for O(1) lookup
    bottom_frames = {rep["bottom_global"] for rep in reps}

    def _current_rep(frame_idx):
        """Return (rep_index_1based, rep dict) for the rep containing frame_idx, or (None, None)."""
        for idx, rep in enumerate(reps, 1):
            if rep["start_global"] <= frame_idx <= rep["end_global"]:
                return idx, rep
        return None, None

    total = len(frames_data)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = _rotate_frame(frame, rotation)
        fdata      = frames_data[frame_idx] if frame_idx < len(frames_data) else None
        fdata_draw = draw_frames[frame_idx]  if frame_idx < len(draw_frames)  else None
        rep_num, cur_rep = _current_rep(frame_idx)

        local_idx = global_to_local.get(frame_idx)

        # Pass 1: dark backing rects only onto overlay, then one blend.
        overlay = frame.copy()
        _draw_backgrounds(overlay, frame.shape[1], h,
                          rep_num, len(reps),
                          fdata, side)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Pass 2: all opaque content directly onto the blended frame.
        # fdata      → depth state logic (unsmoothed, authoritative)
        # fdata_draw → skeleton/trail/markers (smoothed, display only)
        if fdata is not None:
            frame_h = fdata["height"]

            (hc_y, kt_y), _ = _estimated_marker_ys(fdata, side)
            depth_active = hc_y > kt_y
            near_depth   = abs(hc_y - kt_y) < (frame_h * CLOSE_THRESHOLD)
            is_bottom    = frame_idx in bottom_frames

            _draw_skeleton(frame, fdata_draw, side, depth_active, near_depth)
            _draw_graph(frame, smooth_hip_ys, knee_ys, local_idx, h)
            _draw_hud_text(frame, depth_active, near_depth, is_bottom)
            _draw_lights(frame, cur_rep, frame_idx, h)
            _draw_rep_counter(frame, rep_num, len(reps), frame.shape[1], h)
        else:
            _draw_graph(frame, smooth_hip_ys, knee_ys, None, h)
            _draw_lights(frame, cur_rep, frame_idx, h)
            _draw_rep_counter(frame, rep_num, len(reps), frame.shape[1], h)

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

def _draw_skeleton(frame, fdata, side, depth_active, near_depth):
    """Draw joint connections and circles for the selected side."""
    hip      = _pt(fdata[f"{side}_hip"])
    knee     = _pt(fdata[f"{side}_knee"])
    shoulder = _pt(fdata[f"{side}_shoulder"])
    wrist    = _pt(fdata[f"{side}_wrist"])
    heel     = _pt(fdata[f"{side}_heel"])

    (hc_y, kt_y), (hc_x, kt_x) = _estimated_marker_ys(fdata, side)
    hip_crease = (int(hc_x), int(hc_y))
    knee_top   = (int(kt_x), int(kt_y))

    # Marker line color (only the estimated-point line is color-coded)
    if depth_active:
        marker_color = GREEN
    elif near_depth:
        marker_color = YELLOW
    else:
        marker_color = WHITE

    # Torso line (shoulder → hip) — always gray
    cv2.line(frame, shoulder, hip, GRAY, 2, cv2.LINE_AA)

    # Shin line (knee → heel) — always gray
    cv2.line(frame, knee, heel, GRAY, 2, cv2.LINE_AA)

    # Hip → knee segment — always gray (depth state shown only on marker line)
    cv2.line(frame, hip, knee, GRAY, 2, cv2.LINE_AA)

    # Joint circles — all gray
    for pt, radius in [(shoulder, 6), (knee, 7), (heel, 5), (wrist, 5), (hip, 8)]:
        cv2.circle(frame, pt, radius, GRAY, -1, cv2.LINE_AA)

    # ── Estimated anatomical markers ─────────────────────────────────────────
    MARKER_DOT = (60, 60, 60)   # dark fill, easy to distinguish from joints

    # Line between the two markers — color-coded by depth state
    cv2.line(frame, hip_crease, knee_top, marker_color, 2, cv2.LINE_AA)

    cv2.circle(frame, knee_top,   6, MARKER_DOT,    -1, cv2.LINE_AA)
    cv2.circle(frame, knee_top,   6, marker_color,   1, cv2.LINE_AA)
    cv2.circle(frame, hip_crease, 6, MARKER_DOT,    -1, cv2.LINE_AA)
    cv2.circle(frame, hip_crease, 6, marker_color,   1, cv2.LINE_AA)


def _lights_box_coords(frame_h):
    """Return (box_x0, box_x1, box_y0, box_y1, r, h_pad, gap) for the lights box."""
    PAD_L, PAD_B = 12, 12
    GW, GH = 300, 100
    n, h_pad, gap = 3, 12, 12
    r = (GH - 2 * h_pad) // 2
    box_x0 = PAD_L + GW
    box_x1 = box_x0 + 2 * h_pad + n * (2 * r) + (n - 1) * gap
    box_y0 = frame_h - PAD_B - GH
    box_y1 = frame_h - PAD_B
    return box_x0, box_x1, box_y0, box_y1, r, h_pad, gap


def _rep_counter_coords(frame_w, frame_h, rep_num, total_reps):
    """Return (x, y, tw, th, baseline, label) for the rep counter text."""
    label = f"REP {rep_num}/{total_reps}" if total_reps > 1 else f"REP {rep_num}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    PAD = 12
    x = frame_w - tw - PAD
    y = frame_h - PAD
    return x, y, tw, th, baseline, label


def _draw_backgrounds(overlay, frame_w, frame_h, rep_num, total_reps, fdata, side):
    """
    Paint all semi-transparent dark backing rects onto overlay.
    Called once per frame before the single addWeighted blend.
    No opaque content (circles, text, lines) here.
    """
    PAD_L, PAD_B = 12, 12
    GW, GH = 300, 100

    # Graph panel background
    cv2.rectangle(overlay, (PAD_L, frame_h - PAD_B - GH),
                  (PAD_L + GW, frame_h - PAD_B), DARK, -1)

    # Lights box background
    box_x0, box_x1, box_y0, box_y1, _, _, _ = _lights_box_coords(frame_h)
    cv2.rectangle(overlay, (box_x0, box_y0), (box_x1, box_y1), DARK, -1)

    # Rep counter background
    if rep_num is not None:
        x, y, tw, th, baseline, _ = _rep_counter_coords(frame_w, frame_h, rep_num, total_reps)
        cv2.rectangle(overlay, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 6), DARK, -1)

    # HUD text background
    if fdata is not None:
        hip_key = f"{side}_hip"
        hc_y = fdata[f"{side}_shoulder"][1] + (fdata[hip_key][1] - fdata[f"{side}_shoulder"][1]) * HIP_CREASE_FRAC
        kt_y = fdata[f"{side}_heel"][1] + (fdata[f"{side}_knee"][1] - fdata[f"{side}_heel"][1]) * (1.0 + KNEE_TOP_OVERSHOOT)
        depth_active = hc_y > kt_y
        near_depth   = abs(hc_y - kt_y) < (fdata["height"] * CLOSE_THRESHOLD)
        label = "DEPTH  +" if depth_active else ("BORDERLINE" if near_depth else "NO DEPTH")
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(overlay, (12, 12), (12 + tw + 8, 12 + th + baseline + 8), DARK, -1)


def _draw_hud_text(frame, depth_active, near_depth, is_bottom):
    """Top-left depth state text — opaque, drawn after the background blend."""
    if depth_active:
        label, color = "DEPTH  +", GREEN
    elif near_depth:
        label, color = "BORDERLINE", YELLOW
    else:
        label, color = "NO DEPTH", WHITE
    cv2.putText(frame, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    if is_bottom:
        cv2.putText(frame, "BOTTOM", (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, MAGENTA, 2, cv2.LINE_AA)


def _draw_lights(frame, cur_rep, frame_idx, frame_h):
    """Judgment circles — drawn after blend so they appear at full brightness."""
    if cur_rep is None:
        return
    bottom_global = cur_rep["bottom_global"]
    start_global  = cur_rep["start_global"]
    end_global    = cur_rep["end_global"]
    rep_duration  = max(end_global - start_global, 1)
    if frame_idx < bottom_global:
        return
    if frame_idx < start_global + rep_duration * 0.75:
        return

    box_x0, box_x1, box_y0, box_y1, r, h_pad, gap = _lights_box_coords(frame_h)
    light_color = {"pass": WHITE, "fail": RED, "borderline": YELLOW}.get(cur_rep["result"], GRAY)
    cy = (box_y0 + box_y1) // 2
    for i in range(3):
        cx = box_x0 + h_pad + r + i * (2 * r + gap)
        cv2.circle(frame, (cx, cy), r, light_color, -1, cv2.LINE_AA)


def _draw_rep_counter(frame, rep_num, total_reps, frame_w, frame_h):
    """Rep counter text — drawn after blend at full brightness."""
    if rep_num is None:
        return
    x, y, _, _, _, label = _rep_counter_coords(frame_w, frame_h, rep_num, total_reps)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)



def _draw_graph(frame, smooth_hip_ys, knee_ys, current_local, frame_h):
    """
    Bottom-left scrolling line chart: hip_y (white) vs knee_y (yellow).
    Green fill drawn with a single fillPoly call — no per-segment frame copy.
    Background rect is handled by _draw_backgrounds before the blend.
    """
    PAD_L, PAD_B = 12, 12
    GW, GH = 300, 100
    x0 = PAD_L
    y0 = frame_h - PAD_B - GH

    if current_local is None or len(smooth_hip_ys) < 2:
        return

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
        return y0 + int((val - y_min) / y_range * (GH - 4)) + 2

    n = len(hip_slice)
    def to_gx(i):
        return x0 + 2 + int(i / max(n - 1, 1) * (GW - 4))

    hip_pts  = [(to_gx(i), to_px(v)) for i, v in enumerate(hip_slice)]
    knee_pts = [(to_gx(i), to_px(v)) for i, v in enumerate(knee_slice)]

    # Single fillPoly for all depth-active regions
    fill_polys = [
        np.array([hip_pts[i], hip_pts[i+1], knee_pts[i+1], knee_pts[i]], dtype=np.int32)
        for i in range(n - 1)
        if hip_slice[i] > knee_slice[i] or hip_slice[i+1] > knee_slice[i+1]
    ]
    if fill_polys:
        cv2.fillPoly(frame, fill_polys, (0, 180, 40))

    for i in range(n - 1):
        if i % 4 < 2:
            cv2.line(frame, knee_pts[i], knee_pts[i + 1], YELLOW, 1, cv2.LINE_AA)
    for i in range(n - 1):
        cv2.line(frame, hip_pts[i], hip_pts[i + 1], WHITE, 1, cv2.LINE_AA)

    cx = to_gx(n - 1)
    cv2.line(frame, (cx, y0 + 2), (cx, y0 + GH - 2), GREEN, 1, cv2.LINE_AA)
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



def _max_consecutive_true(flags):
    max_count = count = 0
    for f in flags:
        count = count + 1 if f else 0
        max_count = max(max_count, count)
    return max_count


if __name__ == "__main__":
    main()
