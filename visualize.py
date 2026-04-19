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
)
from params import (
    CLOSE_THRESHOLD,
    MIN_DEPTH_FRAMES,
    SMOOTHING_WINDOW,
    KNEE_TOP_OVERSHOOT,
    HIP_CREASE_FRAC,
    REP_SMOOTHING,
    MIN_REP_FRAMES,
    MIN_DESCENT_THRESHOLD,
    DRAW_SMOOTHING,
    TIBIAL_NOTE_DEG,
    TIBIAL_WARN_DEG,
    HOLE_MCV_NOTE,
    HOLE_EXIT_FRACTION,
)
from metrics import compute_tempo, compute_tibial_angle, compute_depth_angle

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
WHITE   = (255, 255, 255)
GREEN   = (80, 230, 0)
YELLOW  = (0, 220, 255)
MAGENTA = (255, 0, 255)
RED     = (60, 60, 220)
DARK    = (20, 20, 20)
GRAY    = (160, 160, 160)

GRAPH_FRAMES = 90   # frames shown in scrolling graph

CYAN   = (200, 200, 0)    # BGR cyan — velocity sparkline, HOLE/MCV labels
PURPLE = (210, 0, 220)    # BGR magenta-pink — tibial angle annotation

# ── Output toggles ────────────────────────────────────────────────────────────
SAVE_VIDEO  = True   # write annotated MP4 alongside input, then auto-open it
SHOW_LIVE   = True   # display frames in a cv2 window as they are rendered



# ── Rep table ─────────────────────────────────────────────────────────────────

def _rep_warnings(rep: dict) -> str:
    """Collect all warnings for a rep: tempo flags + tibial threshold breaches."""
    warns = list(rep["tempo"].get("flags", []))
    max_tib = rep["tibial"].get("max_angle")
    if max_tib is not None:
        if max_tib > TIBIAL_WARN_DEG:
            warns.append("KNEES TOO FORWARD")
        elif max_tib > TIBIAL_NOTE_DEG:
            warns.append("KNEES SLIGHTLY FORWARD")
    return ", ".join(warns) or "--"


def _output_rep_table(reps: list, output_path: str) -> None:
    """
    Write a plain-text rep summary table to output_path.

    Columns: one per rep.
    Rows:
      Result        — PASS / FAIL / BORDERLINE
      Descent time  — full eccentric phase, in seconds
      Hole time     — first 25% of ascent, in seconds
      Ascent time   — full concentric phase, in seconds
      Depth angle   — hip-crease→knee-top angle vs horizontal at bottom (deg)
                      negative = below parallel, positive = above
      Max shin      — peak tibial angle during the rep (deg)
      Warnings      — coaching flags
    """
    n = len(reps)
    if n == 0:
        return

    # ── Build cell data ───────────────────────────────────────────────────────
    def _hole_s(rep):
        t = rep.get("tempo", {})
        bottom = rep["bottom_global"]
        end    = rep["end_global"]
        asc_total = max(end - bottom, 1)
        hole_frames = max(1, int(asc_total * HOLE_EXIT_FRACTION))
        fps_approx = asc_total / max(t.get("ascent_s", 1) or 1, 1e-6)
        hole_s = hole_frames / fps_approx
        return f"{hole_s:.2f}s"

    rows = {
        "Result":       [rep["result"].upper()                                         for rep in reps],
        "Descent time": [f"{rep['tempo'].get('descent_s', 0):.2f}s"                   for rep in reps],
        "Hole time":    [_hole_s(rep)                                                  for rep in reps],
        "Ascent time":  [f"{rep['tempo'].get('ascent_s', 0):.2f}s"                    for rep in reps],
        "Depth angle": [
            (f"{rep['depth_angle']:+.1f}deg" if rep.get("depth_angle") is not None else "--")
            for rep in reps
        ],
        "Max shin":    [
            (f"{rep['tibial']['max_angle']:.0f}deg" if rep["tibial"].get("max_angle") is not None else "--")
            for rep in reps
        ],
        "Warnings":    [
            _rep_warnings(rep) for rep in reps
        ],
    }

    # ── Column widths ─────────────────────────────────────────────────────────
    rep_headers = [f"Rep {i}" for i in range(1, n + 1)]
    label_w = max(len(k) for k in rows)
    col_ws   = [
        max(len(rep_headers[i]), *(len(rows[k][i]) for k in rows))
        for i in range(n)
    ]

    def _row(label, cells):
        return f"  {label:<{label_w}}  " + "  ".join(
            f"{c:^{col_ws[i]}}" for i, c in enumerate(cells)
        )

    sep = "  " + "-" * label_w + "  " + "  ".join("-" * w for w in col_ws)

    lines = []
    lines.append("  " + " " * label_w + "  " + "  ".join(
        f"{h:^{col_ws[i]}}" for i, h in enumerate(rep_headers)
    ))
    lines.append(sep)
    for label, cells in rows.items():
        lines.append(_row(label, cells))
    lines.append("")

    text = "\n".join(lines)
    print("\n" + text)
    with open(output_path, "w") as f:
        f.write(text + "\n")


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
    analysis = _analyze(cap, rotation, fps, force_side)
    cap.release()

    if analysis is None:
        print("ERROR: Could not detect pose or squat bottom. Check camera angle.")
        sys.exit(1)

    frames_data, draw_frames, side, reps, smooth_hip_ys, knee_ys, valid_frame_indices = analysis

    for i, rep in enumerate(reps, 1):
        t = rep["tempo"]
        tib = rep["tibial"]
        print(
            f"  Rep {i}: {rep['result'].upper():12}"
            f"  start={rep['start_global']}  bottom={rep['bottom_global']}  end={rep['end_global']}"
            f"  desc={t['descent_s']:.1f}s  asc={t['ascent_s']:.1f}s"
            f"  hole_v={t['hole_exit_vel']:.3f}  mcv={t['mean_concentric_vel']:.3f}"
            f"  shin_max={tib['max_angle']:.0f}"
            + (f"  [{' '.join(t['flags'])}]" if t["flags"] else "")
        )
    table_path = annotated_dir / (path.stem + "_table.txt")
    _output_rep_table(reps, str(table_path))
    print(f"  Table:  {table_path}")

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
            # Fall back to original if smoothing propagated NaN from a nearby None frame
            if sx != sx:  # NaN check
                sx = f[j][0]
            if sy != sy:
                sy = f[j][1]
            if j in joints_vis:
                df[j] = (sx, sy, f[j][2])   # preserve visibility
            else:
                df[j] = (sx, sy)
        draw_frames.append(df)

    return draw_frames


def _analyze(cap, rotation, fps, force_side=None):
    """
    Run landmark extraction + per-rep depth judgment + coaching metrics.

    Returns:
        (frames_data, draw_frames, side, reps, smooth_hip_ys, knee_ys, valid_frame_indices)
        draw_frames: smoothed copy of frames_data for skeleton/marker rendering only.
        reps: list of {result, bottom_global, start_global, end_global,
                       tempo, tibial, bar_path}
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
            rep["tempo"] = compute_tempo(rep, frames_data, side, fps)
            reps.append(rep)

    if not reps:
        return None

    # Smooth landmarks before computing tibial angles so the displayed arc
    # and angle value are derived from the same 3-frame averaged coordinates.
    draw_frames = _smooth_landmarks_for_drawing(frames_data)
    for rep in reps:
        rep["tibial"]      = compute_tibial_angle(rep, draw_frames, side)
        rep["depth_angle"] = compute_depth_angle(rep, draw_frames, side)

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
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    delay_ms = max(1, int(1000 / fps))  # for SHOW_LIVE playback pacing

    # Build global→local index map for graph lookups
    global_to_local = {g: l for l, g in enumerate(valid_frame_indices)}


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
        _draw_backgrounds(overlay, w, h, rep_num, len(reps), cur_rep)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Pass 2: all opaque content directly onto the blended frame.
        # fdata      → depth state logic (unsmoothed, authoritative)
        # fdata_draw → skeleton/trail/markers (smoothed, display only)
        if fdata is not None:
            frame_h = fdata["height"]

            (hc_y, kt_y), _ = _estimated_marker_ys(fdata, side)
            depth_active = hc_y > kt_y
            near_depth   = abs(hc_y - kt_y) < (frame_h * CLOSE_THRESHOLD)
            tib_angle  = cur_rep["tibial"]["angles"].get(frame_idx) if cur_rep else None
            is_bottom  = cur_rep is not None and frame_idx == cur_rep["bottom_global"]
            _draw_skeleton(frame, fdata_draw, side, depth_active, near_depth, tib_angle, is_bottom)
            _draw_graph(frame, smooth_hip_ys, knee_ys, local_idx, h)
            _draw_metrics_hud(frame, cur_rep, frame_idx, w)
            _draw_coaching_panel(frame, cur_rep, frame_idx, w)
            _draw_lights(frame, cur_rep, frame_idx, h)
            _draw_rep_counter(frame, rep_num, len(reps), w, h)
        else:
            _draw_graph(frame, smooth_hip_ys, knee_ys, None, h)
            _draw_metrics_hud(frame, cur_rep, frame_idx, w)
            _draw_coaching_panel(frame, cur_rep, frame_idx, w)
            _draw_lights(frame, cur_rep, frame_idx, h)
            _draw_rep_counter(frame, rep_num, len(reps), w, h)

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

def _draw_skeleton(frame, fdata, side, depth_active, near_depth, tibial_angle=None, is_bottom=False):
    """Draw joint connections and circles for the selected side."""
    hip      = _pt(fdata[f"{side}_hip"])
    knee     = _pt(fdata[f"{side}_knee"])
    shoulder = _pt(fdata[f"{side}_shoulder"])
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
    for pt, radius in [(shoulder, 6), (knee, 7), (heel, 5), (hip, 8)]:
        cv2.circle(frame, pt, radius, GRAY, -1, cv2.LINE_AA)

    # ── Estimated anatomical markers ─────────────────────────────────────────
    MARKER_DOT = (60, 60, 60)   # dark fill, easy to distinguish from joints

    # Line between the two markers — color-coded by depth state
    cv2.line(frame, hip_crease, knee_top, marker_color, 2, cv2.LINE_AA)

    cv2.circle(frame, knee_top,   6, MARKER_DOT,    -1, cv2.LINE_AA)
    cv2.circle(frame, knee_top,   6, marker_color,   1, cv2.LINE_AA)
    cv2.circle(frame, hip_crease, 6, MARKER_DOT,    -1, cv2.LINE_AA)
    cv2.circle(frame, hip_crease, 6, marker_color,   1, cv2.LINE_AA)

    # ── Bottom frame marker ───────────────────────────────────────────────────
    if is_bottom:
        cv2.circle(frame, hip_crease, 14, MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(frame, "BOTTOM", (hip_crease[0] + 16, hip_crease[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 1, cv2.LINE_AA)

    # ── Tibial angle annotation (active reps only) ────────────────────────────
    if tibial_angle is not None:
        import math as _math
        tib_color = YELLOW if tibial_angle > TIBIAL_WARN_DEG else PURPLE

        # Shin vector length (heel→knee) for scaling the reference line
        shin_len = _math.hypot(knee[0] - heel[0], knee[1] - heel[1])
        ref_len  = max(int(shin_len * 0.55), 20)   # reference line = 55% of shin

        # Vertical reference line upward from heel
        ref_top = (heel[0], heel[1] - ref_len)
        cv2.line(frame, heel, ref_top, GRAY, 1, cv2.LINE_AA)

        # Arc between vertical reference and shin, centered at heel
        arc_r = max(int(shin_len * 0.30), 14)

        # Shin direction angle from vertical (cv2 angles: 0° = right, CCW)
        shin_dx = knee[0] - heel[0]
        shin_dy = -(knee[1] - heel[1])   # flip Y for math coords
        shin_from_vertical = _math.degrees(_math.atan2(shin_dx, _math.hypot(shin_dx, shin_dy) - shin_dy + 1e-6))

        # Simpler: just use atan2 from vertical axis
        angle_rad = _math.atan2(abs(shin_dx), max(abs(knee[1] - heel[1]), 1e-6))
        angle_deg = _math.degrees(angle_rad)

        # cv2.ellipse angles: 0° = 3-o'clock direction, measured clockwise
        # Vertical (upward from heel) = -90°.
        # Shin leans toward the camera side — determine which direction
        start_angle_cv = -90
        if shin_dx >= 0:
            end_angle_cv = -90 + angle_deg
        else:
            end_angle_cv = -90 - angle_deg

        cv2.ellipse(frame, heel, (arc_r, arc_r), 0,
                    min(start_angle_cv, end_angle_cv),
                    max(start_angle_cv, end_angle_cv),
                    tib_color, 1, cv2.LINE_AA)

        # Angle label near the heel
        label = f"{tibial_angle:.0f}deg"
        offset_x = 10 if shin_dx >= 0 else -40
        label_pt = (heel[0] + offset_x, heel[1] + 20)
        cv2.putText(frame, label, label_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, tib_color, 1, cv2.LINE_AA)



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


def _metrics_hud_coords(frame_w):
    """Return (x0, y0, w, h) for the metrics HUD panel (top-right).
    Rows: DESC, ASC, HOLE, MCV, STICK = 5 rows × 16px + 8px padding = 88px."""
    MW, MH = 240, 96
    PAD = 12
    x0 = frame_w - MW - PAD
    y0 = PAD
    return x0, y0, MW, MH


def _coaching_panel_coords(frame_w):
    """Return (x0, y0, w, h) for the coaching insight panel directly below the metrics HUD."""
    x0, y0, mw, mh = _metrics_hud_coords(frame_w)
    GAP = 4
    CW, CH = mw, 72   # up to 4 coaching rows × 16px + 8px padding
    return x0, y0 + mh + GAP, CW, CH


def _draw_backgrounds(overlay, frame_w, frame_h, rep_num, total_reps, cur_rep):
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

    # Metrics HUD + coaching panel backgrounds (top-right) — only when inside a rep
    if cur_rep is not None:
        x0, y0, mw, mh = _metrics_hud_coords(frame_w)
        cv2.rectangle(overlay, (x0, y0), (x0 + mw, y0 + mh), DARK, -1)
        cx0, cy0, cw, ch = _coaching_panel_coords(frame_w)
        cv2.rectangle(overlay, (cx0, cy0), (cx0 + cw, cy0 + ch), DARK, -1)





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



def _draw_metrics_hud(frame, cur_rep, frame_idx, frame_w):
    """
    Top-right panel: raw tempo and velocity numbers.
    Rows: DESC, ASC, HOLE, MCV, STICK.
    Only shown when inside a rep.
    """
    if cur_rep is None:
        return

    x0, y0, _, _ = _metrics_hud_coords(frame_w)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    small = 0.45
    lh    = 16

    tempo  = cur_rep.get("tempo", {})
    desc_s = tempo.get("descent_s")
    asc_s  = tempo.get("ascent_s")
    hole_v = tempo.get("hole_exit_vel")
    mean_v = tempo.get("mean_concentric_vel")

    bottom    = cur_rep.get("bottom_global", 0)
    end       = cur_rep.get("end_global", bottom)
    asc_total = max(end - bottom, 1)
    hole_end  = bottom + max(1, int(asc_total * HOLE_EXIT_FRACTION))

    in_descent = frame_idx < bottom
    in_hole    = bottom <= frame_idx < hole_end

    # Stage label: live indicator of which phase we're in
    if in_descent:
        stage_label, stage_color = "[ DESCENT ]", WHITE
    elif in_hole:
        stage_label, stage_color = "[ HOLE    ]", CYAN
    else:
        stage_label, stage_color = "[ ASCENT  ]", GREEN
    cv2.putText(frame, stage_label, (x0 + 6, y0 + lh), font, small, stage_color, 1, cv2.LINE_AA)

    # DESC / ASC times — always available
    desc_str = f"DESC {desc_s:.1f}s" if desc_s is not None else "DESC --"
    asc_str  = f"ASC  {asc_s:.1f}s"  if asc_s  is not None else "ASC  --"
    cv2.putText(frame, desc_str, (x0 + 6, y0 + lh * 2), font, small, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, asc_str,  (x0 + 6, y0 + lh * 3), font, small, WHITE, 1, cv2.LINE_AA)

    # HOLE / MCV — shown once past the bottom
    if not in_descent:
        if hole_v is not None:
            cv2.putText(frame, f"HOLE  {hole_v:.3f} fh/s", (x0 + 6, y0 + lh * 4), font, small, CYAN, 1, cv2.LINE_AA)
        if mean_v is not None:
            cv2.putText(frame, f"MCV   {mean_v:.3f} fh/s", (x0 + 6, y0 + lh * 5), font, small, CYAN, 1, cv2.LINE_AA)


def _draw_coaching_panel(frame, cur_rep, frame_idx, frame_w):
    """
    Coaching insight panel directly below the metrics HUD.
    Shows tempo flags (FAST DESC, SLOW DESC, GRIND) and velocity-derived
    coaching cues (WEAK HOLE, EARLY STICK, LATE STICK, tibial warning).
    Only shown when inside a rep and past the bottom.
    """
    if cur_rep is None:
        return

    bottom = cur_rep.get("bottom_global", 0)
    if frame_idx < bottom:
        return

    tempo  = cur_rep.get("tempo", {})
    tibial = cur_rep.get("tibial", {})
    flags  = tempo.get("flags", [])
    max_tib = tibial.get("max_angle")

    # Build coaching rows: (text, color)
    rows = []
    for f in flags:
        rows.append((f, YELLOW))

    if max_tib is not None:
        if max_tib > TIBIAL_WARN_DEG:
            rows.append(("KNEE TOO FORWARD", YELLOW))
        elif max_tib > TIBIAL_NOTE_DEG:
            rows.append(("WATCH KNEES", WHITE))

    hole_mcv = tempo.get("hole_mcv_ratio")
    if hole_mcv is not None and hole_mcv < HOLE_MCV_NOTE:
        pct = int(hole_mcv * 100)
        rows.append((f"HOLE {pct}% of MCV", WHITE))

    if not rows:
        return

    x0, y0, _, _ = _coaching_panel_coords(frame_w)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    small = 0.45
    lh    = 16
    for i, (text, color) in enumerate(rows):
        cv2.putText(frame, text, (x0 + 6, y0 + lh * (i + 1)), font, small, color, 1, cv2.LINE_AA)



def _draw_graph(frame, smooth_hip_ys, knee_ys, current_local, frame_h):
    """
    Bottom-left scrolling line chart: hip_y (white) vs knee_y (yellow dashed).
    Green fill where depth is active.
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
