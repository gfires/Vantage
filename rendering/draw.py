"""
draw.py — Frame annotation primitives for the WhiteLights squat analyzer.

All _draw_* functions operate on a BGR numpy frame in-place.  They have no
side effects beyond modifying the frame array and never read or write video
files.  Import from here in both the two-pass _render path and the single-pass
_process_video path.

Coordinate helpers (_pt, _estimated_marker_ys, *_coords) are also here since
they exist purely to support drawing.
"""

import math

import cv2
import numpy as np

from params import (
    CLOSE_THRESHOLD,
    DRAW_SMOOTHING,
    HIP_CREASE_FRAC,
    HOLE_EXIT_FRACTION,
    HOLE_MCV_NOTE,
    KNEE_TOP_OVERSHOOT,
    TIBIAL_NOTE_DEG,
    TIBIAL_WARN_DEG,
)

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
WHITE   = (255, 255, 255)
GREEN   = (80, 230, 0)
YELLOW  = (0, 220, 255)
MAGENTA = (255, 0, 255)
RED     = (60, 60, 220)
DARK    = (20, 20, 20)
GRAY    = (160, 160, 160)
CYAN    = (200, 200, 0)    # velocity sparkline, HOLE/MCV labels
PURPLE  = (210, 0, 220)    # tibial angle annotation

GRAPH_FRAMES = 90   # frames shown in scrolling graph


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _pt(landmark) -> tuple:
    """Convert (x, y, ...) landmark to integer (x, y) point."""
    return (int(landmark[0]), int(landmark[1]))


def _estimated_marker_ys(fdata, side) -> tuple:
    """
    Return ((hc_y, kt_y), (hc_x, kt_x)) using the tuned anatomical offsets.
    Used for both depth detection logic and drawing.
    hc_y > kt_y in screen coords → depth achieved.
    """
    heel     = fdata[f"{side}_heel"]
    knee     = fdata[f"{side}_knee"]
    shoulder = fdata[f"{side}_shoulder"]
    hip      = fdata[f"{side}_hip"]

    kt_y = heel[1] + (knee[1] - heel[1]) * (1.0 + KNEE_TOP_OVERSHOOT)
    kt_x = heel[0] + (knee[0] - heel[0]) * (1.0 + KNEE_TOP_OVERSHOOT)

    hc_y = shoulder[1] + (hip[1] - shoulder[1]) * HIP_CREASE_FRAC
    hc_x = shoulder[0] + (hip[0] - shoulder[0]) * HIP_CREASE_FRAC

    return (hc_y, kt_y), (hc_x, kt_x)


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
    label = f"REP {rep_num}/{total_reps}" if (total_reps is not None and total_reps > 1) else f"REP {rep_num}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    PAD = 12
    x = frame_w - tw - PAD
    y = frame_h - PAD
    return x, y, tw, th, baseline, label


def _side_badge_coords(frame_w, frame_h, side):
    """Return (x0, y0, x1, y1, tx, ty, label) for the side badge box and text anchor."""
    label = f"SIDE: {side.upper()}" if side else "SIDE: ?"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    PAD = 12
    x1 = frame_w - PAD
    y1 = frame_h - PAD - 30
    x0 = x1 - tw - 10
    y0 = y1 - th - baseline - 6
    tx = x0 + 5
    ty = y1 - baseline - 2
    return x0, y0, x1, y1, tx, ty, label


def _metrics_hud_coords(frame_w):
    """Return (x0, y0, w, h) for the metrics HUD panel (top-right)."""
    MW, MH = 240, 96
    PAD = 12
    x0 = frame_w - MW - PAD
    y0 = PAD
    return x0, y0, MW, MH


def _coaching_panel_coords(frame_w):
    """Return (x0, y0, w, h) for the coaching insight panel below the metrics HUD."""
    x0, y0, mw, mh = _metrics_hud_coords(frame_w)
    GAP = 4
    CW, CH = mw, 72
    return x0, y0 + mh + GAP, CW, CH


# ── Draw functions ────────────────────────────────────────────────────────────

def _draw_skeleton(frame, fdata, side, depth_active, near_depth, tibial_angle=None, is_bottom=False):
    """Draw joint connections and circles for the selected side."""
    hip      = _pt(fdata[f"{side}_hip"])
    knee     = _pt(fdata[f"{side}_knee"])
    shoulder = _pt(fdata[f"{side}_shoulder"])
    heel     = _pt(fdata[f"{side}_heel"])

    (hc_y, kt_y), (hc_x, kt_x) = _estimated_marker_ys(fdata, side)
    hip_crease = (int(hc_x), int(hc_y))
    knee_top   = (int(kt_x), int(kt_y))

    if depth_active:
        marker_color = GREEN
    elif near_depth:
        marker_color = YELLOW
    else:
        marker_color = WHITE

    cv2.line(frame, shoulder, hip, GRAY, 2, cv2.LINE_AA)
    cv2.line(frame, knee, heel, GRAY, 2, cv2.LINE_AA)
    cv2.line(frame, hip, knee, GRAY, 2, cv2.LINE_AA)

    for pt, radius in [(shoulder, 6), (knee, 7), (heel, 5), (hip, 8)]:
        cv2.circle(frame, pt, radius, GRAY, -1, cv2.LINE_AA)

    MARKER_DOT = (60, 60, 60)
    cv2.line(frame, hip_crease, knee_top, marker_color, 2, cv2.LINE_AA)
    cv2.circle(frame, knee_top,   6, MARKER_DOT,  -1, cv2.LINE_AA)
    cv2.circle(frame, knee_top,   6, marker_color,  1, cv2.LINE_AA)
    cv2.circle(frame, hip_crease, 6, MARKER_DOT,  -1, cv2.LINE_AA)
    cv2.circle(frame, hip_crease, 6, marker_color,  1, cv2.LINE_AA)

    if is_bottom:
        cv2.circle(frame, hip_crease, 14, MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(frame, "BOTTOM", (hip_crease[0] + 16, hip_crease[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 1, cv2.LINE_AA)

    if tibial_angle is not None:
        tib_color = YELLOW if tibial_angle > TIBIAL_WARN_DEG else PURPLE

        shin_len = math.hypot(knee[0] - heel[0], knee[1] - heel[1])
        ref_len  = max(int(shin_len * 0.55), 20)
        ref_top  = (heel[0], heel[1] - ref_len)
        cv2.line(frame, heel, ref_top, GRAY, 1, cv2.LINE_AA)

        arc_r   = max(int(shin_len * 0.30), 14)
        shin_dx = knee[0] - heel[0]
        angle_rad = math.atan2(abs(shin_dx), max(abs(knee[1] - heel[1]), 1e-6))
        angle_deg = math.degrees(angle_rad)

        start_angle_cv = -90
        if shin_dx >= 0:
            end_angle_cv = -90 + angle_deg
        else:
            end_angle_cv = -90 - angle_deg

        cv2.ellipse(frame, heel, (arc_r, arc_r), 0,
                    min(start_angle_cv, end_angle_cv),
                    max(start_angle_cv, end_angle_cv),
                    tib_color, 1, cv2.LINE_AA)

        label    = f"{tibial_angle:.0f}deg"
        offset_x = 10 if shin_dx >= 0 else -40
        cv2.putText(frame, label, (heel[0] + offset_x, heel[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, tib_color, 1, cv2.LINE_AA)


def _draw_backgrounds(overlay, frame_w, frame_h, rep_num, total_reps, cur_rep):
    """
    Paint all semi-transparent dark backing rects onto overlay.
    Called once per frame before the single addWeighted blend.
    No opaque content (circles, text, lines) here.
    """
    PAD_L, PAD_B = 12, 12
    GW, GH = 300, 100

    cv2.rectangle(overlay, (PAD_L, frame_h - PAD_B - GH),
                  (PAD_L + GW, frame_h - PAD_B), DARK, -1)

    box_x0, box_x1, box_y0, box_y1, _, _, _ = _lights_box_coords(frame_h)
    cv2.rectangle(overlay, (box_x0, box_y0), (box_x1, box_y1), DARK, -1)

    if rep_num is not None:
        x, y, tw, th, baseline, _ = _rep_counter_coords(frame_w, frame_h, rep_num, total_reps)
        cv2.rectangle(overlay, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 6), DARK, -1)

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


def _draw_side_badge(frame, side, frame_w, frame_h):
    """Black box with white text in the bottom-right showing which side is being tracked."""
    x0, y0, x1, y1, tx, ty, label = _side_badge_coords(frame_w, frame_h, side)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


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

    if in_descent:
        stage_label, stage_color = "[ DESCENT ]", WHITE
    elif in_hole:
        stage_label, stage_color = "[ HOLE    ]", CYAN
    else:
        stage_label, stage_color = "[ ASCENT  ]", GREEN
    cv2.putText(frame, stage_label, (x0 + 6, y0 + lh), font, small, stage_color, 1, cv2.LINE_AA)

    desc_str = f"DESC {desc_s:.1f}s" if desc_s is not None else "DESC --"
    asc_str  = f"ASC  {asc_s:.1f}s"  if asc_s  is not None else "ASC  --"
    cv2.putText(frame, desc_str, (x0 + 6, y0 + lh * 2), font, small, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, asc_str,  (x0 + 6, y0 + lh * 3), font, small, WHITE, 1, cv2.LINE_AA)

    if not in_descent:
        if hole_v is not None:
            cv2.putText(frame, f"HOLE  {hole_v:.3f} fh/s", (x0 + 6, y0 + lh * 4), font, small, CYAN, 1, cv2.LINE_AA)
        if mean_v is not None:
            cv2.putText(frame, f"MCV   {mean_v:.3f} fh/s", (x0 + 6, y0 + lh * 5), font, small, CYAN, 1, cv2.LINE_AA)


def _draw_coaching_panel(frame, cur_rep, frame_idx, frame_w):
    """
    Coaching insight panel directly below the metrics HUD.
    Shows tempo flags and tibial warning. Only shown inside a rep, past the bottom.
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

    rows = [(f, YELLOW) for f in flags]

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

    smooth_hip_ys and knee_ys may be either full lists (two-pass path, sliced
    internally) or deque(maxlen=GRAPH_FRAMES) objects (single-pass path, used
    as-is with current_local=None to signal "use entire contents").
    """
    PAD_L, PAD_B = 12, 12
    GW, GH = 300, 100
    x0 = PAD_L
    y0 = frame_h - PAD_B - GH

    if current_local is None:
        # Single-pass path: smooth_hip_ys / knee_ys are already-windowed deques
        hip_slice  = list(smooth_hip_ys)
        knee_slice = list(knee_ys)
    else:
        # Two-pass path: full arrays, slice the last GRAPH_FRAMES ending at current_local
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
