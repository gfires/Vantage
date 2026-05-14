"""
metrics.py — Per-rep coaching metrics computed from frames_data.

All functions take frames_data (list of per-frame dicts or None, as returned
by pose._extract_landmarks) and operate on the selected side.
"""

import math

import numpy as np

from params import (
    DESCENT_FAST_S,
    DESCENT_SLOW_S,
    GRIND_RATIO,
    HOLE_EXIT_FRACTION,
    TIBIAL_NOTE_DEG,
    TIBIAL_WARN_DEG,
    HOLE_MCV_WARN,
)
from pose import estimated_markers, CameraCalibration


def compute_tempo(
    rep: dict, frames_data: list, side: str, fps: float,
    cal: "CameraCalibration | None" = None,
) -> dict:
    """
    Compute phase durations and concentric velocity profile for one rep.

    Phases:
      descent  — rep start → bottom frame
      ascent   — bottom frame → rep end

    Velocity profile is computed over the ascent only: hip-crease Y displacement
    per frame, normalised by frame height, negated so rising = positive.

    Returns:
        descent_s       float   descent duration in seconds
        ascent_s        float   ascent duration in seconds
        velocity        list[float]   per-frame normalised velocity during ascent
                                      (length = ascent_frames - 1)
        sticking_pct    int     0–100, where in the ascent velocity is lowest
                                (within first third — the diagnostic window)
        flags           list[str]   any out-of-range labels
    """
    start  = rep["start_global"]
    bottom = rep["bottom_global"]
    end    = rep["end_global"]

    descent_frames = max(bottom - start, 1)
    ascent_frames  = max(end - bottom, 1)
    descent_s = descent_frames / fps
    ascent_s  = ascent_frames  / fps

    # Hip-crease Y over the ascent window (bottom → end, inclusive)
    frame_height = None
    hc_ys = []
    for i in range(bottom, end + 1):
        f = frames_data[i] if i < len(frames_data) else None
        if f is None:
            hc_ys.append(float("nan"))
            continue
        if frame_height is None:
            frame_height = f["height"]
        hc_y, _, _, _ = estimated_markers(f, side, cal)
        hc_ys.append(hc_y)

    if frame_height is None or len(hc_ys) < 2:
        return {
            "descent_s": round(descent_s, 2),
            "ascent_s":  round(ascent_s, 2),
            "velocity":  [],
            "sticking_pct": None,
            "flags": [],
        }

    hc_arr = np.array(hc_ys, dtype=float)
    # Rising = Y decreasing → negate so positive = moving upward
    velocity = (-np.diff(hc_arr) * fps / frame_height).tolist()

    # ── Velocity quantification ───────────────────────────────────────────────
    valid_v = [v for v in velocity if not math.isnan(v)]

    # Mean concentric velocity: average over full ascent
    mean_concentric_vel = round(float(np.mean(valid_v)), 4) if valid_v else None

    # Hole-exit velocity: mean over first HOLE_EXIT_FRACTION of ascent
    hole_exit_n = max(1, int(len(velocity) * HOLE_EXIT_FRACTION))
    hole_exit_vals = [v for v in velocity[:hole_exit_n] if not math.isnan(v)]
    hole_exit_vel = round(float(np.mean(hole_exit_vals)), 4) if hole_exit_vals else None

    # ── Coaching flags ────────────────────────────────────────────────────────
    flags = []

    if descent_s < DESCENT_FAST_S:
        flags.append("FAST DESC")
    elif descent_s > DESCENT_SLOW_S:
        flags.append("SLOW DESC")
    if ascent_s > descent_s * GRIND_RATIO:
        flags.append("GRIND")

    # Hole-exit quality: HOLE as fraction of MCV
    hole_mcv_ratio = None
    if hole_exit_vel is not None and mean_concentric_vel and mean_concentric_vel > 1e-6:
        hole_mcv_ratio = round(hole_exit_vel / mean_concentric_vel, 3)
        if hole_mcv_ratio < HOLE_MCV_WARN:
            flags.append("WEAK HOLE")

    return {
        "descent_s":           round(descent_s, 2),
        "ascent_s":            round(ascent_s, 2),
        "velocity":            [round(v, 4) if not math.isnan(v) else None for v in velocity],
        "mean_concentric_vel": mean_concentric_vel,
        "hole_exit_vel":       hole_exit_vel,
        "hole_mcv_ratio":      hole_mcv_ratio,
        "flags":               flags,
    }


def compute_tibial_angle(rep: dict, frames_data: list, side: str) -> dict:
    """
    Compute shin angle from vertical for each frame in the rep.

    Tibial angle = atan2(|knee_x - heel_x|, |knee_y - heel_y|) in degrees.
    0° = perfectly vertical shin.  Increases as knee travels forward over toe.

    Returns:
        angles      dict[int, float]   frame_idx → angle in degrees (None frames omitted)
        max_angle   float              peak angle across the rep
        max_frame   int                frame index of peak angle
        flagged     bool               True if max_angle exceeded TIBIAL_WARN_DEG
    """
    start = rep["start_global"]
    end   = rep["end_global"]

    angles = {}
    for i in range(start, end + 1):
        f = frames_data[i] if i < len(frames_data) else None
        if f is None:
            continue
        heel = f[f"{side}_heel"]
        knee = f[f"{side}_knee"]
        dx = abs(knee[0] - heel[0])
        dy = abs(heel[1] - knee[1])   # heel lower than knee → positive
        angle = math.degrees(math.atan2(dx, max(dy, 1e-6)))
        angles[i] = round(angle, 1)

    if not angles:
        return {"angles": {}, "max_angle": None, "max_frame": None, "flagged": False}

    max_frame = max(angles, key=lambda k: angles[k])
    max_angle = angles[max_frame]
    return {
        "angles":    angles,
        "max_angle": max_angle,
        "max_frame": max_frame,
        "flagged":   max_angle > TIBIAL_WARN_DEG,
    }


def compute_depth_angle(
    rep: dict, frames_data: list, side: str,
    cal: "CameraCalibration | None" = None,
) -> float | None:
    """
    Compute the best depth angle sustained for >= MIN_DEPTH_FRAMES consecutive frames.

    Scans all frames in the rep, finds runs where hip crease is below knee top
    (hc_y > kt_y in screen coords) for at least MIN_DEPTH_FRAMES consecutive frames,
    and returns the most positive (deepest) angle within that qualifying window.

    Convention:
      0 deg    = hip crease level with knee top (parallel)
      positive = hip crease below knee top (depth achieved)
      negative = hip crease above knee top (not at depth)

    Falls back to the single best angle across the rep if no qualifying run exists.
    Returns None if no valid frames.
    """
    from params import MIN_DEPTH_FRAMES as _MDF

    start = rep["start_global"]
    end   = rep["end_global"]

    def _angle_for_frame(f):
        hc_y, kt_y, hc_x, kt_x = estimated_markers(f, side, cal)
        # Angle of the hc→kt line against horizontal.
        # Use the full 2D distance as denominator so dx≈0 never blows up.
        # Sign: positive = hc below kt (depth achieved), negative = above.
        dx = kt_x - hc_x
        dy = kt_y - hc_y   # positive when kt is lower than hc (not at depth)
        dist = math.hypot(dx, dy)
        # Signed angle from horizontal: negative means hc is above kt (depth),
        # but we want positive = depth, so flip: use hc_y - kt_y as the rise.
        rise = hc_y - kt_y   # positive in screen coords = hc lower = depth achieved
        angle = math.degrees(math.asin(max(-1.0, min(1.0, rise / dist)))) if dist > 1e-6 else 0.0
        return angle, (hc_y > kt_y)

    frame_angles = []   # list of (frame_idx, angle, at_depth)
    for i in range(start, end + 1):
        f = frames_data[i] if i < len(frames_data) else None
        if f is None:
            continue
        angle, at_depth = _angle_for_frame(f)
        frame_angles.append((i, angle, at_depth))

    if not frame_angles:
        return None

    # Find best angle within any run of >= MIN_DEPTH_FRAMES consecutive at-depth frames
    best_in_run = None
    run_len = 0
    run_best = None
    for _, angle, at_depth in frame_angles:
        if at_depth:
            run_len += 1
            run_best = angle if run_best is None else max(run_best, angle)
            if run_len >= _MDF:
                best_in_run = run_best if best_in_run is None else max(best_in_run, run_best)
        else:
            run_len = 0
            run_best = None

    if best_in_run is not None:
        return round(best_in_run, 1)

    # Fallback: best single-frame angle across the whole rep
    return round(max(a for _, a, _ in frame_angles), 1)


def compute_flags(tempo: dict, tibial: dict) -> list[str]:
    """
    Derive all coaching flags for a completed rep from its tempo and tibial dicts.

    Single source of truth for flag logic — replaces the duplicate _rep_warnings
    functions previously in api.py and visualize.py.

    Checks performed:
      - Descent speed: FAST DESC / SLOW DESC
      - Ascent/descent ratio: GRIND
      - Hole-exit velocity vs MCV: WEAK HOLE
      - Peak tibial angle: KNEES TOO FORWARD / KNEES SLIGHTLY FORWARD

    Args:
        tempo:  dict as returned by compute_tempo() or _build_tempo().
                Keys used: descent_s, ascent_s, hole_mcv_ratio.
        tibial: dict as returned by compute_tibial_angle() or _build_tibial().
                Keys used: max_angle.

    Returns:
        List of flag strings, may be empty.  Order: tempo flags first, tibial last.
    """
    flags = []

    descent_s = tempo.get("descent_s") or 0.0
    ascent_s  = tempo.get("ascent_s")  or 0.0

    if descent_s < DESCENT_FAST_S:
        flags.append("FAST DESC")
    elif descent_s > DESCENT_SLOW_S:
        flags.append("SLOW DESC")

    if descent_s > 0 and ascent_s > descent_s * GRIND_RATIO:
        flags.append("GRIND")

    hole_mcv_ratio = tempo.get("hole_mcv_ratio")
    if hole_mcv_ratio is not None and hole_mcv_ratio < HOLE_MCV_WARN:
        flags.append("WEAK HOLE")

    max_tib = tibial.get("max_angle")
    if max_tib is not None:
        if max_tib > TIBIAL_WARN_DEG:
            flags.append("KNEES TOO FORWARD")
        elif max_tib > TIBIAL_NOTE_DEG:
            flags.append("KNEES SLIGHTLY FORWARD")

    return flags


def compute_back_angle(frames_data: list, side: str) -> dict:
    """
    Compute back angle relative to vertical across all valid frames.

    The back angle is the angle between the shoulder-to-hip vector and the
    vertical axis. 0° = perfectly upright. Larger angle = more forward lean.

    Returns:
        min_angle   float   most upright position (degrees)
        max_angle   float   most forward lean (degrees)
        peak_frame  int     frame index of greatest lean
        angles      list[float]   per-frame angles for graphing
    """
    shoulder_key = f"{side}_shoulder"
    hip_key      = f"{side}_hip"

    valid = [(i, f) for i, f in enumerate(frames_data) if f is not None]
    if not valid:
        return {"min_angle": None, "max_angle": None, "peak_frame": None, "angles": []}

    angles = []
    for _, f in valid:
        sx, sy = f[shoulder_key]
        hx, hy = f[hip_key][:2]
        dx = sx - hx
        dy = sy - hy   # negative when shoulder is above hip (normal)
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))
        angles.append(angle)

    min_angle = min(angles)
    max_angle = max(angles)
    peak_frame_local  = int(np.argmax(angles))
    peak_frame_global = valid[peak_frame_local][0]

    return {
        "min_angle":  round(min_angle, 1),
        "max_angle":  round(max_angle, 1),
        "peak_frame": peak_frame_global,
        "angles":     [round(a, 1) for a in angles],
    }
