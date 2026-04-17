"""
metrics.py — Per-rep coaching metrics computed from frames_data.

All functions take frames_data (list of per-frame dicts or None, as returned
by depth_detector._extract_landmarks) and operate on the selected side.
"""

import math

import numpy as np

from params import (
    HIP_CREASE_FRAC,
    DESCENT_FAST_S,
    DESCENT_SLOW_S,
    GRIND_RATIO,
    HOLE_EXIT_FRACTION,
    TIBIAL_WARN_DEG,
)


def compute_tempo(rep: dict, frames_data: list, side: str, fps: float) -> dict:
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
        shoulder = f[f"{side}_shoulder"]
        hip      = f[f"{side}_hip"]
        hc_y = shoulder[1] + (hip[1] - shoulder[1]) * HIP_CREASE_FRAC
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

    # Sticking point: argmin velocity in the first third of the ascent
    third = max(1, len(velocity) // 3)
    sticking_pct = int(np.nanargmin(velocity[:third]) / max(len(velocity), 1) * 100) if valid_v else None

    flags = []
    if descent_s < DESCENT_FAST_S:
        flags.append("FAST DESC")
    elif descent_s > DESCENT_SLOW_S:
        flags.append("SLOW DESC")
    if ascent_s > descent_s * GRIND_RATIO:
        flags.append("GRIND")

    return {
        "descent_s":          round(descent_s, 2),
        "ascent_s":           round(ascent_s, 2),
        "velocity":           [round(v, 4) if not math.isnan(v) else None for v in velocity],
        "mean_concentric_vel": mean_concentric_vel,   # normalised frame-heights/s
        "hole_exit_vel":       hole_exit_vel,          # mean vel over first ~15% of ascent
        "sticking_pct":        sticking_pct,
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
