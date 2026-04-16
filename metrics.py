"""
Per-lift metrics: back angle and bar path.
Both are computed from the frames_data list returned by depth_detector._extract_landmarks.
"""

import math
import numpy as np


def compute_back_angle(frames_data: list, side: str) -> dict:
    """
    Compute back angle relative to vertical across all valid frames.

    The back angle is the angle between the shoulder-to-hip vector and the
    vertical axis. 0° = perfectly upright. Larger angle = more forward lean.

    Returns:
        min_angle: float (most upright position, degrees)
        max_angle: float (most forward lean, degrees)
        peak_frame: int (frame index where lean was greatest)
        angles: list[float] (per-frame angles for graphing)
    """
    shoulder_key = f"{side}_shoulder"
    hip_key = f"{side}_hip"

    valid = [(i, f) for i, f in enumerate(frames_data) if f is not None]
    if not valid:
        return {"min_angle": None, "max_angle": None, "peak_frame": None, "angles": []}

    angles = []
    for _, f in valid:
        sx, sy = f[shoulder_key]
        hx, hy = f[hip_key][:2]

        # Vector from hip to shoulder
        dx = sx - hx
        dy = sy - hy  # negative = shoulder is above hip (y increases downward)

        # Angle from vertical: arctan(horizontal / vertical component)
        # dy is negative when upright (shoulder above hip), so we negate
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))
        angles.append(angle)

    min_angle = min(angles)
    max_angle = max(angles)
    peak_frame_local = int(np.argmax(angles))
    peak_frame_global = valid[peak_frame_local][0]

    return {
        "min_angle": round(min_angle, 1),
        "max_angle": round(max_angle, 1),
        "peak_frame": peak_frame_global,
        "angles": [round(a, 1) for a in angles],
    }


def compute_bar_path(frames_data: list) -> dict:
    """
    Track bar position (midpoint of both wrists) across all valid frames.
    Returns lateral drift as % of frame width — no pixel-to-cm calibration needed.

    Returns:
        lateral_drift_pct: float (max horizontal deviation from start, % of frame width)
        path: list[tuple[float, float]] (normalized x, y per frame — for overlay)
    """
    valid = [(i, f) for i, f in enumerate(frames_data) if f is not None]
    if not valid:
        return {"lateral_drift_pct": None, "path": []}

    path = []
    for _, f in valid:
        lx, ly = f["left_wrist"]
        rx, ry = f["right_wrist"]
        mid_x = (lx + rx) / 2
        mid_y = (ly + ry) / 2
        w = f["width"]
        h = f["height"]
        path.append((mid_x / w, mid_y / h))  # normalized 0–1

    if not path:
        return {"lateral_drift_pct": None, "path": []}

    start_x = path[0][0]
    max_drift = max(abs(p[0] - start_x) for p in path)
    lateral_drift_pct = round(max_drift * 100, 1)

    return {
        "lateral_drift_pct": lateral_drift_pct,
        "path": path,
    }
