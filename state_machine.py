"""
state_machine.py — Causal per-frame rep segmentation for the single-pass pipeline.

The RepStateMachine replaces the global scipy.signal.find_peaks-based segmentation
used in the two-pass path (_segment_reps / _classify_segment in visualize.py).
It processes one frame at a time, maintaining all state internally, and emits a
completed rep dict the moment a rep ends.

Phase model:
    STANDING → DESCENDING → ASCENDING → STANDING → ...

All transitions are confirmed with a MIN_HOLD_FRAMES consecutive-frame hold to suppress
landmark jitter.  Transitions are retroactive: the event is declared to have occurred
at the frame MIN_HOLD_FRAMES ago, not at the frame where the hold count completes.
This is the pipeline delay — the render loop operates N frames behind inference.

Pipeline delay:
    PIPELINE_DELAY = SMOOTHING_WINDOW + MIN_HOLD_FRAMES
    At 30fps with default params (5 + 4 = 9 frames): ~300ms one-time startup latency.

Signal:
    Smoothed hip_y (y increases downward in screen coords).
    Large hip_y = low position (squat bottom).
    Small hip_y = high position (standing).
    Rolling average of length SMOOTHING_WINDOW maintained incrementally.

Completed rep schema (matches existing two-pass rep dict consumed by metrics, draw, API):
    {
        "result":        "pass" | "fail" | "borderline",
        "start_global":  int,   # frame index where descent began
        "bottom_global": int,   # frame index of squat bottom (retroactive)
        "end_global":    int,   # frame index where standing was re-confirmed
        "depth_flags":   list[bool],  # per-frame hc_y > kt_y across the full rep
        "tempo":         dict,  # see _build_tempo()
        "tibial":        dict,  # see _build_tibial()
        "depth_angle":   float | None,
    }
"""

import math
from collections import deque
from enum import Enum, auto

from params import (
    SMOOTHING_WINDOW,
    MIN_HOLD_FRAMES,
    PIPELINE_DELAY,
    MIN_DEPTH_FRAMES,
    MIN_DESCENT_THRESHOLD,
    ASCENT_RECOVERY_FRAC,
    CLOSE_THRESHOLD,
    HOLE_EXIT_FRACTION,
    STICK_WINDOW_START,
    STICK_WINDOW_END,
    TIBIAL_WARN_DEG,
)
from pose import _max_consecutive_true, CameraCalibration, estimated_markers
from metrics import compute_flags


# ── Phase enum ────────────────────────────────────────────────────────────────

class Phase(Enum):
    STANDING   = auto()
    DESCENDING = auto()
    ASCENDING  = auto()


# ── Geometry helpers (mirrors visualize._estimated_marker_ys and metrics) ─────


def _estimated_markers(
    fdata: dict, side: str, cal: "CameraCalibration | None" = None
) -> tuple[float, float, float, float]:
    """Thin wrapper around pose.estimated_markers for local callers."""
    return estimated_markers(fdata, side, cal)


def _tibial_angle(fdata: dict, side: str, cal: "CameraCalibration | None" = None) -> float:
    """
    Compute shin angle from vertical for one frame (degrees), corrected for camera
    roll and azimuth foreshortening.

    Step 1 — roll correction: rotate (dx, dy) by camera_roll to decompose along the
    true vertical/horizontal axes (adjusts frame of reference, not coordinates).

    Step 2 — azimuth correction: φ is measured from vertical (0°=side profile,
    90°=facing camera). The foreshortening factor is sin(φ) = cos(90°−φ), where
    90°−φ is the angle from side profile.
        tan(θ_obs) = tan(θ_true) × sin(φ)  →  θ_true = atan(tan(θ_obs) / sin(φ))

    Returns:
        Angle in degrees, ≥ 0.
    """
    roll_deg    = cal.roll_deg    if cal is not None else 0.0
    azimuth_deg = cal.azimuth_deg if cal is not None else 0.0

    heel = fdata[f"{side}_heel"]
    knee = fdata[f"{side}_knee"]
    dx = knee[0] - heel[0]
    dy = knee[1] - heel[1]

    roll_rad = math.radians(roll_deg)
    cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
    along_vert  = -dx * sin_r + dy * cos_r
    along_horiz =  dx * cos_r + dy * sin_r
    theta_obs = math.atan2(abs(along_horiz), max(abs(along_vert), 1e-6))
    theta_obs_deg = math.degrees(theta_obs)

    # phi is measured from vertical (0°=side profile, 90°=facing camera).
    # Foreshortening factor is cos(angle_from_side_profile) = cos(90° - phi) = sin(phi).
    sin_az = math.sin(math.radians(azimuth_deg))
    if sin_az > 1e-6:
        return math.degrees(math.atan(math.tan(theta_obs) / sin_az))
    return theta_obs_deg


def _depth_angle_at_frame(fdata: dict, side: str, cal: "CameraCalibration | None" = None) -> float:
    """
    Compute the signed depth angle at a single frame, corrected for camera roll
    and azimuth foreshortening.

    Convention:
        positive = hip crease below knee top (depth achieved)
        negative = hip crease above knee top (short)

    Step 1 — roll correction: decompose hc→kt along true vertical/horizontal axes.
    Step 2 — azimuth correction: foreshortening factor is sin(φ) where φ is
    measured from vertical (0°=side profile, 90°=facing camera).
        θ_true = atan(tan(θ_obs) / sin(φ))
    Sign is preserved: the correction only magnifies the absolute angle.

    Returns:
        Angle in degrees, signed.
    """
    roll_deg    = cal.roll_deg    if cal is not None else 0.0
    azimuth_deg = cal.azimuth_deg if cal is not None else 0.0

    hc_y, kt_y, hc_x, kt_x = _estimated_markers(fdata, side, cal)
    dx = hc_x - kt_x
    dy = hc_y - kt_y
    roll_rad = math.radians(roll_deg)
    cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
    along_vert  = -dx * sin_r + dy * cos_r
    along_horiz =  dx * cos_r + dy * sin_r
    theta_obs = math.atan2(along_vert, max(abs(along_horiz), 1e-6))

    sin_az = math.sin(math.radians(azimuth_deg))
    if sin_az > 1e-6:
        return math.degrees(math.atan(math.tan(theta_obs) / sin_az))
    return math.degrees(theta_obs)


# ── Metric builders ───────────────────────────────────────────────────────────

def _build_tempo(
    descent_frames: int,
    ascent_frames: int,
    ascent_hip_ys: list[float],
    frame_height: float,
    fps: float,
) -> dict:
    """
    Build the tempo dict from accumulated per-phase frame counts and raw hip
    Y positions recorded during the ascent.

    Mirrors compute_tempo() in metrics.py but operates on pre-accumulated data
    rather than doing a post-hoc pass over frames_data.

    Args:
        descent_frames:  number of frames from rep start to bottom (inclusive).
        ascent_frames:   number of frames from bottom to rep end (inclusive).
        ascent_hip_ys:    list of raw hip Y values recorded each ascent frame.
                         Length = ascent_frames - MIN_HOLD_FRAMES.
        frame_height:    full-resolution frame height in pixels (for normalisation).
        fps:             video frame rate.

    Returns:
        Dict with keys: descent_s, ascent_s, velocity, mean_concentric_vel,
        hole_exit_vel, hole_mcv_ratio, sticking_pct, sticking_vel_pct.
        Flags are intentionally absent — caller passes this to compute_flags().
    """
    descent_s = descent_frames / fps
    ascent_s  = ascent_frames  / fps

    # Per-frame normalised velocity during ascent (length = len(ascent_hip_ys) - 1).
    # hip Y decreases as the lifter rises → negate so positive = moving upward.
    # Normalised by frame_height so values are comparable across resolutions.
    if len(ascent_hip_ys) >= 2:
        velocity = [
            -(ascent_hip_ys[i] - ascent_hip_ys[i - 1]) / frame_height * fps
            for i in range(1, len(ascent_hip_ys))
        ]
    else:
        velocity = []

    valid_v = [v for v in velocity if v == v]  # exclude any NaN placeholders

    mean_concentric_vel = round(sum(valid_v) / len(valid_v), 4) if valid_v else None

    # Hole-exit velocity: mean over first HOLE_EXIT_FRACTION of ascent frames.
    hole_exit_n    = max(1, int(len(velocity) * HOLE_EXIT_FRACTION))
    hole_exit_vals = [v for v in velocity[:hole_exit_n] if v == v]  # filter NaN from window
    hole_exit_vel  = round(sum(hole_exit_vals) / len(hole_exit_vals), 4) if hole_exit_vals else None

    hole_mcv_ratio = None
    if hole_exit_vel is not None and mean_concentric_vel and mean_concentric_vel > 1e-6:
        hole_mcv_ratio = round(hole_exit_vel / mean_concentric_vel, 3)

    # Sticking point: velocity minimum in the 25%–80% ascent window.
    # Use a 3-frame centred average to suppress single-frame jitter before finding the min.
    sticking_pct = None
    sticking_vel_pct = None
    n = len(velocity)
    if n >= 2 and mean_concentric_vel and mean_concentric_vel > 1e-6:
        w_start = int(n * STICK_WINDOW_START)
        w_end   = int(n * STICK_WINDOW_END)
        hw = 8  # half-window → 21-frame centred average
        smoothed = [
            sum(velocity[max(0, i - hw):min(n, i + hw + 1)]) / len(velocity[max(0, i - hw):min(n, i + hw + 1)])
            for i in range(n)
        ]
        # Search only within [w_start, w_end] — smoothing reads freely from full list
        window = [(i, smoothed[i]) for i in range(w_start, w_end)]
        if window:
            stick_idx, stick_vel_smooth = min(window, key=lambda x: x[1])
            sticking_pct     = round(stick_idx / n * 100)
            sticking_vel_pct = round(stick_vel_smooth / mean_concentric_vel * 100)

    return {
        "descent_s":           round(descent_s, 2),
        "ascent_s":            round(ascent_s, 2),
        "velocity":            [round(v, 4) for v in velocity],
        "mean_concentric_vel": mean_concentric_vel,
        "hole_exit_vel":       hole_exit_vel,
        "hole_mcv_ratio":      hole_mcv_ratio,
        "sticking_pct":        sticking_pct,
        "sticking_vel_pct":    sticking_vel_pct,
        # flags intentionally absent — caller passes this dict to compute_flags()
        # which owns all flag logic and returns them separately.
    }


def _build_tibial(
    per_frame_angles: dict[int, float],
) -> dict:
    """
    Build the tibial dict from per-frame angles accumulated during the rep.

    Mirrors compute_tibial_angle() in metrics.py but uses pre-accumulated data.

    Args:
        per_frame_angles: mapping of frame_idx → tibial angle (degrees),
                          populated for every valid frame in the rep.

    Returns:
        Dict with keys: angles, max_angle, max_frame, flagged.
        Matches the schema returned by metrics.compute_tibial_angle().
    """
    if not per_frame_angles:
        return {"angles": {}, "max_angle": None, "max_frame": None, "flagged": False}

    max_frame = max(per_frame_angles, key=lambda k: per_frame_angles[k])
    max_angle = per_frame_angles[max_frame]
    return {
        "angles":    {k: round(v, 1) for k, v in per_frame_angles.items()},
        "max_angle": max_angle,
        "max_frame": max_frame,
        "flagged":   max_angle > TIBIAL_WARN_DEG,
    }


def _build_depth_result(
    depth_flags: list[bool],
    min_gap_px: float,
    frame_height: float,
) -> str:
    """
    Determine pass / fail / borderline from accumulated depth flags.

    Mirrors the classification logic in visualize._classify_segment().

    Args:
        depth_flags:  list of per-frame bool (hc_y > kt_y) across the full rep.
        min_gap_px:   minimum absolute pixel gap between hc_y and kt_y seen
                      across all rep frames (used for borderline detection).
        frame_height: full-resolution frame height in pixels.

    Returns:
        "pass" | "fail" | "borderline"
    """
    # Count consecutive depth frames to determine pass/fail.
    max_consec = _max_consecutive_true(depth_flags)
    if max_consec >= MIN_DEPTH_FRAMES:
        return "pass"
    if min_gap_px < (frame_height * CLOSE_THRESHOLD) or max_consec > 0:
        return "borderline"
    return "fail"


# ── RepStateMachine ───────────────────────────────────────────────────────────

class RepStateMachine:
    """
    Causal, per-frame rep segmenter and classifier.

    Feed one frame at a time via feed().  The machine tracks the current squat
    phase (STANDING / DESCENDING / ASCENDING) and emits a completed rep dict
    each time a rep finishes.

    All transitions require MIN_HOLD_FRAMES consecutive confirming frames before
    they fire, and are declared retroactively (at the frame MIN_HOLD_FRAMES ago).
    This makes every transition both noise-resistant and causally computable.

    Usage:
        sm = RepStateMachine(frame_height=1920, fps=30.0, side="left")
        for frame_idx, fdata in enumerate(frames_data):
            completed_rep = sm.feed(frame_idx, fdata)
            if completed_rep is not None:
                # rep finished — use completed_rep dict
                ...
            # sm.phase, sm.rep_start, sm.bottom_frame available at any time
            # for live drawing decisions
    """

    def __init__(self, frame_height: float, fps: float, side: str, cal: CameraCalibration | None = None) -> None:
        """
        Args:
            frame_height: full-resolution frame height in pixels.
            fps:          video frame rate.
            side:         "left" | "right".
            cal:          CameraCalibration with roll_deg and azimuth_deg.
                          Defaults to zero correction if None.
        """
        self.frame_height = frame_height
        self.fps          = fps
        self.side         = side
        self.cal          = cal or CameraCalibration()

        # ── Public state (safe to read between feed() calls for live drawing) ──
        self.phase: Phase          = Phase.STANDING
        self.rep_start:    int | None = None   # global frame idx where descent began
        self.bottom_frame: int | None = None   # global frame idx of confirmed bottom

        # ── Smoothing buffer ──────────────────────────────────────────────────
        # Rolling window of raw hip_y values; mean = smoothed signal.
        self._hip_buf: deque[float] = deque(maxlen=SMOOTHING_WINDOW)

        # ── Transition hold counters and candidates ───────────────────────────
        self._hold_count:      int   = 0     # frames held in current direction
        self._prev_smooth:     float | None = None  # previous smoothed hip_y

        # ── STANDING state ────────────────────────────────────────────────────
        self._standing_peak:   float | None = None  # min smoothed hip_y while standing
                                                     # (small = tall = standing position)

        # ── DESCENDING state ──────────────────────────────────────────────────
        self._bottom_candidate_val:   float | None = None  # smoothed hip_y at candidate bottom
        self._bottom_candidate_frame: int   | None = None  # frame idx of candidate bottom
        self._bottom_fdata:           dict  | None = None  # raw fdata at candidate bottom

        # ── Per-rep accumulators (reset on each new rep) ──────────────────────
        self._descent_frames:  int         = 0
        self._ascent_frames:   int         = 0
        self._ascent_hip_ys:    list[float] = []   # raw hip Y per ascent frame
        self._depth_flags:     list[bool]  = []   # hc_y > kt_y per rep frame
        self._min_gap_px:      float       = float("inf")  # min |hc_y - kt_y| across rep
        self._best_depth_av:   float       = -float("inf") # max along_vert seen this rep
        self._best_depth_fdata: dict | None = None          # fdata at that frame
        self._tibial_angles:   dict[int, float] = {}       # frame_idx → tibial angle
        self._frame_height_obs: float | None = None        # observed from first valid frame


    def feed(self, frame_idx: int, fdata: dict | None) -> dict | None:
        """
        Process one frame and advance the state machine.

        This is called with the *tail* frame from the pipeline delay buffer —
        i.e., the frame PIPELINE_DELAY frames behind the most recently inferred
        frame.  The caller handles the buffering; this method sees frames in
        presentation order at a fixed delay.

        Args:
            frame_idx: global frame index (0-based, monotonically increasing).
            fdata:     landmark dict for this frame as returned by
                       pose._extract_landmarks(), or None if MediaPipe
                       produced no pose for this frame.

        Returns:
            A completed rep dict (see module docstring for schema) when the
            machine transitions ASCENDING → STANDING, confirming a rep has ended.
            None on all other frames.

        Side effects:
            Updates self.phase, self.rep_start, self.bottom_frame.
        """
        if fdata is None:
            return self._handle_none_frame(frame_idx)

        # Update observed frame height (first valid frame wins)
        if self._frame_height_obs is None:
            self._frame_height_obs = fdata["height"]

        hip_y      = fdata[f"{self.side}_hip"][1]
        smooth_val = self._update_smooth(hip_y)

        # Geometry for depth tracking (every frame inside a rep)
        if self.phase != Phase.STANDING:
            self._accumulate_rep_frame(frame_idx, fdata, smooth_val)

        result = None
        if self.phase == Phase.STANDING:
            result = self._step_standing(frame_idx, fdata, smooth_val)
        elif self.phase == Phase.DESCENDING:
            result = self._step_descending(frame_idx, fdata, smooth_val)
        elif self.phase == Phase.ASCENDING:
            result = self._step_ascending(frame_idx, fdata, smooth_val)

        self._prev_smooth = smooth_val
        return result


    # ── Phase step methods ────────────────────────────────────────────────────

    def _step_standing(
        self, frame_idx: int, fdata: dict, smooth_val: float
    ) -> None:
        """
        Advance one frame while in STANDING phase.

        Tracks the rolling minimum of the smoothed hip_y signal (the standing
        reference height).  Transitions to DESCENDING when the signal drops
        MIN_HOLD_FRAMES consecutive frames AND has fallen more than
        MIN_DESCENT_THRESHOLD * frame_height below the standing peak.

        The transition is retroactive: DESCENDING is declared to have started
        MIN_HOLD_FRAMES ago when the hold count completes.

        Args:
            frame_idx:  current (tail) frame index.
            fdata:      landmark dict for this frame.
            smooth_val: current smoothed hip_y.

        Returns:
            Always None — STANDING never emits a completed rep.

        Side effects:
            May set self.phase = Phase.DESCENDING and self.rep_start.
        """
        # Track rolling min of smooth hip_y while standing.
        # Small hip_y = high physical position = standing reference.
        # standing_peak holds the lowest (most upright) hip_y seen while standing,
        # which is the baseline from which we measure descent depth.
        if self._standing_peak is None or smooth_val < self._standing_peak:
            self._standing_peak = smooth_val

        if self._prev_smooth is None:
            return None

        # Count consecutive frames where hip_y is increasing (hips going lower = descent).
        if smooth_val > self._prev_smooth:
            self._hold_count += 1
        else:
            self._hold_count = 0

        if self._hold_count >= MIN_HOLD_FRAMES:
            # Check that the drop is significant: hip_y must exceed standing baseline
            # by at least MIN_DESCENT_THRESHOLD * frame_height.
            if (smooth_val - self._standing_peak) >= MIN_DESCENT_THRESHOLD * self.frame_height:
                # Retroactive: descent started MIN_HOLD_FRAMES ago.
                # Seed _descent_frames to account for hold frames already elapsed.
                self.rep_start       = frame_idx - MIN_HOLD_FRAMES + 1
                self.phase           = Phase.DESCENDING
                self._hold_count     = 0
                self._descent_frames = MIN_HOLD_FRAMES - 1
                # Seed the bottom candidate at the current frame
                self._bottom_candidate_val   = smooth_val
                self._bottom_candidate_frame = frame_idx
                self._bottom_fdata           = fdata

        return None



    def _step_descending(
        self, frame_idx: int, fdata: dict, smooth_val: float
    ) -> None:
        """
        Advance one frame while in DESCENDING phase.

        Tracks the rolling minimum of the smoothed signal to find the squat
        bottom candidate.  Transitions to ASCENDING when the signal rises
        MIN_HOLD_FRAMES consecutive frames.

        On transition: retroactively declares bottom_frame = the frame at which
        the minimum was last seen (MIN_HOLD_FRAMES ago).  Stores the raw fdata
        at that frame for depth_angle computation.

        Args:
            frame_idx:  current (tail) frame index.
            fdata:      landmark dict for this frame.
            smooth_val: current smoothed hip_y.

        Returns:
            Always None — DESCENDING never emits a completed rep.

        Side effects:
            May set self.phase = Phase.ASCENDING and self.bottom_frame.
        """
        # Track rolling maximum — large hip_y = lowest physical position = squat bottom.
        if (self._bottom_candidate_val is None
                or smooth_val >= self._bottom_candidate_val):
            self._bottom_candidate_val   = smooth_val
            self._bottom_candidate_frame = frame_idx
            self._bottom_fdata           = fdata

        if self._prev_smooth is None:
            return None

        # Count consecutive frames where hip_y is decreasing (hips rising = ascent).
        if smooth_val < self._prev_smooth:
            self._hold_count += 1
        else:
            self._hold_count = 0

        if self._hold_count >= MIN_HOLD_FRAMES:
            # Retroactively declare bottom at the candidate (the true minimum)
            self.bottom_frame = self._bottom_candidate_frame
            self.phase        = Phase.ASCENDING
            self._hold_count  = 0
            # The MIN_HOLD_FRAMES confirmation frames were already ascending —
            # seed ascent_frames so they're included in the timing.
            self._ascent_frames = MIN_HOLD_FRAMES
            # Seed with current frame's hip Y so the first diff in _step_ascending
            # is a clean 1-frame step (bottom fdata is 4 frames back — using it
            # would produce a 4x spike on the first velocity sample).
            self._ascent_hip_ys = [fdata[f"{self.side}_hip"][1]]

        return None


    def _step_ascending(
        self, frame_idx: int, fdata: dict, smooth_val: float
    ) -> dict | None:
        """
        Advance one frame while in ASCENDING phase.

        Accumulates hip-crease Y values for velocity computation.  Transitions
        to STANDING when the recovered fraction of the descent exceeds 0.9:
            (smooth_val - bottom_value) / (standing_peak - bottom_value) > 0.9

        On transition: finalises all rep metrics and emits the completed rep dict.
        Resets all per-rep accumulators and updates standing_peak.

        Args:
            frame_idx:  current (tail) frame index.
            fdata:      landmark dict for this frame.
            smooth_val: current smoothed hip_y.

        Returns:
            Completed rep dict when the rep ends, None otherwise.

        Side effects:
            May set self.phase = Phase.STANDING, reset all _rep accumulators.
        """
        # Accumulate raw hip Y for velocity computation
        self._ascent_hip_ys.append(fdata[f"{self.side}_hip"][1])
        self._ascent_frames += 1

        # Recovered fraction: how far hip_y has returned toward standing (small) value.
        # bottom_candidate_val is large (hips low); standing_peak is small (hips high).
        # recovered = (bottom_val - smooth_val) / (bottom_val - standing_peak)
        bottom_val    = self._bottom_candidate_val or 0.0
        standing_ref  = self._standing_peak or 0.0
        denom = bottom_val - standing_ref
        if denom > 1e-6:
            recovered = (bottom_val - smooth_val) / denom
        else:
            recovered = 0.0

        if recovered > ASCENT_RECOVERY_FRAC:
            completed = self._finalise_rep(frame_idx)
            self._reset_rep_accumulators()
            self._standing_peak = smooth_val
            self.phase          = Phase.STANDING
            self._hold_count    = 0
            return completed

        return None


    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update_smooth(self, hip_y: float) -> float:
        """
        Push hip_y into the rolling buffer and return the current smoothed value.

        Uses a simple uniform (box) average of up to SMOOTHING_WINDOW values.
        The buffer fills over the first SMOOTHING_WINDOW frames; until full,
        the average is computed over however many values are present.

        Args:
            hip_y: raw hip joint Y coordinate for this frame (pixels).

        Returns:
            Smoothed hip_y (mean of buffer contents).
        """
        self._hip_buf.append(hip_y)
        return sum(self._hip_buf) / len(self._hip_buf)


    def _accumulate_rep_frame(
        self, frame_idx: int, fdata: dict, smooth_val: float
    ) -> None:
        """
        Update all per-rep accumulators for one frame inside an active rep.

        Called every frame while phase is DESCENDING or ASCENDING.  Tracks:
          - depth_flags (hc_y > kt_y per frame)
          - min_gap_px (minimum |hc_y - kt_y| seen so far)
          - tibial_angles (frame_idx → degrees)

        Args:
            frame_idx:  current (tail) frame index.
            fdata:      landmark dict for this frame.
            smooth_val: current smoothed hip_y (unused here, available if needed).
        """
        hc_y, kt_y, hc_x, kt_x = _estimated_markers(fdata, self.side, self.cal)
        dx = hc_x - kt_x
        dy = hc_y - kt_y
        roll_rad = math.radians(self.cal.roll_deg)
        along_vert = -dx * math.sin(roll_rad) + dy * math.cos(roll_rad)
        at_depth = along_vert > 0
        self._depth_flags.append(at_depth)

        gap = abs(along_vert)
        if gap < self._min_gap_px:
            self._min_gap_px = gap
        if along_vert > self._best_depth_av:
            self._best_depth_av    = along_vert
            self._best_depth_fdata = fdata


        angle = _tibial_angle(fdata, self.side, self.cal)
        self._tibial_angles[frame_idx] = round(angle, 1)

        if self.phase == Phase.DESCENDING:
            self._descent_frames += 1


    def _handle_none_frame(self, frame_idx: int) -> None:
        """
        Handle a frame where MediaPipe produced no pose detection.

        None frames are treated as neutral — they do not advance hold counters
        in either direction and do not reset them.  The machine holds its
        current transition progress across gaps.  None frames are skipped
        entirely: no depth_flag is appended, preserving the rep's flag sequence
        as if the missing frame did not occur.

        Args:
            frame_idx: current (tail) frame index.

        Returns:
            Always None.
        """
        # Hold counters are not advanced or reset on None frames (neutral).
        # No depth_flag appended — the frame is simply skipped.
        return None


    def _reset_rep_accumulators(self) -> None:
        """
        Reset all per-rep state in preparation for the next rep.

        Called immediately after emitting a completed rep dict in _step_ascending.
        Does not reset phase-independent state (standing_peak, smoothing buffer,
        prev_smooth).
        """
        self.rep_start                = None
        self.bottom_frame             = None
        self._bottom_candidate_val    = None
        self._bottom_candidate_frame  = None
        self._bottom_fdata            = None
        self._descent_frames          = 0
        self._ascent_frames           = 0
        self._ascent_hip_ys            = []
        self._depth_flags             = []
        self._min_gap_px              = float("inf")
        self._best_depth_av           = -float("inf")
        self._best_depth_fdata        = None
        self._tibial_angles           = {}
        self._hold_count              = 0


    def _finalise_rep(self, end_frame: int) -> dict:
        """
        Build and return the completed rep dict from all accumulated state.

        Called once per rep from _step_ascending when ASCENDING → STANDING fires.
        Invokes _build_tempo, _build_tibial, _build_depth_result, and
        _depth_angle_at_frame to assemble the full rep schema.

        Args:
            end_frame: global frame index of the last frame of the rep
                       (the frame where STANDING was re-confirmed).

        Returns:
            Completed rep dict matching the schema in the module docstring.
        """
        frame_h = self._frame_height_obs or self.frame_height

        tempo  = _build_tempo(
            self._descent_frames,
            self._ascent_frames,
            self._ascent_hip_ys,
            frame_h,
            self.fps,
        )
        tibial = _build_tibial(self._tibial_angles)

        flags = compute_flags(tempo, tibial)
        tempo["flags"] = flags

        depth_ref = self._best_depth_fdata or self._bottom_fdata
        depth_angle = (
            _depth_angle_at_frame(depth_ref, self.side, self.cal)
            if depth_ref is not None
            else None
        )

        result = _build_depth_result(
            self._depth_flags,
            self._min_gap_px,
            frame_h,
        )


        return {
            "result":        result,
            "start_global":  self.rep_start,
            "bottom_global": self.bottom_frame,
            "end_global":    end_frame,
            "depth_flags":   list(self._depth_flags),
            "tempo":         tempo,
            "tibial":        tibial,
            "depth_angle":   round(depth_angle, 1) if depth_angle is not None else None,
        }
