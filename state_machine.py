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
    CLOSE_THRESHOLD,
    HOLE_EXIT_FRACTION,
    HIP_CREASE_FRAC,
    KNEE_TOP_OVERSHOOT,
    DESCENT_FAST_S,
    DESCENT_SLOW_S,
    GRIND_RATIO,
    HOLE_MCV_WARN,
    TIBIAL_WARN_DEG,
)


# ── Phase enum ────────────────────────────────────────────────────────────────

class Phase(Enum):
    STANDING   = auto()
    DESCENDING = auto()
    ASCENDING  = auto()


# ── Geometry helpers (mirrors visualize._estimated_marker_ys and metrics) ─────

def _hip_crease_y(fdata: dict, side: str) -> float:
    """
    Estimate anatomical hip-crease Y from shoulder and hip joint centers.

    Uses HIP_CREASE_FRAC: the crease sits HIP_CREASE_FRAC of the way from
    shoulder to hip joint center (slightly above the joint center).

    Args:
        fdata: per-frame landmark dict from depth_detector._extract_landmarks.
        side:  "left" | "right"

    Returns:
        Hip-crease Y in full-resolution pixel coordinates (y increases downward).
    """
    shoulder = fdata[f"{side}_shoulder"]
    hip      = fdata[f"{side}_hip"]
    return shoulder[1] + (hip[1] - shoulder[1]) * HIP_CREASE_FRAC


def _estimated_markers(fdata: dict, side: str) -> tuple[float, float, float, float]:
    """
    Estimate anatomical hip-crease and knee-top positions.

    Hip crease: HIP_CREASE_FRAC of the way from shoulder to hip joint center.
    Knee top:   heel→knee vector extended past the knee by KNEE_TOP_OVERSHOOT.

    Args:
        fdata: per-frame landmark dict.
        side:  "left" | "right"

    Returns:
        (hc_y, kt_y, hc_x, kt_x) — all in full-resolution pixel coordinates.
        hc_y > kt_y in screen coords means hip crease is below knee top (depth achieved).
    """
    heel     = fdata[f"{side}_heel"]
    knee     = fdata[f"{side}_knee"]
    shoulder = fdata[f"{side}_shoulder"]
    hip      = fdata[f"{side}_hip"]

    kt_y = heel[1] + (knee[1] - heel[1]) * (1.0 + KNEE_TOP_OVERSHOOT)
    kt_x = heel[0] + (knee[0] - heel[0]) * (1.0 + KNEE_TOP_OVERSHOOT)
    hc_y = shoulder[1] + (hip[1] - shoulder[1]) * HIP_CREASE_FRAC
    hc_x = shoulder[0] + (hip[0] - shoulder[0]) * HIP_CREASE_FRAC

    return hc_y, kt_y, hc_x, kt_x


def _tibial_angle(fdata: dict, side: str) -> float:
    """
    Compute shin angle from vertical for one frame (degrees).

    Tibial angle = atan2(|knee_x - heel_x|, |knee_y - heel_y|).
    0° = perfectly vertical shin. Increases as knee travels forward over toe.

    Args:
        fdata: per-frame landmark dict.
        side:  "left" | "right"

    Returns:
        Angle in degrees, ≥ 0.
    """
    heel = fdata[f"{side}_heel"]
    knee = fdata[f"{side}_knee"]
    dx   = abs(knee[0] - heel[0])
    dy   = abs(heel[1] - knee[1])
    return math.degrees(math.atan2(dx, max(dy, 1e-6)))


def _depth_angle_at_frame(fdata: dict, side: str) -> float:
    """
    Compute the signed depth angle at a single frame.

    Convention:
        positive = hip crease below knee top (depth achieved)
        negative = hip crease above knee top (short)
        0        = exactly parallel

    Args:
        fdata: per-frame landmark dict (typically the bottom frame).
        side:  "left" | "right"

    Returns:
        Angle in degrees, signed.
    """
    hc_y, kt_y, hc_x, kt_x = _estimated_markers(fdata, side)
    dx   = kt_x - hc_x
    dy   = kt_y - hc_y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0
    rise = hc_y - kt_y   # positive in screen coords = hc lower = depth achieved
    return math.degrees(math.asin(max(-1.0, min(1.0, rise / dist))))


# ── Metric builders ───────────────────────────────────────────────────────────

def _build_tempo(
    descent_frames: int,
    ascent_frames: int,
    ascent_hc_ys: list[float],
    frame_height: float,
    fps: float,
) -> dict:
    """
    Build the tempo dict from accumulated per-phase frame counts and hip-crease
    Y positions recorded during the ascent.

    Mirrors compute_tempo() in metrics.py but operates on pre-accumulated data
    rather than doing a post-hoc pass over frames_data.

    Args:
        descent_frames:  number of frames from rep start to bottom (inclusive).
        ascent_frames:   number of frames from bottom to rep end (inclusive).
        ascent_hc_ys:    list of hip-crease Y values recorded each ascent frame,
                         starting at the bottom frame.  Length = ascent_frames.
        frame_height:    full-resolution frame height in pixels (for normalisation).
        fps:             video frame rate.

    Returns:
        Dict with keys: descent_s, ascent_s, velocity, mean_concentric_vel,
        hole_exit_vel, hole_mcv_ratio, flags.
        Matches the schema returned by metrics.compute_tempo().
    """
    ...


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
    ...


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
    ...


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

    def __init__(self, frame_height: float, fps: float, side: str) -> None:
        """
        Args:
            frame_height: full-resolution frame height in pixels.
                          Used to compute threshold distances as fractions of frame.
            fps:          video frame rate.  Used to convert frame counts to seconds.
            side:         "left" | "right" — which landmark side to use.
                          Determined by _select_side() before the loop begins.
        """
        self.frame_height = frame_height
        self.fps          = fps
        self.side         = side

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
        self._standing_peak:   float | None = None  # max smoothed hip_y while standing
                                                     # (large = tall = standing position)

        # ── DESCENDING state ──────────────────────────────────────────────────
        self._bottom_candidate_val:   float | None = None  # smoothed hip_y at candidate bottom
        self._bottom_candidate_frame: int   | None = None  # frame idx of candidate bottom
        self._bottom_fdata:           dict  | None = None  # raw fdata at candidate bottom

        # ── Per-rep accumulators (reset on each new rep) ──────────────────────
        self._descent_frames:  int         = 0
        self._ascent_frames:   int         = 0
        self._ascent_hc_ys:    list[float] = []   # hip-crease Y per ascent frame
        self._depth_flags:     list[bool]  = []   # hc_y > kt_y per rep frame
        self._min_gap_px:      float       = float("inf")  # min |hc_y - kt_y| across rep
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
                       depth_detector._extract_landmarks(), or None if MediaPipe
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

        Tracks the rolling maximum of the smoothed hip_y signal (the standing
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
        ...


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
        ...


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
        ...


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
        ...


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
        ...


    def _handle_none_frame(self, frame_idx: int) -> None:
        """
        Handle a frame where MediaPipe produced no pose detection.

        None frames are treated as neutral — they do not advance hold counters
        in either direction and do not reset them.  The machine holds its
        current transition progress across gaps.  A depth_flag of False is
        appended for None frames inside active reps (conservative: no depth
        credit for frames where pose is invisible).

        Args:
            frame_idx: current (tail) frame index.

        Returns:
            Always None.
        """
        ...


    def _reset_rep_accumulators(self) -> None:
        """
        Reset all per-rep state in preparation for the next rep.

        Called immediately after emitting a completed rep dict in _step_ascending.
        Does not reset phase-independent state (standing_peak, smoothing buffer,
        prev_smooth).
        """
        ...


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
        ...
