"""
pipeline.py — Single-pass inference + render pipeline for WhiteLights.

Replaces the two-pass _analyze + _render sequence in visualize.py with a
unified forward pass that streams annotated frames as they are inferred.

The core function is _process_video().  Supporting helpers live here too:
_infer_one_frame() (single-frame MediaPipe wrapper) and _smooth_one_frame()
(incremental draw-smoothing).

Used by:
    visualize.py  — CLI path (main() calls _process_video with output_path)
    api.py        — streaming path (_process calls _process_video with on_frame)
"""

import math
from collections import deque
from pathlib import Path

import cv2

from depth_detector import (
    JOINTS,
    _ensure_model,
    _get_rotation,
    _infer_one_frame,
    _make_detector,
    _rotate_frame,
    _select_side,
)
from rendering.draw import (
    GRAPH_FRAMES,
    _draw_backgrounds,
    _draw_coaching_panel,
    _draw_graph,
    _draw_lights,
    _draw_metrics_hud,
    _draw_phase_box,
    _draw_rep_counter,
    _draw_side_badge,
    _draw_skeleton,
    _estimated_marker_ys,
)
from params import (
    CLOSE_THRESHOLD,
    DRAW_SMOOTHING,
    PIPELINE_DELAY,
)
from state_machine import RepStateMachine



def _smooth_one_frame(
    fdata: dict | None,
    smooth_bufs: dict,
) -> dict | None:
    """
    Apply incremental draw-smoothing to one frame's landmark coordinates.

    Maintains a per-joint rolling buffer (deque of length DRAW_SMOOTHING) that
    is updated on every call.  Returns a new fdata dict with smoothed x/y
    coordinates suitable for skeleton rendering, while leaving the original
    fdata unchanged for classification.

    This is the incremental, per-frame equivalent of
    visualize._smooth_landmarks_for_drawing, which does a global pass over all
    frames.  The math is identical: uniform box average of the last DRAW_SMOOTHING
    values, with NaN fallback to the raw value for frames near None gaps.

    Args:
        fdata:       Raw landmark dict for this frame, or None if no pose detected.
                     None frames push NaN into every joint buffer (preserving the
                     smoothing window continuity) and return None.
        smooth_bufs: Dict of {joint_name: deque(maxlen=DRAW_SMOOTHING)} maintained
                     across calls.  Must be initialized before the loop with one
                     deque per joint (see _make_smooth_bufs()).  Modified in-place.

    Returns:
        Smoothed copy of fdata with x/y replaced by buffer means, or None if
        fdata is None.  Visibility and z fields are copied from the original
        without smoothing.
    """
    NAN = float("nan")

    if fdata is None:
        for bufs in smooth_bufs.values():
            bufs["x"].append(NAN)
            bufs["y"].append(NAN)
        return None

    df = dict(fdata)  # shallow copy — width/height/frame_idx unchanged
    for joint, bufs in smooth_bufs.items():
        raw = fdata[joint]
        bufs["x"].append(raw[0])
        bufs["y"].append(raw[1])

        valid_x = [v for v in bufs["x"] if v == v]
        valid_y = [v for v in bufs["y"] if v == v]
        sx = sum(valid_x) / len(valid_x) if valid_x else raw[0]
        sy = sum(valid_y) / len(valid_y) if valid_y else raw[1]

        # Preserve any extra fields (visibility, z) verbatim from original
        df[joint] = (sx, sy) + raw[2:]

    return df



def _make_smooth_bufs() -> dict:
    """
    Allocate the per-joint rolling buffers used by _smooth_one_frame.

    Driven by the JOINTS registry in depth_detector so the joint list stays
    in one place.  Extra tuple fields (visibility, z) are handled generically
    via raw[2:] in _smooth_one_frame — no per-joint metadata needed here.

    Returns:
        Dict mapping joint name to {"x": deque(maxlen=DRAW_SMOOTHING), "y": deque(...)}.
        Must be called once before the render loop.
    """
    return {
        name: {"x": deque(maxlen=DRAW_SMOOTHING), "y": deque(maxlen=DRAW_SMOOTHING)}
        for name in JOINTS
    }


def _process_video(
    cap: cv2.VideoCapture,
    rotation: int,
    fps: float,
    output_path: str | Path | None = None,
    on_frame=None,
    force_side: str | None = None,
) -> list[dict]:
    """
    Unified single-pass inference + render loop.

    Replaces the two-pass _analyze + _render sequence.  Frames are inferred
    and drawn in a single forward pass using RepStateMachine for causal rep
    detection.  A PIPELINE_DELAY-frame ring buffer absorbs the lookahead needed
    for smoothing and transition confirmation, so annotated frames are emitted
    with a one-time startup latency of PIPELINE_DELAY / fps seconds (~300ms at
    30fps default params).

    Side selection:
        The first PIPELINE_DELAY frames are probed before the main loop begins.
        These frames are pushed into the ring buffers and processed normally —
        the probe adds no extra latency beyond the startup cost that already exists.
        _select_side is called on the probe's valid frames.  If force_side is
        given, side selection is skipped entirely.

    Output:
        If output_path is given, an annotated MP4 is written to that path.
        The VideoWriter is opened lazily on the first emitted frame (dimensions
        not known until then) and closed after the flush phase completes.
        If on_frame is given, it is called with JPEG-encoded bytes for every
        emitted frame (used for MJPEG streaming in api.py).
        Both may be active simultaneously.  Neither is required.

    Args:
        cap:         Open cv2.VideoCapture positioned at frame 0.  The caller
                     retains ownership and must release it after this returns.
        rotation:    Rotation angle in degrees from _get_rotation(); applied to
                     each decoded frame before inference and drawing.
        fps:         Video frame rate.  Used for tempo computation and for
                     generating the monotonically-increasing timestamps required
                     by MediaPipe VIDEO mode.
        output_path: Destination path for the annotated MP4, or None to skip
                     file output.
        on_frame:    Optional callable(jpeg_bytes: bytes) -> None.  Called once
                     per emitted frame with JPEG-encoded annotated bytes.  Must
                     be non-blocking; exceptions raised inside are not caught.
                     Called with None once after the flush phase completes (stream
                     sentinel, matches existing _render behaviour).
        force_side:  "left" | "right" to override automatic side selection, or
                     None to detect from z-coordinates of the probe frames.

    Returns:
        List of completed rep dicts in emission order, matching the schema in
        state_machine.py.  Empty list if no reps were detected.

    Pipeline per frame:
        1. Decode + rotate                        → raw BGR frame
        2. _infer_one_frame(frame, detector, ...)  → fdata | None
        3. Push (frame_idx, raw_frame, fdata)      → frame_buf, fdata_buf
        4. Pop tail from buffers                   → tail_frame, tail_fdata
        5. sm.feed(tail_idx, tail_fdata)           → completed_rep | None
        6. _smooth_one_frame(tail_fdata, bufs)     → fdata_draw
        7. Draw overlays onto tail_frame
        8. JPEG-encode tail_frame
        9. on_frame(jpeg_bytes) if provided
       10. VideoWriter.write(tail_frame) if output_path provided

    Flush phase:
        After the decode loop ends, drain the remaining PIPELINE_DELAY - 1
        frames still in the ring buffers.  No new inference; fdata values are
        already buffered.  Each buffered frame is drawn and emitted normally.
    """
    _ensure_model()

    # ── Ring buffers ──────────────────────────────────────────────────────────
    frame_buf: deque[tuple[int, object]] = deque()   # (frame_idx, BGR ndarray)
    fdata_buf: deque                     = deque()   # fdata | None

    # ── State ─────────────────────────────────────────────────────────────────
    bufs            = _make_smooth_bufs()
    completed_reps: list[dict] = []
    last_rep: dict | None      = None   # most recently completed rep (for HUD)
    out: cv2.VideoWriter | None = None
    w = h = 0

    # ── Scrolling graph deques (single-pass path in _draw_graph) ─────────────
    hip_y_deque  = deque(maxlen=GRAPH_FRAMES)
    knee_y_deque = deque(maxlen=GRAPH_FRAMES)

    # ── Inner helpers ─────────────────────────────────────────────────────────
    def _emit(frame_idx: int, frame, fdata, side: str, sm: RepStateMachine) -> None:
        """Draw overlays onto frame and write/stream it."""
        nonlocal out, w, h

        fdata_draw = _smooth_one_frame(fdata, bufs)

        if fdata is not None:
            hip_y_deque.append(fdata[f"{side}_hip"][1])
            knee_y_deque.append(fdata[f"{side}_knee"][1])

        rep_num = len(completed_reps) + 1 if last_rep is not None else 1

        overlay = frame.copy()
        _draw_backgrounds(overlay, w, h, rep_num, None, last_rep)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if fdata_draw is not None:
            frame_h     = fdata["height"]
            (hc_y, kt_y), _ = _estimated_marker_ys(fdata, side)
            depth_active = hc_y > kt_y
            near_depth   = abs(hc_y - kt_y) < (frame_h * CLOSE_THRESHOLD)

            tib_angle = None
            if last_rep is not None:
                tib_angle = last_rep["tibial"]["angles"].get(frame_idx)
            if tib_angle is None and fdata is not None:
                heel = fdata[f"{side}_heel"]
                knee = fdata[f"{side}_knee"]
                dx   = abs(knee[0] - heel[0])
                dy   = abs(heel[1] - knee[1])
                tib_angle = math.degrees(math.atan2(dx, max(dy, 1e-6)))

            is_bottom = (sm.bottom_frame is not None and frame_idx == sm.bottom_frame)

            _draw_skeleton(frame, fdata_draw, side, depth_active, near_depth, tib_angle, is_bottom)
            _draw_graph(frame, hip_y_deque, knee_y_deque, None, h)
            _draw_phase_box(frame, sm.phase.name, h)
            _draw_metrics_hud(frame, last_rep, frame_idx, w)
            _draw_coaching_panel(frame, last_rep, frame_idx, w)
            _draw_lights(frame, last_rep, frame_idx, h, fps)
            _draw_rep_counter(frame, rep_num, None, w, h)
            _draw_side_badge(frame, side, w, h)
        else:
            _draw_graph(frame, hip_y_deque, knee_y_deque, None, h)
            _draw_phase_box(frame, sm.phase.name, h)
            _draw_metrics_hud(frame, last_rep, frame_idx, w)
            _draw_coaching_panel(frame, last_rep, frame_idx, w)
            _draw_lights(frame, last_rep, frame_idx, h, fps)
            _draw_rep_counter(frame, rep_num, None, w, h)
            _draw_side_badge(frame, side, w, h)

        if output_path is not None:
            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            out.write(frame)

        if on_frame is not None:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            on_frame(jpeg.tobytes())

    # ── Probe phase: fill ring buffer, select side ────────────────────────────
    side: str | None = force_side
    sm: RepStateMachine | None = None

    with _make_detector() as detector:
        frame_idx = 0

        # Read probe frames to fill PIPELINE_DELAY-deep buffer and select side
        probe_valid: list[tuple[int, dict]] = []
        while frame_idx < PIPELINE_DELAY:
            ret, raw = cap.read()
            if not ret:
                break
            raw = _rotate_frame(raw, rotation)
            if frame_idx == 0:
                h, w = raw.shape[:2]
            fdata = _infer_one_frame(raw, detector, frame_idx, fps)
            frame_buf.append((frame_idx, raw))
            fdata_buf.append(fdata)
            if fdata is not None:
                probe_valid.append((frame_idx, fdata))
            frame_idx += 1

        if side is None:
            if probe_valid:
                side = _select_side(probe_valid)
            if side is None:
                side = "right"
                # TODO: fix this edge case by falling back to a non-side-specific heuristic (e.g. which hip is more visible on average across the probe frames) instead of just defaulting to right and hoping for the best.

        if h == 0:
            # No frames decoded at all
            return []

        sm = RepStateMachine(frame_height=h, fps=fps, side=side)

        # ── Main decode loop ──────────────────────────────────────────────────
        while True:
            ret, raw = cap.read()
            if not ret:
                break
            raw   = _rotate_frame(raw, rotation)
            fdata = _infer_one_frame(raw, detector, frame_idx, fps)

            frame_buf.append((frame_idx, raw))
            fdata_buf.append(fdata)
            frame_idx += 1

            # Pop the oldest buffered frame (PIPELINE_DELAY frames behind)
            if len(frame_buf) > PIPELINE_DELAY:
                tail_idx, tail_frame = frame_buf.popleft()
                tail_fdata           = fdata_buf.popleft()

                completed = sm.feed(tail_idx, tail_fdata)
                if completed is not None:
                    completed_reps.append(completed)
                    last_rep = completed

                _emit(tail_idx, tail_frame, tail_fdata, side, sm)

    # ── Flush phase: drain remaining buffered frames ──────────────────────────
    while frame_buf:
        tail_idx, tail_frame = frame_buf.popleft()
        tail_fdata           = fdata_buf.popleft()

        completed = sm.feed(tail_idx, tail_fdata)
        if completed is not None:
            completed_reps.append(completed)
            last_rep = completed

        _emit(tail_idx, tail_frame, tail_fdata, side, sm)

    # ── Teardown ──────────────────────────────────────────────────────────────
    if out is not None:
        out.release()
    if on_frame is not None:
        on_frame(None)

    return completed_reps

