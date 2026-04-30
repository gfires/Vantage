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

from collections import deque
from pathlib import Path

import cv2

from depth_detector import (
    _ensure_model,
    _get_rotation,
    _rotate_frame,
    _select_side,
)
from draw import (
    GRAPH_FRAMES,
    _draw_backgrounds,
    _draw_coaching_panel,
    _draw_graph,
    _draw_lights,
    _draw_metrics_hud,
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


def _infer_one_frame(
    frame,
    detector,
    frame_idx: int,
    fps: float,
) -> dict | None:
    """
    Run MediaPipe BlazePose inference on a single already-decoded BGR frame.

    This is the single-frame equivalent of depth_detector._extract_landmarks,
    which processes an entire video in one call.  The caller owns the detector
    lifecycle (create once before the loop, close after).

    Args:
        frame:     BGR numpy array (already rotated to display orientation).
        detector:  An open mediapipe.tasks.python.vision.PoseLandmarker instance
                   created in VIDEO running mode.  Must be the same instance used
                   for all frames in this video — VIDEO mode is stateful.
        frame_idx: 0-based frame index.  Used to compute the timestamp_ms passed
                   to detect_for_video(); must be strictly monotonically increasing.
        fps:       Video frame rate.  Used to compute timestamp_ms = frame_idx * 1000 / fps.

    Returns:
        fdata dict matching the schema from depth_detector._extract_landmarks:
            {frame_idx, left_hip, right_hip, left_knee, right_knee,
             left_wrist, right_wrist, left_shoulder, right_shoulder,
             left_heel, right_heel, width, height}
        or None if MediaPipe detected no pose in this frame.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def _make_smooth_bufs() -> dict:
    """
    Allocate the per-joint rolling buffers used by _smooth_one_frame.

    Returns:
        Dict mapping each joint name to a fresh deque(maxlen=DRAW_SMOOTHING).
        Must be called once before the render loop and passed to every
        _smooth_one_frame call.
    """
    joints = [
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_shoulder", "right_shoulder", "left_heel", "right_heel",
    ]
    return {j: {"x": deque(maxlen=DRAW_SMOOTHING), "y": deque(maxlen=DRAW_SMOOTHING)}
            for j in joints}


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

    Raises:
        ValueError: if no pose is detected during the probe phase and force_side
                    is None (cannot proceed without a side selection).

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
    raise NotImplementedError
