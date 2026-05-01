"""
bench_inference.py — Measure MediaPipe inference time vs. INFERENCE_HEIGHT.

Usage:
    # Run all four resolutions against a single video:
    python bench_inference.py tests/raw_videos/valid_1.MOV

    # Run against all valid_* videos (accuracy + timing):
    python bench_inference.py --all

Results printed as a table: resolution → ms/frame, Phase 1 total, depth calls per video.
Depth call accuracy is compared against the None (full-res) baseline.
"""

import sys
import time
from pathlib import Path

import cv2
import params  # mutated directly to avoid re-importing the module tree

from depth_detector import _ensure_model, _get_rotation, _extract_landmarks
from visualization.visualize import _analyze

RESOLUTIONS = [None, 720, 480, 360]   # None = full resolution baseline

RAW_DIR = Path(__file__).parent / "tests" / "raw_videos"
ALL_VIDEOS = sorted(RAW_DIR.glob("valid_*.MOV")) + sorted(RAW_DIR.glob("valid_*.mov"))


# ── Timing harness ────────────────────────────────────────────────────────────

def _time_inference(video_path: Path, inf_height: int | None) -> dict:
    """
    Run _extract_landmarks on video_path at the given inference height.
    Returns timing data and frame count.
    """
    params.INFERENCE_HEIGHT = inf_height  # mutate module-level constant

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rotation = _get_rotation(cap)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # Probe source resolution
    ret, first = cap.read()
    if ret:
        from depth_detector import _rotate_frame
        first = _rotate_frame(first, rotation)
        src_h, src_w = first.shape[:2]
    else:
        src_h, src_w = 0, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    t0 = time.monotonic()
    frames_data = _extract_landmarks(cap, rotation)
    elapsed = time.monotonic() - t0
    cap.release()

    valid = sum(1 for f in frames_data if f is not None)
    ms_per_frame = elapsed * 1000 / max(len(frames_data), 1)

    return {
        "inf_height": inf_height,
        "src_res": f"{src_w}×{src_h}",
        "frames": len(frames_data),
        "valid_frames": valid,
        "elapsed_s": elapsed,
        "ms_per_frame": ms_per_frame,
        "fps": fps,
        "total_frames": total_frames,
    }


# ── Depth call harness ────────────────────────────────────────────────────────

def _depth_call(video_path: Path, inf_height: int | None) -> str | None:
    """
    Run the full _analyze pipeline and return the depth result of the first rep,
    or None if analysis failed.
    """
    params.INFERENCE_HEIGHT = inf_height

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rotation = _get_rotation(cap)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    analysis = _analyze(cap, rotation, fps, force_side=None)
    cap.release()

    if analysis is None:
        return None
    _, _, _, reps, _, _, _ = analysis
    if not reps:
        return None
    # Return comma-separated results for multi-rep videos
    return ",".join(r["result"] for r in reps)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt(val, baseline=None, is_time=True):
    if is_time and baseline is not None:
        speedup = baseline / val if val > 0 else 0
        return f"{val:6.1f}ms  ({speedup:.2f}×)"
    return f"{val:6.1f}ms"


def run_single(video_path: Path):
    print(f"\nVideo: {video_path.name}")
    print(f"{'Resolution':>12}  {'Inf height':>10}  {'ms/frame':>20}  {'Phase1 total':>12}  {'Valid/Total':>12}")
    print("─" * 80)

    baseline_ms = None
    for res in RESOLUTIONS:
        r = _time_inference(video_path, res)
        label = "full-res" if res is None else f"{res}p"
        if baseline_ms is None:
            baseline_ms = r["ms_per_frame"]
            speedup_str = "  (baseline)"
        else:
            speedup = baseline_ms / r["ms_per_frame"] if r["ms_per_frame"] > 0 else 0
            speedup_str = f"  ({speedup:.2f}×)"
        print(
            f"  {label:>10}  {r['src_res']:>10}  "
            f"{r['ms_per_frame']:6.1f}ms{speedup_str:<14}  "
            f"{r['elapsed_s']:6.1f}s total   "
            f"{r['valid_frames']}/{r['frames']} frames"
        )


def run_all():
    if not ALL_VIDEOS:
        print("No valid_*.MOV files found in tests/raw_videos/")
        return

    print(f"\nRunning against {len(ALL_VIDEOS)} videos: {', '.join(v.name for v in ALL_VIDEOS)}")

    # Phase 1: timing (use first video only for speed — timing is resolution-dependent, not video-dependent)
    timing_video = ALL_VIDEOS[0]
    print(f"\n── Timing (on {timing_video.name}) ──────────────────────────────────────")
    run_single(timing_video)

    # Phase 2: depth call accuracy across all videos
    print(f"\n── Depth call accuracy (all {len(ALL_VIDEOS)} videos) ──────────────────")
    print(f"{'Video':<20}", end="")
    for res in RESOLUTIONS:
        label = "full-res" if res is None else f"{res}p"
        print(f"  {label:>10}", end="")
    print("  diverges?")
    print("─" * (20 + len(RESOLUTIONS) * 12 + 12))

    total_divergences = 0
    for vp in ALL_VIDEOS:
        results = {}
        for res in RESOLUTIONS:
            results[res] = _depth_call(vp, res)

        baseline = results[None]
        diverges = any(results[r] != baseline for r in RESOLUTIONS[1:] if results[r] is not None)
        if diverges:
            total_divergences += 1

        print(f"{vp.name:<20}", end="")
        for res in RESOLUTIONS:
            val = results[res] or "none"
            marker = " *" if (res is not None and results[res] != baseline) else "  "
            print(f"  {val:>9}{marker}", end="")
        print(f"  {'YES ← ' if diverges else ''}")

    print(f"\nDivergences vs full-res baseline: {total_divergences}/{len(ALL_VIDEOS)} videos")
    print("* = differs from full-res baseline")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _ensure_model()

    args = sys.argv[1:]
    if "--all" in args:
        run_all()
    elif args:
        run_single(Path(args[0]))
    else:
        # Default: time on first available video, accuracy on all
        run_all()
