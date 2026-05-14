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

from pose import (
    _ensure_model,
    _get_rotation,
)
from params import (
    HOLE_EXIT_FRACTION,
)
from metrics import compute_flags
from rendering.pipeline import _process_video


# ── Rep table ─────────────────────────────────────────────────────────────────

def _rep_warnings(rep: dict) -> str:
    """Collect all warnings for a rep as a display string for the rep table."""
    return ", ".join(compute_flags(rep["tempo"], rep["tibial"])) or "--"


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
    if not args:
        raise ValueError("Usage: visualize.py <video> [--side left|right]")
    video_path = args[0]

    path = Path(video_path)
    annotated_dir = path.parent.parent / "annotated_videos"
    annotated_dir.mkdir(exist_ok=True)
    output_path = annotated_dir / (path.stem + "_annotated.mp4")
    table_path  = annotated_dir / (path.stem + "_table.txt")

    print(f"\nInput:  {path}")
    print(f"Output: {output_path}")
    if force_side:
        print(f"Side:   {force_side} (forced)")

    _ensure_model()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open {path}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    rotation = _get_rotation(cap)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print("Processing video (single pass)...")
    try:
        reps = _process_video(
            cap, rotation, fps,
            output_path=str(output_path),
            force_side=force_side,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        cap.release()

    if not reps:
        print("ERROR: No reps detected. Check camera angle.")
        sys.exit(1)

    for i, rep in enumerate(reps, 1):
        t   = rep["tempo"]
        tib = rep["tibial"]
        print(
            f"  Rep {i}: {rep['result'].upper():12}"
            f"  start={rep['start_global']}  bottom={rep['bottom_global']}  end={rep['end_global']}"
            f"  desc={t['descent_s']:.1f}s  asc={t['ascent_s']:.1f}s"
            f"  hole_v={t['hole_exit_vel']:.3f}  mcv={t['mean_concentric_vel']:.3f}"
            f"  shin_max={tib['max_angle']:.0f}"
            + (f"  [{' '.join(t['flags'])}]" if t.get("flags") else "")
        )

    _output_rep_table(reps, str(table_path))
    print(f"  Table:  {table_path}")
    print(f"  Saved:  {output_path}")
    print("\nDone.")
    subprocess.run(["open", str(output_path)])


if __name__ == "__main__":
    main()
