"""
Test harness for depth_detector.py against labeled squat videos.

Usage:
    python tests/test_depth.py

Success criterion: >= 80% accuracy on labeled test set.
"borderline" results are counted as incorrect (conservative).
"""

import json
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from depth_detector import analyze_video

VIDEOS_DIR = Path(__file__).parent / "videos"
LABELS_FILE = Path(__file__).parent / "labels.json"
PASS_THRESHOLD = 0.80


def run():
    labels = json.loads(LABELS_FILE.read_text())

    results = []
    print(f"\nRunning depth detection on {len(labels)} labeled videos...\n")

    for filename, expected in labels.items():
        video_path = VIDEOS_DIR / filename
        if not video_path.exists():
            print(f"  SKIP  {filename} (file not found)")
            continue

        t0 = time.time()
        prediction = analyze_video(str(video_path))
        elapsed = time.time() - t0

        got = prediction["result"]
        correct = got == expected

        status = "PASS" if correct else "FAIL"
        marker = "✓" if correct else "✗"

        detail = ""
        if got == "indeterminate":
            detail = f"  [{prediction.get('error', 'unknown error')}]"
        elif not correct:
            detail = f"  [hip_y={prediction.get('hip_y', '?'):.1f}, knee_y={prediction.get('knee_y', '?'):.1f}, consec={prediction.get('max_consecutive_depth_frames', '?')}]"

        print(f"  {marker} {filename:<20} expected={expected:<12} got={got:<12} ({elapsed:.1f}s){detail}")
        results.append({"file": filename, "expected": expected, "got": got, "correct": correct})

    total = len(results)
    correct_count = sum(r["correct"] for r in results)
    accuracy = correct_count / total if total > 0 else 0

    print(f"\n{'─' * 60}")
    print(f"  Accuracy: {correct_count}/{total} ({accuracy:.0%})")

    # Break down by category
    passes = [r for r in results if r["expected"] == "pass"]
    fails  = [r for r in results if r["expected"] == "fail"]
    pass_acc  = sum(r["correct"] for r in passes) / len(passes) if passes else 0
    fail_acc  = sum(r["correct"] for r in fails)  / len(fails)  if fails  else 0
    print(f"  Valid squats (pass): {sum(r['correct'] for r in passes)}/{len(passes)} ({pass_acc:.0%})")
    print(f"  Bad squats  (fail): {sum(r['correct'] for r in fails)}/{len(fails)} ({fail_acc:.0%})")

    # Indeterminate breakdown
    indeterminate = [r for r in results if r["got"] == "indeterminate"]
    if indeterminate:
        print(f"\n  Indeterminate ({len(indeterminate)} videos — check camera angle or video quality):")
        for r in indeterminate:
            print(f"    - {r['file']}")

    # Borderline breakdown
    borderline = [r for r in results if r["got"] == "borderline"]
    if borderline:
        print(f"\n  Borderline ({len(borderline)} videos — close calls, manual review recommended):")
        for r in borderline:
            print(f"    - {r['file']} (expected: {r['expected']})")

    print(f"{'─' * 60}\n")

    if accuracy >= PASS_THRESHOLD:
        print(f"  ✓ PASSED — {accuracy:.0%} >= {PASS_THRESHOLD:.0%} threshold. Ready to build the UI.\n")
        sys.exit(0)
    else:
        print(f"  ✗ BELOW THRESHOLD — {accuracy:.0%} < {PASS_THRESHOLD:.0%}. Debug the detector before building UI.\n")
        sys.exit(1)


if __name__ == "__main__":
    run()
