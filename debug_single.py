"""
Debug a single video: print frame dimensions, rotation, landmark y-values
across all frames so we can see where depth is/isn't happening.
"""
import sys
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from depth_detector import (
    _get_rotation, _rotate_frame, _extract_landmarks, _select_side,
    _rolling_average, _find_bottom_frame, _ensure_model,
    LEFT_HIP, LEFT_KNEE, RIGHT_HIP, RIGHT_KNEE, VISIBILITY_THRESHOLD
)

video = sys.argv[1] if len(sys.argv) > 1 else "tests/videos/valid_1.MOV"

_ensure_model()
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # prevent double-rotate on macOS
rotation = _get_rotation(cap)
fps = cap.get(cv2.CAP_PROP_FPS)

# Check first frame dimensions
ret, frame = cap.read()
if ret:
    raw_h, raw_w = frame.shape[:2]
    rotated = _rotate_frame(frame, rotation)
    rot_h, rot_w = rotated.shape[:2]
    print(f"\nFile: {video}")
    print(f"Rotation metadata: {rotation}°")
    print(f"Raw frame: {raw_w}x{raw_h},  After rotation: {rot_w}x{rot_h}  (fps={fps:.0f})")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frames_data = _extract_landmarks(cap, rotation)
cap.release()

valid = [(i, f) for i, f in enumerate(frames_data) if f is not None]
print(f"Total frames: {len(frames_data)}, with pose: {len(valid)}")

if not valid:
    print("No pose detected at all.")
    sys.exit(1)

left_vis  = np.mean([f["left_hip"][2]  for _, f in valid])
right_vis = np.mean([f["right_hip"][2] for _, f in valid])
side = _select_side(valid)
print(f"Avg visibility — left_hip: {left_vis:.2f}, right_hip: {right_vis:.2f} → side: {side}")

if not side:
    print("Side selection failed.")
    sys.exit(1)

hip_key  = f"{side}_hip"
knee_key = f"{side}_knee"

hip_ys   = [f[hip_key][1]  for _, f in valid]
knee_ys  = [f[knee_key][1] for _, f in valid]
smooth   = _rolling_average(hip_ys, 5)
bottom   = _find_bottom_frame(smooth)

print(f"\nBottom frame (local index in valid frames): {bottom}")
if bottom is not None:
    print(f"Bottom frame (global): {valid[bottom][0]}")
    print(f"  hip_y={hip_ys[bottom]:.1f}  knee_y={knee_ys[bottom]:.1f}  depth={hip_ys[bottom]>knee_ys[bottom]}")

# Show hip_y vs knee_y for all frames, marking depth condition
print(f"\nFrame-by-frame (every 3rd frame):")
print(f"  {'idx':>4}  {'hip_y':>7}  {'knee_y':>7}  {'depth?':>7}  {'margin':>7}")
for j, (i, f) in enumerate(valid):
    if j % 3 != 0:
        continue
    hy = f[hip_key][1]
    ky = f[knee_key][1]
    depth = hy > ky
    margin = hy - ky
    marker = "✓" if depth else " "
    b_marker = " <-- BOTTOM" if j == bottom else ""
    print(f"  {i:>4}  {hy:>7.1f}  {ky:>7.1f}  {marker:>7}  {margin:>+7.1f}{b_marker}")
