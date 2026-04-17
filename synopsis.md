# WhiteLights — Project Synopsis

Squat depth analysis tool for powerlifting. Analyzes side-profile video to determine whether a lifter achieves IPF-legal depth (hip crease below top of kneecap), and renders an annotated overlay video with per-rep judgment lights.

Built for the Rice powerlifting team.

---

## Stack

| Layer | Technology |
|---|---|
| Pose estimation | MediaPipe BlazePose Tasks API (`mediapipe >= 0.10`) — `pose_landmarker_full.task` |
| Video I/O | OpenCV (`opencv-python >= 4.8`) |
| Numerical ops | NumPy |
| Runtime | Python 3.12 (`/usr/local/bin/python3.12`) |
| Future UI | Streamlit (in requirements, not yet built) |
| Future AI | Anthropic SDK (in requirements, not yet built) |

---

## Project Structure

```
vantage/
├── depth_detector.py       # Core detection engine (landmark extraction, depth logic)
├── visualize.py            # Annotated video renderer + rep segmentation
├── metrics.py              # Back angle + bar path calculations
├── debug_single.py         # CLI debug tool (prints per-frame hip/knee values)
├── requirements.txt
├── .gitignore              # Excludes tests/raw_videos/, tests/annotated_videos/, models/
├── CLAUDE.md               # gstack skill routing config
├── synopsis.md             # This file
├── models/
│   └── pose_landmarker_full.task   # Auto-downloaded ~5MB MediaPipe model
└── tests/
    ├── raw_videos/         # Input iPhone MOV files (gitignored)
    ├── annotated_videos/   # Output annotated MP4s (gitignored)
    ├── labels.json         # Ground truth: 13 pass + 8 fail videos
    └── test_depth.py       # Test harness (80% accuracy gate)
```

---

## Key Design Decisions

### Landmark extraction (`depth_detector.py`)

Extracts 8 joints per frame from MediaPipe's 33-landmark BlazePose model:

| Joint | Landmark index |
|---|---|
| Hip (L/R) | 23/24 |
| Knee (L/R) | 25/26 |
| Shoulder (L/R) | 11/12 |
| Wrist (L/R) | 15/16 |
| Heel (L/R) | 29/30 |

Each frame stored as a dict: `{frame_idx, left_hip, right_hip, ..., width, height}`. Hip/knee include visibility score; others are (x, y) only. Frames with no detected pose store `None`.

**macOS gotcha**: `cv2.VideoCapture` auto-applies rotation metadata on macOS (`CAP_PROP_ORIENTATION_AUTO = 1` by default), which double-rotates iPhone MOV files. Fix: `cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)` immediately after opening every `VideoCapture`.

### Side selection

Picks left or right landmarks based on which hip has average visibility > 0.7. Can be overridden with `--side right` / `--side left` CLI flag in `visualize.py`.

### Depth detection

IPF rule: hip crease must pass below the top of the kneecap.

MediaPipe gives joint centers, not anatomical landmarks — landmark 23 is the femoral head (deep in pelvis), not the hip crease; landmark 25 is the tibiofemoral midpoint, not the patella top. The tool compensates with **estimated anatomical markers**:

- **Knee-top marker**: extends the heel→knee vector `KNEE_TOP_OVERSHOOT = 0.18` (18%) past the knee joint center
- **Hip-crease marker**: `HIP_CREASE_FRAC = 0.88` of the way along the shoulder→hip vector

Depth is judged on these estimated markers, not the raw joints:
- `hip_crease_y > knee_top_y` for `MIN_DEPTH_FRAMES = 3` consecutive frames → **PASS**
- Markers never get within `CLOSE_THRESHOLD = 0.02` (2% of frame height) → **FAIL**
- Within 2% but not crossing → **BORDERLINE**

Checked across the entire video (not just a window around the detected bottom), so the result matches what the overlay shows.

### Rep segmentation (`visualize.py`)

Uses **shoulder Y** (not hip Y) as the segmentation signal. Shoulder Y is low when standing, high at squat bottom — its local minima are standing positions between reps.

1. Extract per-frame shoulder Y for the tracked side
2. Smooth with `REP_SMOOTHING = 15` frame rolling average
3. Find local minima → rep boundary indices
4. Reject segments shorter than `MIN_REP_FRAMES = 15` frames (noise)
5. Reject segments where the estimated markers never got within `MIN_DESCENT_THRESHOLD = 0.10` (10% of frame height) of parallel — eliminates hip-hinge warmup movements that look like a rep boundary but aren't real squats
6. Fallback: entire video as one rep if no boundaries found

Each valid segment is classified independently using `_classify_segment()`.

### Two-pass rendering

Pass 1 (analysis): open video, extract all landmarks, segment reps, classify each.
Pass 2 (render): re-open video, iterate frames, draw overlays, write output.

Avoids holding all raw frames in RAM. Trade-off: video is decoded twice.

---

## visualize.py — Overlay Elements

| Element | Description |
|---|---|
| Skeleton | Gray lines: shoulder→hip, hip→knee, knee→heel. Gray dots at all joints. |
| Estimated marker line | Colored line between hip-crease and knee-top markers. **Green** = depth active, **Yellow** = borderline, **White** = above parallel. This line (not the skeleton) drives the depth color. |
| Hip trail | Last 60 hip positions as fading dots. Green when depth was active at that frame. |
| Magenta ring | Appears around the hip joint on the detected bottom frame of each rep. |
| Depth HUD (top-left) | "DEPTH +" / "BORDERLINE" / "NO DEPTH" text label. |
| Graph HUD (bottom-left) | 300×100px scrolling chart of smoothed hip_y (white) vs knee_y (yellow dashed) over last 90 frames. Green fill where depth is active. |
| Lights box (right of graph) | Always-visible dark box, same height as graph. Shows "REP N/M" counter bottom-right. After 75% of a rep's segment (post-bottom): 3 large circles appear — **white** = pass, **red** = fail, **yellow** = borderline. |

### Output toggles (top of visualize.py)

```python
SAVE_VIDEO = True    # write annotated MP4 to tests/annotated_videos/, auto-open after
SHOW_LIVE  = True    # display in cv2.imshow() window while rendering
```

### Tuning constants (top of visualize.py)

```python
KNEE_TOP_OVERSHOOT    = 0.18   # how far past knee joint the knee-top marker sits
HIP_CREASE_FRAC       = 0.88   # how far down shoulder→hip the crease marker sits
REP_SMOOTHING         = 15     # shoulder Y smoothing window for rep boundaries
MIN_REP_FRAMES        = 15     # min frames for a segment to count as a rep
MIN_DESCENT_THRESHOLD = 0.10   # min proximity to parallel to count as a real rep
```

Detection constants live in `depth_detector.py`:
```python
MIN_DEPTH_FRAMES  = 3     # consecutive depth frames required for pass
CLOSE_THRESHOLD   = 0.02  # within 2% of frame height = borderline
SMOOTHING_WINDOW  = 5     # hip Y rolling average for bottom detection
VISIBILITY_THRESHOLD = 0.7
```

---

## Usage

```bash
# Single rep, force right side (most common — lifter faces left toward camera)
python3.12 visualize.py tests/raw_videos/valid_1.MOV --side right

# Auto side-select (uses visibility scores)
python3.12 visualize.py tests/raw_videos/valid_1.MOV

# Debug: print per-frame hip/knee values without rendering
python3.12 debug_single.py tests/raw_videos/valid_1.MOV

# Run accuracy test suite (must hit ≥80% on 21 labeled videos)
python3.12 tests/test_depth.py
```

Output always goes to `tests/annotated_videos/<stem>_annotated.mp4`.

---

## What's Not Yet Built

- `app.py` — Streamlit UI (upload → analyze → display results)
- Claude AI coaching breakdown (Anthropic SDK is in requirements, not wired)
- Multi-rep support in `depth_detector.py`'s `analyze_video()` — it's still single-rep; multi-rep logic lives only in `visualize.py`
- Performance optimization — MediaPipe extraction runs ~24ms/frame sequentially; for longer sets this adds up
