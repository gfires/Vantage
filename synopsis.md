# WhiteLights — Project Synopsis

Squat depth analysis tool for powerlifting. Analyzes side-profile video to determine whether a lifter achieves IPF-legal depth (hip crease below top of kneecap), renders an annotated overlay video with per-rep judgment lights and coaching metrics, and outputs a plain-text rep summary table.

Built for the Rice powerlifting team (low-bar focus).

---

## Stack

| Layer | Technology |
|---|---|
| Pose estimation | MediaPipe BlazePose Tasks API (`mediapipe >= 0.10`) — `pose_landmarker_full.task` |
| Video I/O | OpenCV (`opencv-python >= 4.8`) |
| Numerical ops | NumPy, SciPy (`find_peaks` for rep segmentation) |
| Runtime | Python 3.12 (`/usr/local/bin/python3.12`) |
| Future UI | Streamlit (in requirements, not yet built) |
| Future AI | Anthropic SDK (in requirements, not yet built) |

---

## Project Structure

```
vantage/
├── depth_detector.py       # Core detection engine (landmark extraction, depth logic)
├── visualize.py            # Annotated video renderer + rep segmentation + table output
├── metrics.py              # Coaching metrics: tempo, tibial angle, depth angle
├── params.py               # Single source of truth for all tunable constants
├── debug_single.py         # CLI debug tool (prints per-frame hip/knee values)
├── requirements.txt
├── .gitignore              # Excludes tests/raw_videos/, tests/annotated_videos/, models/
├── CLAUDE.md               # gstack skill routing config
├── synopsis.md             # This file
├── models/
│   └── pose_landmarker_full.task   # Auto-downloaded ~5MB MediaPipe model
└── tests/
    ├── raw_videos/         # Input iPhone MOV files (gitignored)
    ├── annotated_videos/   # Output annotated MP4s + rep tables (gitignored)
    ├── labels.json         # Ground truth: 13 pass + 8 fail videos
    └── test_depth.py       # Test harness (80% accuracy gate)
```

---

## Key Design Decisions

### Landmark extraction (`depth_detector.py`)

Extracts 7 joints per frame from MediaPipe's 33-landmark BlazePose model (wrist tracking removed as inaccurate):

| Joint | Landmark index |
|---|---|
| Hip (L/R) | 23/24 |
| Knee (L/R) | 25/26 |
| Shoulder (L/R) | 11/12 |
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

### Bottom frame detection (`depth_detector.py: _find_bottom_frame`)

Each rep segment starts and ends at a standing position (peak of the height signal). The true bottom is the deepest hip position — `argmax(hip_y)` — within the middle 80% of the segment. The 10% margin at each edge prevents standing-position noise from triggering a false bottom near the start or end of the segment.

### Rep segmentation (`visualize.py`)

Uses **hip-crease Y relative to heel Y** as the segmentation signal. This normalizes for the lifter's position in frame and uses the same anatomical marker that drives depth detection.

Signal: `heel_y − hip_crease_y` — large when standing tall, small at squat bottom.

1. Smooth with `REP_SMOOTHING = 15` frame rolling average
2. Find peaks of the smoothed signal (standing positions) using `scipy.signal.find_peaks` with:
   - `prominence ≥ 8% of frame height` — rejects noise and partial movements
   - `distance ≥ MIN_REP_FRAMES` — enforces minimum spacing between reps
3. Wrap detected peaks with sentinel boundaries `[0, peaks..., n]` so the first and last reps are not dropped
4. State machine confirmation: each candidate segment must show a descent of at least `MIN_DESCENT_THRESHOLD = 0.10` (10% of frame height) from its standing peak to its valley — rejects hip-hinge setup movements
5. Fallback: entire video as one rep if no peaks found

Each valid segment is classified independently using `_classify_segment()`.

### Drawing smoothing

Raw MediaPipe landmarks jitter slightly frame-to-frame even during still positions. To eliminate this visually without affecting classification accuracy, a separate `draw_frames` list is computed during analysis:

- Each joint's x/y is smoothed with a `DRAW_SMOOTHING = 3` frame rolling average
- `draw_frames` is used exclusively for skeleton and marker rendering, and for tibial angle computation (so the on-video arc and displayed value are consistent)
- `frames_data` (unsmoothed) is used for depth classification, depth state logic, and depth angle computation

### Two-pass rendering

Pass 1 (analysis): open video, extract all landmarks, build `frames_data` and `draw_frames`, segment reps, classify each, compute all metrics.
Pass 2 (render): re-open video, iterate frames, draw overlays, write output.

Avoids holding all raw frames in RAM. Trade-off: video is decoded twice.

### Render loop: single overlay blend

All semi-transparent dark backing rects (graph panel, lights box, rep counter, metrics HUD, coaching panel) are drawn onto a single `overlay = frame.copy()` per frame, then blended once with `cv2.addWeighted`. All opaque content (skeleton, graph lines, circles, text) is drawn directly onto the post-blend frame.

---

## Coaching Metrics (`metrics.py`)

Computed per rep after classification. All use raw `frames_data` except tibial angle (uses `draw_frames` for display consistency).

### Tempo (`compute_tempo`)

| Metric | Description |
|---|---|
| DESC | Descent duration (rep start → bottom), seconds |
| ASC | Ascent duration (bottom → rep end), seconds |
| HOLE | Mean velocity over first 25% of ascent (fh/s) — the out-of-hole drive window |
| MCV | Mean concentric velocity over full ascent (fh/s) — VBT autoregulation proxy |
| hole_mcv_ratio | HOLE ÷ MCV — hole-exit quality relative to overall ascent speed |

Velocity units are normalized frame-heights per second (`fh/s`) — dimensionless, relative within a lifter's own data.

**Tempo flags** (appear in coaching panel and warnings table):

| Flag | Condition |
|---|---|
| FAST DESC | Descent < 1.5s |
| SLOW DESC | Descent > 4.0s |
| GRIND | Ascent > 1.5× descent time |
| WEAK HOLE | HOLE < 60% of MCV |

### Tibial angle (`compute_tibial_angle`)

Shin angle from vertical: `atan2(|knee_x − heel_x|, |heel_y − knee_y|)` in degrees. 0° = perfectly vertical shin; increases as knee travels forward.

Computed using smoothed `draw_frames` landmarks so the displayed arc on video matches the reported value.

**Thresholds (low-bar calibrated):**

| Range | Flag |
|---|---|
| > 35° | KNEE TOO FORWARD |
| > 25° | WATCH KNEES |

### Depth angle (`compute_depth_angle`)

Angle of the hip-crease → knee-top line against horizontal, at the best sustained depth position.

Scans all frames in the rep for runs of ≥ `MIN_DEPTH_FRAMES` consecutive frames where `hc_y > kt_y`. Returns the most negative angle within any qualifying run (most negative = deepest below parallel). Falls back to best single-frame angle if no qualifying run exists.

Convention: **positive = hip crease below knee top (at depth)**, negative = above parallel.

---

## visualize.py — Overlay Elements

| Element | Description |
|---|---|
| Skeleton | Gray lines: shoulder→hip, hip→knee, knee→heel. Gray dots at all joints. |
| Estimated marker line | Colored line between hip-crease and knee-top markers. **Green** = depth active, **Yellow** = borderline, **White** = above parallel. |
| Bottom marker | Magenta ring on hip-crease + "BOTTOM" label at the detected hole frame. |
| Tibial angle annotation | Purple arc between shin line and vertical reference, with angle label in degrees near the heel. Yellow when > warn threshold. Only shown during active reps. |
| Graph HUD (bottom-left) | 300×100px scrolling chart of smoothed hip_y (white) vs knee_y (yellow dashed) over last 90 frames. Green fill where depth is active. |
| Lights box (right of graph) | Three large circles after 75% of rep duration — **white** = pass, **red** = fail, **yellow** = borderline. |
| Rep counter (bottom-right) | "REP N/M". |
| Metrics HUD (top-right) | Live phase label (DESCENT / HOLE / ASCENT), DESC/ASC times, HOLE and MCV velocities (fh/s). Shown during active reps only. |
| Coaching panel (below metrics HUD) | Coaching flags: FAST DESC, SLOW DESC, GRIND, WEAK HOLE, KNEE TOO FORWARD, WATCH KNEES. Shown after the bottom frame. |

### Output toggles (top of `visualize.py`)

```python
SAVE_VIDEO = True    # write annotated MP4 to tests/annotated_videos/, auto-open after
SHOW_LIVE  = True    # display in cv2.imshow() window while rendering
```

---

## Rep Summary Table

In addition to the annotated video, a plain-text table is written to `tests/annotated_videos/<stem>_table.txt` and printed to the terminal.

Columns: one per rep. Rows:

| Row | Content |
|---|---|
| Result | PASS / FAIL / BORDERLINE |
| Hole time | Duration of first 25% of ascent (seconds) |
| Ascent time | Full concentric phase (seconds) |
| Depth angle | Hip-crease→knee-top angle vs horizontal at best sustained depth (degrees) |
| Max shin | Peak tibial angle during the rep (degrees) |
| Warnings | All coaching flags for the rep |

---

## All Tunable Constants (`params.py`)

```python
# Anatomical markers
KNEE_TOP_OVERSHOOT    = 0.18   # how far past knee joint the knee-top marker sits
HIP_CREASE_FRAC       = 0.88   # how far down shoulder→hip the crease marker sits

# Depth detection
MIN_DEPTH_FRAMES      = 3      # consecutive depth frames required for PASS
CLOSE_THRESHOLD       = 0.02   # within 2% of frame height = BORDERLINE
SMOOTHING_WINDOW      = 5      # hip Y rolling average for bottom detection
VISIBILITY_THRESHOLD  = 0.7    # min landmark visibility to use a side

# Rep segmentation
REP_SMOOTHING         = 15     # hip-crease height signal smoothing for rep boundaries
MIN_REP_FRAMES        = 15     # min frames between standing peaks / min segment length
MIN_DESCENT_THRESHOLD = 0.10   # min hip drop fraction to count as a rep

# Drawing
DRAW_SMOOTHING        = 3      # rolling average for skeleton/marker rendering (display only)

# Tempo thresholds
DESCENT_FAST_S        = 1.5
DESCENT_SLOW_S        = 4.0
GRIND_RATIO           = 1.5    # ascent > descent × this → GRIND

# Tibial angle (low-bar calibrated)
TIBIAL_NOTE_DEG       = 25.0   # WATCH KNEES
TIBIAL_WARN_DEG       = 35.0   # KNEE TOO FORWARD

# Velocity
HOLE_EXIT_FRACTION    = 0.25   # first 25% of ascent = hole-exit window
HOLE_MCV_WARN         = 0.60   # HOLE < 60% of MCV → WEAK HOLE
HOLE_MCV_NOTE         = 0.80   # informational threshold
```

---

## Performance Profile

Measured on 1920×1080 @ 28fps video (Apple M3):

| Phase | Cost |
|---|---|
| MediaPipe landmark extraction | ~21.8ms/frame (dominant cost, CPU-bound) |
| Render loop total | ~10.8ms/frame |
| — VideoWriter encode | ~4.1ms/frame |
| — Decode (pass 2) | ~3.1ms/frame |
| — Rotate | ~2.1ms/frame |
| — Overlay blend | ~1.5ms/frame |
| — All drawing combined | ~0.3ms/frame |

Real-time budget at 28fps is 36ms/frame. The render loop is well within budget; MediaPipe is the ceiling.

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

Outputs go to `tests/annotated_videos/`:
- `<stem>_annotated.mp4` — annotated video
- `<stem>_table.txt` — rep summary table

---

## What's Not Yet Built

- `app.py` — Streamlit UI (upload → analyze → display results)
- Claude AI coaching breakdown (Anthropic SDK is in requirements, not wired)
- Multi-rep support in `depth_detector.py`'s `analyze_video()` — it's still single-rep; multi-rep logic lives only in `visualize.py`
- GPU inference — MediaPipe BlazePose supports CoreML/GPU delegates; would reduce extraction from ~21.8ms to ~2–5ms/frame
