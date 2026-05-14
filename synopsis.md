# WhiteLights — Project Synopsis

Squat depth analysis tool for powerlifting. Analyzes side-profile video to determine whether a lifter achieves IPF-legal depth (hip crease below top of kneecap), renders an annotated overlay video with per-rep judgment lights and coaching metrics, and outputs a plain-text rep summary table.

Built for the Rice powerlifting team (low-bar focus). Targeting extension to arbitrary lifters and camera setups via camera calibration.

---

## Stack

| Layer | Technology |
|---|---|
| Pose estimation | MediaPipe BlazePose Tasks API (`mediapipe >= 0.10`) — `pose_landmarker_full.task` |
| Video I/O | OpenCV (`opencv-python >= 4.8`) |
| Numerical ops | NumPy |
| Runtime | Python 3.12 |
| Web backend | FastAPI + Uvicorn (`api.py`) |
| Web frontend | Plain HTML/CSS/JS (`index.html`) — no framework |
| Future AI | Anthropic SDK (in requirements, not yet wired) |

---

## Project Structure

```
vantage/
├── pose.py       # Landmark extraction, depth logic, side selection
├── state_machine.py        # Causal per-frame rep segmentation state machine
├── metrics.py              # Coaching metrics: tempo, tibial angle, depth angle
├── params.py               # Single source of truth for all tunable constants
├── api.py                  # FastAPI web backend (upload, MJPEG stream, status, download)
├── index.html              # Single-page frontend
├── debug_single.py         # CLI debug tool (per-frame hip/knee values)
├── bench_inference.py      # MediaPipe inference benchmarking
├── requirements.txt
├── synopsis.md             # This file
├── CLAUDE.md               # gstack skill routing config
├── rendering/
│   ├── pipeline.py         # Single-pass inference + render loop (used by both CLI and API)
│   ├── draw.py             # All frame annotation primitives
│   └── visualize.py        # CLI entry point — outputs annotated MP4 + rep table
├── scripts/
│   └── calibrate.py        # Diagnostic: 3-axis camera calibration (vertical + azimuth + sagittal, 5 probe frames)
├── models/
│   └── pose_landmarker_full.task   # Auto-downloaded ~9MB MediaPipe model
└── tests/
    ├── raw_videos/         # Input iPhone MOV files (gitignored)
    ├── annotated_videos/   # Output annotated MP4s + rep tables (gitignored)
    ├── labels.json         # Ground truth: 13 pass + 8 fail videos
    └── test_depth.py       # Test harness (80% accuracy gate on 21 labeled videos)
```

---

## Key Design Decisions

### Signs and numerical conventions

Y increases downward (screen coordinates). Everything follows from that:

| Signal | Small value | Large value |
|---|---|---|
| `hip_y` | hips high (standing) | hips low (squat bottom) |
| `hc_y` | hip crease high | hip crease low |
| `kt_y` | knee top high | knee top low |

- **Depth flag**: `hc_y > kt_y` → hip crease is below knee top in screen coords → depth achieved
- **Depth angle**: `rise = hc_y - kt_y` → positive = depth achieved, negative = short
- **Tibial angle**: roll-corrected then azimuth-corrected: `atan(tan(θ_obs) / sin(φ))` — always ≥ 0°, increases as knee travels forward
- **Depth angle**: same two-step correction applied to the hc→kt vector angle against horizontal
- **Azimuth φ**: measured from vertical; 0° = pure side profile, 90° = facing camera. `sin(φ)` is the foreshortening factor (sin≈1 at side-on → minimal correction)
- **Velocity**: `−Δhc_y × fps / frame_height` — negated so upward motion is positive

### State machine transitions (`state_machine.py`)

```
STANDING → DESCENDING → ASCENDING → STANDING → ...
```

- `STANDING → DESCENDING`: smoothed hip_y increases for `MIN_HOLD_FRAMES` consecutive frames AND hip has dropped ≥ `MIN_DESCENT_THRESHOLD` below the standing peak
- `DESCENDING → ASCENDING`: smoothed hip_y decreases for `MIN_HOLD_FRAMES` consecutive frames
- `ASCENDING → STANDING`: recovered fraction `(bottom_val − smooth_val) / (bottom_val − standing_peak)` exceeds `ASCENT_RECOVERY_FRAC`

`standing_peak` = rolling minimum of smoothed hip_y while STANDING (highest the hips sit at rest).
`bottom_candidate` = maximum smoothed hip_y seen during descent (deepest the hips reach).

Transitions are **retroactive**: when the hold count completes, the event is declared to have occurred `MIN_HOLD_FRAMES` frames in the past. This is absorbed by the pipeline ring buffer.

### Single-pass pipeline (`rendering/pipeline.py`)

Replaces the old two-pass analyze→render sequence. A single forward sweep does inference, state machine updates, and rendering simultaneously. A `PIPELINE_DELAY`-frame ring buffer absorbs the lookahead needed for smoothing and transition confirmation.

**Per-frame loop:**
1. Decode + rotate → raw BGR frame
2. `_infer_one_frame` (MediaPipe) → `fdata | None`
3. Push `(frame_idx, raw_frame, fdata)` → ring buffers
4. Pop tail (PIPELINE_DELAY frames behind)
5. `sm.feed(tail_idx, tail_fdata)` → `completed_rep | None`
6. `_smooth_one_frame(tail_fdata, bufs)` → `fdata_draw`
7. Draw all overlays onto `tail_frame`
8. JPEG-encode and stream / write to MP4

**Probe phase** (before main loop): reads first `PIPELINE_DELAY` frames, selects side via z-coordinates, runs camera calibration on the first `CAL_PROBE_FRAMES = 3` of those frames (vertical tilt via Hough + azimuth via heel-vector median), constructs `RepStateMachine` with a `CameraCalibration` object. Zero additional latency — calibration runs within the frames already buffered.

**Flush phase** (after decode ends): drains the remaining buffered frames without new inference.

### Camera calibration (`pose.py`, `scripts/calibrate.py`, `rendering/pipeline.py`)

`CameraCalibration` dataclass (in `pose.py`) carries two values determined during the probe phase:

| Field | Source | Convention |
|---|---|---|
| `roll_deg` | Hough line on rack upright; median of probe frames, clamped ±4° | positive = top of upright leans right |
| `azimuth_deg` | Heel-vector angle from vertical; median of probe frames; wrists as fallback | 0° = pure side profile, 90° = facing camera |

**Roll correction** (frame of reference): the `(dx, dy)` heel→knee or hc→kt vector is decomposed onto the true vertical/horizontal axes via a 2D rotation by `roll_deg`. The coordinate system rotates; the landmarks do not.

**Azimuth correction** (foreshortening): the sagittal plane is foreshortened by `sin(φ)` when the lifter is `φ` degrees from pure side-on. Applied after roll:
```
tan(θ_true) = tan(θ_obs) / sin(φ)
```
- Depth pass/fail (`hc_y > kt_y`) is azimuth-invariant (Y-comparison; both landmarks compress equally in X) — no correction applied.
- `min_gap_px` (borderline threshold) uses roll-corrected pixel units — azimuth correction not needed since it's compared against a relative threshold.

### Landmark extraction (`pose.py`)

Extracts 10 joints per frame from MediaPipe's 33-landmark BlazePose model:

| Joint | Landmark index | Fields |
|---|---|---|
| Hip (L/R) | 23/24 | x, y, visibility, z |
| Knee (L/R) | 25/26 | x, y, visibility |
| Shoulder (L/R) | 11/12 | x, y |
| Heel (L/R) | 29/30 | x, y, visibility |
| Wrist (L/R) | 15/16 | x, y, visibility |

Hip z-depth is used for side selection (negative = closer to camera).

**macOS gotcha**: `cv2.VideoCapture` auto-applies rotation metadata on macOS, double-rotating iPhone MOV files. Fix: `cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)` immediately after opening.

### Side selection (`pose._select_side`)

Picks whichever hip's z-coordinate is more negative (closer to camera) from the first valid probe frame. Can be overridden with `force_side` in `_process_video`.

### Depth detection

IPF rule: hip crease must pass below the top of the kneecap. MediaPipe gives joint centers, not anatomical landmarks, so the tool estimates them:

- **Knee-top marker**: extends the heel→knee vector `KNEE_TOP_OVERSHOOT = 0.18` (18%) past the knee joint center
- **Hip-crease marker**: `HIP_CREASE_FRAC = 0.88` of the way along the shoulder→hip vector

Classification per rep:
- `hc_y > kt_y` for ≥ `MIN_DEPTH_FRAMES = 3` consecutive frames → **PASS**
- Closest approach within `CLOSE_THRESHOLD = 0.02` (2% of frame height) → **BORDERLINE**
- Otherwise → **FAIL**

### Draw smoothing

MediaPipe landmarks jitter frame-to-frame. A separate `_smooth_one_frame` pass applies a `DRAW_SMOOTHING = 3` frame rolling box average per joint, producing `fdata_draw` used exclusively for skeleton rendering and tibial arc display. Classification always uses raw `fdata`.

### Axes compass overlay (`rendering/draw.py`)

Top-left debug box showing three calibrated axes as arrows:
- **V** (green) — true vertical after roll correction
- **H** (white) — sagittal horizontal axis
- **Az** (orange) — heel-vector azimuth, φ° from vertical in the frontal plane

---

## Coaching Metrics (`metrics.py`)

Computed per rep after classification.

### Tempo (`compute_tempo`)

| Metric | Description |
|---|---|
| `descent_s` | Descent duration (rep start → bottom), seconds |
| `ascent_s` | Ascent duration (bottom → rep end), seconds |
| `mean_concentric_vel` | Mean hip-crease velocity over full ascent (frame-heights/s) |
| `hole_exit_vel` | Mean velocity over first 25% of ascent (out-of-hole drive window) |
| `hole_mcv_ratio` | `hole_exit_vel / mean_concentric_vel` — hole quality vs overall speed |

**Flags:**

| Flag | Condition |
|---|---|
| FAST DESC | `descent_s < 1.0s` |
| SLOW DESC | `descent_s > 4.0s` |
| GRIND | `ascent_s > descent_s × 2` |
| WEAK HOLE | `hole_mcv_ratio < 0.60` |

### Tibial angle (`compute_tibial_angle`)

`atan2(|knee_x − heel_x|, |heel_y − knee_y|)` in degrees. 0° = vertical shin; increases as knee travels forward.

| Range | Flag |
|---|---|
| > 35° | KNEES TOO FORWARD |
| > 25° | KNEES SLIGHTLY FORWARD |

### Depth angle (`compute_depth_angle`)

Angle of the hip-crease → knee-top line against horizontal at the best sustained depth. Scans all frames in the rep for runs of ≥ `MIN_DEPTH_FRAMES` consecutive depth frames; returns the most positive angle within any qualifying run. Falls back to best single-frame angle if no qualifying run.

**Convention**: positive = hip crease below knee top (at depth), negative = above parallel.

---

## Rendering Overlays (`rendering/draw.py`)

All overlays drawn in-place onto the BGR numpy frame. Semi-transparent elements use a single `cv2.addWeighted` blend per frame.

| Element | Description |
|---|---|
| Skeleton | Shoulder→hip, hip→knee, knee→heel lines + joint dots |
| Marker line | Hip-crease↔knee-top line. Green = depth active, yellow = borderline, white = above parallel |
| Bottom marker | Magenta ring on hip-crease at detected hole frame |
| Tibial arc | Purple arc between shin line and vertical, with angle label near heel |
| Graph HUD | 90-frame scrolling chart of hip_y (white) vs knee_y (yellow dashed). Green fill where depth active |
| Lights box | Pass/fail/borderline ring indicator. Shown after 75% of rep duration |
| Phase box | Current state: DESCENDING / ASCENDING |
| Metrics HUD | Live DESC/ASC times, HOLE and MCV velocities |
| Coaching panel | Active flags displayed after bottom frame |
| Rep counter | "REP N" |
| Side badge | "LEFT" / "RIGHT" |

---

## All Tunable Constants (`params.py`)

```python
# Anatomical markers
KNEE_TOP_OVERSHOOT    = 0.18   # fraction past knee joint for knee-top marker
HIP_CREASE_FRAC       = 0.88   # fraction along shoulder→hip for crease marker

# Depth detection
MIN_DEPTH_FRAMES      = 3      # consecutive depth frames required for PASS
CLOSE_THRESHOLD       = 0.02   # within 2% of frame height → BORDERLINE
SMOOTHING_WINDOW      = 5      # hip Y rolling average for state machine signal

# State machine
MIN_HOLD_FRAMES       = 4      # consecutive frames to confirm phase transition
MIN_DESCENT_THRESHOLD = 0.02   # min hip drop (fraction of frame height) to enter DESCENDING
ASCENT_RECOVERY_FRAC  = 0.90   # fraction of descent recovered to exit ASCENDING
PIPELINE_DELAY        = 9      # SMOOTHING_WINDOW + MIN_HOLD_FRAMES (derived, do not set directly)

# Drawing
DRAW_SMOOTHING        = 3      # rolling average for skeleton rendering (display only)

# Tempo thresholds
DESCENT_FAST_S        = 1.0
DESCENT_SLOW_S        = 4.0
GRIND_RATIO           = 2      # ascent > descent × this → GRIND

# Tibial angle (low-bar calibrated)
TIBIAL_NOTE_DEG       = 25.0   # KNEES SLIGHTLY FORWARD
TIBIAL_WARN_DEG       = 35.0   # KNEES TOO FORWARD

# Velocity
HOLE_EXIT_FRACTION    = 0.25   # first 25% of ascent = hole-exit window
HOLE_MCV_WARN         = 0.60   # HOLE < 60% of MCV → WEAK HOLE
HOLE_MCV_NOTE         = 0.80   # informational threshold

# Camera calibration
CAL_PROBE_FRAMES      = 3      # frames to sample for upright/azimuth detection (subset of PIPELINE_DELAY buffer)
CAL_BLUR_KERNEL       = (5, 5) # Gaussian blur kernel before Canny
CAL_CANNY_LOW         = 50
CAL_CANNY_HIGH        = 150
CAL_HOUGH_THRESHOLD   = 40     # HoughLinesP accumulator votes (at half-res)
CAL_HOUGH_MIN_LENGTH  = 40     # minimum line length in pixels (at half-res)
CAL_HOUGH_MAX_GAP     = 10     # maximum collinear gap in pixels (at half-res)
CAL_UPRIGHT_TOL_DEG   = 20     # max deviation from vertical to count as upright candidate
CAL_TILT_MAX_DEG      = 4.0    # detected tilt clamped to this range; fallback 0.0 if no upright found
```

---

## Performance Profile

Measured on 1920×1080 @ 28fps (Apple M3, CPU inference):

| Operation | Cost |
|---|---|
| MediaPipe inference | ~21.8ms/frame (dominant, CPU-bound) |
| Full render loop | ~10.8ms/frame |
| — VideoWriter encode | ~4.1ms/frame |
| — Decode | ~3.1ms/frame |
| — Rotate | ~2.1ms/frame |
| — Overlay blend | ~1.5ms/frame |
| — All drawing | ~0.3ms/frame |
| Hough calibration (half-res) | ~1.8ms/frame at 1080p, ~6.9ms at 4K |

Real-time budget at 28fps is 36ms/frame. MediaPipe is the ceiling. Pre-render probe latency is ~300ms at 30fps (`PIPELINE_DELAY` frames × ~22ms + decode overhead).

---

## Usage

### Web UI

```bash
.venv/bin/python -m uvicorn api:app --reload --port 8000
# open http://localhost:8000
```

Upload a side-profile squat video. The UI streams annotated frames live via MJPEG, then swaps to a pauseable H.264 player on completion. Rep table appears as soon as analysis completes.

### CLI

```bash
# Single video, auto side-select
.venv/bin/python rendering/visualize.py path/to/video.MOV

# Force side
.venv/bin/python rendering/visualize.py path/to/video.MOV --side right

# Debug: per-frame hip/knee values without rendering
.venv/bin/python debug_single.py path/to/video.MOV

# Accuracy test suite (≥80% gate on 21 labeled videos)
.venv/bin/python tests/test_depth.py
```

### Camera calibration diagnostic

```bash
.venv/bin/python scripts/calibrate.py path/to/video.MOV
```

Runs 3-axis calibration on the first 5 frames. Prints per-frame vertical tilt and median to terminal. Writes `<stem>_calibration.jpg` — 5 annotated frames stacked vertically, each showing:

| Axis | Color | Method |
|---|---|---|
| Vertical | Green | Longest near-vertical Hough line (rack upright). Extended full frame height + white reference line. |
| Azimuth | Cyan | BlazePose heel landmark vector (wrists as fallback). Centered arrow. |
| Sagittal | Orange | Perpendicular to the detected vertical line through its midpoint, extended to frame width. |

All three axes degrade gracefully: sagittal requires vertical; azimuth requires pose detection above 0.5 visibility.

---

## Web Architecture (`api.py`)

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve `index.html` |
| `POST` | `/upload` | Save video, start background job, return `{"job_id", "duration_s"}` |
| `GET` | `/stream/{job_id}` | MJPEG stream (`multipart/x-mixed-replace`) |
| `GET` | `/status/{job_id}` | `{"state": "analyzing"\|"done"\|"error", "reps": [...]}` |
| `GET` | `/download/{job_id}` | Serve final H.264 MP4 |

### Job lifecycle

1. Upload saves to temp dir, probes metadata, spawns background thread
2. Thread calls `_process_video` from `rendering/pipeline.py` with `on_frame` callback
3. `on_frame` puts each JPEG frame on a `queue.Queue(maxsize=30)`; `/stream` drains it as MJPEG
4. On completion: `state=done`, MP4 available at `/download`

---

## In Progress / Not Yet Built

- **Claude AI coaching**: Anthropic SDK in requirements, not wired
- **GPU inference**: MediaPipe supports CoreML delegates; would reduce ~21.8ms → ~2–5ms/frame
- **Side selection UI**: `api.py` currently hardcodes `force_side="right"`
