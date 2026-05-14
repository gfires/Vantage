"""
params.py — All tunable constants for WhiteLights squat analysis.

Centralised here so depth detection, metrics, and visualisation share a
single source of truth.  Import from this module rather than defining
duplicates in pose, metrics, or visualize.
"""

# ── Anatomical marker estimation ──────────────────────────────────────────────
# Knee-top marker: extends the heel→knee vector this fraction *past* the knee
# joint center.  0.18 = 18% of heel-to-knee distance above the knee center.
KNEE_TOP_OVERSHOOT = 0.18

# Hip-crease marker: this fraction of the way along the shoulder→hip vector.
# 0.88 = 88% down from shoulder (≈ 12% above the hip joint center).
HIP_CREASE_FRAC = 0.88

# ── Depth detection ───────────────────────────────────────────────────────────
MIN_DEPTH_FRAMES     = 3     # consecutive depth frames required for PASS
SMOOTHING_WINDOW     = 5     # rolling average for hip-Y bottom detection
CLOSE_THRESHOLD      = 0.02  # within 2% of frame height → BORDERLINE

# ── State machine ─────────────────────────────────────────────────────────────
MIN_HOLD_FRAMES       = 4     # consecutive frames a direction must hold before a
                               # transition is confirmed (suppresses jitter)
MIN_DESCENT_THRESHOLD = 0.02  # min hip drop (fraction of frame height) to confirm descent
ASCENT_RECOVERY_FRAC  = 0.90  # fraction of descent recovered before ASCENDING→STANDING fires

# Pipeline delay = how many frames behind the render loop operates relative to
# inference.  Derived — do not set directly.  Changing SMOOTHING_WINDOW or
# MIN_HOLD_FRAMES automatically adjusts the delay.
PIPELINE_DELAY = SMOOTHING_WINDOW + MIN_HOLD_FRAMES  # currently 5 + 4 = 9 frames

# ── Drawing smoothing (display only — never affects classification) ────────────
DRAW_SMOOTHING = 3           # rolling average for skeleton/marker rendering coords

# ── Speed & tempo thresholds ──────────────────────────────────────────────────
DESCENT_FAST_S   = 1.0   # descent faster than this → flag "FAST DESC"
DESCENT_SLOW_S   = 4.0   # descent slower than this → flag "SLOW DESC"
# Ascent/descent ratio above which the rep is flagged as a grind
GRIND_RATIO      = 2   # ascent_time > descent_time * GRIND_RATIO → "GRIND"

# ── Tibial angle ──────────────────────────────────────────────────────────────
# Shin angle from vertical (degrees), roll- and azimuth-corrected.
# Low-bar squat context: upright shin is expected; >35 deg is notable.
TIBIAL_NOTE_DEG = 25.0   # above this → forward knee travel worth watching (low-bar)
TIBIAL_WARN_DEG = 35.0   # above this → outside low-bar norms, ankle mobility / stance issue

# ── Coaching thresholds (low-bar powerlifting) ────────────────────────────────
# HOLE/MCV ratio: HOLE velocity as a fraction of MCV.
# Good hole exit = driving hard from the bottom; low ratio = stalling at the bottom.
HOLE_MCV_WARN  = 0.60    # HOLE < 60% of MCV → weak hole exit
HOLE_MCV_NOTE  = 0.80    # HOLE < 80% of MCV → slight hole stall (info only)


# ── Velocity ──────────────────────────────────────────────────────────────────
# Fraction of the ascent used as the "out of hole" window.
HOLE_EXIT_FRACTION = 0.25   # first 25% of ascent frames — covers the true out-of-hole drive window

# ── Camera calibration (upright tilt detection) ───────────────────────────────
CAL_PROBE_FRAMES   = 3       # frames to sample for upright/azimuth detection (uses the pipeline delay buffer)
CAL_BLUR_KERNEL    = (5, 5)  # Gaussian blur kernel for edge pre-processing
CAL_CANNY_LOW      = 50      # Canny lower threshold
CAL_CANNY_HIGH     = 150     # Canny upper threshold
CAL_HOUGH_THRESHOLD   = 40   # HoughLinesP accumulator votes (at half-res)
CAL_HOUGH_MIN_LENGTH  = 40   # minimum line length in pixels (at half-res)
CAL_HOUGH_MAX_GAP     = 10   # maximum gap between collinear segments (at half-res)
CAL_UPRIGHT_TOL_DEG   = 20   # lines within this many degrees of vertical are upright candidates
CAL_TILT_MAX_DEG      = 4.0  # detected tilt clamped to this range; fallback 0.0 if no upright found
