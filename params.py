"""
params.py — All tunable constants for WhiteLights squat analysis.

Centralised here so depth detection, metrics, and visualisation share a
single source of truth.  Import from this module rather than defining
duplicates in depth_detector, metrics, or visualize.
"""

# ── Anatomical marker estimation ──────────────────────────────────────────────
# Knee-top marker: extends the heel→knee vector this fraction *past* the knee
# joint center.  0.18 = 18% of heel-to-knee distance above the knee center.
KNEE_TOP_OVERSHOOT = 0.18

# Hip-crease marker: this fraction of the way along the shoulder→hip vector.
# 0.88 = 88% down from shoulder (≈ 12% above the hip joint center).
HIP_CREASE_FRAC = 0.88

# ── Depth detection ───────────────────────────────────────────────────────────
VISIBILITY_THRESHOLD = 0.7   # min landmark visibility to use a side
DEPTH_WINDOW         = 5     # frames around bottom to check depth condition
MIN_DEPTH_FRAMES     = 3     # consecutive depth frames required for PASS
SMOOTHING_WINDOW     = 5     # rolling average for hip-Y bottom detection
CLOSE_THRESHOLD      = 0.02  # within 2% of frame height → BORDERLINE

# ── Rep segmentation ──────────────────────────────────────────────────────────
REP_SMOOTHING         = 15   # smoothing window for hip-crease height signal
MIN_REP_FRAMES        = 15   # min frames between standing peaks / min segment length
MIN_DESCENT_THRESHOLD = 0.10 # min hip drop (fraction of frame height) to count as a rep

# ── Drawing smoothing (display only — never affects classification) ────────────
DRAW_SMOOTHING = 3           # rolling average for skeleton/marker rendering coords

# ── Speed & tempo thresholds ──────────────────────────────────────────────────
DESCENT_FAST_S   = 1.29   # descent faster than this → flag "FAST DESC"
DESCENT_SLOW_S   = 4.0   # descent slower than this → flag "SLOW DESC"
# Ascent/descent ratio above which the rep is flagged as a grind
GRIND_RATIO      = 2   # ascent_time > descent_time * GRIND_RATIO → "GRIND"

# ── Tibial angle ──────────────────────────────────────────────────────────────
# Shin angle from vertical (degrees).  Measured as atan2(|knee_x - heel_x|, |knee_y - heel_y|).
# Low-bar squat context: upright shin is expected; >35 deg is notable, >45 deg is problematic.
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
