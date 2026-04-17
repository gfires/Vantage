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
DESCENT_FAST_S   = 1.5   # descent faster than this → flag "FAST DESC"
DESCENT_SLOW_S   = 4.0   # descent slower than this → flag "SLOW DESC"
# Ascent/descent ratio above which the rep is flagged as a grind
GRIND_RATIO      = 1.5   # ascent_time > descent_time * GRIND_RATIO → "GRIND"

# ── Tibial angle ──────────────────────────────────────────────────────────────
# Shin angle from vertical (degrees).  Measured as atan2(|knee_x - heel_x|, |knee_y - heel_y|).
TIBIAL_WARN_DEG = 40.0   # above this during descent → forward lean / ankle mobility flag

# ── Velocity ──────────────────────────────────────────────────────────────────
# Fraction of the ascent used as the "out of hole" window.
HOLE_EXIT_FRACTION = 0.15   # first 15% of ascent frames
