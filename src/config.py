"""
config.py
=========
Central configuration for the CE5033 Final Project (Bikez motorcycle dataset).

Holds: file paths, the list of numeric features we extract from the raw text,
their physically-plausible value ranges (used for outlier handling), and a few
analysis constants. Importing this single module keeps every analysis script
consistent and fully reproducible.

Author : (your name)
Course : CE5033 Statistical Methods and Data Mining, NCU CSIE
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# 1. Project paths
# --------------------------------------------------------------------------- #
# src/  -> project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = Path("/home/u5534225/archive")          # location of the raw download

RAW_CSV      = RAW_DIR / "all_bikez_raw.csv"
BRANDS_CSV   = RAW_DIR / "bikez_brands.csv"

DATA_DIR     = PROJECT_ROOT / "data"
FIG_DIR      = PROJECT_ROOT / "figures"
RESULTS_DIR  = PROJECT_ROOT / "results"

# Cleaned / derived datasets produced by 01_preprocessing.py
CLEAN_CSV    = DATA_DIR / "bikez_clean.csv"      # all rows, extracted numeric + categorical
MODEL_CSV    = DATA_DIR / "bikez_model.csv"      # modelling subset (complete-ish, top categories)

for _d in (DATA_DIR, FIG_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 2. Feature definitions
# --------------------------------------------------------------------------- #
# Numeric features extracted from free-text spec strings, with the SI unit token
# that precedes/marks the number, and a physically plausible [min, max] range.
# Values outside the range are treated as data-entry errors and set to NaN.
#
#   key            : raw column            unit token   plausible range
NUMERIC_SPECS = {
    "displacement_ccm":   ("Displacement",        "ccm",     (25,   3000)),
    "power_hp":           ("Power",               "HP",      (1,    400)),
    "torque_nm":          ("Torque",              "Nm",      (1,    600)),
    "top_speed_kmh":      ("Top speed",           "km/h",    (20,   450)),
    "dry_weight_kg":      ("Dry weight",          "kg",      (20,   700)),
    "fuel_capacity_l":    ("Fuel capacity",       "litres",  (1,    60)),
    "seat_height_mm":     ("Seat height",         "mm",      (400,  1100)),
    "wheelbase_mm":       ("Wheelbase",           "mm",      (600,  2200)),
    "power_weight_ratio": ("Power/weight ratio",  "HP/kg",   (0.0,  3.0)),
}

# Engine RPM at which peak power / peak torque is reached ("... @ 8500 RPM").
RPM_SPECS = {
    "power_rpm":  ("Power",  (500, 25000)),
    "torque_rpm": ("Torque", (100, 25000)),
}

# "Bore x stroke" -> two numbers in mm.
BORE_STROKE_COL   = "Bore x stroke"
BORE_RANGE        = (20, 160)
STROKE_RANGE      = (15, 160)

# "Compression"  -> "x:1"
COMPRESSION_COL   = "Compression"
COMPRESSION_RANGE = (4, 18)

# "Rating"       -> leading float on a 0-5 scale
RATING_COL        = "Rating"
RATING_RANGE      = (0, 5)

# "Year"         -> numeric model year
YEAR_COL          = "Year"
YEAR_RANGE        = (1894, 2026)

# Categorical features (cleaned to a small, tidy set of levels).
CATEGORICAL_SPECS = {
    "category":     "Category",            # motorcycle type  -> classification target
    "cooling":      "Cooling system",      # Air / Liquid / Oil & air
    "transmission": "Transmission type",   # Chain / Belt / Shaft final drive
}

# The full list of numeric analysis columns (extracted + simple-numeric).
NUMERIC_COLS = (
    list(NUMERIC_SPECS.keys())
    + list(RPM_SPECS.keys())
    + ["bore_mm", "stroke_mm", "compression_ratio", "rating", "year"]
)

# --------------------------------------------------------------------------- #
# 3. Analysis constants
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42          # reproducible train/test splits, K-means, RF, ...
TEST_SIZE    = 0.25
TOP_N_CATEGORIES = 8       # keep the 8 most frequent Category levels for classification
ALPHA = 0.05               # significance level for hypothesis tests

# A compact, high-signal feature set used by the data-mining models.
MODEL_FEATURES = [
    "displacement_ccm", "power_hp", "torque_nm", "top_speed_kmh",
    "dry_weight_kg", "fuel_capacity_l", "seat_height_mm", "wheelbase_mm",
    "bore_mm", "stroke_mm", "compression_ratio", "power_weight_ratio",
    "power_rpm",
]
