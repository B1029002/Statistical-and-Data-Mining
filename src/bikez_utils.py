"""
bikez_utils.py
==============
Reusable helpers shared by every analysis script:

  * regex parsers that pull a clean number out of a free-text spec string
    (e.g. "241.5 ccm (14.74 cubic inches)" -> 241.5),
  * a brand extractor that uses the official brand list for a longest-prefix
    match on the `Model` column,
  * small plotting / IO conveniences (consistent figure style, safe saving).

All parsers are NaN-safe and return numpy.nan when nothing can be parsed.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend - no display needed
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------- #
# Global plotting style (one place, applied everywhere)
# --------------------------------------------------------------------------- #
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"]      = 110
plt.rcParams["savefig.dpi"]     = 150
plt.rcParams["savefig.bbox"]    = "tight"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["font.size"]       = 11


# --------------------------------------------------------------------------- #
# 1. Numeric extraction
# --------------------------------------------------------------------------- #
def num_before_unit(value, unit: str):
    """Return the float that appears immediately before `unit`.

    Examples
    --------
    >>> num_before_unit("241.5 ccm (14.74 cubic inches)", "ccm")
    241.5
    >>> num_before_unit("0.0749 HP/kg", "HP/kg")
    0.0749
    """
    if pd.isna(value):
        return np.nan
    # escape regex-special characters in the unit token (e.g. "HP/kg", "km/h")
    pattern = r"([0-9]+\.?[0-9]*)\s*" + re.escape(unit)
    m = re.search(pattern, str(value))
    return float(m.group(1)) if m else np.nan


def rpm_after_at(value):
    """Extract the engine speed in '... @ 8500 RPM' -> 8500.0 ."""
    if pd.isna(value):
        return np.nan
    m = re.search(r"@\s*([0-9]+)\s*RPM", str(value))
    return float(m.group(1)) if m else np.nan


def bore_stroke(value):
    """Parse 'Bore x stroke' -> (bore_mm, stroke_mm).

    '107.1 x 100.0 mm (4.2 x 3.9 inches)' -> (107.1, 100.0)
    """
    if pd.isna(value):
        return (np.nan, np.nan)
    m = re.search(r"([0-9]+\.?[0-9]*)\s*x\s*([0-9]+\.?[0-9]*)\s*mm", str(value))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def compression_ratio(value):
    """Parse 'Compression' -> the leading ratio. '9.6:1' -> 9.6 ."""
    if pd.isna(value):
        return np.nan
    m = re.search(r"([0-9]+\.?[0-9]*)\s*:\s*1", str(value))
    return float(m.group(1)) if m else np.nan


def leading_float(value):
    """Return the leading float of a string ('3.3  Check out ...' -> 3.3).

    Rows such as 'Do you know this bike? ...' have no leading number and
    therefore return NaN, which is exactly what we want for the Rating column.
    """
    if pd.isna(value):
        return np.nan
    m = re.match(r"\s*([0-9]+\.?[0-9]*)", str(value))
    return float(m.group(1)) if m else np.nan


def clip_to_range(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Set values outside [lo, hi] to NaN (treat them as data-entry errors)."""
    return series.where((series >= lo) & (series <= hi))


# --------------------------------------------------------------------------- #
# 2. Categorical helpers
# --------------------------------------------------------------------------- #
def clean_fuel_system(value):
    """Collapse the very messy 'Fuel system' text to its main technology."""
    if pd.isna(value):
        return np.nan
    head = str(value).split(".")[0].split(",")[0].strip().title()
    if "Inject" in head:
        return "Injection"
    if "Carburett" in head or "Carburet" in head:
        return "Carburettor"
    if "Turbo" in head:
        return "Turbo"
    return "Other"


def clean_transmission(value):
    """Final-drive type: Chain / Belt / Shaft (drop the '(final drive)' tail)."""
    if pd.isna(value):
        return np.nan
    head = str(value).split("(")[0].strip().title()
    if head.startswith("Chain"):
        return "Chain"
    if head.startswith("Belt"):
        return "Belt"
    if head.startswith("Shaft"):
        return "Shaft"
    return "Other"


def build_brand_extractor(brands: list[str]):
    """Return a function Model-string -> brand using longest-prefix matching.

    The brand list may contain multi-word names ('FB Mondial', 'GAS GAS',
    'Harley-Davidson'); matching the *longest* candidate first avoids picking
    the wrong single-word brand.
    """
    # longest brand names first so multi-word brands win over their first token
    ordered = sorted({b.strip() for b in brands if isinstance(b, str) and b.strip()},
                     key=len, reverse=True)
    lowered = [(b.lower(), b) for b in ordered]

    def extract(model):
        if pd.isna(model):
            return np.nan
        s = str(model).strip().lower()
        for low, original in lowered:
            if s.startswith(low):
                return original
        # fall back to the first whitespace-separated token
        return str(model).strip().split()[0] if str(model).strip() else np.nan

    return extract


# --------------------------------------------------------------------------- #
# 3. IO / plotting conveniences
# --------------------------------------------------------------------------- #
def savefig(fig, path, *, close=True):
    """Save a figure and (optionally) close it to free memory."""
    fig.savefig(path)
    if close:
        plt.close(fig)
    print(f"  [fig] saved {path.name}")


def save_table(df: pd.DataFrame, path, *, index=True, float_format="%.4f"):
    """Persist a results table as CSV and echo where it went."""
    df.to_csv(path, index=index, float_format=float_format)
    print(f"  [tbl] saved {path.name}  ({df.shape[0]}x{df.shape[1]})")


def section(title: str):
    """Pretty console banner so script logs are easy to read."""
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")
