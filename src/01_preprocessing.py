"""
01_preprocessing.py
===================
Step 1 of the pipeline -- DATA PREPROCESSING (Instruction #1).

What it does
------------
1.  Loads the raw Bikez dump (42,564 motorcycles x 105 mostly-text columns).
2.  Parses numbers out of the free-text spec strings with regex
    (e.g. "241.5 ccm (14.74 cubic inches)" -> 241.5) and splits compound
    fields (Bore x stroke, Power @ RPM).
3.  Applies physically-plausible bounds so impossible values
    (dry weight = 0 kg, top speed = 650 km/h, ...) become NaN.
4.  Cleans the key categorical fields to a small, tidy set of levels and
    derives the manufacturer brand from the official brand list.
5.  Quantifies and reports missingness, then builds:
      * data/bikez_clean.csv  -- every row, tidy numeric + categorical columns
      * data/bikez_model.csv  -- modelling subset: the TOP_N categories with the
        core specs present (median-imputed) for the data-mining stage.
6.  Saves a missingness bar chart and a missingness summary table.

Run:  python 01_preprocessing.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
import bikez_utils as U


def main():
    U.section("STEP 1  |  DATA PREPROCESSING")

    # ----------------------------------------------------------------- load
    print(f"Reading raw data: {C.RAW_CSV}")
    raw = pd.read_csv(C.RAW_CSV, low_memory=False)
    print(f"  raw shape          : {raw.shape[0]:,} rows x {raw.shape[1]} cols")

    # remove exact duplicate spec rows (same URL = same bike page)
    n0 = len(raw)
    if "URL" in raw.columns:
        raw = raw.drop_duplicates(subset="URL").reset_index(drop=True)
    print(f"  after de-duplication: {len(raw):,} rows  (removed {n0 - len(raw):,})")

    df = pd.DataFrame(index=raw.index)

    # ------------------------------------------------- numeric extraction
    print("\nExtracting numeric specs from free text ...")
    for key, (col, unit, (lo, hi)) in C.NUMERIC_SPECS.items():
        parsed = raw[col].apply(lambda v: U.num_before_unit(v, unit))
        df[key] = U.clip_to_range(parsed, lo, hi)

    # peak-power / peak-torque engine speed
    for key, (col, (lo, hi)) in C.RPM_SPECS.items():
        parsed = raw[col].apply(U.rpm_after_at)
        df[key] = U.clip_to_range(parsed, lo, hi)

    # bore x stroke -> two columns
    bs = raw[C.BORE_STROKE_COL].apply(U.bore_stroke)
    df["bore_mm"]   = U.clip_to_range(bs.apply(lambda t: t[0]), *C.BORE_RANGE)
    df["stroke_mm"] = U.clip_to_range(bs.apply(lambda t: t[1]), *C.STROKE_RANGE)

    # compression ratio, rating, year
    df["compression_ratio"] = U.clip_to_range(
        raw[C.COMPRESSION_COL].apply(U.compression_ratio), *C.COMPRESSION_RANGE)
    df["rating"] = U.clip_to_range(
        raw[C.RATING_COL].apply(U.leading_float), *C.RATING_RANGE)
    df["year"] = U.clip_to_range(
        pd.to_numeric(raw[C.YEAR_COL], errors="coerce"), *C.YEAR_RANGE)

    # ------------------------------------------------- categorical cleaning
    print("Cleaning categorical fields ...")
    df["category"]     = raw[C.CATEGORICAL_SPECS["category"]].astype("string").str.strip()
    df["cooling"]      = raw[C.CATEGORICAL_SPECS["cooling"]].astype("string").str.strip()
    df["transmission"] = raw[C.CATEGORICAL_SPECS["transmission"]].apply(U.clean_transmission)
    df["fuel_system"]  = raw["Fuel system"].apply(U.clean_fuel_system)

    # brand from the official brand list (longest-prefix match)
    brands = pd.read_csv(C.BRANDS_CSV)["Brand"].tolist()
    extract_brand = U.build_brand_extractor(brands)
    df["brand"] = raw["Model"].apply(extract_brand)

    # drop the rare "Unspecified category" so the target is meaningful
    df.loc[df["category"].isin(["Unspecified category"]), "category"] = pd.NA

    # ----------------------------------------------------- missingness report
    U.section("Missingness report")
    miss = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    miss_tbl = miss.rename("missing_%").to_frame()
    miss_tbl["non_null"] = df.notna().sum()
    U.save_table(miss_tbl, C.RESULTS_DIR / "missingness.csv")
    print(miss_tbl.to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    miss.sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missingness by extracted feature")
    U.savefig(fig, C.FIG_DIR / "01_missingness.png")

    # ------------------------------------------------------ save clean table
    U.save_table(df, C.CLEAN_CSV, index=False)
    print(f"  clean dataset shape: {df.shape[0]:,} x {df.shape[1]}")

    # ----------------------------------------------- build modelling subset
    U.section("Building modelling subset (top categories + core specs)")
    top_cats = (df["category"].value_counts()
                .head(C.TOP_N_CATEGORIES).index.tolist())
    print("Top categories kept:", top_cats)

    core = ["displacement_ccm", "dry_weight_kg"]          # must be present
    sub = df[df["category"].isin(top_cats) & df[core].notna().all(axis=1)].copy()
    print(f"  rows with a top category + core specs: {len(sub):,}")

    # NOTE: we deliberately DO NOT impute here.  The remaining missing values are
    # left as NaN so that imputation can happen *inside* the cross-validation
    # pipelines in 04_datamining.py, which prevents information from the test
    # folds leaking into the training statistics.
    feat = [c for c in C.MODEL_FEATURES if c in sub.columns]
    keep = feat + ["category", "cooling", "transmission", "fuel_system",
                   "brand", "rating", "year"]
    sub = sub[keep].reset_index(drop=True)
    U.save_table(sub, C.MODEL_CSV, index=False)
    print(f"  modelling subset shape: {sub.shape[0]:,} x {sub.shape[1]}")

    # quick sanity print
    U.section("Sanity check -- descriptive stats of extracted numerics")
    print(df[C.NUMERIC_COLS].describe().T.round(2).to_string())

    print("\nSTEP 1 complete. Outputs in data/, figures/, results/.")


if __name__ == "__main__":
    main()
