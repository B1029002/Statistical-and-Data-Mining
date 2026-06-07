"""
02_eda.py
=========
Step 2 of the pipeline -- EXPLORATORY DATA ANALYSIS (Instruction #2).

Produces statistical summaries and a gallery of visualisations that build
intuition before any formal modelling:

  * numeric summary table (count/mean/std/quantiles/skew)  -> results/eda_numeric_summary.csv
  * distribution grid (histograms + KDE) of the core specs
  * boxplots of engine power across motorcycle categories
  * Pearson + Spearman correlation heatmaps
  * displacement-vs-power scatter coloured by cooling system (log-log)
  * technology trends over model year (injection share, mean displacement/power)
  * category and cooling-system frequency bars

Run (after 01):  python 02_eda.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as C
import bikez_utils as U


def load():
    if not C.CLEAN_CSV.exists():
        raise FileNotFoundError("Run 01_preprocessing.py first.")
    return pd.read_csv(C.CLEAN_CSV)


def numeric_summary(df):
    U.section("Numeric summary statistics")
    desc = df[C.NUMERIC_COLS].describe().T
    desc["skew"] = df[C.NUMERIC_COLS].skew(numeric_only=True)
    desc["missing_%"] = (df[C.NUMERIC_COLS].isna().mean() * 100).round(2)
    desc = desc.round(3)
    U.save_table(desc, C.RESULTS_DIR / "eda_numeric_summary.csv")
    print(desc.to_string())
    return desc


def distribution_grid(df):
    cols = ["displacement_ccm", "power_hp", "torque_nm", "top_speed_kmh",
            "dry_weight_kg", "fuel_capacity_l", "seat_height_mm",
            "compression_ratio", "rating"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for ax, c in zip(axes.ravel(), cols):
        data = df[c].dropna()
        sns.histplot(data, bins=40, kde=True, ax=ax, color="#4C72B0")
        ax.set_title(f"{c}\n(n={len(data):,}, skew={data.skew():.2f})", fontsize=12)
        ax.set_xlabel("")
    fig.suptitle("Distributions of core numeric specifications", y=1.01)
    fig.tight_layout()
    U.savefig(fig, C.FIG_DIR / "02_distributions.png")


def power_by_category(df):
    sub = df.dropna(subset=["power_hp", "category"])
    order = (sub.groupby("category")["power_hp"].median()
             .sort_values(ascending=False).index)
    order = [c for c in order if sub["category"].value_counts()[c] >= 100]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=sub[sub["category"].isin(order)], x="category", y="power_hp",
                order=order, ax=ax, showfliers=False, palette="viridis", hue="category", legend=False)
    ax.set_title("Engine power (HP) by motorcycle category")
    ax.set_xlabel(""); ax.set_ylabel("Power (HP)")
    ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    U.savefig(fig, C.FIG_DIR / "02_power_by_category.png")


def correlation_heatmaps(df):
    cols = ["displacement_ccm", "power_hp", "torque_nm", "top_speed_kmh",
            "dry_weight_kg", "fuel_capacity_l", "seat_height_mm", "wheelbase_mm",
            "bore_mm", "stroke_mm", "compression_ratio", "power_weight_ratio",
            "power_rpm", "rating", "year"]
    pear = df[cols].corr(method="pearson")
    spear = df[cols].corr(method="spearman")
    U.save_table(pear.round(3), C.RESULTS_DIR / "eda_corr_pearson.csv")

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    for ax, mat, name in zip(axes, [pear, spear], ["Pearson", "Spearman"]):
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, ax=ax, annot_kws={"size": 8},
                    cbar_kws={"shrink": 0.7})
        ax.set_title(f"{name} correlation")
    fig.tight_layout()
    U.savefig(fig, C.FIG_DIR / "02_correlation_heatmaps.png")


def displacement_vs_power(df):
    sub = df.dropna(subset=["displacement_ccm", "power_hp", "cooling"])
    sub = sub[sub["cooling"].isin(["Air", "Liquid", "Oil & air"])]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(data=sub, x="displacement_ccm", y="power_hp", hue="cooling",
                    alpha=0.35, s=22, ax=ax, edgecolor=None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Engine power vs displacement (log-log), by cooling system")
    ax.set_xlabel("Displacement (ccm)"); ax.set_ylabel("Power (HP)")
    ax.legend(title="Cooling")
    U.savefig(fig, C.FIG_DIR / "02_displacement_vs_power.png")


def trends_over_time(df):
    sub = df.dropna(subset=["year"])
    sub = sub[sub["year"] >= 1980]
    grp = sub.groupby(sub["year"].astype(int))
    mean_disp  = grp["displacement_ccm"].mean()
    mean_power = grp["power_hp"].mean()
    inj_share  = grp["fuel_system"].apply(
        lambda s: (s == "Injection").mean() if s.notna().any() else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    mean_disp.plot(ax=axes[0], marker="o", color="#4C72B0")
    axes[0].set_title("Mean displacement by year"); axes[0].set_ylabel("ccm")
    mean_power.plot(ax=axes[1], marker="o", color="#C44E52")
    axes[1].set_title("Mean power by year"); axes[1].set_ylabel("HP")
    (inj_share * 100).plot(ax=axes[2], marker="o", color="#55A868")
    axes[2].set_title("Fuel injection adoption"); axes[2].set_ylabel("% injection")
    for ax in axes:
        ax.set_xlabel("Model year")
    fig.tight_layout()
    U.savefig(fig, C.FIG_DIR / "02_trends_over_time.png")


def category_and_cooling_bars(df):
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    vc = df["category"].value_counts().head(12)
    sns.barplot(x=vc.values, y=vc.index, ax=axes[0], palette="crest", hue=vc.index, legend=False)
    axes[0].set_title("Top motorcycle categories"); axes[0].set_xlabel("Count")
    cc = df["cooling"].value_counts()
    sns.barplot(x=cc.values, y=cc.index, ax=axes[1], palette="flare", hue=cc.index, legend=False)
    axes[1].set_title("Cooling systems"); axes[1].set_xlabel("Count")
    fig.tight_layout()
    U.savefig(fig, C.FIG_DIR / "02_category_cooling_bars.png")


def main():
    U.section("STEP 2  |  EXPLORATORY DATA ANALYSIS")
    df = load()
    numeric_summary(df)
    distribution_grid(df)
    power_by_category(df)
    correlation_heatmaps(df)
    displacement_vs_power(df)
    trends_over_time(df)
    category_and_cooling_bars(df)
    print("\nSTEP 2 complete. Figures in figures/, tables in results/.")


if __name__ == "__main__":
    main()
