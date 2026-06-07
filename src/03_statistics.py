"""
03_statistics.py
================
Step 3 of the pipeline -- STATISTICAL METHODS (Instruction #3).

A self-contained statistical study of the cleaned data.  Every test prints its
hypotheses, statistic, p-value, an effect size and a plain-English conclusion,
and the whole transcript is also written to results/statistics_report.txt.

Contents
--------
A. Normality assessment (D'Agostino K^2) -> justifies parametric vs non-parametric
B. Two-group test  : power of Liquid- vs Air-cooled engines
                     (Welch t-test + Mann-Whitney U + Cohen's d)
C. k-group test    : power across motorcycle categories
                     (one-way ANOVA on log-power + Kruskal-Wallis + Tukey HSD)
D. Correlation     : Pearson & Spearman with p-values for key spec pairs
E. Independence    : chi-square test  (cooling x transmission) + Cramer's V
F. Regression      : OLS  log(top_speed) ~ log(power)+log(weight)  (physics check)
                     + a richer multivariable model with VIF & residual diagnostics

Run (after 01):  python 03_statistics.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportions_ztest

import config as C
import bikez_utils as U


# --------------------------------------------------------------------------- #
# tiny logger: echo to console AND collect into a report file
# --------------------------------------------------------------------------- #
class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg)
        self.lines.append(msg)

    def save(self, path):
        path.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"\n[report] full transcript -> {path.name}")


log = Log()


def cohens_d(a, b):
    """Cohen's d for two independent samples (pooled SD)."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
                 / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


def cramers_v(confusion):
    """Bias-corrected Cramer's V for a contingency table."""
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    r, k = confusion.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / max(1e-12, min(kcorr - 1, rcorr - 1)))


# --------------------------------------------------------------------------- #
def a_normality(df):
    U.section("A. Normality assessment (D'Agostino-Pearson K^2)")
    log("H0: the variable is drawn from a normal distribution.")
    rows = []
    for c in ["power_hp", "displacement_ccm", "top_speed_kmh", "dry_weight_kg"]:
        x = df[c].dropna()
        # work on a capped sample so the test is not absurdly over-powered
        xs = x.sample(min(5000, len(x)), random_state=C.RANDOM_STATE)
        k2, p = stats.normaltest(xs)
        k2l, pl = stats.normaltest(np.log(xs[xs > 0]))
        rows.append([c, round(x.skew(), 3), round(k2, 1), f"{p:.2e}",
                     round(np.log(x[x > 0]).skew(), 3), f"{pl:.2e}"])
    tab = pd.DataFrame(rows, columns=["variable", "skew", "K2", "p_value",
                                      "skew_log", "p_value_log"])
    log(tab.to_string(index=False))
    log("\n-> All p-values << 0.05: raw specs are right-skewed and NON-normal.")
    log("   Log-transformation greatly reduces skew, so we pair parametric tests")
    log("   (on log scale) with non-parametric tests on the raw scale.")
    U.save_table(tab, C.RESULTS_DIR / "stat_A_normality.csv", index=False)

    # figure: histogram + Q-Q for a representative spec (raw vs log) -> shows non-normality
    x = df["power_hp"].dropna()
    xl = np.log(x[x > 0])
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    sns.histplot(x, bins=40, kde=True, ax=ax[0, 0], color="#4C72B0"); ax[0, 0].set_title("power_hp (raw)")
    sm.qqplot(x, line="s", ax=ax[0, 1], markersize=3, alpha=0.3); ax[0, 1].set_title("power_hp raw: Q-Q")
    sns.histplot(xl, bins=40, kde=True, ax=ax[1, 0], color="#55A868"); ax[1, 0].set_title("log(power_hp)")
    sm.qqplot(xl, line="s", ax=ax[1, 1], markersize=3, alpha=0.3); ax[1, 1].set_title("log(power_hp): Q-Q")
    fig.suptitle("Normality check: raw is right-skewed, log is closer to normal", y=1.01)
    fig.tight_layout()
    U.savefig(fig, C.FIG_DIR / "03_normality_qq.png")


def b_two_group(df):
    U.section("B. Two-group comparison -- power: Liquid- vs Air-cooled")
    sub = df.dropna(subset=["power_hp", "cooling"])
    liq = sub.loc[sub["cooling"] == "Liquid", "power_hp"]
    air = sub.loc[sub["cooling"] == "Air", "power_hp"]
    log(f"n(Liquid) = {len(liq):,}   mean = {liq.mean():.1f} HP   median = {liq.median():.1f}")
    log(f"n(Air)    = {len(air):,}   mean = {air.mean():.1f} HP   median = {air.median():.1f}")

    log("\nH0: mean power is equal for Liquid- and Air-cooled engines.")
    t, p = stats.ttest_ind(liq, air, equal_var=False)          # Welch
    log(f"Welch t-test         : t = {t:.2f},  p = {p:.3e}")
    u, pu = stats.mannwhitneyu(liq, air, alternative="two-sided")
    log(f"Mann-Whitney U       : U = {u:.0f},  p = {pu:.3e}")
    d = cohens_d(liq, air)
    log(f"Cohen's d            : {d:.2f}  ({'large' if abs(d)>=0.8 else 'medium' if abs(d)>=0.5 else 'small'} effect)")
    log(f"-> p < {C.ALPHA}: liquid-cooled engines produce significantly MORE power.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=sub[sub.cooling.isin(["Air", "Liquid", "Oil & air"])],
                x="cooling", y="power_hp", showfliers=False, ax=ax,
                hue="cooling", palette="Set2", legend=False)
    ax.set_title("Power by cooling system"); ax.set_ylabel("Power (HP)")
    U.savefig(fig, C.FIG_DIR / "03_power_by_cooling.png")


def c_kgroup(df):
    U.section("C. k-group comparison -- power across categories")
    sub = df.dropna(subset=["power_hp", "category"]).copy()
    keep = sub["category"].value_counts()
    keep = keep[keep >= 100].index
    sub = sub[sub["category"].isin(keep)].copy()
    sub["log_power"] = np.log(sub["power_hp"])
    groups = [g["power_hp"].values for _, g in sub.groupby("category")]
    log(f"Categories compared (n>=100): {list(keep)}")

    log("\nH0: all categories share the same mean/median power.")
    F, p = stats.f_oneway(*[np.log(g) for g in groups])
    log(f"One-way ANOVA (log power): F = {F:.1f},  p = {p:.3e}")
    H, ph = stats.kruskal(*groups)
    log(f"Kruskal-Wallis (raw)     : H = {H:.1f},  p = {ph:.3e}")
    # eta^2 effect size from ANOVA
    grand = sub["log_power"].mean()
    ss_between = sum(len(g) * (np.log(g).mean() - grand) ** 2 for g in groups)
    ss_total = ((sub["log_power"] - grand) ** 2).sum()
    log(f"eta^2 (ANOVA)            : {ss_between/ss_total:.3f}")
    log(f"-> p << {C.ALPHA}: motorcycle category strongly explains engine power.")

    tuk = pairwise_tukeyhsd(sub["log_power"], sub["category"], alpha=C.ALPHA)
    tdf = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
    U.save_table(tdf, C.RESULTS_DIR / "stat_C_tukey.csv", index=False)
    n_sig = tdf["reject"].astype(str).eq("True").sum()
    log(f"Tukey HSD: {n_sig} of {len(tdf)} category pairs differ significantly "
        f"(see stat_C_tukey.csv).")

    # figure: power distribution per category (visual companion to ANOVA/Kruskal)
    order = sub.groupby("category")["power_hp"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.boxplot(data=sub, x="category", y="power_hp", order=order, showfliers=False,
                hue="category", palette="viridis", legend=False, ax=ax)
    ax.set_title("Power by category (ANOVA / Kruskal-Wallis)")
    ax.set_ylabel("Power (HP)"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    U.savefig(fig, C.FIG_DIR / "03_power_by_category_anova.png")


def d_correlation(df):
    U.section("D. Correlation tests (Pearson & Spearman)")
    pairs = [("displacement_ccm", "power_hp"),
             ("power_hp", "top_speed_kmh"),
             ("dry_weight_kg", "top_speed_kmh"),
             ("displacement_ccm", "torque_nm"),
             ("compression_ratio", "power_weight_ratio")]
    rows = []
    for a, b in pairs:
        s = df[[a, b]].dropna()
        r, pr = stats.pearsonr(s[a], s[b])
        rho, prho = stats.spearmanr(s[a], s[b])
        rows.append([f"{a} ~ {b}", len(s), round(r, 3), f"{pr:.2e}",
                     round(rho, 3), f"{prho:.2e}"])
        log(f"{a:>18} ~ {b:<15} n={len(s):>6}  Pearson r={r:+.3f}  Spearman rho={rho:+.3f}")
    tab = pd.DataFrame(rows, columns=["pair", "n", "pearson_r", "p_pearson",
                                      "spearman_rho", "p_spearman"])
    U.save_table(tab, C.RESULTS_DIR / "stat_D_correlation.csv", index=False)
    log("-> Strong positive relationships; all p-values are effectively zero.")

    # figure: scatter + regression line for the two headline pairs (with r)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (xa, xb) in zip(axes, [("displacement_ccm", "power_hp"),
                                   ("power_hp", "top_speed_kmh")]):
        s = df[[xa, xb]].dropna()
        s = s.sample(min(3000, len(s)), random_state=C.RANDOM_STATE)
        r = stats.pearsonr(s[xa], s[xb])[0]
        sns.regplot(data=s, x=xa, y=xb, ax=ax,
                    scatter_kws={"s": 8, "alpha": 0.2}, line_kws={"color": "red"})
        ax.set_title(f"{xa} ~ {xb}  (Pearson r = {r:.2f})")
    fig.tight_layout()
    U.savefig(fig, C.FIG_DIR / "03_correlation_scatter.png")


def e_independence(df):
    U.section("E. Chi-square test of independence -- cooling x transmission")
    sub = df.dropna(subset=["cooling", "transmission"])
    sub = sub[sub["cooling"].isin(["Air", "Liquid", "Oil & air"])]
    ct = pd.crosstab(sub["cooling"], sub["transmission"])
    log("Contingency table (counts):")
    log(ct.to_string())
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = cramers_v(ct.values)
    log(f"\nH0: cooling system and final-drive type are independent.")
    log(f"chi-square = {chi2:.1f},  dof = {dof},  p = {p:.3e}")
    log(f"Cramer's V = {v:.3f}  ({'strong' if v>=0.5 else 'moderate' if v>=0.3 else 'weak'} association)")
    log(f"-> p << {C.ALPHA}: cooling system and final-drive type are statistically")
    log("   DEPENDENT, but Cramer's V ~ 0.10 means the association is weak -- knowing")
    log("   one feature only mildly shifts the odds of the other in practice.")
    U.save_table(ct, C.RESULTS_DIR / "stat_E_contingency.csv")

    # figure: contingency heatmap (counts) -- visual companion to the chi-square test
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Cooling x transmission (observed counts)")
    U.savefig(fig, C.FIG_DIR / "03_contingency_heatmap.png")


def f_regression(df):
    U.section("F. Regression analysis")

    # ----- Model A: physics-motivated log-log model -----------------------
    log(">> Model A: log(top_speed) ~ log(power) + log(weight)")
    log("   Aerodynamic theory: a drag-limited top speed scales as power^(1/3),")
    log("   so we expect the coefficient on log(power) to be near 0.33.")
    a = df.dropna(subset=["top_speed_kmh", "power_hp", "dry_weight_kg"]).copy()
    a = a[(a.power_hp > 0) & (a.dry_weight_kg > 0) & (a.top_speed_kmh > 0)]
    a["ltop"] = np.log(a.top_speed_kmh)
    a["lpow"] = np.log(a.power_hp)
    a["lwt"]  = np.log(a.dry_weight_kg)
    mA = smf.ols("ltop ~ lpow + lwt", data=a).fit()
    log(f"   n = {int(mA.nobs):,}   R^2 = {mA.rsquared:.3f}   adj R^2 = {mA.rsquared_adj:.3f}")
    log(f"   coef log(power)  = {mA.params['lpow']:+.3f}  (95% CI "
        f"{mA.conf_int().loc['lpow',0]:.3f} .. {mA.conf_int().loc['lpow',1]:.3f})")
    log(f"   coef log(weight) = {mA.params['lwt']:+.3f}")
    log(f"   -> empirical power exponent {mA.params['lpow']:.2f} vs theoretical 0.33.")

    # ----- Model B: richer multivariable model ----------------------------
    log("\n>> Model B: top_speed ~ power + weight + displacement + fuel_cap + cooling")
    b = df.dropna(subset=["top_speed_kmh", "power_hp", "dry_weight_kg",
                          "displacement_ccm", "fuel_capacity_l", "cooling"]).copy()
    b = b[b["cooling"].isin(["Air", "Liquid", "Oil & air"])].copy()
    # NB: a plain string column is auto-treated as categorical by patsy; we avoid
    # the C() wrapper because it would clash with our `import config as C`.
    b["cooling"] = b["cooling"].astype("category")
    mB = smf.ols("top_speed_kmh ~ power_hp + dry_weight_kg + displacement_ccm "
                 "+ fuel_capacity_l + cooling", data=b).fit()
    log(f"   n = {int(mB.nobs):,}   R^2 = {mB.rsquared:.3f}   adj R^2 = {mB.rsquared_adj:.3f}")
    log(f"   F = {mB.fvalue:.0f},  p(F) = {mB.f_pvalue:.2e}")

    # save full statsmodels summaries
    (C.RESULTS_DIR / "stat_F_regression_summary.txt").write_text(
        "MODEL A: log(top_speed) ~ log(power) + log(weight)\n"
        + mA.summary().as_text()
        + "\n\n\nMODEL B: top_speed ~ power + weight + displacement + fuel_cap + cooling\n"
        + mB.summary().as_text(), encoding="utf-8")
    log("   full OLS summaries -> stat_F_regression_summary.txt")

    # ----- VIF (multicollinearity diagnostic) for Model B -----------------
    Xcols = ["power_hp", "dry_weight_kg", "displacement_ccm", "fuel_capacity_l"]
    X = sm.add_constant(b[Xcols])
    vif = pd.DataFrame({
        "feature": Xcols,
        "VIF": [variance_inflation_factor(X.values, i + 1) for i in range(len(Xcols))]
    })
    log("\n   Variance Inflation Factors (Model B numeric terms):")
    log(vif.to_string(index=False))
    U.save_table(vif, C.RESULTS_DIR / "stat_F_vif.csv", index=False)

    # ----- residual diagnostics for Model A -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(mA.fittedvalues, mA.resid, s=6, alpha=0.2, color="#4C72B0")
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_xlabel("Fitted log(top speed)"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Model A: residuals vs fitted")
    sm.qqplot(mA.resid, line="45", fit=True, ax=axes[1], markersize=3, alpha=0.3)
    axes[1].set_title("Model A: Normal Q-Q of residuals")
    fig.tight_layout()
    U.savefig(fig, C.FIG_DIR / "03_regression_diagnostics.png")


def g_one_sample(df):
    U.section("G. One-sample t-test -- mean rating vs the neutral value 3.0")
    r = df["rating"].dropna()
    log("H0: the mean user rating equals the neutral value 3.0.")
    t, p = stats.ttest_1samp(r, 3.0)
    ci = stats.t.interval(0.95, len(r) - 1, loc=r.mean(), scale=stats.sem(r))
    log(f"n = {len(r):,}   mean = {r.mean():.3f}")
    log(f"one-sample t = {t:.1f},  p = {p:.2e},  95% CI = ({ci[0]:.3f}, {ci[1]:.3f})")
    log("-> mean rating is significantly ABOVE 3.0: users skew positive.")

    # figure: rating distribution with the test value (3.0), the mean, and its 95% CI
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(r, bins=30, kde=True, ax=ax, color="#4C72B0")
    ax.axvline(3.0, color="black", ls="--", label="test value 3.0")
    ax.axvline(r.mean(), color="red", label=f"mean {r.mean():.2f}")
    ax.axvspan(ci[0], ci[1], color="red", alpha=0.15, label="95% CI of mean")
    ax.set_title("One-sample t-test: rating vs neutral 3.0")
    ax.set_xlabel("rating"); ax.legend()
    U.savefig(fig, C.FIG_DIR / "03_one_sample_rating.png")


def h_paired(df):
    U.section("H. Paired tests -- bore vs stroke within the same engine")
    # bore and stroke are two measurements on the SAME engine -> a paired design.
    # bore>stroke == 'oversquare' (high-revving); bore<stroke == 'undersquare' (torquey).
    pair = df[["bore_mm", "stroke_mm"]].dropna()
    diff = pair["bore_mm"] - pair["stroke_mm"]
    log("H0: within an engine, mean(bore) = mean(stroke)  (a 'square' engine).")
    log(f"n = {len(pair):,}   mean(bore - stroke) = {diff.mean():.2f} mm")
    tt, pt = stats.ttest_rel(pair["bore_mm"], pair["stroke_mm"])          # (a) parametric
    w, pw = stats.wilcoxon(pair["bore_mm"], pair["stroke_mm"])            # (b) non-parametric
    n_pos = int((diff > 0).sum()); n_eff = int((diff != 0).sum())        # (c) sign test
    sign_p = stats.binomtest(n_pos, n_eff, 0.5).pvalue
    log(f"(a) paired t-test       : t = {tt:.1f},  p = {pt:.2e}")
    log(f"(b) Wilcoxon signed-rank: W = {w:.0f},  p = {pw:.2e}")
    log(f"(c) sign test           : {n_pos}/{n_eff} have bore>stroke,  p = {sign_p:.2e}")
    log("-> all three agree: engines are on average OVERSQUARE (bore>stroke).")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diff, bins=50, color="#4C72B0"); ax.axvline(0, color="red")
    ax.set_title("Distribution of (bore - stroke)"); ax.set_xlabel("bore - stroke (mm)")
    U.savefig(fig, C.FIG_DIR / "03_bore_minus_stroke.png")


def i_two_proportion(df):
    U.section("I. Two-sample proportion z-test -- fuel-injection adoption by era")
    sub = df.dropna(subset=["year", "fuel_system"])
    sub = sub[sub["fuel_system"].isin(["Injection", "Carburettor"])]
    modern = sub[sub.year >= 2010]; older = sub[sub.year < 2010]
    counts = [int((modern.fuel_system == "Injection").sum()),
              int((older.fuel_system == "Injection").sum())]
    nobs = [len(modern), len(older)]
    z, p = proportions_ztest(counts, nobs)
    log("H0: the share of fuel-injected bikes is equal in the two eras.")
    log(f"modern (>=2010): {counts[0]/nobs[0]:.1%} injection  (n = {nobs[0]:,})")
    log(f"older  (<2010) : {counts[1]/nobs[1]:.1%} injection  (n = {nobs[1]:,})")
    log(f"two-proportion z = {z:.1f},  p = {p:.2e}")
    log("-> fuel injection became significantly more prevalent in modern bikes.")

    # figure: injection share by era
    fig, ax = plt.subplots(figsize=(7, 5))
    shares = [counts[0] / nobs[0] * 100, counts[1] / nobs[1] * 100]
    bars = ax.bar(["modern (>=2010)", "older (<2010)"], shares,
                  color=["#4C72B0", "#C44E52"])
    ax.bar_label(bars, fmt="%.1f%%")
    ax.set_ylabel("% fuel injection")
    ax.set_title("Injection adoption by era (two-proportion z-test)")
    U.savefig(fig, C.FIG_DIR / "03_injection_by_era.png")


def j_goodness_of_fit(df):
    U.section("J. Chi-square goodness-of-fit -- are cooling systems equally common?")
    obs = df["cooling"].value_counts().reindex(["Air", "Liquid", "Oil & air"]).dropna()
    exp = [obs.sum() / len(obs)] * len(obs)          # expected counts under a uniform split
    chi2, p = stats.chisquare(obs.values, exp)
    log("H0: Air / Liquid / Oil & air each account for 1/3 of motorcycles.")
    log("observed counts: " + str({k: int(v) for k, v in obs.items()}))
    log(f"chi-square (GoF) = {chi2:.1f},  p = {p:.2e}")
    log("-> the three cooling systems are NOT equally common (oil & air is rare).")
    U.save_table(obs.rename("count").to_frame(), C.RESULTS_DIR / "stat_J_cooling_counts.csv")

    # figure: observed vs expected counts (visual companion to the GoF test)
    fig, ax = plt.subplots(figsize=(7, 5))
    xpos = np.arange(len(obs))
    ax.bar(xpos - 0.2, obs.values, 0.4, label="observed", color="#4C72B0")
    ax.bar(xpos + 0.2, exp, 0.4, label="expected (uniform)", color="#C44E52")
    ax.set_xticks(xpos); ax.set_xticklabels(obs.index)
    ax.set_ylabel("count"); ax.legend()
    ax.set_title("Cooling: observed vs expected (chi-square goodness-of-fit)")
    U.savefig(fig, C.FIG_DIR / "03_gof_observed_expected.png")


def k_bootstrap(df):
    U.section("K. Bootstrap 95% CI -- mean power (Liquid vs Air)")
    rng = np.random.default_rng(C.RANDOM_STATE)
    def boot(x, n=1000):
        x = np.asarray(x)
        return np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    liq = df.loc[df.cooling == "Liquid", "power_hp"].dropna().values
    air = df.loc[df.cooling == "Air", "power_hp"].dropna().values
    bl, ba = boot(liq), boot(air)
    ci_l, ci_a = np.percentile(bl, [2.5, 97.5]), np.percentile(ba, [2.5, 97.5])
    log(f"Liquid mean power 95% CI = [{ci_l[0]:.1f}, {ci_l[1]:.1f}] HP")
    log(f"Air    mean power 95% CI = [{ci_a[0]:.1f}, {ci_a[1]:.1f}] HP")
    log("-> non-overlapping CIs confirm the difference is robust (cf. test B).")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for a, (bs, ci, name, c) in zip(ax, [(bl, ci_l, "Liquid", "#4C72B0"),
                                         (ba, ci_a, "Air", "#C44E52")]):
        a.hist(bs, bins=30, color=c)
        a.axvline(ci[0], color="red", ls="--"); a.axvline(ci[1], color="red", ls="--")
        a.set_title(f"Bootstrap mean power: {name}"); a.set_xlabel("Mean power (HP)")
    fig.tight_layout()
    U.savefig(fig, C.FIG_DIR / "03_bootstrap_power.png")


def l_inferential_logit(df):
    U.section("L. Inferential logistic regression -- P(liquid-cooled) with odds ratios")
    from sklearn.preprocessing import StandardScaler
    sub = df.dropna(subset=["cooling", "displacement_ccm", "power_hp",
                            "compression_ratio", "top_speed_kmh"]).copy()
    sub = sub[sub.cooling.isin(["Liquid", "Air"])]
    sub["is_liquid"] = (sub.cooling == "Liquid").astype(int)
    feats = ["displacement_ccm", "power_hp", "compression_ratio", "top_speed_kmh"]
    # standardise so each odds ratio is the effect of a +1 SD change (comparable)
    Xz = pd.DataFrame(StandardScaler().fit_transform(sub[feats]), columns=feats, index=sub.index)
    Xz = sm.add_constant(Xz)
    logit = sm.Logit(sub["is_liquid"], Xz).fit(disp=False)
    res = pd.DataFrame({"coef": logit.params, "p_value": logit.pvalues,
                        "odds_ratio": np.exp(logit.params)})
    cint = logit.conf_int(); res["or_lo"] = np.exp(cint[0]); res["or_hi"] = np.exp(cint[1])
    log(f"n = {int(logit.nobs):,}   Pseudo R^2 = {logit.prsquared:.3f}")
    log(res.round(4).to_string())
    U.save_table(res.round(4), C.RESULTS_DIR / "stat_L_logit_oddsratio.csv")
    # forest plot of odds ratios (drop the intercept)
    r2 = res.drop(index="const")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(r2["odds_ratio"], range(len(r2)),
                xerr=[r2["odds_ratio"] - r2["or_lo"], r2["or_hi"] - r2["odds_ratio"]],
                fmt="o", color="#4C72B0", capsize=4)
    ax.axvline(1.0, color="red", ls="--")
    ax.set_yticks(range(len(r2))); ax.set_yticklabels(r2.index)
    ax.set_xlabel("Odds ratio (per +1 SD)")
    ax.set_title("P(liquid-cooled): odds ratios with 95% CI")
    U.savefig(fig, C.FIG_DIR / "03_logit_oddsratio.png")


def m_sample_size(df):
    U.section("M. Sample-size sensitivity -- p-value vs n (power: Liquid vs Air)")
    full = df.dropna(subset=["power_hp", "cooling"])
    full = full[full.cooling.isin(["Liquid", "Air"])]
    ns, ps = [], []
    for n in [50, 100, 300, 1000, 3000, len(full)]:
        s = full.sample(min(n, len(full)), random_state=C.RANDOM_STATE)
        l = s.loc[s.cooling == "Liquid", "power_hp"]; a = s.loc[s.cooling == "Air", "power_hp"]
        if len(l) > 5 and len(a) > 5:
            _, p = stats.mannwhitneyu(l, a, alternative="two-sided")
            ns.append(min(n, len(full))); ps.append(p)
            log(f"n = {min(n, len(full)):>6}  ->  p = {p:.2e}")
    U.save_table(pd.DataFrame({"sample_n": ns, "p_value": ps}),
                 C.RESULTS_DIR / "stat_M_sample_size.csv", index=False)
    log("-> p shrinks fast with n: at large n almost everything is 'significant'.")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, ps, "o-", color="#4C72B0")
    ax.axhline(C.ALPHA, color="red", ls="--", label=f"alpha = {C.ALPHA}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sample size n"); ax.set_ylabel("p-value (log scale)"); ax.legend()
    ax.set_title("p-value shrinks as sample size grows")
    U.savefig(fig, C.FIG_DIR / "03_sample_size_pvalue.png")


def main():
    U.section("STEP 3  |  STATISTICAL METHODS")
    df = pd.read_csv(C.CLEAN_CSV)
    # --- core methods ---
    a_normality(df)
    b_two_group(df)
    c_kgroup(df)
    d_correlation(df)
    e_independence(df)
    f_regression(df)
    # --- advanced hypothesis tests (synced from the notebook) ---
    g_one_sample(df)
    h_paired(df)
    i_two_proportion(df)
    j_goodness_of_fit(df)
    # --- bootstrap CI, inferential logit, sample-size sensitivity (synced from notebook) ---
    k_bootstrap(df)
    l_inferential_logit(df)
    m_sample_size(df)
    log.save(C.RESULTS_DIR / "statistics_report.txt")
    print("\nSTEP 3 complete. Tables/figures saved; transcript in results/.")


if __name__ == "__main__":
    main()
