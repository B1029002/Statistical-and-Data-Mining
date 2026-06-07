# CE5033 Final Project — What Makes a Motorcycle? <br>A Hybrid Statistical & Data-Mining Study of 42,564 Motorcycles

**統計方法與資料採礦 期末專題報告**

Course: CE5033 Statistical Methods and Data Mining (NCU CSIE) · Instructor: Chia-Ru Chung
Author: *(your name / student ID)* · Date: 2026

---

## 0. Executive Summary 摘要

We analyse the **Bikez.com motorcycle specifications catalogue** (42,564 motorcycles, 105 raw
fields, model years up to 2025) to answer one driving question:

> **What technical specifications define a motorcycle's "type", and how do engine, weight and
> performance variables relate to one another?**

Using a *hybrid* methodology that combines classical statistics with data-mining algorithms, we find:

1. **Engine power is far from random** — it is strongly determined by displacement
   (Spearman ρ = 0.92) and differs hugely across cooling systems (liquid-cooled engines
   average **74.9 HP** vs **30.2 HP** for air-cooled, Welch *t* = 37.6, *p* < 10⁻²⁶³, Cohen's *d* = 1.03).
2. **Physics shows up in the data** — a log–log regression of top speed on power recovers an
   exponent of **0.43**, close to the cube-root (0.33) predicted by aerodynamic drag theory
   (*R²* = 0.89).
3. **A motorcycle's category is highly predictable** from its specs — a Random Forest classifies
   8 categories with **79.8 % accuracy** (macro-F1 = 0.76), and **seat height + weight** are the
   most discriminative features.
4. **Natural market segments exist** — K-means finds **3 clusters** that map cleanly onto
   *small commuters/scooters*, *heavy cruisers*, and *high-performance sport bikes*.
5. **Interpretable design "rules"** emerge — e.g. *{large displacement + belt drive} ⇒ cruiser*
   (confidence 86 %, lift 6.1) and *{small displacement + belt drive} ⇒ scooter* (confidence 98 %).
6. **Engineering & temporal signatures** confirm the data's validity — engines are on average
   **oversquare** (bore − stroke = +8.0 mm; paired *t*/Wilcoxon/sign tests all *p* ≈ 0), fuel
   injection rose from **23 % to 64 %** of bikes (≥2010 vs earlier), and a Random-Forest model
   predicts top speed to within **RMSE 17.6 km/h (R² = 0.91)**.

> **中文重點**：本研究以 42,564 筆機車規格資料，結合假設檢定／迴歸（統計）與分類／分群／關聯規則
> （資料採礦）。主要發現：(1) 水冷引擎馬力顯著高於氣冷（74.9 vs 30.2 HP，效應量大）；
> (2) 極速對馬力的 log–log 迴歸指數 0.43，呼應空氣阻力的立方根理論；(3) 隨機森林可由規格
> 預測車種，準確率 79.8%；(4) K-means 自然分出「通勤小車／巡航重機／運動性能車」三群；
> (5) 關聯規則找出可解讀的設計法則（大排氣量＋皮帶傳動⇒巡航車）；(6) 引擎平均為大口徑短行程、
> 噴射普及率由 23% 升至 64%、隨機森林預測極速 RMSE 17.6 km/h（R²=0.91），佐證資料品質與方法廣度。

---

## 1. Dataset & Requirement Compliance 資料集與規格符合性

**Source.** Bikez.com motorcycle specification database, distributed on Kaggle
(*"all_bikez"* motorcycle dataset). Bikez is a long-running public catalogue of technical
specifications for virtually every production motorcycle. *(Please paste the exact Kaggle URL,
author and download date here for the citation.)*

| Requirement (from the brief) | Required | This dataset | Status |
|---|---|---|---|
| Scientific / innovative domain | yes | engineering / mechanical design | ✅ |
| Min. instances | ≥ 1,500 | **42,564** (modelling subset 9,842) | ✅ |
| Min. variables | ≥ 10 | **105 raw → 21 cleaned analytic features** | ✅ |
| Published 2023–2026 | yes | actively-maintained catalogue, model years up to **2025** | ✅ † |
| Hybrid: statistics + data mining | yes | hypothesis tests + regression **and** classification + clustering + association | ✅ |

† The catalogue contains 2023–2025 model-year bikes and is continuously updated; confirm the
exact Kaggle "last updated" date on the dataset page when you cite it.

**Why this dataset is a good fit.** It is genuinely *messy and complex*: every numeric quantity is
buried inside a free-text string with mixed units (`"241.5 ccm (14.74 cubic inches)"`,
`"23.1 HP (16.8 kW)) @ 8500 RPM"`), missingness ranges from 10 % to 84 % across fields, and it
contains real data-entry errors. This makes the **preprocessing** stage substantive rather than a
formality, and it gives both numeric and categorical variables for the full range of techniques.

---

## 2. Methodology 方法總覽

A reproducible 4-stage pipeline (one script per stage; see `src/`):

```
 raw CSV (42,564 × 105, free text)
        │  01_preprocessing.py  ── regex unit-extraction, physical bounds, cleaning
        ▼
 bikez_clean.csv (42,564 × 21)  +  bikez_model.csv (9,842 × 20, top-8 categories)
        │
        ├─ 02_eda.py          ── summaries + 7 visualisation figures
        ├─ 03_statistics.py   ── normality, t-tests (1-sample / 2-sample / paired),
        │                        ANOVA+Tukey, Kruskal-Wallis, Mann-Whitney, Wilcoxon,
        │                        sign, 2-proportion z, correlation, χ² (independence +
        │                        goodness-of-fit), OLS regression
        └─ 04_datamining.py   ── classification, clustering, association rules, PCA,
                                 class-imbalance handling, leakage demo, predictive regression
        ▼
 figures/  (16 PNG)   results/  (21 tables + transcripts)
```

Everything is driven by a single config (`src/config.py`) and re-run with `bash run_all.sh`.

---

## 3. Step 1 — Data Preprocessing 資料前處理  *(Instruction #1)*

**3.1 Unit extraction.** Numeric values were parsed out of the spec strings with carefully-built
regular expressions (`src/bikez_utils.py`). Compound fields were split: `Bore x stroke` → two
columns; `Power`/`Torque` → magnitude **and** the engine RPM at peak (`"… @ 8500 RPM"`).

**3.2 Physical-bounds outlier handling.** Each extracted variable was clipped to a
domain-plausible range (defined once in `config.py`); out-of-range values are treated as
data-entry errors and set to `NaN`. This is not cosmetic — e.g. the raw record
`"72.0 x 552.0 mm"` has an impossible 552 mm stroke (almost certainly a misplaced decimal for
55.2 mm); the bound `[15, 160] mm` correctly rejects it while keeping the valid 72 mm bore.

**3.3 Missing-value strategy.** Missingness was quantified for every feature
(`figures/01_missingness.png`, `results/missingness.csv`):

| highest missingness | % | lowest missingness | % |
|---|---|---|---|
| power_rpm | 84.3 | cooling | 10.2 |
| power_hp | 82.0 | transmission | 13.6 |
| year | 74.2 | fuel_capacity_l | 16.8 |
| top_speed_kmh | 67.2 | bore_mm | 25.2 |

Rather than impute globally (which would leak information), we **defer imputation to the
modelling pipelines**, where `SimpleImputer(median)` is fit *inside* each cross-validation fold.
For statistical tests we use complete-case (pairwise) deletion.

**3.4 Categorical cleaning.** The chaotic `Fuel system` text was collapsed to
*Injection / Carburettor / Turbo*; `Transmission type` to *Chain / Belt / Shaft*; and the
manufacturer **Brand** was recovered from the `Model` string using a **longest-prefix match**
against the official 400+ brand list (so multi-word brands like *"FB Mondial"*, *"GAS GAS"* are
not truncated).

**Output.** A tidy single table `data/bikez_clean.csv` (42,564 × 21) plus an analysis-ready
modelling subset `data/bikez_model.csv` (9,842 rows in the 8 most common categories with the two
core specs present).

---

## 4. Step 2 — Exploratory Data Analysis 探索式分析  *(Instruction #2)*

| Figure | What it shows | Key takeaway |
|---|---|---|
| `02_distributions.png` | histograms + KDE of 9 specs | engine/performance specs are **right-skewed** (skew 1.1–2.3) → motivates log-transform & non-parametric tests |
| `02_correlation_heatmaps.png` | Pearson & Spearman matrices | displacement–torque (ρ = 0.98) and power–top-speed (ρ = 0.95) are nearly deterministic |
| `02_power_by_category.png` | power boxplots per category | Sport / Sport-touring sit at the top; Scooters at the bottom |
| `02_displacement_vs_power.png` | log–log scatter by cooling | a tight power-law band; liquid-cooled bikes occupy the high-power region |
| `02_trends_over_time.png` | yearly means since 1980 | mean displacement & power rise over time; **fuel injection displaces carburettors** |
| `02_category_cooling_bars.png` | category & cooling counts | Scooter is the largest class; Air vs Liquid cooling are ~ balanced |

**Descriptive snapshot** (`results/eda_numeric_summary.csv`): median displacement 395 ccm,
median power 30 HP, median top speed 130 km/h, median dry weight 145 kg.

---

## 5. Step 3 — Statistical Methods 統計方法  *(Instruction #3)*

Full transcript: `results/statistics_report.txt`; per-test tables in `results/stat_*.csv`.

### A. Normality assessment 常態性檢定
D'Agostino–Pearson *K²* rejects normality for every raw spec (*p* < 10⁻³³). Log-transformation
shrinks the skew toward 0 (e.g. power skew 1.51 → −0.33). **Decision:** report parametric tests on
the log scale *and* non-parametric tests on the raw scale, so conclusions don't hinge on a
distributional assumption.

### B. Two-group test — power by cooling system 雙組檢定
H₀: liquid- and air-cooled engines have equal power.

| group | n | mean HP | median |
|---|---|---|---|
| Liquid | 2,807 | **74.9** | 58.0 |
| Air | 3,826 | **30.2** | 17.0 |

Welch *t* = 37.6 (*p* ≈ 2.4 × 10⁻²⁶³); Mann–Whitney *U* (*p* ≈ 1.6 × 10⁻²⁹³); **Cohen's *d* = 1.03
(large)**. → Liquid cooling is strongly associated with higher engine output — both tests agree,
so the result is robust to non-normality. (`figures/03_power_by_cooling.png`)

### C. k-group test — power across categories 多組檢定
One-way **ANOVA** on log-power: *F* = 421.6 (*p* ≈ 0); **Kruskal–Wallis**: *H* = 2597 (*p* ≈ 0);
effect size **η² = 0.37** (category explains 37 % of power variance). **Tukey HSD** post-hoc:
**48 of 55** category pairs differ significantly (`results/stat_C_tukey.csv`). → Category is a
first-order driver of engine power, not a cosmetic label.

### D. Correlation tests 相關性檢定 (`results/stat_D_correlation.csv`)

| pair | Pearson *r* | Spearman ρ |
|---|---|---|
| displacement ~ torque | 0.968 | 0.978 |
| power ~ top speed | 0.893 | 0.948 |
| displacement ~ power | 0.804 | 0.924 |
| compression ~ power/weight | 0.618 | 0.618 |
| dry weight ~ top speed | 0.509 | 0.762 |

All *p*-values are effectively zero. The gap between Pearson and Spearman for
displacement–power (0.80 vs 0.92) confirms a **monotone but non-linear** (power-law) relationship.

### E. Chi-square test of independence 卡方獨立性檢定
H₀: cooling system ⫫ final-drive type. χ² = 682.1, df = 4, *p* ≈ 2.7 × 10⁻¹⁴⁶ →
**statistically dependent**, but **Cramér's V = 0.10 (weak)**: a significant association that is
small in practice — a good teaching example that *p*-value ≠ effect size at *n* ≈ 35 k.

### F. Regression analysis 迴歸分析 (`results/stat_F_regression_summary.txt`)

**Model A (physics-motivated, log–log):** `log(top_speed) ~ log(power) + log(weight)`
*n* = 1,648, **R² = 0.893**. Coefficient on log(power) = **0.43** (95 % CI 0.42–0.44),
on log(weight) = −0.11. Aerodynamic drag theory predicts a drag-limited top speed
∝ power^(1/3) ≈ 0.33; the empirical 0.43 is slightly higher because more powerful bikes also tend
to be more aerodynamic and longer-geared — **the data quantitatively echoes the physics.**

**Model B (multivariable):** `top_speed ~ power + weight + displacement + fuel_capacity + cooling`
*n* = 1,081, **R² = 0.864**, *F* = 1,134 (*p* ≈ 0). **VIF** for all numeric predictors < 5
(max = 4.98 for displacement) → no harmful multicollinearity. Residual diagnostics
(`figures/03_regression_diagnostics.png`) show approximately homoscedastic, near-normal residuals
for the log model.

### G–J. Additional hypothesis tests 進階假設檢定
A further battery of tests, chosen by the *type* of data, rounds out the statistical toolbox
(full transcript in `results/statistics_report.txt`):

| # | Test 檢定 | Question | Result |
|---|---|---|---|
| G | **One-sample t-test** | Is the mean rating ≠ the neutral 3.0? | mean = 3.413, *t* = 167.4, *p* ≈ 0, 95% CI (3.408, 3.418) → users skew **positive** |
| H | **Paired t-test + Wilcoxon signed-rank + sign test** | Within an engine, bore ≠ stroke? | mean(bore−stroke) = **+8.03 mm**; all three *p* ≈ 0; 21,061/30,158 oversquare → engines are on average **oversquare** (high-revving) |
| I | **Two-sample proportion z-test** | Injection share modern (≥2010) vs older? | **64.0 %** vs **22.7 %**, *z* = 34.3, *p* ≈ 4×10⁻²⁵⁸ → injection displaced carburettors |
| J | **Chi-square goodness-of-fit** | Are the 3 cooling systems equally common? | χ² = 14,135.6, *p* ≈ 0 → **not** uniform (oil & air is rare) |

Test H is a neat illustration of the parametric → non-parametric → sign-test progression on a
single, mechanically-meaningful comparison (`figures/03_bore_minus_stroke.png`).

---

## 6. Step 4 — Data Mining 資料採礦  *(Instruction #4)*

### 6.1 Classification — predict category from specs 分類
Modelling subset: 9,842 bikes, 8 categories, 13 numeric features. Each model is a leakage-safe
`Pipeline(SimpleImputer → [StandardScaler] → classifier)`, evaluated with 5-fold stratified CV and
a held-out 25 % test set.

| model | CV macro-F1 | test accuracy | test macro-F1 |
|---|---|---|---|
| Logistic Regression | 0.43 ± 0.02 | 0.570 | 0.438 |
| **Random Forest** | **0.75 ± 0.01** | **0.798** | **0.761** |

The large RF-vs-LR gap shows the spec→category boundary is **non-linear** (feature interactions
matter). Per-class F1 (`results/dm_classification_report.csv`): **Scooter 0.91, Custom/cruiser 0.90,
Enduro 0.80**. The confusion matrix (`figures/04_confusion_matrix.png`) shows the only sizeable
mix-up is **Super-motard ↔ Enduro** (31 %) — sensible, since supermotards *are* converted enduro
bikes. Most discriminative features (`figures/04_feature_importance.png`): **seat height (0.166),
dry weight (0.122), fuel capacity (0.115), displacement (0.110)**.

### 6.2 Clustering — unsupervised segments 分群
K-means on standardised specs; *k* chosen by silhouette (`figures/04_kmeans_selection.png`).
**k = 3** (silhouette 0.34). Cluster profiles (`results/dm_cluster_profile.csv`):

| cluster | n | displ. | power | top speed | weight | identity |
|---|---|---|---|---|---|---|
| 0 | 6,705 | 236 ccm | 17 HP | 102 km/h | 127 kg | **small commuters / scooters** (34 % scooter) |
| 1 | 1,169 | 1,370 ccm | 76 HP | 152 km/h | 289 kg | **heavy cruisers** (79 % custom/cruiser, shaft drive) |
| 2 | 1,968 | 937 ccm | 110 HP | 226 km/h | 191 kg | **high-performance sport** (32 % sport, 32 % naked; compression 11.8) |

The clusters were discovered *without* using the category label, yet they recover the real market
segmentation — strong evidence that the engineering specs carry the type structure.

### 6.3 Association rules — design "grammar" 關聯規則
Apriori on quantile-binned specs + categorical fields (9,842 transactions, market-basket encoding
that keeps every bike; `results/dm_association_rules.csv`). Highest-lift, interpretable rules:

| antecedent ⇒ consequent | confidence | lift |
|---|---|---|
| {displacement = High, drive = Belt} ⇒ **Custom/cruiser** | 0.86 | 6.1 |
| {displacement = High, seat height = Low} ⇒ **Custom/cruiser** | 0.78 | 5.5 |
| {weight = Low, seat height = High} ⇒ **Enduro/offroad** | 0.74 | 4.9 |
| {displacement = Low, drive = Belt} ⇒ **Scooter** | 0.98 | 4.2 |
| {cooling = Liquid, weight = Low} ⇒ **Enduro/offroad** | 0.65 | 4.3 |

These read like an engineer's design grammar: *big engine + belt + low seat* ⇒ cruiser;
*small engine + belt* ⇒ scooter; *light + tall seat* ⇒ off-road.

### 6.4–6.7 Additional data-mining techniques 進階資料採礦

| # | Technique 技術 | Result |
|---|---|---|
| 6.4 | **Dimensionality reduction** — PCA scree + loadings | first 2 PCs explain **58.3 %** of variance; PC1 is a "size & power" axis (displacement / torque / weight load highest) — `figures/04_pca_scree_loadings.png` |
| 6.5 | **Class-imbalance handling** — `class_weight='balanced'` | lifts the smallest class's recall (e.g. 0.61 → 0.66) for only a ~0.01 drop in overall macro-F1 — a transparent precision/recall trade-off (`results/dm_class_imbalance.csv`) |
| 6.6 | **Data-leakage demonstration** — wrong vs right preprocessing | CV accuracy 0.5644 (preprocess on all data) vs 0.5629 (preprocess inside the pipeline); the small gap is honest for median imputation but would be large for feature selection / SMOTE / target encoding (`results/dm_leakage_demo.csv`) |
| 6.7 | **Predictive regression** — Random-Forest top-speed model | held-out **RMSE = 17.6 km/h, MAE = 9.4 km/h, R² = 0.907** (`figures/04_topspeed_regression.png`) — a strong predictive complement to the inferential OLS model of §5 |

---

## 7. Key Insights & Actionable Conclusions 重要洞察

1. **Cooling technology is a power proxy** (large, robust effect): for buyers and for data
   imputation, knowing "liquid-cooled" already implies a much higher expected power band.
2. **Specs ⇒ type with 80 % accuracy**: a catalogue with missing/incorrect category labels could be
   auto-tagged from numeric specs; *seat height + weight + fuel capacity* are enough to get most of
   the signal — useful for cleaning marketplace listings.
3. **Three real market segments** (commuter / cruiser / sport) emerge unsupervised — a clean basis
   for pricing, marketing or recommendation.
4. **The physics is in the data**: top speed follows a near cube-root law in power, validating that
   the cleaned variables are physically meaningful (a strong data-quality check).

---

## 8. Limitations & Future Work 限制與未來方向

- **Missingness is high** for performance fields (power 82 %, top speed 67 %); the regression and
  two-group analyses therefore run on the better-documented bikes (*n* ≈ 1–8 k) and may slightly
  over-represent popular models. A sensitivity analysis with multiple imputation (MICE) would
  strengthen them.
- **Selection bias**: Bikez coverage skews toward European/recent models.
- **Top speed** also depends on gearing/aerodynamics not in the data, capping Model A's *R²*.
- **Future**: gradient-boosting + SHAP for richer feature attribution; hierarchical clustering and
  brand-level association rules; price modelling using `Price as new`.

---

## 9. Code Review & Self-Evaluation 程式碼檢視與評估

*(Requested deliverable — an honest assessment of the code that produced these results.)*

**Correctness — verified, not assumed.**
- Extraction was **spot-checked against raw strings** (e.g. `"205.0 HP … @ 13000 RPM"` →
  `power_hp = 205, power_rpm = 13000`); every parse matched.
- Physical bounds were shown to catch a **genuine data-entry error** (552 mm stroke → `NaN`),
  confirming the outlier logic does real work.
- Post-clean numeric ranges are all physically sensible (displacement 25–2,575 ccm, power 1–357 HP).

**Statistical rigour.**
- Non-normality is *tested*, and every effect is reported with both parametric and non-parametric
  tests **plus an effect size** (Cohen's d, η², Cramér's V) — so we never confuse a tiny *p*-value
  with a large effect (made explicit in the χ² result).
- Regression checks assumptions: **VIF** for multicollinearity and **residual/Q-Q diagnostics**.

**Machine-learning hygiene.**
- Imputation and scaling live **inside** CV pipelines → **no train/test leakage**.
- Stratified splits/folds; two models compared; macro-F1 used because classes are imbalanced.
- Clustering uses standardised inputs and a principled *k* (silhouette), then is **validated**
  against the held-out category label.

**Engineering quality.** Single source of config, reusable utilities, deterministic
(`random_state = 42`), one-command reproducibility (`run_all.sh`), pinned versions
(`requirements.txt`), and **zero warnings** from project code.

**Where it could be stronger (honest):** global median-impute is used for the *unsupervised*
clustering (acceptable but not leak-proof in the strict sense); regression *n* is limited by
missing performance data; hyper-parameters are sensible defaults rather than tuned via grid search.
None of these change the qualitative conclusions, and each is an explicit, low-risk follow-up.

**Overall:** the pipeline is correct, reproducible, and statistically defensible; results are
consistent across methods (e.g. category matters in ANOVA, in classification, and in clustering),
which is the best available evidence that the findings are real rather than artefacts.

---

## 10. Reproducibility 重現方式

```bash
# one-time environment
conda create -n data_mining python=3.11 -y
conda activate data_mining
pip install -r requirements.txt

# reproduce everything (≈ 1–2 min)
bash run_all.sh
```
Outputs are written to `data/`, `figures/`, `results/`. See `README.md` for the file map.

---

## 11. References 參考文獻

1. **Bikez.com** motorcycle specification database (dataset via Kaggle, *all_bikez*). *(add exact URL/author/date)*
2. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12:2825–2830.
3. Seabold & Perktold (2010). *statsmodels: Econometric and statistical modeling with Python.* SciPy.
4. Virtanen et al. (2020). *SciPy 1.0.* Nature Methods 17:261–272.
5. Raschka (2018). *MLxtend.* JOSS 3(24):638.
6. Hunter (2007). *Matplotlib.* CiSE 9(3):90–95. · Waskom (2021). *seaborn.* JOSS 6(60):3021.
7. Agrawal & Srikant (1994). *Fast Algorithms for Mining Association Rules.* VLDB.
8. Breiman (2001). *Random Forests.* Machine Learning 45(1):5–32.
9. McKinney (2010). *Data Structures for Statistical Computing in Python (pandas).* SciPy.

---

## Appendix A — Presentation outline (15 min) 簡報大綱

| min | slide | content | figure |
|---|---|---|---|
| 0–2 | Motivation & question | "what defines a motorcycle's type?" + dataset scale | — |
| 2–4 | Data & preprocessing | messy text → tidy table; the 552 mm error example | `01_missingness` |
| 4–7 | EDA | skewed specs, correlation, power-law scatter, time trends | `02_correlation_heatmaps`, `02_displacement_vs_power` |
| 7–10 | Statistics | cooling t-test (d=1.03), ANOVA η²=0.37, log–log regression vs physics; advanced tests (paired bore-vs-stroke, injection 23%→64%) | `03_power_by_cooling`, `03_regression_diagnostics`, `03_bore_minus_stroke` |
| 10–13 | Data mining | RF 80 % accuracy + importances; 3 clusters; association rules; PCA + predictive regression (R²=0.91) | `04_confusion_matrix`, `04_clusters_pca`, `04_topspeed_regression` |
| 13–15 | Insights & limitations | 4 takeaways + honest caveats (incl. class-imbalance & leakage notes) | — |

**Anticipated Q&A.**
- *Why is LR so much worse than RF?* The spec→type boundary is non-linear; RF models interactions
  (e.g. high power **and** low seat ⇒ cruiser).
- *Isn't 82 % missing power a problem?* Yes — we restrict power-based analyses to documented bikes
  and report it openly; the multi-method agreement guards against bias.
- *Why is the regression exponent 0.43, not 0.33?* Idealised drag theory ignores gearing and the
  fact that powerful bikes are also more aerodynamic; 0.43 is the realistic, data-driven value.
- *Is the χ² result important?* It's significant but weak (V = 0.10) — a deliberate illustration
  that significance ≠ practical importance at large *n*.
