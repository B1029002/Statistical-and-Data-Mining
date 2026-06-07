"""
04_datamining.py
================
Step 4 of the pipeline -- DATA MINING (Instruction #4).

Three classic families of techniques on the modelling subset
(data/bikez_model.csv: 9,842 motorcycles, top-8 categories):

  1. CLASSIFICATION  -- predict the motorcycle Category from numeric specs.
     * Logistic Regression (scaled) and Random Forest, each inside a
       Pipeline(SimpleImputer -> [Scaler] -> model) so imputation/scaling are
       fit on training folds only (no leakage).
     * 5-fold stratified CV, held-out test accuracy & macro-F1,
       classification report, confusion matrix and RF feature importances.

  2. CLUSTERING -- K-means on standardised specs.
     * elbow (inertia) + silhouette scan over k,
     * PCA(2D) projection coloured by cluster,
     * per-cluster spec profile and cluster x category cross-tab.

  3. ASSOCIATION RULES -- Apriori on quantile-binned specs + categorical fields.
     * frequent itemsets, rules ranked by lift, top rules exported.

Run (after 01):  python 04_datamining.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, recall_score,
                             mean_absolute_error, r2_score, root_mean_squared_error)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from mlxtend.frequent_patterns import apriori, association_rules

import config as C
import bikez_utils as U


# =========================================================================== #
# 1. CLASSIFICATION
# =========================================================================== #
def classification(df):
    U.section("1. CLASSIFICATION -- predict motorcycle category from specs")
    feat = [c for c in C.MODEL_FEATURES if c in df.columns]
    X = df[feat]
    y = df["category"]
    print(f"Features ({len(feat)}): {feat}")
    print(f"Samples: {len(X):,}   Classes: {y.nunique()}")
    print("Class balance:\n", y.value_counts().to_string())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)

    models = {
        "LogisticRegression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=C.RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400, random_state=C.RANDOM_STATE, n_jobs=-1)),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    summary = []
    fitted = {}
    for name, pipe in models.items():
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, pred)
        f1m = f1_score(y_te, pred, average="macro")
        summary.append([name, cv_f1.mean(), cv_f1.std(), acc, f1m])
        fitted[name] = (pipe, pred)
        print(f"\n{name}: CV macro-F1 = {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}"
              f" | test acc = {acc:.3f} | test macro-F1 = {f1m:.3f}")

    summ = pd.DataFrame(summary, columns=["model", "cv_f1_mean", "cv_f1_std",
                                          "test_accuracy", "test_macro_f1"])
    U.save_table(summ, C.RESULTS_DIR / "dm_classification_scores.csv", index=False)

    # detailed report + confusion matrix for the best model (by test macro-F1)
    best = summ.sort_values("test_macro_f1", ascending=False).iloc[0]["model"]
    print(f"\nBest model: {best}")
    pipe, pred = fitted[best]
    rep = classification_report(y_te, pred, zero_division=0, output_dict=True)
    U.save_table(pd.DataFrame(rep).T, C.RESULTS_DIR / "dm_classification_report.csv")
    print(classification_report(y_te, pred, zero_division=0))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_te, pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax, cbar_kws={"label": "row-normalised"})
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (row-normalised) -- {best}")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    U.savefig(fig, C.FIG_DIR / "04_confusion_matrix.png")

    # RF feature importance
    if "RandomForest" in fitted:
        rf = fitted["RandomForest"][0].named_steps["clf"]
        imp = pd.Series(rf.feature_importances_, index=feat).sort_values()
        U.save_table(imp.rename("importance").to_frame(),
                     C.RESULTS_DIR / "dm_feature_importance.csv")
        fig, ax = plt.subplots(figsize=(10, 8))
        imp.plot.barh(ax=ax, color="#55A868")
        ax.set_title("Random-Forest feature importance (category prediction)")
        ax.set_xlabel("Mean decrease in impurity")
        U.savefig(fig, C.FIG_DIR / "04_feature_importance.png")


# =========================================================================== #
# 2. CLUSTERING
# =========================================================================== #
def clustering(df):
    U.section("2. CLUSTERING -- K-means on standardised specs")
    feat = [c for c in C.MODEL_FEATURES if c in df.columns]
    # median-impute then standardise (unsupervised, so a global fit is fine here)
    X = df[feat].copy()
    X = X.fillna(X.median(numeric_only=True))
    Xs = StandardScaler().fit_transform(X)

    ks = range(2, 11)
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE)
        lab = km.fit_predict(Xs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Xs, lab, sample_size=4000,
                                    random_state=C.RANDOM_STATE))
        print(f"  k={k}: inertia={km.inertia_:.0f}  silhouette={sil[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(list(ks), inertia, "o-", color="#4C72B0")
    axes[0].set_title("Elbow method"); axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[1].plot(list(ks), sil, "o-", color="#C44E52")
    axes[1].set_title("Silhouette score"); axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
    fig.tight_layout()
    U.savefig(fig, C.FIG_DIR / "04_kmeans_selection.png")

    best_k = list(ks)[int(np.argmax(sil))]
    print(f"\nChosen k (max silhouette) = {best_k}")
    km = KMeans(n_clusters=best_k, n_init=10, random_state=C.RANDOM_STATE)
    df = df.copy()
    df["cluster"] = km.fit_predict(Xs)

    # PCA 2D projection
    pca = PCA(n_components=2, random_state=C.RANDOM_STATE)
    pcs = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=df["cluster"], cmap="tab10",
                    s=10, alpha=0.5)
    ax.set_title(f"K-means clusters (k={best_k}) in PCA space\n"
                 f"PC1+PC2 explain {pca.explained_variance_ratio_[:2].sum()*100:.1f}% var")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    legend = ax.legend(*sc.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    U.savefig(fig, C.FIG_DIR / "04_clusters_pca.png")

    # cluster profile (mean of original-scale features) + size
    profile = df.groupby("cluster")[feat].mean().round(1)
    profile.insert(0, "size", df["cluster"].value_counts().sort_index())
    U.save_table(profile, C.RESULTS_DIR / "dm_cluster_profile.csv")
    print("\nCluster profile (mean specs):")
    print(profile.to_string())

    # which categories fall in which cluster
    ctab = pd.crosstab(df["cluster"], df["category"], normalize="index").round(2)
    U.save_table(ctab, C.RESULTS_DIR / "dm_cluster_vs_category.csv")
    print("\nCluster x category (row-normalised):")
    print(ctab.to_string())


# =========================================================================== #
# 3. ASSOCIATION RULES
# =========================================================================== #
def association(df):
    U.section("3. ASSOCIATION RULES -- Apriori on binned specs + categoricals")
    # quantile-bin a few numeric specs into Low/Med/High
    to_bin = ["displacement_ccm", "power_hp", "dry_weight_kg",
              "top_speed_kmh", "fuel_capacity_l", "seat_height_mm"]
    items = pd.DataFrame(index=df.index)
    for c in to_bin:
        if c in df.columns:
            try:
                binned = pd.qcut(df[c], 3, labels=["Low", "Med", "High"])
                items[c] = c + "=" + binned.astype("string")
            except ValueError:
                pass  # not enough distinct values to form 3 bins
    # add categorical fields directly
    for c in ["category", "cooling", "transmission", "fuel_system"]:
        if c in df.columns:
            items[c] = c + "=" + df[c].astype("string")

    # Market-basket transactions: each motorcycle contributes ONLY the items it
    # actually has (a missing spec is simply absent, not False) -- this keeps all
    # rows instead of discarding any bike with a single missing field.
    long = (items.reset_index().melt(id_vars="index", value_name="item")
            .dropna(subset=["item"]))
    onehot = pd.crosstab(long["index"], long["item"]).astype(bool)
    print(f"Transactions used: {onehot.shape[0]:,}")
    print(f"One-hot item table: {onehot.shape[0]:,} x {onehot.shape[1]} items")

    # max_len keeps itemsets small and rules interpretable (no giant permutations)
    freq = apriori(onehot, min_support=0.05, use_colnames=True, max_len=3)
    print(f"Frequent itemsets (support>=0.05, len<=3): {len(freq):,}")

    try:
        rules = association_rules(freq, metric="lift", min_threshold=1.2)
    except TypeError:
        rules = association_rules(freq, metric="lift", min_threshold=1.2,
                                  num_itemsets=len(onehot))

    # keep concise, high-quality rules: <=2 antecedents -> exactly 1 consequent
    rules = rules[(rules["confidence"] >= 0.6) & (rules["lift"] >= 1.2)
                  & (rules["antecedents"].apply(len) <= 2)
                  & (rules["consequents"].apply(len) == 1)]
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    rules = (rules.drop_duplicates(subset=["antecedents", "consequents"])
             .sort_values("lift", ascending=False))
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    top = rules[cols].head(20).reset_index(drop=True)
    U.save_table(top, C.RESULTS_DIR / "dm_association_rules.csv", index=False)
    print(f"\nTop association rules (by lift): {len(rules):,} rules total")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(top.to_string())


# =========================================================================== #
# 4. DIMENSIONALITY REDUCTION (PCA scree + loadings)
# =========================================================================== #
def dimensionality_reduction(df):
    U.section("4. DIMENSIONALITY REDUCTION -- PCA scree plot + loadings")
    feat = [c for c in C.MODEL_FEATURES if c in df.columns]
    X = df[feat].fillna(df[feat].median(numeric_only=True))
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    ev = pca.explained_variance_ratio_
    print(f"PC1+PC2 explain {ev[:2].sum()*100:.1f}% of total variance")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(range(1, len(ev)+1), ev.cumsum()*100, "o-", color="#4C72B0")
    axes[0].axhline(80, color="red", ls="--")
    axes[0].set_title("PCA cumulative explained variance (%)")
    axes[0].set_xlabel("Number of components"); axes[0].set_ylabel("Cumulative %")
    load = pd.DataFrame(pca.components_[:2].T, index=feat, columns=["PC1", "PC2"])
    sns.heatmap(load, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1])
    axes[1].set_title("PCA loadings (PC1/PC2)")
    fig.tight_layout()
    U.savefig(fig, C.FIG_DIR / "04_pca_scree_loadings.png")
    U.save_table(load.round(3), C.RESULTS_DIR / "dm_pca_loadings.csv")


# =========================================================================== #
# 5. CLASS IMBALANCE (effect of class_weight='balanced')
# =========================================================================== #
def class_imbalance(df):
    U.section("5. CLASS IMBALANCE -- effect of class_weight='balanced'")
    feat = [c for c in C.MODEL_FEATURES if c in df.columns]
    X, y = df[feat], df["category"]
    print("Class counts (imbalanced):\n", y.value_counts().to_string())
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    minority = y.value_counts().index[-1]            # smallest class
    rows = []
    for tag, weight in [("default", None), ("balanced", "balanced")]:
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", RandomForestClassifier(
                             n_estimators=400, n_jobs=-1,
                             random_state=C.RANDOM_STATE, class_weight=weight))])
        pipe.fit(X_tr, y_tr); pred = pipe.predict(X_te)
        macro = f1_score(y_te, pred, average="macro")
        rec_min = recall_score(y_te, pred, labels=[minority], average="macro")
        rows.append([tag, round(macro, 3), round(rec_min, 3)])
        print(f"  {tag:9s}: macro-F1 = {macro:.3f} | minority '{minority}' recall = {rec_min:.3f}")
    tab = pd.DataFrame(rows, columns=["class_weight", "macro_f1", f"recall[{minority}]"])
    U.save_table(tab, C.RESULTS_DIR / "dm_class_imbalance.csv", index=False)
    print("-> 'balanced' raises minority recall at a small cost to overall F1.")


# =========================================================================== #
# 6. DATA-LEAKAGE DEMONSTRATION (wrong vs right preprocessing)
# =========================================================================== #
def leakage_demo(df):
    U.section("6. DATA-LEAKAGE DEMO -- preprocessing on all data vs inside a pipeline")
    feat = [c for c in C.MODEL_FEATURES if c in df.columns]
    X, y = df[feat], df["category"]
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    # WRONG: fit imputer + scaler on the WHOLE training set, then cross-validate
    leaky = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(X_tr))
    acc_leak = cross_val_score(LogisticRegression(max_iter=2000), leaky, y_tr,
                               cv=cv, scoring="accuracy").mean()
    # RIGHT: preprocessing inside the pipeline -> fit on training folds only
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000))])
    acc_ok = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
    print(f"  (wrong) preprocessing on all data : CV accuracy = {acc_leak:.4f}")
    print(f"  (right) preprocessing in pipeline : CV accuracy = {acc_ok:.4f}")
    print(f"  gap = {acc_leak - acc_ok:+.4f}  (small for median imputation; large for "
          "feature selection / SMOTE / target encoding)")
    U.save_table(pd.DataFrame({"method": ["leaky_all_data", "pipeline_correct"],
                               "cv_accuracy": [round(acc_leak, 4), round(acc_ok, 4)]}),
                 C.RESULTS_DIR / "dm_leakage_demo.csv", index=False)


# =========================================================================== #
# 7. PREDICTIVE REGRESSION (top speed) with RMSE / MAE
# =========================================================================== #
def predictive_regression(df_clean):
    U.section("7. PREDICTIVE REGRESSION -- predict top speed, report RMSE / MAE")
    feat = [c for c in C.MODEL_FEATURES if c in df_clean.columns and c != "top_speed_kmh"]
    reg = df_clean.dropna(subset=["top_speed_kmh"]).copy()
    X, y = reg[feat], reg["top_speed_kmh"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
    model = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("rf", RandomForestRegressor(n_estimators=400, n_jobs=-1,
                                                   random_state=C.RANDOM_STATE))]).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = root_mean_squared_error(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    print(f"n = {len(reg):,} | test RMSE = {rmse:.1f} km/h, MAE = {mae:.1f} km/h, R2 = {r2:.3f}")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_te, pred, s=8, alpha=.3)
    lims = [y_te.min(), y_te.max()]; ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual top speed (km/h)"); ax.set_ylabel("Predicted top speed (km/h)")
    ax.set_title(f"Top speed prediction (R2={r2:.2f})")
    U.savefig(fig, C.FIG_DIR / "04_topspeed_regression.png")
    U.save_table(pd.DataFrame({"metric": ["RMSE", "MAE", "R2"],
                               "value": [round(rmse, 2), round(mae, 2), round(r2, 3)]}),
                 C.RESULTS_DIR / "dm_regression_metrics.csv", index=False)


def main():
    U.section("STEP 4  |  DATA MINING")
    if not C.MODEL_CSV.exists():
        raise FileNotFoundError("Run 01_preprocessing.py first.")
    df = pd.read_csv(C.MODEL_CSV)
    # --- core techniques ---
    classification(df)
    clustering(df)
    association(df)
    # --- advanced techniques (synced from the notebook) ---
    dimensionality_reduction(df)
    class_imbalance(df)
    leakage_demo(df)
    predictive_regression(pd.read_csv(C.CLEAN_CSV))
    print("\nSTEP 4 complete. Models evaluated; figures/tables saved.")


if __name__ == "__main__":
    main()
