#!/usr/bin/env python3
"""
=====================================================================
KETA COASTAL FLOOD PREDICTION -- XGBoost PIPELINE v4.0
Companion to the GEE v4.0 Random Forest script.
=====================================================================

This script loads the GEE-exported sample CSVs and trains XGBoost
classifiers for both Model A (flood susceptibility) and Model B
(driver attribution). It replicates every evaluation from the GEE
RF pipeline: terrain ablation, partial leakage, strict leakage.

Usage:
    python keta_xgboost_pipeline.py

Prerequisites:
    pip install xgboost scikit-learn pandas numpy matplotlib seaborn
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# PATHS
# =====================================================================
MODEL_A_CSV = r"C:\Users\LEE\Downloads\keta_samples_v4_0_all_splits.csv"
MODEL_B_CSV = r"C:\Users\LEE\Downloads\keta_samples_driver_v4_0_all_splits.csv"
OUTPUT_DIR  = r"C:\Users\LEE\Documents\Keta Intelligence system"

# =====================================================================
# FEATURE SETS (matching GEE v4.0 exactly)
# =====================================================================
INPUT_FEATURES_ALL = [
    "elevation", "slope", "aspect", "dist_water", "lt3", "lt1", "lt0",
    "imerg_24h", "imerg_peak_intensity", "imerg_3d",
    "chirps_7d", "chirps_30d",
    "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
    "soil_moisture",
    "tide_max_fes", "tide_range_fes", "spring_flag_fes", "freeboard_fes",
    "compound_risk", "season_wet"
]

# Model A terrain ablation (remove top-3 static terrain)
INPUT_FEATURES_NO_TERRAIN = [
    f for f in INPUT_FEATURES_ALL
    if f not in ["slope", "aspect", "dist_water"]
]

# Model B: dynamic-only (no terrain at all)
INPUT_FEATURES_DRIVER_ONLY = [
    "imerg_24h", "imerg_peak_intensity", "imerg_3d",
    "chirps_7d", "chirps_30d",
    "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
    "soil_moisture",
    "tide_max_fes", "tide_range_fes", "spring_flag_fes", "freeboard_fes",
    "compound_risk", "season_wet"
]

# Model B partial leakage: remove direct label-rule inputs + terrain
INPUT_FEATURES_NO_LEAK = [
    "imerg_peak_intensity", "imerg_3d",
    "chirps_7d", "chirps_30d",
    "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
    "soil_moisture",
    "tide_range_fes", "spring_flag_fes",
    "season_wet"
]

# Model B strict leakage: only features with ZERO role in RFI/CII
INPUT_FEATURES_STRICT = [
    "imerg_peak_intensity",
    "chirps_7d",
    "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
    "spring_flag_fes",
    "season_wet"
]


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================
def separator(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def print_binary_metrics(y_true, y_pred, label=""):
    """Print confusion matrix + accuracy/kappa/precision/recall/F1."""
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  {label}")
    print(f"  Confusion Matrix:")
    print(f"    {cm.tolist()}")
    print(f"  Overall Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Kappa:             {kap:.4f}")
    print(f"  Precision:         {prec:.4f}")
    print(f"  Recall:            {rec:.4f}")
    print(f"  F1-Score:          {f1:.4f}")
    return {"acc": acc, "kap": kap, "prec": prec, "rec": rec, "f1": f1}


def print_multiclass_metrics(y_true, y_pred, class_names, label=""):
    """Print confusion matrix + accuracy/kappa + per-class recall/precision."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    acc = accuracy_score(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred)

    print(f"\n  {label}")
    print(f"  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"    Classes: {class_names}")
    for i, row in enumerate(cm):
        print(f"    {class_names[i]:>18s}: {row.tolist()}")
    print(f"  Overall Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Kappa:             {kap:.4f}")

    print(f"  Per-class metrics:")
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"    {name:>18s}:  Precision={p:.4f}  Recall={r:.4f}  Support={cm[i,:].sum()}")

    return {"acc": acc, "kap": kap}


def plot_feature_importance(model, feature_names, title, filepath):
    """Save a horizontal bar chart of XGBoost feature importance."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.35)))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="#4C72B0")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  [SAVED] {filepath}")


def compute_sample_weights(y, classes):
    """Compute per-sample weights for balanced multiclass training."""
    counts = {}
    for c in classes:
        counts[c] = np.sum(y == c)
    max_count = max(counts.values())
    weights = np.ones(len(y), dtype=float)
    for c in classes:
        if counts[c] > 0:
            weights[y == c] = max_count / counts[c]
    return weights


# =====================================================================
# LOAD DATA
# =====================================================================
separator("LOADING DATA")

df_a = pd.read_csv(MODEL_A_CSV)
df_b = pd.read_csv(MODEL_B_CSV)

print(f"  Model A CSV: {len(df_a)} rows, {len(df_a.columns)} columns")
print(f"  Model B CSV: {len(df_b)} rows, {len(df_b.columns)} columns")

# Convert numeric columns
for col in INPUT_FEATURES_ALL + ["flood", "flood_driver"]:
    if col in df_a.columns:
        df_a[col] = pd.to_numeric(df_a[col], errors="coerce")
    if col in df_b.columns:
        df_b[col] = pd.to_numeric(df_b[col], errors="coerce")

# Split
train_a = df_a[df_a["split"] == "train"].copy()
test_a  = df_a[df_a["split"] == "test"].copy()
val_a   = df_a[df_a["split"] == "validation"].copy()

train_b = df_b[df_b["split"] == "train"].copy()
test_b  = df_b[df_b["split"] == "test"].copy()
val_b   = df_b[df_b["split"] == "validation"].copy()

print(f"\n  Model A splits: train={len(train_a)}, test={len(test_a)}, val={len(val_a)}")
print(f"  Model B splits: train={len(train_b)}, test={len(test_b)}, val={len(val_b)}")

# Class counts
print(f"\n  Model A train flood distribution:")
print(f"    Flood=0: {(train_a['flood']==0).sum()}")
print(f"    Flood=1: {(train_a['flood']==1).sum()}")

print(f"\n  Model B train driver distribution:")
for c in [0, 1, 2]:
    print(f"    Class {c}: {(train_b['flood_driver']==c).sum()}")


# =====================================================================
# MODEL A: FLOOD SUSCEPTIBILITY (XGBoost)
# =====================================================================
separator("MODEL A: FLOOD SUSCEPTIBILITY (XGBoost)")

X_train_a = train_a[INPUT_FEATURES_ALL].values
y_train_a = train_a["flood"].astype(int).values
X_test_a  = test_a[INPUT_FEATURES_ALL].values
y_test_a  = test_a["flood"].astype(int).values
X_val_a   = val_a[INPUT_FEATURES_ALL].values
y_val_a   = val_a["flood"].astype(int).values

# Compute class weight
n_neg = (y_train_a == 0).sum()
n_pos = (y_train_a == 1).sum()
spw = n_neg / n_pos if n_pos > 0 else 1.0
print(f"  scale_pos_weight = {spw:.3f} ({n_neg} neg / {n_pos} pos)")

xgb_a = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=spw,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False,
    verbosity=0
)

print("  Training Model A...")
xgb_a.fit(X_train_a, y_train_a)
print("  Done.")

# --- Evaluate ---
y_pred_test_a  = xgb_a.predict(X_test_a)
y_pred_val_a   = xgb_a.predict(X_val_a)

res_a_test = print_binary_metrics(y_test_a, y_pred_test_a, "TEST SET (2023)")
res_a_val  = print_binary_metrics(y_val_a, y_pred_val_a, "VALIDATION SET (2024-2025)")

# --- Feature importance ---
print("\n  Feature Importance (sorted, highest first):")
imp_a = xgb_a.feature_importances_
sorted_idx = np.argsort(imp_a)[::-1]
for i in sorted_idx:
    print(f"    {INPUT_FEATURES_ALL[i]:>25s}: {imp_a[i]:.4f}")

plot_feature_importance(
    xgb_a, INPUT_FEATURES_ALL,
    "Model A: XGBoost Feature Importance (Flood Susceptibility)",
    os.path.join(OUTPUT_DIR, "xgb_model_a_feature_importance.png")
)


# =====================================================================
# MODEL A: TERRAIN ABLATION CHECK
# =====================================================================
separator("MODEL A: TERRAIN ABLATION (no slope/aspect/dist_water)")

X_train_a_nt = train_a[INPUT_FEATURES_NO_TERRAIN].values
X_test_a_nt  = test_a[INPUT_FEATURES_NO_TERRAIN].values
X_val_a_nt   = val_a[INPUT_FEATURES_NO_TERRAIN].values

xgb_a_nt = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=spw, random_state=42,
    eval_metric="logloss", use_label_encoder=False, verbosity=0
)

print("  Training terrain-ablated Model A...")
xgb_a_nt.fit(X_train_a_nt, y_train_a)

y_pred_test_a_nt = xgb_a_nt.predict(X_test_a_nt)
y_pred_val_a_nt  = xgb_a_nt.predict(X_val_a_nt)

res_a_nt_test = print_binary_metrics(y_test_a, y_pred_test_a_nt, "TERRAIN-ABLATED TEST SET (2023)")
res_a_nt_val  = print_binary_metrics(y_val_a, y_pred_val_a_nt,  "TERRAIN-ABLATED VALIDATION SET (2024-2025)")


# =====================================================================
# MODEL B: DRIVER ATTRIBUTION (3-class, dynamic-only)
# =====================================================================
separator("MODEL B: DRIVER ATTRIBUTION (XGBoost, terrain-free)")

X_train_b = train_b[INPUT_FEATURES_DRIVER_ONLY].values
y_train_b = train_b["flood_driver"].astype(int).values
X_test_b  = test_b[INPUT_FEATURES_DRIVER_ONLY].values
y_test_b  = test_b["flood_driver"].astype(int).values
X_val_b   = val_b[INPUT_FEATURES_DRIVER_ONLY].values
y_val_b   = val_b["flood_driver"].astype(int).values

# Compute sample weights for 3-class balance
sw_train_b = compute_sample_weights(y_train_b, [0, 1, 2])
print(f"  Class weights applied:")
for c in [0, 1, 2]:
    n = (y_train_b == c).sum()
    w = sw_train_b[y_train_b == c][0] if n > 0 else 0
    print(f"    Class {c}: {n} samples, weight={w:.3f}")

DRIVER_NAMES = ["0: no-flood", "1: rain-dominant", "2: coastal-dominant"]

xgb_b = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=3,
    random_state=42,
    eval_metric="mlogloss",
    use_label_encoder=False,
    verbosity=0
)

print("  Training Model B (dynamic-only features)...")
xgb_b.fit(X_train_b, y_train_b, sample_weight=sw_train_b)
print("  Done.")

y_pred_test_b = xgb_b.predict(X_test_b)
y_pred_val_b  = xgb_b.predict(X_val_b)

res_b_test = print_multiclass_metrics(y_test_b, y_pred_test_b, DRIVER_NAMES, "MODEL B TEST SET (2023)")
res_b_val  = print_multiclass_metrics(y_val_b, y_pred_val_b, DRIVER_NAMES, "MODEL B VALIDATION SET (2024-2025)")

# --- Feature importance ---
print("\n  Model B Feature Importance (sorted):")
imp_b = xgb_b.feature_importances_
sorted_idx_b = np.argsort(imp_b)[::-1]
for i in sorted_idx_b:
    print(f"    {INPUT_FEATURES_DRIVER_ONLY[i]:>25s}: {imp_b[i]:.4f}")

plot_feature_importance(
    xgb_b, INPUT_FEATURES_DRIVER_ONLY,
    "Model B: XGBoost Feature Importance (Driver Attribution, Dynamic-Only)",
    os.path.join(OUTPUT_DIR, "xgb_model_b_feature_importance.png")
)


# =====================================================================
# MODEL B: PARTIAL LEAKAGE CHECK
# =====================================================================
separator("MODEL B: PARTIAL LEAKAGE CHECK")

X_train_b_nl = train_b[INPUT_FEATURES_NO_LEAK].values
X_test_b_nl  = test_b[INPUT_FEATURES_NO_LEAK].values
X_val_b_nl   = val_b[INPUT_FEATURES_NO_LEAK].values

xgb_b_nl = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    objective="multi:softmax", num_class=3, random_state=42,
    eval_metric="mlogloss", use_label_encoder=False, verbosity=0
)

print("  Training partial-leakage Model B...")
xgb_b_nl.fit(X_train_b_nl, y_train_b, sample_weight=sw_train_b)

y_pred_test_b_nl = xgb_b_nl.predict(X_test_b_nl)
y_pred_val_b_nl  = xgb_b_nl.predict(X_val_b_nl)

res_b_nl_test = print_multiclass_metrics(y_test_b, y_pred_test_b_nl, DRIVER_NAMES, "PARTIAL LEAKAGE TEST (2023)")
res_b_nl_val  = print_multiclass_metrics(y_val_b, y_pred_val_b_nl, DRIVER_NAMES, "PARTIAL LEAKAGE VALIDATION (2024-2025)")

print("\n  Partial Leakage Feature Importance:")
imp_b_nl = xgb_b_nl.feature_importances_
for i in np.argsort(imp_b_nl)[::-1]:
    print(f"    {INPUT_FEATURES_NO_LEAK[i]:>25s}: {imp_b_nl[i]:.4f}")


# =====================================================================
# MODEL B: STRICT LEAKAGE CHECK
# =====================================================================
separator("MODEL B: STRICT LEAKAGE CHECK (9 independent features)")

X_train_b_st = train_b[INPUT_FEATURES_STRICT].values
X_test_b_st  = test_b[INPUT_FEATURES_STRICT].values
X_val_b_st   = val_b[INPUT_FEATURES_STRICT].values

xgb_b_st = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    objective="multi:softmax", num_class=3, random_state=42,
    eval_metric="mlogloss", use_label_encoder=False, verbosity=0
)

print("  Training strict-leakage Model B...")
xgb_b_st.fit(X_train_b_st, y_train_b, sample_weight=sw_train_b)

y_pred_test_b_st = xgb_b_st.predict(X_test_b_st)
y_pred_val_b_st  = xgb_b_st.predict(X_val_b_st)

res_b_st_test = print_multiclass_metrics(y_test_b, y_pred_test_b_st, DRIVER_NAMES, "STRICT LEAKAGE TEST (2023)")
res_b_st_val  = print_multiclass_metrics(y_val_b, y_pred_val_b_st, DRIVER_NAMES, "STRICT LEAKAGE VALIDATION (2024-2025)")

print("\n  Strict Leakage Feature Importance:")
imp_b_st = xgb_b_st.feature_importances_
for i in np.argsort(imp_b_st)[::-1]:
    print(f"    {INPUT_FEATURES_STRICT[i]:>25s}: {imp_b_st[i]:.4f}")


# =====================================================================
# COMPARISON TABLE: RANDOM FOREST vs XGBoost
# =====================================================================
separator("COMPARISON: RANDOM FOREST (GEE) vs XGBoost (Python)")

# RF baseline numbers from the GEE v4.0 console run
rf_results = {
    "Model A Test Acc":        0.9302,
    "Model A Test Kappa":      0.8518,
    "Model A Val Acc":         0.9331,
    "Model A Ablated Test":    0.7747,
    "Model A Ablated Val":     0.6828,
    "Model B Test Acc":        0.5835,
    "Model B Test Kappa":      0.2447,
    "Model B Val Acc":         0.6178,
    "Model B Val Kappa":       0.1661,
    "Partial Leak Test Acc":   0.5502,
    "Partial Leak Test Kappa": 0.0179,
    "Partial Leak Val Acc":    0.5750,
    "Partial Leak Val Kappa":  0.0000,
    "Strict Leak Test Acc":    0.4871,
    "Strict Leak Test Kappa": -0.0092,
    "Strict Leak Val Acc":     0.5810,
    "Strict Leak Val Kappa":   0.0193,
}

xgb_results = {
    "Model A Test Acc":        res_a_test["acc"],
    "Model A Test Kappa":      res_a_test["kap"],
    "Model A Val Acc":         res_a_val["acc"],
    "Model A Ablated Test":    res_a_nt_test["acc"],
    "Model A Ablated Val":     res_a_nt_val["acc"],
    "Model B Test Acc":        res_b_test["acc"],
    "Model B Test Kappa":      res_b_test["kap"],
    "Model B Val Acc":         res_b_val["acc"],
    "Model B Val Kappa":       res_b_val["kap"],
    "Partial Leak Test Acc":   res_b_nl_test["acc"],
    "Partial Leak Test Kappa": res_b_nl_test["kap"],
    "Partial Leak Val Acc":    res_b_nl_val["acc"],
    "Partial Leak Val Kappa":  res_b_nl_val["kap"],
    "Strict Leak Test Acc":    res_b_st_test["acc"],
    "Strict Leak Test Kappa":  res_b_st_test["kap"],
    "Strict Leak Val Acc":     res_b_st_val["acc"],
    "Strict Leak Val Kappa":   res_b_st_val["kap"],
}

print(f"\n  {'Metric':<30s} {'RF (GEE)':>10s} {'XGBoost':>10s} {'Delta':>10s}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
for key in rf_results:
    rf_val  = rf_results[key]
    xgb_val = xgb_results[key]
    delta   = xgb_val - rf_val
    arrow = "+" if delta > 0.005 else ("-" if delta < -0.005 else "=")
    print(f"  {key:<30s} {rf_val:>9.4f}  {xgb_val:>9.4f}  {delta:>+8.4f} {arrow}")


# =====================================================================
# DONE
# =====================================================================
separator("PIPELINE COMPLETE")
print(f"  Feature importance plots saved to: {OUTPUT_DIR}")
print(f"  Files:")
print(f"    - xgb_model_a_feature_importance.png")
print(f"    - xgb_model_b_feature_importance.png")
print()
print("  Copy these results into your research paper to present")
print("  an RF vs XGBoost comparison table in Section 4.")
print()
