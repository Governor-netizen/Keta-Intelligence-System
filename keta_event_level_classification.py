#!/usr/bin/env python3
"""
=====================================================================
KETA COASTAL FLOOD PREDICTION -- EVENT-LEVEL DRIVER CLASSIFICATION
Aggregates pixel-level features to per-event vectors, then uses
Leave-One-Out Cross-Validation (LOOCV) with XGBoost to classify
each event as rain-dominant (1) vs coastal-dominant (2).
=====================================================================
"""
import os, warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

# =====================================================================
# PATHS
# =====================================================================
MODEL_B_CSV = r"C:\Users\LEE\Downloads\keta_samples_driver_v4_0_all_splits.csv"
OUT_DIR     = r"C:\Users\LEE\Documents\Keta Intelligence system"

# =====================================================================
# FEATURES TO AGGREGATE PER EVENT
# =====================================================================
# These are the dynamic meteorological/tidal features
AGG_FEATURES = [
    "imerg_24h", "imerg_peak_intensity", "imerg_3d",
    "chirps_7d", "chirps_30d",
    "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
    "soil_moisture",
    "tide_max_fes", "tide_range_fes", "spring_flag_fes",
    "freeboard_fes", "compound_risk", "season_wet"
]

# Strict features: exclude ALL direct RFI/CII formula inputs and components
LEAK_VARS = [
    "imerg_24h", "freeboard_fes", "tide_max_fes", "tide_range_fes",
    "chirps_30d", "soil_moisture", "compound_risk"
]
STRICT_FEATURES = [f for f in AGG_FEATURES if f not in LEAK_VARS]

def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

# =====================================================================
# LOAD & AGGREGATE
# =====================================================================
sep("LOADING PIXEL-LEVEL DATA")
df = pd.read_csv(MODEL_B_CSV)

# Coerce numeric
for col in AGG_FEATURES + ["flood_driver", "flood", "RFI", "CII"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"  Total pixel samples: {len(df)}")
print(f"  Events: {df['event_id'].nunique()}")
print(f"  Driver class distribution:\n{df['flood_driver'].value_counts().to_string()}")

# Keep only flooded pixels (driver != 0) for event-level aggregation
df_flood = df[df["flood_driver"].isin([1, 2])].copy()
print(f"\n  Flooded pixel samples (class 1+2): {len(df_flood)}")

sep("AGGREGATING TO EVENT-LEVEL FEATURES")

# For each event, compute: mean, max, std of each feature across flooded pixels
# Also compute the event's majority driver label
event_rows = []
for eid, grp in df_flood.groupby("event_id"):
    row = {"event_id": eid, "split": grp["split"].iloc[0]}

    # Majority-vote driver label for this event
    driver_counts = grp["flood_driver"].value_counts()
    row["driver_label"] = driver_counts.idxmax()  # 1 or 2
    row["n_pixels"] = len(grp)
    row["rain_fraction"] = (grp["flood_driver"] == 1).mean()
    row["coastal_fraction"] = (grp["flood_driver"] == 2).mean()

    # Aggregate each feature
    for f in AGG_FEATURES:
        if f in grp.columns:
            vals = grp[f].dropna()
            if len(vals) > 0:
                row[f"{f}_mean"] = vals.mean()
                row[f"{f}_max"]  = vals.max()
                row[f"{f}_std"]  = vals.std() if len(vals) > 1 else 0.0
            else:
                row[f"{f}_mean"] = 0.0
                row[f"{f}_max"]  = 0.0
                row[f"{f}_std"]  = 0.0

    # Mean RFI and CII for diagnostics
    if "RFI" in grp.columns:
        row["RFI_mean"] = grp["RFI"].mean()
    if "CII" in grp.columns:
        row["CII_mean"] = grp["CII"].mean()

    event_rows.append(row)

events_df = pd.DataFrame(event_rows)
print(f"\n  Aggregated to {len(events_df)} events")
print(f"\n  Event-level driver labels:")
print(events_df[["event_id", "split", "driver_label", "n_pixels",
                  "rain_fraction", "coastal_fraction"]].to_string(index=False))

# =====================================================================
# FEATURE MATRIX
# =====================================================================
# Build feature columns from aggregated stats
feature_cols_full = []
for f in AGG_FEATURES:
    for stat in ["mean", "max", "std"]:
        col = f"{f}_{stat}"
        if col in events_df.columns:
            feature_cols_full.append(col)

feature_cols_strict = []
for f in STRICT_FEATURES:
    for stat in ["mean", "max", "std"]:
        col = f"{f}_{stat}"
        if col in events_df.columns:
            feature_cols_strict.append(col)

# Remap labels: 1 -> 0 (rain), 2 -> 1 (coastal) for binary classification
events_df["label_binary"] = (events_df["driver_label"] == 2).astype(int)
CLASS_NAMES = ["Rain-dominant", "Coastal-dominant"]

sep("EVENT-LEVEL CLASSIFICATION -- LOOCV")

def run_loocv(events_df, feature_cols, label_col, title):
    """Run Leave-One-Out CV with XGBoost and report results."""
    X = events_df[feature_cols].values
    y = events_df[label_col].values
    event_ids = events_df["event_id"].values

    # Standardize features
    scaler = StandardScaler()

    loo = LeaveOneOut()
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    event_results = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            objective="binary:logistic", random_state=42,
            eval_metric="logloss", use_label_encoder=False, verbosity=0,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=2.0  # stronger regularization for small n
        )
        model.fit(X_train_s, y_train)

        pred = model.predict(X_test_s)[0]
        prob = model.predict_proba(X_test_s)[0]

        y_true_all.append(y_test[0])
        y_pred_all.append(pred)
        y_prob_all.append(prob)

        event_results.append({
            "event_id": event_ids[test_idx[0]],
            "true": CLASS_NAMES[y_test[0]],
            "predicted": CLASS_NAMES[pred],
            "correct": "Y" if y_test[0] == pred else "N",
            "prob_rain": f"{prob[0]:.3f}",
            "prob_coastal": f"{prob[1]:.3f}"
        })

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    acc = accuracy_score(y_true_all, y_pred_all)
    kap = cohen_kappa_score(y_true_all, y_pred_all)
    cm  = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

    print(f"\n  {title}")
    print(f"  {'-'*60}")

    # Per-event results table
    res_df = pd.DataFrame(event_results)
    print(f"\n  Per-event predictions:")
    print(f"  {res_df.to_string(index=False)}")

    print(f"\n  Confusion Matrix:")
    print(f"    {'':>18s}{'Rain-dom':>14s}{'Coastal-dom':>14s}")
    print(f"    {'Rain-dom':>18s}{cm[0,0]:>14d}{cm[0,1]:>14d}")
    print(f"    {'Coastal-dom':>18s}{cm[1,0]:>14d}{cm[1,1]:>14d}")

    print(f"\n  Accuracy:  {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Kappa:     {kap:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true_all, y_pred_all,
                                target_names=CLASS_NAMES, digits=4, zero_division=0))

    # Feature importance (from last fold's model -- indicative)
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:15]
    print(f"  Top features (from last fold):")
    for i in top_idx:
        if imp[i] > 0.001:
            print(f"    {feature_cols[i]:>35s}: {imp[i]:.4f}")

    return {
        "acc": acc, "kap": kap, "cm": cm,
        "y_true": y_true_all, "y_pred": y_pred_all,
        "results_df": res_df, "feature_cols": feature_cols,
        "last_model": model
    }

# Run with FULL dynamic features
res_full = run_loocv(events_df, feature_cols_full, "label_binary",
                     "FULL DYNAMIC FEATURES (LOOCV)")

# Run with STRICT features (no direct RFI/CII inputs)
res_strict = run_loocv(events_df, feature_cols_strict, "label_binary",
                       "STRICT LEAKAGE CHECK (LOOCV)")

# =====================================================================
# FEATURE IMPORTANCE PLOT
# =====================================================================
def plot_event_importance(model, features, title, filename):
    imp = model.feature_importances_
    idx = np.argsort(imp)
    # Only show features with importance > 0
    mask = imp[idx] > 0.001
    idx = idx[mask]

    fig, ax = plt.subplots(figsize=(9, max(4, len(idx)*0.4)))
    ax.barh(range(len(idx)), imp[idx], color="#4A90D9")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([features[i] for i in idx], fontsize=9)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {path}")

plot_event_importance(res_full["last_model"], feature_cols_full,
                      "Event-Level Full Features Importance",
                      "xgb_event_full_importance.png")

plot_event_importance(res_strict["last_model"], feature_cols_strict,
                      "Event-Level Strict Features Importance",
                      "xgb_event_strict_importance.png")

# =====================================================================
# CONFUSION MATRIX PLOT
# =====================================================================
for label, res, fname in [
    ("Event-Level Full CM", res_full, "xgb_event_full_cm.png"),
    ("Event-Level Strict CM", res_strict, "xgb_event_strict_cm.png"),
]:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(label, fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {path}")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n======================================================================")
print("  SUMMARY: EVENT-LEVEL CLASSIFICATION (n=19 events, LOOCV)")
print("======================================================================")
print(f"  Full Dynamic Features:   Accuracy: {res_full['acc']*100:.1f}%    Kappa: {res_full['kap']:.4f}")
print(f"  Strict Leakage Check:    Accuracy: {res_strict['acc']*100:.1f}%    Kappa: {res_strict['kap']:.4f}")
print("\n  Compare to pixel-level Model B:")
print("    Pixel Standard:  55.5% acc, Kappa 0.15")
print("    Pixel Strict:    52.6% acc, Kappa 0.04")
print(f"\n  All plots saved to: {OUT_DIR}")
