#!/usr/bin/env python3
"""
=====================================================================
KETA COASTAL FLOOD PREDICTION -- XGBoost EVALUATION (v4.0 DATA)
Comprehensive evaluation of Model A and Model B with multiple
leakage check tiers.
=====================================================================
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, cohen_kappa_score,
    classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

# =====================================================================
# PATHS
# =====================================================================
MODEL_A_CSV = r"C:\Users\LEE\Downloads\keta_samples_v4_0_all_splits.csv"
MODEL_B_CSV = r"C:\Users\LEE\Downloads\keta_samples_driver_v4_0_all_splits.csv"
OUT_DIR     = r"C:\Users\LEE\Documents\Keta Intelligence system"

# =====================================================================
# FEATURE DEFINITIONS
# =====================================================================
TERRAIN_FEATURES = ["elevation", "slope", "aspect", "dist_water", "lt3", "lt1", "lt0"]

DYNAMIC_FEATURES = [
    "imerg_24h", "imerg_peak_intensity", "imerg_3d",
    "chirps_7d", "chirps_30d",
    "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
    "soil_moisture",
    "tide_max_fes", "tide_range_fes", "spring_flag_fes",
    "freeboard_fes", "compound_risk", "season_wet"
]

# Model A: ALL features (terrain + dynamic)
FEATURES_MODEL_A = TERRAIN_FEATURES + DYNAMIC_FEATURES

# Model B standard: dynamic-only (architectural change from v4.0)
FEATURES_MODEL_B = DYNAMIC_FEATURES

# Model B no-leak: remove the 2 direct rule variables (imerg_24h, freeboard_fes)
# but keep compound_risk which partially derives from them
FEATURES_NO_LEAK = [f for f in DYNAMIC_FEATURES if f not in ["imerg_24h", "freeboard_fes"]]

# Model B strict: remove ALL variables that directly enter RFI/CII formulas
LEAK_VARS = ["imerg_24h", "freeboard_fes", "chirps_30d", "soil_moisture", "tide_range_fes", "compound_risk"]
FEATURES_STRICT = [f for f in DYNAMIC_FEATURES if f not in LEAK_VARS]

# =====================================================================
# HELPERS
# =====================================================================
def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def compute_sample_weights(y, classes):
    counts = {c: int(np.sum(y == c)) for c in classes}
    mx = max(counts.values())
    w = np.ones(len(y), dtype=float)
    for c in classes:
        if counts[c] > 0:
            w[y == c] = mx / counts[c]
    return w

def eval_model(y_true, y_pred, class_names, label=""):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    acc = accuracy_score(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred)
    print(f"\n  --- {label} ---")
    print(f"  Confusion Matrix (rows=actual, cols=predicted):")
    header = "".join(f"{n:>16s}" for n in class_names)
    print(f"  {'':>16s}{header}")
    for i, row in enumerate(cm):
        vals = "".join(f"{v:>16d}" for v in row)
        print(f"  {class_names[i]:>16s}{vals}")
    print(f"\n  Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Kappa:     {kap:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
    return {"acc": acc, "kap": kap, "cm": cm}

def plot_importance(model, features, title, filename):
    imp = model.feature_importances_
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8, max(4, len(features)*0.35)))
    ax.barh(range(len(features)), imp[idx], color="#4A90D9")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in idx], fontsize=9)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {path}")

def plot_confusion(cm, class_names, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {path}")

# =====================================================================
# LOAD DATA
# =====================================================================
sep("LOADING DATA")
df_a = pd.read_csv(MODEL_A_CSV)
df_b = pd.read_csv(MODEL_B_CSV)
print(f"  Model A CSV: {df_a.shape[0]} samples, {df_a.shape[1]} columns")
print(f"  Model B CSV: {df_b.shape[0]} samples, {df_b.shape[1]} columns")

# Coerce numeric
for col in FEATURES_MODEL_A + ["flood"]:
    if col in df_a.columns:
        df_a[col] = pd.to_numeric(df_a[col], errors="coerce")

for col in FEATURES_MODEL_B + ["flood_driver"]:
    if col in df_b.columns:
        df_b[col] = pd.to_numeric(df_b[col], errors="coerce")

# Split
train_a = df_a[df_a["split"] == "train"].dropna(subset=FEATURES_MODEL_A + ["flood"])
test_a  = df_a[df_a["split"] == "test"].dropna(subset=FEATURES_MODEL_A + ["flood"])
val_a   = df_a[df_a["split"] == "validation"].dropna(subset=FEATURES_MODEL_A + ["flood"])

train_b = df_b[df_b["split"] == "train"].dropna(subset=FEATURES_MODEL_B + ["flood_driver"])
test_b  = df_b[df_b["split"] == "test"].dropna(subset=FEATURES_MODEL_B + ["flood_driver"])
val_b   = df_b[df_b["split"] == "validation"].dropna(subset=FEATURES_MODEL_B + ["flood_driver"])

print(f"\n  Model A splits: train={len(train_a)}, test={len(test_a)}, val={len(val_a)}")
print(f"  Model B splits: train={len(train_b)}, test={len(test_b)}, val={len(val_b)}")

# =====================================================================
# MODEL A: FLOOD SUSCEPTIBILITY (Binary: 0=no-flood, 1=flood)
# =====================================================================
sep("MODEL A: FLOOD SUSCEPTIBILITY (XGBoost)")

X_tr_a = train_a[FEATURES_MODEL_A].values
y_tr_a = train_a["flood"].astype(int).values
X_te_a = test_a[FEATURES_MODEL_A].values
y_te_a = test_a["flood"].astype(int).values
X_va_a = val_a[FEATURES_MODEL_A].values
y_va_a = val_a["flood"].astype(int).values

sw_a = compute_sample_weights(y_tr_a, [0, 1])

xgb_a = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    objective="binary:logistic", random_state=42,
    eval_metric="logloss", use_label_encoder=False, verbosity=0,
    subsample=0.8, colsample_bytree=0.8
)
xgb_a.fit(X_tr_a, y_tr_a, sample_weight=sw_a)

FLOOD_NAMES = ["No-Flood", "Flood"]
res_a_test = eval_model(y_te_a, xgb_a.predict(X_te_a), FLOOD_NAMES, "Model A TEST (2023)")
res_a_val  = eval_model(y_va_a, xgb_a.predict(X_va_a), FLOOD_NAMES, "Model A VALIDATION (2024-2025)")

plot_importance(xgb_a, FEATURES_MODEL_A, "Model A Feature Importance (XGBoost)", "xgb_model_a_importance.png")
plot_confusion(res_a_test["cm"], FLOOD_NAMES, "Model A Test Confusion Matrix", "xgb_model_a_cm_test.png")
plot_confusion(res_a_val["cm"], FLOOD_NAMES, "Model A Validation Confusion Matrix", "xgb_model_a_cm_val.png")

# =====================================================================
# MODEL B: DRIVER ATTRIBUTION -- STANDARD (dynamic-only features)
# =====================================================================
sep("MODEL B: DRIVER ATTRIBUTION -- STANDARD (Dynamic-Only Features)")

DRIVER_NAMES = ["No-flood", "Rain-dom", "Coastal-dom"]

X_tr_b = train_b[FEATURES_MODEL_B].values
y_tr_b = train_b["flood_driver"].astype(int).values
X_te_b = test_b[FEATURES_MODEL_B].values
y_te_b = test_b["flood_driver"].astype(int).values
X_va_b = val_b[FEATURES_MODEL_B].values
y_va_b = val_b["flood_driver"].astype(int).values

sw_b = compute_sample_weights(y_tr_b, [0, 1, 2])

xgb_b = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    objective="multi:softmax", num_class=3, random_state=42,
    eval_metric="mlogloss", use_label_encoder=False, verbosity=0,
    subsample=0.8, colsample_bytree=0.8
)
xgb_b.fit(X_tr_b, y_tr_b, sample_weight=sw_b)

res_b_test = eval_model(y_te_b, xgb_b.predict(X_te_b), DRIVER_NAMES, "Model B STANDARD TEST")
res_b_val  = eval_model(y_va_b, xgb_b.predict(X_va_b), DRIVER_NAMES, "Model B STANDARD VALIDATION")

plot_importance(xgb_b, FEATURES_MODEL_B, "Model B Standard Feature Importance", "xgb_model_b_std_importance.png")
plot_confusion(res_b_test["cm"], DRIVER_NAMES, "Model B Standard Test CM", "xgb_model_b_std_cm_test.png")

# =====================================================================
# MODEL B: NO-LEAK (remove imerg_24h + freeboard_fes)
# =====================================================================
sep("MODEL B: NO-LEAK CHECK (removed imerg_24h, freeboard_fes)")

avail_nl = [f for f in FEATURES_NO_LEAK if f in train_b.columns]
X_tr_nl = train_b[avail_nl].values
X_te_nl = test_b[avail_nl].values
X_va_nl = val_b[avail_nl].values

xgb_nl = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    objective="multi:softmax", num_class=3, random_state=42,
    eval_metric="mlogloss", use_label_encoder=False, verbosity=0,
    subsample=0.8, colsample_bytree=0.8
)
xgb_nl.fit(X_tr_nl, y_tr_b, sample_weight=sw_b)

res_nl_test = eval_model(y_te_b, xgb_nl.predict(X_te_nl), DRIVER_NAMES, "Model B NO-LEAK TEST")
res_nl_val  = eval_model(y_va_b, xgb_nl.predict(X_va_nl), DRIVER_NAMES, "Model B NO-LEAK VALIDATION")

plot_importance(xgb_nl, avail_nl, "Model B No-Leak Feature Importance", "xgb_model_b_noleak_importance.png")

# =====================================================================
# MODEL B: STRICT LEAKAGE CHECK (remove ALL RFI/CII source vars)
# =====================================================================
sep("MODEL B: STRICT LEAKAGE CHECK")
print(f"  Removed features: {LEAK_VARS}")

avail_st = [f for f in FEATURES_STRICT if f in train_b.columns]
print(f"  Remaining features ({len(avail_st)}): {avail_st}")

X_tr_st = train_b[avail_st].values
X_te_st = test_b[avail_st].values
X_va_st = val_b[avail_st].values

xgb_st = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    objective="multi:softmax", num_class=3, random_state=42,
    eval_metric="mlogloss", use_label_encoder=False, verbosity=0,
    subsample=0.8, colsample_bytree=0.8
)
xgb_st.fit(X_tr_st, y_tr_b, sample_weight=sw_b)

res_st_test = eval_model(y_te_b, xgb_st.predict(X_te_st), DRIVER_NAMES, "Model B STRICT TEST")
res_st_val  = eval_model(y_va_b, xgb_st.predict(X_va_st), DRIVER_NAMES, "Model B STRICT VALIDATION")

plot_importance(xgb_st, avail_st, "Model B Strict Leakage Feature Importance", "xgb_model_b_strict_importance.png")
plot_confusion(res_st_test["cm"], DRIVER_NAMES, "Model B Strict Test CM", "xgb_model_b_strict_cm_test.png")

# =====================================================================
# SUMMARY TABLE
# =====================================================================
sep("SUMMARY COMPARISON TABLE")

rows = [
    ("Model A (Susceptibility)", "XGBoost", "All 24", f"{res_a_test['acc']*100:.2f}%", f"{res_a_test['kap']:.4f}", f"{res_a_val['acc']*100:.2f}%", f"{res_a_val['kap']:.4f}"),
    ("Model B Standard", "XGBoost", f"Dynamic {len(FEATURES_MODEL_B)}", f"{res_b_test['acc']*100:.2f}%", f"{res_b_test['kap']:.4f}", f"{res_b_val['acc']*100:.2f}%", f"{res_b_val['kap']:.4f}"),
    ("Model B No-Leak", "XGBoost", f"No-Leak {len(avail_nl)}", f"{res_nl_test['acc']*100:.2f}%", f"{res_nl_test['kap']:.4f}", f"{res_nl_val['acc']*100:.2f}%", f"{res_nl_val['kap']:.4f}"),
    ("Model B Strict", "XGBoost", f"Strict {len(avail_st)}", f"{res_st_test['acc']*100:.2f}%", f"{res_st_test['kap']:.4f}", f"{res_st_val['acc']*100:.2f}%", f"{res_st_val['kap']:.4f}"),
]

print(f"\n  {'Model':<28s} {'Algo':<9s} {'Features':<14s} {'Test Acc':>10s} {'Test K':>10s} {'Val Acc':>10s} {'Val K':>10s}")
print(f"  {'-'*28} {'-'*9} {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for r in rows:
    print(f"  {r[0]:<28s} {r[1]:<9s} {r[2]:<14s} {r[3]:>10s} {r[4]:>10s} {r[5]:>10s} {r[6]:>10s}")

print("\n  DONE. All plots saved to:", OUT_DIR)
