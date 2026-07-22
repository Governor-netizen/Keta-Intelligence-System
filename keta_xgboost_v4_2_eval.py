#!/usr/bin/env python3
"""
=====================================================================
KETA COASTAL FLOOD PREDICTION -- XGBoost EVALUATION (v4.2 DATA)
First evaluation on fully corrected data:
  - real SMAP soil moisture for all 23 events (SPL4SMGP/008)
  - surface_pressure + onshore_wind + tide_anomaly + rain_intensity_ratio
  - news-verified event windows and regenerated FES2022 tide table
Model A (susceptibility), Model B (driver attribution) at four leakage
tiers, and event-level LOOCV attribution.
=====================================================================
"""
import os, warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
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

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_A_CSV = os.path.join(BASE, "keta_samples_v4_2_all_splits.csv")
MODEL_B_CSV = os.path.join(BASE, "keta_samples_driver_v4_2_all_splits.csv")
OUT_DIR = BASE

# =====================================================================
# FEATURE SETS (v4.2: 28 input features)
# =====================================================================
TERRAIN = ["elevation", "slope", "aspect", "dist_water", "lt3", "lt1", "lt0"]

DYNAMIC = [
    "imerg_24h", "imerg_peak_intensity", "imerg_3d",
    "chirps_7d", "chirps_30d",
    "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
    "surface_pressure", "onshore_wind",
    "soil_moisture",
    "tide_max_fes", "tide_range_fes", "spring_flag_fes", "tide_anomaly",
    "freeboard_fes", "compound_risk", "season_wet", "rain_intensity_ratio"
]

FEATURES_A = TERRAIN + DYNAMIC                       # Model A: 28
FEATURES_B_STD = DYNAMIC                             # Model B standard: 21

# No-leak: drop the two direct rule variables
NOLEAK_DROP = ["imerg_24h", "freeboard_fes"]
FEATURES_B_NOLEAK = [f for f in DYNAMIC if f not in NOLEAK_DROP]

# Strict (v4.0-comparable): drop ALL direct RFI/CII formula inputs
STRICT_DROP = ["imerg_24h", "freeboard_fes", "chirps_30d",
               "soil_moisture", "tide_range_fes", "compound_risk"]
FEATURES_B_STRICT = [f for f in DYNAMIC if f not in STRICT_DROP]

# Physical-only (harshest, new in v4.2): additionally drop every
# tide-table quantity (tide_max/anomaly/spring all derive from the same
# FES numbers the CII rule uses) and rain_intensity_ratio (its
# denominator is chirps_30d, an RFI input). What remains is pure
# independently-measured weather: satellite rain rates, wind, pressure,
# solar, season. If attribution works HERE, it is beyond dispute.
PHYSICAL_DROP = STRICT_DROP + ["tide_max_fes", "tide_anomaly",
                               "spring_flag_fes", "rain_intensity_ratio"]
FEATURES_B_PHYSICAL = [f for f in DYNAMIC if f not in PHYSICAL_DROP]

FLOOD_NAMES = ["No-Flood", "Flood"]
# The driver CSV holds flooded pixels only (classes 1/2) -> binary task,
# remapped to 0=rain-dominant, 1=coastal-dominant.
DRIVER_NAMES = ["Rain-dom", "Coastal-dom"]

def sep(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")

def weights(y, classes):
    counts = {c: int(np.sum(y == c)) for c in classes}
    mx = max(counts.values())
    w = np.ones(len(y), dtype=float)
    for c in classes:
        if counts[c] > 0:
            w[y == c] = mx / counts[c]
    return w

def evaluate(y_true, y_pred, names, label):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    acc = accuracy_score(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred)
    print(f"\n  --- {label} ---")
    hdr = "".join(f"{n:>14s}" for n in names)
    print(f"  {'':>14s}{hdr}")
    for i, row in enumerate(cm):
        print(f"  {names[i]:>14s}" + "".join(f"{v:>14d}" for v in row))
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)   Kappa: {kap:.4f}")
    print(classification_report(y_true, y_pred, target_names=names,
                                digits=4, zero_division=0))
    return {"acc": acc, "kap": kap, "cm": cm}

def plot_imp(model, feats, title, fname):
    imp = model.feature_importances_
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feats)*0.32)))
    ax.barh(range(len(feats)), imp[idx], color="#4A90D9")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([feats[i] for i in idx], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {fname}")

def plot_cm(cm, names, title, fname):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {fname}")

def make_xgb(objective, nclass=None):
    kw = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
              random_state=42, verbosity=0, subsample=0.8,
              colsample_bytree=0.8)
    if nclass:
        return XGBClassifier(objective=objective, num_class=nclass,
                             eval_metric="mlogloss", **kw)
    return XGBClassifier(objective=objective, eval_metric="logloss", **kw)

# =====================================================================
# LOAD
# =====================================================================
sep("LOADING v4.2 DATA")
df_a = pd.read_csv(MODEL_A_CSV)
df_b = pd.read_csv(MODEL_B_CSV)
for col in FEATURES_A + ["flood"]:
    df_a[col] = pd.to_numeric(df_a[col], errors="coerce")
for col in DYNAMIC + ["flood_driver", "RFI", "CII"]:
    if col in df_b.columns:
        df_b[col] = pd.to_numeric(df_b[col], errors="coerce")

tr_a = df_a[df_a.split == "train"].dropna(subset=FEATURES_A + ["flood"])
te_a = df_a[df_a.split == "test"].dropna(subset=FEATURES_A + ["flood"])
va_a = df_a[df_a.split == "validation"].dropna(subset=FEATURES_A + ["flood"])
tr_b = df_b[df_b.split == "train"].dropna(subset=DYNAMIC + ["flood_driver"])
te_b = df_b[df_b.split == "test"].dropna(subset=DYNAMIC + ["flood_driver"])
va_b = df_b[df_b.split == "validation"].dropna(subset=DYNAMIC + ["flood_driver"])
print(f"  Model A: train={len(tr_a)}  test={len(te_a)}  val={len(va_a)}")
print(f"  Model B: train={len(tr_b)}  test={len(te_b)}  val={len(va_b)}")

# =====================================================================
# MODEL A
# =====================================================================
sep("MODEL A: FLOOD SUSCEPTIBILITY (28 features)")
Xtr, ytr = tr_a[FEATURES_A].values, tr_a.flood.astype(int).values
xgb_a = make_xgb("binary:logistic")
xgb_a.fit(Xtr, ytr, sample_weight=weights(ytr, [0, 1]))
res_a_te = evaluate(te_a.flood.astype(int).values,
                    xgb_a.predict(te_a[FEATURES_A].values),
                    FLOOD_NAMES, "Model A TEST (2023)")
res_a_va = evaluate(va_a.flood.astype(int).values,
                    xgb_a.predict(va_a[FEATURES_A].values),
                    FLOOD_NAMES, "Model A VALIDATION (2024-2026)")
plot_imp(xgb_a, FEATURES_A, "Model A Importance (v4.2)", "xgb42_model_a_importance.png")
plot_cm(res_a_te["cm"], FLOOD_NAMES, "Model A Test CM (v4.2)", "xgb42_model_a_cm_test.png")
plot_cm(res_a_va["cm"], FLOOD_NAMES, "Model A Val CM (v4.2)", "xgb42_model_a_cm_val.png")

# =====================================================================
# MODEL B TIERS
# =====================================================================
ytr_b = tr_b.flood_driver.astype(int).values - 1
yte_b = te_b.flood_driver.astype(int).values - 1
yva_b = va_b.flood_driver.astype(int).values - 1
sw_b = weights(ytr_b, [0, 1])

tier_results = {}
for tier, feats in [("STANDARD", FEATURES_B_STD),
                    ("NO-LEAK", FEATURES_B_NOLEAK),
                    ("STRICT", FEATURES_B_STRICT),
                    ("PHYSICAL", FEATURES_B_PHYSICAL)]:
    sep(f"MODEL B: {tier} ({len(feats)} features)")
    print(f"  Features: {feats}")
    m = make_xgb("binary:logistic")
    m.fit(tr_b[feats].values, ytr_b, sample_weight=sw_b)
    r_te = evaluate(yte_b, m.predict(te_b[feats].values), DRIVER_NAMES, f"B {tier} TEST")
    r_va = evaluate(yva_b, m.predict(va_b[feats].values), DRIVER_NAMES, f"B {tier} VALIDATION")
    tier_results[tier] = (r_te, r_va)
    tag = tier.lower().replace("-", "")
    plot_imp(m, feats, f"Model B {tier} Importance (v4.2)", f"xgb42_model_b_{tag}_importance.png")
    plot_cm(r_te["cm"], DRIVER_NAMES, f"Model B {tier} Test CM (v4.2)", f"xgb42_model_b_{tag}_cm_test.png")

# =====================================================================
# EVENT-LEVEL LOOCV
# =====================================================================
sep("EVENT-LEVEL DRIVER ATTRIBUTION -- LOOCV (23 events)")
dfl = df_b[df_b.flood_driver.isin([1, 2])].copy()
rows = []
for eid, g in dfl.groupby("event_id"):
    row = {"event_id": eid,
           "label": int((g.flood_driver == 2).mean() > 0.5),
           "n_pixels": len(g)}
    for f in DYNAMIC:
        v = g[f].dropna()
        row[f"{f}_mean"] = v.mean() if len(v) else 0.0
        row[f"{f}_max"] = v.max() if len(v) else 0.0
        row[f"{f}_std"] = v.std() if len(v) > 1 else 0.0
    rows.append(row)
ev = pd.DataFrame(rows)
CLASSES = ["Rain-dominant", "Coastal-dominant"]
print(f"  Events: {len(ev)}  |  Coastal-dominant: {ev.label.sum()}, Rain-dominant: {(1-ev.label).sum()}")

def loocv(feat_base, title):
    cols = [f"{f}_{s}" for f in feat_base for s in ("mean", "max", "std")
            if f"{f}_{s}" in ev.columns]
    X, y = ev[cols].values, ev.label.values
    preds, details = [], []
    scaler = StandardScaler()
    last = None
    for tr_i, te_i in LeaveOneOut().split(X):
        Xtr_s = scaler.fit_transform(X[tr_i])
        Xte_s = scaler.transform(X[te_i])
        m = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                          objective="binary:logistic", random_state=42,
                          eval_metric="logloss", verbosity=0,
                          subsample=0.8, colsample_bytree=0.8,
                          reg_alpha=1.0, reg_lambda=2.0)
        m.fit(Xtr_s, y[tr_i])
        p = m.predict(Xte_s)[0]
        preds.append(p)
        details.append((ev.event_id.iloc[te_i[0]], CLASSES[y[te_i[0]]],
                        CLASSES[p], "Y" if p == y[te_i[0]] else "N"))
        last = m
    preds = np.array(preds)
    acc = accuracy_score(y, preds)
    kap = cohen_kappa_score(y, preds)
    print(f"\n  {title}  ({len(cols)} aggregated features)")
    for d in details:
        print(f"    {d[0]:>20s}  true={d[1]:<17s} pred={d[2]:<17s} {d[3]}")
    cm = confusion_matrix(y, preds, labels=[0, 1])
    print(f"    CM [[{cm[0,0]},{cm[0,1]}],[{cm[1,0]},{cm[1,1]}]]  "
          f"Accuracy: {acc*100:.1f}%   Kappa: {kap:.4f}")
    return acc, kap

ev_full = loocv(DYNAMIC, "EVENT-LEVEL: FULL DYNAMIC")
ev_strict = loocv(FEATURES_B_STRICT, "EVENT-LEVEL: STRICT")
ev_phys = loocv(FEATURES_B_PHYSICAL, "EVENT-LEVEL: PHYSICAL-ONLY")

# =====================================================================
# SUMMARY
# =====================================================================
sep("SUMMARY -- v4.2 (corrected data)")
print(f"\n  {'Model':<26s}{'Test Acc':>10s}{'Test K':>9s}{'Val Acc':>10s}{'Val K':>9s}")
print(f"  {'-'*26}{'-'*10}{'-'*9}{'-'*10}{'-'*9}")
print(f"  {'A Susceptibility':<26s}{res_a_te['acc']*100:>9.2f}%{res_a_te['kap']:>9.3f}{res_a_va['acc']*100:>9.2f}%{res_a_va['kap']:>9.3f}")
for tier in ["STANDARD", "NO-LEAK", "STRICT", "PHYSICAL"]:
    r_te, r_va = tier_results[tier]
    print(f"  {'B ' + tier:<26s}{r_te['acc']*100:>9.2f}%{r_te['kap']:>9.3f}{r_va['acc']*100:>9.2f}%{r_va['kap']:>9.3f}")
print(f"\n  {'Event-level (LOOCV)':<26s}{'Acc':>10s}{'Kappa':>9s}")
print(f"  {'Full dynamic':<26s}{ev_full[0]*100:>9.1f}%{ev_full[1]:>9.3f}")
print(f"  {'Strict':<26s}{ev_strict[0]*100:>9.1f}%{ev_strict[1]:>9.3f}")
print(f"  {'Physical-only':<26s}{ev_phys[0]*100:>9.1f}%{ev_phys[1]:>9.3f}")
print(f"\n  v4.0 reference: B Standard 55.5%/K0.15 | B Strict 52.6%/K0.04")
print("  DONE.")
