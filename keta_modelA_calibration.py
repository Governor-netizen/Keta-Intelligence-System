#!/usr/bin/env python3
"""
=====================================================================
KETA v4.2 -- MODEL A PROBABILITY CALIBRATION + ALERT THRESHOLDS
Isotonic calibration fitted on out-of-fold training predictions
(5-fold CV), applied to the untouched test and validation splits.
Outputs Brier scores (raw vs calibrated), a reliability diagram, and
an operating-point table from which advisory/watch/warning alert
thresholds can be chosen.
=====================================================================
"""
import os, warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
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
FEATS = TERRAIN + DYNAMIC

def sep(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")

def weights(y):
    c0, c1 = (y == 0).sum(), (y == 1).sum()
    m = max(c0, c1)
    return np.where(y == 0, m / c0, m / c1)

def make_xgb():
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                         objective="binary:logistic", random_state=42,
                         eval_metric="logloss", verbosity=0,
                         subsample=0.8, colsample_bytree=0.8)

sep("LOADING")
df = pd.read_csv(os.path.join(BASE, "keta_samples_v4_2_all_splits.csv"))
for c in FEATS + ["flood"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
tr = df[df.split == "train"].dropna(subset=FEATS + ["flood"])
te = df[df.split == "test"].dropna(subset=FEATS + ["flood"])
va = df[df.split == "validation"].dropna(subset=FEATS + ["flood"])
Xtr, ytr = tr[FEATS].values, tr.flood.astype(int).values
print(f"  train={len(tr)}  test={len(te)}  val={len(va)}")

# ---------------------------------------------------------------------
# Out-of-fold training probabilities -> isotonic calibrator
# ---------------------------------------------------------------------
sep("FITTING ISOTONIC CALIBRATOR ON OUT-OF-FOLD TRAIN PREDICTIONS")
oof = np.zeros(len(ytr))
for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, ytr):
    m = make_xgb()
    m.fit(Xtr[tr_i], ytr[tr_i], sample_weight=weights(ytr[tr_i]))
    oof[te_i] = m.predict_proba(Xtr[te_i])[:, 1]
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof, ytr)

# Platt scaling: logistic fit on the OOF log-odds. Smooth and monotone,
# so it supports a graded advisory/watch/warning threshold scheme where
# the isotonic step function cannot.
from sklearn.linear_model import LogisticRegression
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
platt = LogisticRegression(C=1e6)
platt.fit(logit(oof).reshape(-1, 1), ytr)

final = make_xgb()
final.fit(Xtr, ytr, sample_weight=weights(ytr))

results = {}
for name, d in [("TEST (2023)", te), ("VALIDATION (2024-26)", va)]:
    y = d.flood.astype(int).values
    p_raw = final.predict_proba(d[FEATS].values)[:, 1]
    p_cal = iso.predict(p_raw)
    p_pl = platt.predict_proba(logit(p_raw).reshape(-1, 1))[:, 1]
    results[name] = (y, p_raw, p_cal, p_pl)
    print(f"\n  {name}: base rate {y.mean():.3f}")
    print(f"    Brier raw:        {brier_score_loss(y, p_raw):.4f}")
    print(f"    Brier isotonic:   {brier_score_loss(y, p_cal):.4f}")
    print(f"    Brier Platt:      {brier_score_loss(y, p_pl):.4f}")
    print(f"    Brier climatology: {brier_score_loss(y, np.full_like(p_raw, y.mean())):.4f}")

# ---------------------------------------------------------------------
# Reliability diagram (validation)
# ---------------------------------------------------------------------
y, p_raw, p_cal, p_pl = results["VALIDATION (2024-26)"]
fig, ax = plt.subplots(figsize=(6.5, 6))
for p, lbl, c in [(p_raw, "raw", "#d62728"), (p_cal, "isotonic", "#1f77b4"),
                  (p_pl, "Platt", "#2ca02c")]:
    bins = np.linspace(0, 1, 11)
    mids, obs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m_ = (p >= lo) & (p < hi)
        if m_.sum() >= 20:
            mids.append(p[m_].mean())
            obs.append(y[m_].mean())
    ax.plot(mids, obs, "o-", color=c, label=lbl)
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
ax.set_xlabel("Predicted flood probability")
ax.set_ylabel("Observed flood frequency")
ax.set_title("Model A reliability (validation 2024-2026)", fontsize=11, fontweight="bold")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(BASE, "keta_modelA_reliability.png"), dpi=200)
plt.close(fig)
print("\n  [SAVED] keta_modelA_reliability.png")

# ---------------------------------------------------------------------
# Operating points on calibrated validation probabilities
# ---------------------------------------------------------------------
sep("ALERT OPERATING POINTS (PLATT-calibrated probabilities, validation)")
print("  Platt gives a smooth monotone mapping, so thresholds grade")
print("  continuously (the isotonic map is stepped and plateaus).")
print(f"  {'threshold':>10s} {'flagged%':>9s} {'recall':>8s} {'precision':>10s}")
rows = []
for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
          0.55, 0.60, 0.65, 0.70, 0.80]:
    pred = (p_pl >= t).astype(int)
    rec = recall_score(y, pred, zero_division=0)
    prec = precision_score(y, pred, zero_division=0)
    rows.append({"threshold": t, "flagged_frac": pred.mean(),
                 "recall": rec, "precision": prec})
    print(f"  {t:>10.2f} {pred.mean()*100:>8.1f}% {rec:>8.3f} {prec:>10.3f}")
pd.DataFrame(rows).to_csv(os.path.join(BASE, "keta_modelA_operating_points.csv"), index=False)
print("\n  [SAVED] keta_modelA_operating_points.csv")
print("  Choose advisory/watch/warning from this table (e.g., recall-first")
print("  advisory, balanced watch, precision-first warning).")
