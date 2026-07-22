#!/usr/bin/env python3
"""
=====================================================================
KETA v4.3 -- TWL-AUGMENTED ATTRIBUTION + COMPOUND DETECTION
Adds the hourly total-water-level features (twl_features.csv) to the
event-level LOOCV ablation and the two-detector compound framework.
Focus questions:
  1. Does TWL improve the coastal-forcing detector?
  2. Is may2025 now diagnosed as compound/coastal?
  3. How do the mixed events reorganize with phase information?
=====================================================================
"""
import os, warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))

RAIN_ONLY = ["imerg_peak_intensity", "imerg_3d", "chirps_7d",
             "solar_rad", "season_wet"]
WAVE_VARS = ["swh_max", "swh_mean", "swell_hs_max", "swell_per_max",
             "wave_power_max"]
TWL_VARS = ["twl_max", "twl_p95", "hours_ge_190", "phase_align"]
TIDE_VARS = ["tide_max_fes", "tide_range_fes", "spring_flag_fes", "tide_anomaly"]
FLUVIAL = ["oct2023"]

def sep(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")

# ---------------------------------------------------------------------
sep("BUILDING EVENT TABLE (pixels + FES + WW3 + TWL)")
df = pd.read_csv(os.path.join(BASE, "keta_samples_driver_v4_2_all_splits.csv"))
for c in RAIN_ONLY + TIDE_VARS + ["imerg_24h", "flood_driver"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df.flood_driver.isin([1, 2])]

rows = []
for eid, g in df.groupby("event_id"):
    row = {"event_id": eid,
           "coastal_fraction": float((g.flood_driver == 2).mean())}
    for f in RAIN_ONLY:
        v = g[f].dropna()
        row[f"{f}_mean"] = v.mean() if len(v) else 0.0
        row[f"{f}_max"] = v.max() if len(v) else 0.0
        row[f"{f}_std"] = v.std() if len(v) > 1 else 0.0
    for f in TIDE_VARS:
        row[f] = g[f].iloc[0]
    rows.append(row)
ev = pd.DataFrame(rows)

waves = pd.read_csv(os.path.join(BASE, "ww3_extracted_waves.csv")).rename(columns={"id": "event_id"})
twl = pd.read_csv(os.path.join(BASE, "twl_features.csv")).rename(columns={"id": "event_id"})
ev = ev.merge(waves[["event_id"] + WAVE_VARS], on="event_id", how="left")
ev = ev.merge(twl[["event_id"] + TWL_VARS], on="event_id", how="left")

ev["majority"] = ev.coastal_fraction.where(ev.coastal_fraction >= 0.5,
                                          1 - ev.coastal_fraction)
ev["hard_label"] = (ev.coastal_fraction >= 0.5).astype(int)
conf = ev[(ev.majority >= 0.60) & (~ev.event_id.isin(FLUVIAL))].reset_index(drop=True)
print(f"  events={len(ev)}, confident={len(conf)}")

def cols_for(base):
    out = []
    for f in base:
        if f in ev.columns:
            out.append(f)
        else:
            out += [f"{f}_{s}" for s in ("mean", "max", "std") if f"{f}_{s}" in ev.columns]
    return out

def new_model():
    return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                         objective="binary:logistic", random_state=42,
                         eval_metric="logloss", verbosity=0,
                         subsample=0.8, colsample_bytree=0.8,
                         reg_alpha=1.0, reg_lambda=2.0)

def loocv(base, title):
    cols = cols_for(base)
    X, y = conf[cols].values, conf.hard_label.values
    preds = []
    sc = StandardScaler()
    for tr, te in LeaveOneOut().split(X):
        m = new_model()
        m.fit(sc.fit_transform(X[tr]), y[tr])
        preds.append(m.predict(sc.transform(X[te]))[0])
    preds = np.array(preds)
    acc = accuracy_score(y, preds)
    kap = cohen_kappa_score(y, preds)
    miss = [conf.event_id.iloc[i] for i in range(len(y)) if preds[i] != y[i]]
    print(f"  {title:<28s} Acc {acc*100:5.1f}%  Kappa {kap:6.3f}  missed: {miss if miss else 'none'}")

sep("EVENT-LEVEL LOOCV (n=14) WITH TWL")
loocv(RAIN_ONLY, "RAIN-ONLY (reference)")
loocv(TWL_VARS, "TWL-ONLY")
loocv(TWL_VARS + TIDE_VARS, "TWL+TIDE detector")
loocv(WAVE_VARS + TIDE_VARS + TWL_VARS, "WAVE+TIDE+TWL detector")
loocv(RAIN_ONLY + TWL_VARS, "RAIN+TWL")
loocv(RAIN_ONLY + WAVE_VARS + TIDE_VARS + TWL_VARS, "ALL FORCING")

# ---------------------------------------------------------------------
sep("TWO-DETECTOR COMPOUND ANALYSIS WITH TWL (all 23 events)")
COASTAL_DET = WAVE_VARS + TIDE_VARS + TWL_VARS

def evidence(base, invert):
    cols = cols_for(base)
    sc = StandardScaler()
    Xc, yc = conf[cols].values, conf.hard_label.values
    p = pd.Series(index=ev.index, dtype=float)
    for tr, te in LeaveOneOut().split(Xc):
        m = new_model()
        m.fit(sc.fit_transform(Xc[tr]), yc[tr])
        pr = m.predict_proba(sc.transform(Xc[te]))[0, 1]
        eid = conf.event_id.iloc[te[0]]
        p[ev.index[ev.event_id == eid][0]] = pr
    m = new_model()
    m.fit(sc.fit_transform(Xc), yc)
    for i in ev.index:
        if pd.isna(p[i]):
            p[i] = m.predict_proba(sc.transform(ev.loc[[i], cols].values))[0, 1]
    return (1 - p) if invert else p

ev["rain_evidence"] = evidence(RAIN_ONLY, invert=True)
ev["coastal_evidence"] = evidence(COASTAL_DET, invert=False)
ev["compound_index"] = np.minimum(ev.rain_evidence, ev.coastal_evidence)

def regime(r):
    if r.rain_evidence >= 0.5 and r.coastal_evidence >= 0.5:
        return "COMPOUND"
    if r.coastal_evidence >= 0.5:
        return "coastal"
    if r.rain_evidence >= 0.5:
        return "rain"
    return "weak/uncertain"
ev["regime"] = ev.apply(regime, axis=1)

out = ev[["event_id", "coastal_fraction", "twl_max", "phase_align",
          "rain_evidence", "coastal_evidence", "compound_index", "regime"]]
out = out.sort_values("coastal_evidence", ascending=False)
print(out.to_string(index=False, formatters={c: "{:.3f}".format for c in
      ["coastal_fraction", "twl_max", "phase_align", "rain_evidence",
       "coastal_evidence", "compound_index"]}))
out.to_csv(os.path.join(BASE, "keta_compound_regimes_twl.csv"), index=False)
print("\n  [SAVED] keta_compound_regimes_twl.csv")

fig, ax = plt.subplots(figsize=(8, 7))
colors = {"rain": "#2ca02c", "coastal": "#1f77b4", "COMPOUND": "#d62728",
          "weak/uncertain": "#7f7f7f"}
for reg, g in ev.groupby("regime"):
    ax.scatter(g.rain_evidence, g.coastal_evidence, c=colors[reg], label=reg, s=60)
for _, r in ev.iterrows():
    ax.annotate(r.event_id, (r.rain_evidence, r.coastal_evidence),
                fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.axhline(0.5, color="k", lw=0.7, ls="--")
ax.axvline(0.5, color="k", lw=0.7, ls="--")
ax.set_xlabel("Rain-forcing evidence")
ax.set_ylabel("Coastal-forcing evidence (waves + tide + hourly TWL)")
ax.set_title("Keta flood events: forcing diagnosis with TWL", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(BASE, "keta_compound_quadrant_twl.png"), dpi=200)
plt.close(fig)
print("  [SAVED] keta_compound_quadrant_twl.png")
