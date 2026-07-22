#!/usr/bin/env python3
"""
=====================================================================
KETA v4.2 -- CONFIDENCE-AWARE EVENT-LEVEL DRIVER ATTRIBUTION
1. Events get a hard rain/coastal label only when the pixel-majority
   is decisive (>= threshold); the rest are 'mixed/compound'.
2. oct2023 (Akosombo dam spillage) is relabelled 'fluvial' and held
   out of rain-vs-coastal training entirely.
3. LOOCV on the confident events at two thresholds (0.60, 0.65),
   at three feature tiers (full / strict / physical-only).
4. A final physical-only model produces P(coastal) for ALL events ->
   keta_event_attribution_table.csv + paper figure.
=====================================================================
"""
import os, warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_B_CSV = os.path.join(BASE, "keta_samples_driver_v4_2_all_splits.csv")

DYNAMIC = [
    "imerg_24h", "imerg_peak_intensity", "imerg_3d",
    "chirps_7d", "chirps_30d",
    "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
    "surface_pressure", "onshore_wind",
    "soil_moisture",
    "tide_max_fes", "tide_range_fes", "spring_flag_fes", "tide_anomaly",
    "freeboard_fes", "compound_risk", "season_wet", "rain_intensity_ratio"
]
STRICT_DROP = ["imerg_24h", "freeboard_fes", "chirps_30d",
               "soil_moisture", "tide_range_fes", "compound_risk"]
STRICT = [f for f in DYNAMIC if f not in STRICT_DROP]
PHYSICAL = [f for f in STRICT if f not in
            ["tide_max_fes", "tide_anomaly", "spring_flag_fes",
             "rain_intensity_ratio"]]

# Mechanism ablation (the 'are you sure' check):
# SURGE_ONLY tests whether local pressure/wind carry surge signal (they
# don't -- kappa ~0: Keta's surges are remotely generated swell, invisible
# to local ERA5 meteorology). RAIN_ONLY is where all leak-free attribution
# skill lives: coastal floods are identified by rainfall ABSENCE
# ("hydrometeorological elimination"), not by detecting the surge itself.
SURGE_ONLY = ["surface_pressure", "onshore_wind", "u_wind_10m",
              "v_wind_10m", "wind_speed", "wind_dir"]
RAIN_ONLY = ["imerg_peak_intensity", "imerg_3d", "chirps_7d",
             "solar_rad", "season_wet"]

def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

FLUVIAL_EVENTS = ["oct2023"]  # Akosombo/Kpong dam spillage reaching the lagoon

NEWS_VERIFIED = {
    "nov2021": "tidal surge 2021-11-07", "apr2022": "tidal waves 2022-04-03/04",
    "jun2023": "rain floods, 5000+ displaced by Jun 11", "jul2023": "rains continued into July",
    "sep2023": "surge Sep 20 + spill onset (rain-dom label)", "oct2023": "Akosombo spillage (FLUVIAL)",
    "feb2024": "tidal surge 2024-02-15", "jan2025": "tidal waves 2025-01-16",
    "feb2025": "coastal flood 2025-02-01", "mar2025": "tidal surge 2025-03-01",
    "may2025": "tidal waves 2025-05-26", "sep2025_market": "market flood 2025-09-12 (rain)",
    "jun2025_lawoshime": "island communities flooded Jun 2025 (rain)",
    "jun2026_floodgates": "lagoon flooding + Jun 29 2026 storm",
}

def sep(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")

# =====================================================================
# AGGREGATE PIXELS -> EVENTS
# =====================================================================
sep("AGGREGATING v4.2 DRIVER SAMPLES TO EVENT LEVEL")
df = pd.read_csv(MODEL_B_CSV)
for c in DYNAMIC + ["flood_driver"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df.flood_driver.isin([1, 2])]

rows = []
for eid, g in df.groupby("event_id"):
    row = {"event_id": eid, "n_pixels": len(g),
           "coastal_fraction": float((g.flood_driver == 2).mean())}
    for f in DYNAMIC:
        v = g[f].dropna()
        row[f"{f}_mean"] = v.mean() if len(v) else 0.0
        row[f"{f}_max"] = v.max() if len(v) else 0.0
        row[f"{f}_std"] = v.std() if len(v) > 1 else 0.0
    rows.append(row)
ev = pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True)
ev["majority"] = ev.coastal_fraction.where(ev.coastal_fraction >= 0.5,
                                          1 - ev.coastal_fraction)
ev["hard_label"] = (ev.coastal_fraction >= 0.5).astype(int)  # 1=coastal

print(f"  {'event':>20s} {'pixels':>7s} {'coastal_frac':>13s} {'majority':>9s}")
for _, r in ev.iterrows():
    tag = " FLUVIAL" if r.event_id in FLUVIAL_EVENTS else ""
    print(f"  {r.event_id:>20s} {r.n_pixels:>7d} {r.coastal_fraction:>13.3f} {r.majority:>9.3f}{tag}")

def agg_cols(feat_base):
    return [f"{f}_{s}" for f in feat_base for s in ("mean", "max", "std")
            if f"{f}_{s}" in ev.columns]

def new_model():
    return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                         objective="binary:logistic", random_state=42,
                         eval_metric="logloss", verbosity=0,
                         subsample=0.8, colsample_bytree=0.8,
                         reg_alpha=1.0, reg_lambda=2.0)

# =====================================================================
# LOOCV ON CONFIDENT EVENTS
# =====================================================================
CLASSES = ["Rain-dominant", "Coastal-dominant"]
for thr in (0.60, 0.65):
    conf = ev[(ev.majority >= thr) & (~ev.event_id.isin(FLUVIAL_EVENTS))].reset_index(drop=True)
    n_c, n_r = int(conf.hard_label.sum()), int((1 - conf.hard_label).sum())
    sep(f"LOOCV: CONFIDENT EVENTS ONLY (majority >= {thr:.2f})  "
        f"n={len(conf)} ({n_r} rain, {n_c} coastal)")
    for tier, feats in [("FULL", DYNAMIC), ("STRICT", STRICT),
                        ("PHYSICAL", PHYSICAL), ("SURGE-ONLY", SURGE_ONLY),
                        ("RAIN-ONLY", RAIN_ONLY)]:
        cols = agg_cols(feats)
        X, y = conf[cols].values, conf.hard_label.values
        preds = []
        scaler = StandardScaler()
        for tr_i, te_i in LeaveOneOut().split(X):
            m = new_model()
            m.fit(scaler.fit_transform(X[tr_i]), y[tr_i])
            preds.append(m.predict(scaler.transform(X[te_i]))[0])
        preds = np.array(preds)
        acc = accuracy_score(y, preds)
        kap = cohen_kappa_score(y, preds)
        cm = confusion_matrix(y, preds, labels=[0, 1])
        lo, hi = wilson_ci(int((preds == y).sum()), len(y))
        wrong = [conf.event_id.iloc[i] for i in range(len(y)) if preds[i] != y[i]]
        print(f"  {tier:<10s}: Acc {acc*100:5.1f}% (95% CI {lo*100:.0f}-{hi*100:.0f}%)  "
              f"Kappa {kap:6.3f}  CM [[{cm[0,0]},{cm[0,1]}],[{cm[1,0]},{cm[1,1]}]]  "
              f"missed: {wrong if wrong else 'none'}")

# =====================================================================
# SOFT PROBABILITIES FOR ALL EVENTS (rain-elimination model, thr 0.60)
# RAIN_ONLY is the best-performing leak-free tier (see ablation above);
# it attributes coastal floods by the absence of rainfall forcing.
# =====================================================================
sep("SOFT P(coastal) FOR ALL 23 EVENTS -- rain-elimination model")
thr = 0.60
conf = ev[(ev.majority >= thr) & (~ev.event_id.isin(FLUVIAL_EVENTS))].reset_index(drop=True)
cols = agg_cols(RAIN_ONLY)

# LOOCV probabilities for confident events (no self-prediction);
# single model trained on all confident events for the rest.
prob = pd.Series(index=ev.index, dtype=float)
scaler = StandardScaler()
conf_idx = {r.event_id: i for i, r in conf.iterrows()}
Xc, yc = conf[cols].values, conf.hard_label.values
for tr_i, te_i in LeaveOneOut().split(Xc):
    m = new_model()
    m.fit(scaler.fit_transform(Xc[tr_i]), yc[tr_i])
    p = m.predict_proba(scaler.transform(Xc[te_i]))[0, 1]
    eid = conf.event_id.iloc[te_i[0]]
    prob[ev.index[ev.event_id == eid][0]] = p

m_all = new_model()
Xc_s = scaler.fit_transform(Xc)
m_all.fit(Xc_s, yc)
for i, r in ev.iterrows():
    if pd.isna(prob[i]):
        prob[i] = m_all.predict_proba(scaler.transform(ev.loc[[i], cols].values))[0, 1]
ev["P_coastal"] = prob

def category(r):
    if r.event_id in FLUVIAL_EVENTS:
        return "fluvial (excluded)"
    if r.majority >= thr:
        return "coastal" if r.hard_label == 1 else "rain"
    return "mixed/compound"
ev["category"] = ev.apply(category, axis=1)
ev["news"] = ev.event_id.map(NEWS_VERIFIED).fillna("(seasonal proxy, unverified)")

out = ev[["event_id", "n_pixels", "coastal_fraction", "majority",
          "category", "P_coastal", "news"]].sort_values("P_coastal")
print(out.to_string(index=False,
      formatters={"coastal_fraction": "{:.3f}".format,
                  "majority": "{:.3f}".format, "P_coastal": "{:.3f}".format}))
out.to_csv(os.path.join(BASE, "keta_event_attribution_table.csv"), index=False)
print("\n  [SAVED] keta_event_attribution_table.csv")

# =====================================================================
# PAPER FIGURE: P(coastal) by event
# =====================================================================
colors = {"coastal": "#1f77b4", "rain": "#2ca02c",
          "mixed/compound": "#ff7f0e", "fluvial (excluded)": "#9467bd"}
o = out.reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(range(len(o)), o.P_coastal,
        color=[colors[c] for c in o.category])
ax.set_yticks(range(len(o)))
ax.set_yticklabels(o.event_id, fontsize=9)
ax.axvline(0.5, color="k", lw=0.8, ls="--")
ax.set_xlabel("P(coastal-dominant)  --  rain-elimination model (rain, solar, season only)")
ax.set_title("Keta flood events: leak-free driver attribution (v4.2)",
             fontsize=11, fontweight="bold")
handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()]
ax.legend(handles, colors.keys(), loc="lower right", fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(BASE, "keta_event_attribution_figure.png"), dpi=200)
plt.close(fig)
print("  [SAVED] keta_event_attribution_figure.png")
