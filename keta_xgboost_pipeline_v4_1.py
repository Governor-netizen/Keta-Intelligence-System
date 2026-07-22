#!/usr/bin/env python3
"""
=====================================================================
KETA COASTAL FLOOD PREDICTION -- XGBoost PIPELINE v4.1
Evaluates Model A and Model B using v4.1 sample exports containing:
  - surface_pressure (ERA5)
  - onshore_wind (ERA5 engineered SE vector)
  - rain_intensity_ratio (GPM / CHIRPS)
  - tide_anomaly (FES2022)
=====================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# PATHS (Prefers v4.2 exports, then v4.1, then v4.0 fallback)
# =====================================================================
def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[-1]

MODEL_A_CSV = _first_existing(
    r"C:\Users\LEE\Downloads\keta_samples_v4_2_all_splits.csv",
    r"C:\Users\LEE\Downloads\keta_samples_v4_1_all_splits.csv",
    r"C:\Users\LEE\Downloads\keta_samples_v4_0_all_splits.csv",
)
MODEL_B_CSV = _first_existing(
    r"C:\Users\LEE\Downloads\keta_samples_driver_v4_2_all_splits.csv",
    r"C:\Users\LEE\Downloads\keta_samples_driver_v4_1_all_splits.csv",
    r"C:\Users\LEE\Downloads\keta_samples_driver_v4_0_all_splits.csv",
)
OUTPUT_DIR  = r"C:\Users\LEE\Documents\Keta Intelligence system"

# =====================================================================
# FEATURE SETS (v4.1 ENHANCED)
# =====================================================================
INPUT_FEATURES_ALL = [
    "elevation", "slope", "aspect", "dist_water", "lt3", "lt1", "lt0",
    "imerg_24h", "imerg_peak_intensity", "imerg_3d",
    "chirps_7d", "chirps_30d",
    "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
    "surface_pressure", "onshore_wind",
    "soil_moisture",
    "tide_max_fes", "tide_range_fes", "spring_flag_fes", "tide_anomaly",
    "freeboard_fes", "compound_risk", "season_wet", "rain_intensity_ratio"
]

# Model B: dynamic-only (no terrain at all)
INPUT_FEATURES_DRIVER_ONLY = [
    "imerg_24h", "imerg_peak_intensity", "imerg_3d",
    "chirps_7d", "chirps_30d",
    "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
    "surface_pressure", "onshore_wind",
    "soil_moisture",
    "tide_max_fes", "tide_range_fes", "spring_flag_fes", "tide_anomaly",
    "freeboard_fes", "compound_risk", "season_wet", "rain_intensity_ratio"
]

# Model B strict leakage: includes the NEW physical proxies (pressure, onshore wind, tide anomaly)
# while excluding direct label-rule variables (imerg_24h, freeboard_fes)
INPUT_FEATURES_STRICT_V41 = [
    "imerg_peak_intensity", "imerg_3d",
    "chirps_7d", "chirps_30d",
    "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
    "surface_pressure", "onshore_wind",
    "soil_moisture",
    "tide_range_fes", "spring_flag_fes", "tide_anomaly",
    "season_wet", "rain_intensity_ratio"
]

def separator(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

def print_multiclass_metrics(y_true, y_pred, class_names, label=""):
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
    return {"acc": acc, "kap": kap}

def compute_sample_weights(y, classes):
    counts = {c: np.sum(y == c) for c in classes}
    max_count = max(counts.values())
    weights = np.ones(len(y), dtype=float)
    for c in classes:
        if counts[c] > 0:
            weights[y == c] = max_count / counts[c]
    return weights

separator("PIPELINE v4.1 EXECUTION")
print(f"  Loading file: {MODEL_B_CSV}")
df_b = pd.read_csv(MODEL_B_CSV)

# Check which features exist in CSV
available_features = [f for f in INPUT_FEATURES_STRICT_V41 if f in df_b.columns]
print(f"  Available strict features ({len(available_features)}): {available_features}")

for col in available_features + ["flood_driver", "split"]:
    df_b[col] = pd.to_numeric(df_b[col], errors="coerce")

train_b = df_b[df_b["split"] == "train"].copy()
test_b  = df_b[df_b["split"] == "test"].copy()
val_b   = df_b[df_b["split"] == "validation"].copy()

X_train_b = train_b[available_features].values
y_train_b = train_b["flood_driver"].astype(int).values
X_test_b  = test_b[available_features].values
y_test_b  = test_b["flood_driver"].astype(int).values
X_val_b   = val_b[available_features].values
y_val_b   = val_b["flood_driver"].astype(int).values

sw_train_b = compute_sample_weights(y_train_b, [0, 1, 2])
DRIVER_NAMES = ["0: no-flood", "1: rain-dominant", "2: coastal-dominant"]

xgb_b = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    objective="multi:softmax", num_class=3, random_state=42,
    eval_metric="mlogloss", use_label_encoder=False, verbosity=0
)

print("  Training Model B on available strict leakage features...")
xgb_b.fit(X_train_b, y_train_b, sample_weight=sw_train_b)

y_pred_test_b = xgb_b.predict(X_test_b)
y_pred_val_b  = xgb_b.predict(X_val_b)

res_test = print_multiclass_metrics(y_test_b, y_pred_test_b, DRIVER_NAMES, "STRICT LEAKAGE TEST SET")
res_val  = print_multiclass_metrics(y_val_b, y_pred_val_b, DRIVER_NAMES, "STRICT LEAKAGE VALIDATION SET")

print("\n  Feature Importances:")
imp = xgb_b.feature_importances_
for i in np.argsort(imp)[::-1]:
    print(f"    {available_features[i]:>25s}: {imp[i]:.4f}")
