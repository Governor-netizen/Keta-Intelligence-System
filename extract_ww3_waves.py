#!/usr/bin/env python3
"""
=====================================================================
KETA v4.3 -- WAVEWATCH III WAVE FORCING EXTRACTION
Fetches 3-hourly wave conditions at an offshore Keta point from the
PacIOOS ERDDAP mirror of the NOAA WaveWatch III global hindcast
(dataset ww3_global, 0.5 deg, 2017-01 to present, no account needed).

Rationale: the v4.2 mechanism ablation showed local ERA5 pressure/wind
carry NO surge signal at Keta (kappa -0.02) because damaging surges
arrive as long-period swell from distant South Atlantic storms. Wave
model output is the observable that carries that signal.

Outputs per event (window matching the v4.2 catalogue EXACTLY):
  swh_max / swh_mean : significant wave height, total sea (Thgt), m
  swell_hs_max       : swell partition height (shgt), m
  swell_per_max      : swell peak period (sper), s  (long = remote swell)
  wave_power_max     : 0.49 * Hs^2 * Tper, kW/m proxy at window peak

Writes ww3_extracted_waves.csv and prints a WW3_DATA JS block for the
GEE script (v4.3).
=====================================================================
"""
import os
import io
import time
import requests
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SERVER = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global.csv"

# Offshore Keta: ~30 km south of the barrier, on the 0.5-deg grid.
LAT = 5.5
LON = 1.0  # dataset uses 0-360; 1.0E is 1.0

# Event windows: keep identical to keta_flood_pipeline_v4_2.js
EVENTS = [
    {"id": "jun2019", "start": "2019-06-01", "end": "2019-06-20"},
    {"id": "jul2019", "start": "2019-07-01", "end": "2019-07-20"},
    {"id": "jun2020", "start": "2020-06-01", "end": "2020-06-20"},
    {"id": "jul2020", "start": "2020-07-01", "end": "2020-07-20"},
    {"id": "may2021", "start": "2021-05-01", "end": "2021-05-15"},
    {"id": "jun2021", "start": "2021-06-01", "end": "2021-06-15"},
    {"id": "nov2021", "start": "2021-11-06", "end": "2021-11-16"},
    {"id": "apr2022", "start": "2022-04-03", "end": "2022-04-14"},
    {"id": "jun2022", "start": "2022-06-01", "end": "2022-06-15"},
    {"id": "jul2022", "start": "2022-07-01", "end": "2022-07-15"},
    {"id": "jun2023", "start": "2023-06-01", "end": "2023-06-15"},
    {"id": "jul2023", "start": "2023-07-01", "end": "2023-07-15"},
    {"id": "sep2023", "start": "2023-09-15", "end": "2023-09-25"},
    {"id": "oct2023", "start": "2023-10-15", "end": "2023-10-25"},
    {"id": "feb2024", "start": "2024-02-10", "end": "2024-02-25"},
    {"id": "jan2025", "start": "2025-01-16", "end": "2025-01-26"},
    {"id": "feb2025", "start": "2025-02-01", "end": "2025-02-11"},
    {"id": "mar2025", "start": "2025-03-01", "end": "2025-03-15"},
    {"id": "may2025", "start": "2025-05-24", "end": "2025-06-03"},
    {"id": "sep2025_market", "start": "2025-09-12", "end": "2025-09-15"},
    {"id": "jun2025_lawoshime", "start": "2025-06-25", "end": "2025-07-05"},
    {"id": "may2026_downpour", "start": "2026-05-06", "end": "2026-05-18"},
    {"id": "jun2026_floodgates", "start": "2026-06-15", "end": "2026-07-03"},
]

VARS = ["Thgt", "Tper", "shgt", "sper"]


def fetch_window(start, end):
    """Return a DataFrame of 3-hourly wave variables for [start, end]."""
    sub4 = f"[({start}T00:00:00Z):({end}T23:59:59Z)][(0.0):(0.0)][({LAT}):({LAT})][({LON}):({LON})]"
    sub3 = f"[({start}T00:00:00Z):({end}T23:59:59Z)][({LAT}):({LAT})][({LON}):({LON})]"
    for sub in (sub4, sub3):  # dataset has a z dim; fall back if not
        url = SERVER + "?" + ",".join(v + sub for v in VARS)
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text), skiprows=[1])  # row 1 = units
            return df
    raise RuntimeError(f"ERDDAP request failed ({r.status_code}): {r.text[:300]}")


def main():
    rows = []
    for e in EVENTS:
        for attempt in range(3):
            try:
                df = fetch_window(e["start"], e["end"])
                break
            except Exception as ex:
                if attempt == 2:
                    raise
                time.sleep(5)
        thgt = pd.to_numeric(df["Thgt"], errors="coerce").dropna()
        tper = pd.to_numeric(df["Tper"], errors="coerce").dropna()
        shgt = pd.to_numeric(df["shgt"], errors="coerce").dropna()
        sper = pd.to_numeric(df["sper"], errors="coerce").dropna()
        power = 0.49 * thgt ** 2 * tper.reindex(thgt.index).fillna(tper.mean())
        rows.append({
            "id": e["id"],
            "n_steps": len(thgt),
            "swh_max": round(float(thgt.max()), 3),
            "swh_mean": round(float(thgt.mean()), 3),
            "swell_hs_max": round(float(shgt.max()), 3) if len(shgt) else np.nan,
            "swell_per_max": round(float(sper.max()), 2) if len(sper) else np.nan,
            "wave_power_max": round(float(power.max()), 2),
        })
        print(f"{e['id']:>20s}: n={len(thgt):3d}  swh_max={rows[-1]['swh_max']:.2f} m  "
              f"swell_hs_max={rows[-1]['swell_hs_max']:.2f} m  "
              f"swell_per_max={rows[-1]['swell_per_max']:.1f} s  "
              f"power_max={rows[-1]['wave_power_max']:.1f}")

    out = pd.DataFrame(rows).set_index("id")
    out.to_csv(os.path.join(BASE, "ww3_extracted_waves.csv"))
    print("\nSaved ww3_extracted_waves.csv")

    print("\n=== Paste into the GEE script (v4.3) as WW3_DATA ===\n")
    print("var WW3_DATA = {")
    for i, e in enumerate(EVENTS):
        r = out.loc[e["id"]]
        comma = "," if i < len(EVENTS) - 1 else ""
        print(f'  "{e["id"]}":{{ swh_max:{r.swh_max:.3f}, swell_hs_max:{r.swell_hs_max:.3f}, '
              f'swell_per_max:{r.swell_per_max:.2f}, wave_power_max:{r.wave_power_max:.2f} }}{comma}')
    print("};")


if __name__ == "__main__":
    main()
