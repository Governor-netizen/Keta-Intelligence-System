#!/usr/bin/env python3
"""
=====================================================================
KETA v4.3 -- HOURLY TOTAL WATER LEVEL (TWL) FEATURES
For each event: hourly FES2022 geocentric tide + WW3 waves
(interpolated 3-hourly -> hourly) combined into

    TWL(t) = tide(t) + R2(t),   R2(t) = 0.043 * sqrt(H0(t) * L0(t)),
    L0(t) = g * T(t)^2 / (2*pi)  (deep-water wavelength)

R2 is the Stockdon et al. (2006) dissipative-beach 2% runup term. The
constant is shared across events, so TWL is a valid RELATIVE measure of
wave-tide phase coincidence even without a surveyed beach profile.

Window statistics (v4.2) could not distinguish "high tide and high
swell somewhere in the window" from "high swell arriving ON high
tide". The hourly product can.

Outputs twl_features.csv:
    twl_max        : peak total water level in the window (m)
    twl_p95        : 95th percentile of hourly TWL (sustained extreme)
    hours_ge_190   : hours with TWL >= 1.90 m (exposure duration)
    phase_align    : twl_max / (tide_max + R2_max) -- 1.0 means the
                     swell peak arrived exactly on the tide peak
=====================================================================
"""
import os
import numpy as np
import pandas as pd

from extract_fes2022_tides import (
    discover_constituents, load_models, compute_event_tide,
    ALL_EVENTS, SITE_LON, SITE_LAT, SITE_BBOX,
    OCEAN_TIDE_DIR, LOAD_TIDE_DIR,
)
from extract_ww3_waves import fetch_window

BASE = os.path.dirname(os.path.abspath(__file__))
G = 9.81


def runup(h0, t0):
    l0 = G * t0 ** 2 / (2 * np.pi)
    return 0.043 * np.sqrt(h0 * l0)


def main():
    print("Loading FES2022 models...")
    ocean = discover_constituents(OCEAN_TIDE_DIR)
    load = discover_constituents(LOAD_TIDE_DIR)
    tide_model, radial_model = load_models(ocean, load, SITE_BBOX)
    print("  loaded.")

    rows = []
    for e in ALL_EVENTS:
        tide = compute_event_tide(tide_model, radial_model,
                                  e["start"], e["end"], SITE_LON, SITE_LAT)
        tide = tide.set_index("datetime")

        wav = fetch_window(e["start"], e["end"])
        wav["time"] = pd.to_datetime(wav["time"]).dt.tz_localize(None)
        wav = wav.set_index("time")[["Thgt", "Tper"]].apply(
            pd.to_numeric, errors="coerce").dropna()

        # interpolate 3-hourly waves onto the hourly tide index
        wavh = wav.reindex(wav.index.union(tide.index)).interpolate(
            method="time").reindex(tide.index).dropna()
        joined = tide.join(wavh, how="inner").dropna()

        twl = joined["combined_m"] + runup(joined["Thgt"], joined["Tper"])
        r2 = runup(joined["Thgt"], joined["Tper"])

        twl_max = float(twl.max())
        potential = float(joined["combined_m"].max()) + float(r2.max())
        rows.append({
            "id": e["id"],
            "n_hours": len(twl),
            "twl_max": round(twl_max, 3),
            "twl_p95": round(float(twl.quantile(0.95)), 3),
            "hours_ge_190": int((twl >= 1.90).sum()),
            "phase_align": round(twl_max / potential, 4),
            "tide_at_twl_peak": round(float(joined["combined_m"][twl.idxmax()]), 3),
            "hs_at_twl_peak": round(float(joined["Thgt"][twl.idxmax()]), 3),
            "twl_peak_time": str(twl.idxmax()),
        })
        r = rows[-1]
        print(f"{e['id']:>20s}: TWL_max={r['twl_max']:.2f} m  p95={r['twl_p95']:.2f}  "
              f"hrs>=1.9m={r['hours_ge_190']:3d}  align={r['phase_align']:.3f}  "
              f"peak@{r['twl_peak_time']} (tide {r['tide_at_twl_peak']:.2f} + Hs {r['hs_at_twl_peak']:.2f})")

    out = pd.DataFrame(rows).set_index("id")
    out.to_csv(os.path.join(BASE, "twl_features.csv"))
    print("\nSaved twl_features.csv")


if __name__ == "__main__":
    main()
