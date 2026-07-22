#!/usr/bin/env python3
"""
=====================================================================
KETA v4.3 -- TWL TIME-SERIES FIGURE (paper figure)
Hourly total water level for the three forensically decisive events:
  nov2021           -- phase_align = 1.000, surge at dawn Nov 7 2021
  may2025           -- TWL peak on the documented day (May 26 2025)
  jun2025_lawoshime -- TWL peak inside the GhMet tidal warning window
Saves per-event series CSVs and a 3-panel figure.
=====================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from extract_fes2022_tides import (
    discover_constituents, load_models, compute_event_tide,
    SITE_LON, SITE_LAT, SITE_BBOX, OCEAN_TIDE_DIR, LOAD_TIDE_DIR,
)
from extract_ww3_waves import fetch_window

BASE = os.path.dirname(os.path.abspath(__file__))
G = 9.81

EVENTS = [
    {"id": "nov2021", "start": "2021-11-06", "end": "2021-11-16",
     "title": "nov2021 -- tidal surge, ~4,000 displaced",
     "mark": ("2021-11-07 05:00", "surge at dawn, 7 Nov 2021"),
     "shade": None},
    {"id": "may2025", "start": "2025-05-24", "end": "2025-06-03",
     "title": "may2025 -- tidal waves, Denu road submerged",
     "mark": ("2025-05-26 12:00", "documented event, 26 May 2025"),
     "shade": None},
    {"id": "jun2025_lawoshime", "start": "2025-06-25", "end": "2025-07-05",
     "title": "jun2025_lawoshime -- island communities flooded",
     "mark": None,
     "shade": ("2025-06-27", "2025-06-30", "GhMet tidal-wave warning")},
]


def runup(h0, t0):
    l0 = G * t0 ** 2 / (2 * np.pi)
    return 0.043 * np.sqrt(h0 * l0)


def main():
    print("Loading FES2022 models...")
    ocean = discover_constituents(OCEAN_TIDE_DIR)
    load = discover_constituents(LOAD_TIDE_DIR)
    tide_model, radial_model = load_models(ocean, load, SITE_BBOX)
    print("  loaded.")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for ax, e in zip(axes, EVENTS):
        tide = compute_event_tide(tide_model, radial_model,
                                  e["start"], e["end"], SITE_LON, SITE_LAT)
        tide = tide.set_index("datetime")

        wav = fetch_window(e["start"], e["end"])
        wav["time"] = pd.to_datetime(wav["time"]).dt.tz_localize(None)
        wav = wav.set_index("time")[["Thgt", "Tper"]].apply(
            pd.to_numeric, errors="coerce").dropna()
        wavh = wav.reindex(wav.index.union(tide.index)).interpolate(
            method="time").reindex(tide.index).dropna()
        j = tide.join(wavh, how="inner").dropna()
        j["R2"] = runup(j["Thgt"], j["Tper"])
        j["TWL"] = j["combined_m"] + j["R2"]
        j[["combined_m", "Thgt", "Tper", "R2", "TWL"]].to_csv(
            os.path.join(BASE, f"keta_twl_series_{e['id']}.csv"))

        ax.plot(j.index, j["combined_m"], color="#999999", lw=0.9,
                label="FES2022 tide")
        ax.plot(j.index, j["TWL"], color="#1f77b4", lw=1.6,
                label="TWL = tide + swell runup")
        ax.axhline(1.9, color="#d62728", lw=0.8, ls=":",
                   label="1.9 m reference")

        pk = j["TWL"].idxmax()
        ax.plot([pk], [j['TWL'][pk]], "o", color="#1f77b4", ms=6)
        ax.annotate(f"TWL peak {j['TWL'][pk]:.2f} m",
                    (pk, j["TWL"][pk]), xytext=(8, 6),
                    textcoords="offset points", fontsize=8)

        if e["mark"]:
            t0, lbl = e["mark"]
            t0 = pd.Timestamp(t0)
            ax.axvline(t0, color="#d62728", lw=1.2, ls="--")
            ax.annotate(lbl, (t0, ax.get_ylim()[0]), xytext=(6, 14),
                        textcoords="offset points", fontsize=8, color="#d62728")
        if e["shade"]:
            s0, s1, lbl = e["shade"]
            ax.axvspan(pd.Timestamp(s0), pd.Timestamp(s1),
                       color="#d62728", alpha=0.12)
            ax.annotate(lbl, (pd.Timestamp(s0), ax.get_ylim()[1]),
                        xytext=(6, -14), textcoords="offset points",
                        fontsize=8, color="#d62728")

        ax.set_ylim(-1.1, 2.35)
        ax.set_ylabel("water level (m)")
        ax.set_title(e["title"], fontsize=10, fontweight="bold", loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.legend(fontsize=7, loc="lower left", ncol=3)
        print(f"  {e['id']}: TWL peak {j['TWL'].max():.2f} m at {j['TWL'].idxmax()}")

    fig.suptitle("Hourly total water level at Keta: wave-tide phase coincidence",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(BASE, "keta_twl_series_figure.png"), dpi=200)
    plt.close(fig)
    print("Saved keta_twl_series_figure.png")


if __name__ == "__main__":
    main()
