"""
extract_fes2022_tides.py

Regenerates the ENTIRE FES2022_DATA object (all 19 events) in
keta_flood_prediction_v3_2_3.js using real PyFES-computed tide values at
the Keta site: lon=0.97E, lat=5.90N. Recomputing all 19 from the same
method — rather than only filling the 5 placeholder-zero validation rows
— ensures the whole table is on a consistent basis (geocentric tide,
hourly resolution, same script), since the provenance of the original
14 values (2019-2023) wasn't confirmed to match this method.

REQUIRES:
    pip install pyfes numpy pandas --break-system-packages
    (or: conda install -c conda-forge pyfes)

FOLDER LAYOUT ASSUMED (matches your screenshot):
    load_tide/                <- per-constituent load tide grids (m2.nc, s2.nc, ...)
    ocean_tide_extrapolated/  <- per-constituent ocean tide grids, extrapolated to coast
    mask_fes2022b.nc          <- land/ocean mask (referenced by config, not read directly here)

WHAT THIS COMPUTES, PER EVENT:
    - tide time series at hourly resolution across the event's [start, end) window
    - max:   maximum GEOCENTRIC tide elevation (m) in that window
             (geocentric tide = ocean_tide_extrapolated + load_tide, the
             standard FES2022/altimetry definition of total sea-surface
             elevation referenced to the geocenter)
    - range: max - min geocentric tide elevation (m) in that window
    - spring: 1 if this event's range is in the upper half of all 19 events, else 0
      (proxy rule — swap in your original threshold logic if it differs from
      how the 2019-2023 FES values were derived)

NOTE ON TPXO:
    TPXO has been dropped from this pipeline. FES2022 is now the sole tidal
    data source, per project decision — GEE script updated accordingly
    (tide_max_tpxo / tide_range_tpxo / spring_flag_tpxo bands removed).
"""

import glob
import os
import numpy as np
import pandas as pd
import pyfes
from pyfes.config import Cartesian

# ---------------------------------------------------------------------------
# CONFIG — adjust paths if your folders live somewhere else
# ---------------------------------------------------------------------------
# Resolve relative to this script's location so it works from any CWD.
_BASE = os.path.dirname(os.path.abspath(__file__))
OCEAN_TIDE_DIR = os.path.join(_BASE, "ocean_tide_extrapolated")
LOAD_TIDE_DIR = os.path.join(_BASE, "load_tide")

SITE_LON = 0.97   # degrees East
SITE_LAT = 5.90   # degrees North

# Bounding box around Keta and its coastal hinterland (lon_min, lat_min,
# lon_max, lat_max).  A 2°×2° window is more than enough for a single-point
# extraction and keeps peak RAM ~50 MB instead of ~15 GB (global grids).
SITE_BBOX = (0.0, 4.9, 2.0, 6.9)

# Geocentric tide (ocean + load) is the standard FES2022/altimetry
# definition of total sea-surface elevation. Not a toggle — this is the
# correct quantity for "max"/"range" as used elsewhere in FES2022_DATA.

# ALL 23 events, with windows matching the GEE event catalogue in
# keta_flood_pipeline_v4_1.js / v4_2.js EXACTLY. Earlier revisions of this
# script used different windows for several events (may2021, nov2021,
# may2022, jul2022, jul2023, sep2023, oct2023, may2024, jan2025, feb2025),
# which meant FES2022_DATA tide values were computed over different date
# ranges than every other feature in the pipeline. Keep this list in sync
# with the GEE catalogue — it is the single source of truth for windows.
ALL_EVENTS = [
    {"id": "jun2019", "start": "2019-06-01", "end": "2019-06-20"},
    {"id": "jul2019", "start": "2019-07-01", "end": "2019-07-20"},
    {"id": "jun2020", "start": "2020-06-01", "end": "2020-06-20"},
    {"id": "jul2020", "start": "2020-07-01", "end": "2020-07-20"},
    {"id": "may2021", "start": "2021-05-01", "end": "2021-05-15"},
    {"id": "jun2021", "start": "2021-06-01", "end": "2021-06-15"},
    # Surge hit at dawn on Sunday 2021-11-07 (news-verified); window starts
    # the day before to catch the spring-tide build-up (new moon Nov 4).
    {"id": "nov2021", "start": "2021-11-06", "end": "2021-11-16"},
    # Renamed from may2022: the documented 2022 tidal-wave event at
    # Agavedzi/Salakope was 2022-04-03/04, not May (news-verified).
    {"id": "apr2022", "start": "2022-04-03", "end": "2022-04-14"},
    {"id": "jun2022", "start": "2022-06-01", "end": "2022-06-15"},
    {"id": "jul2022", "start": "2022-07-01", "end": "2022-07-15"},
    {"id": "jun2023", "start": "2023-06-01", "end": "2023-06-15"},
    {"id": "jul2023", "start": "2023-07-01", "end": "2023-07-15"},
    {"id": "sep2023", "start": "2023-09-15", "end": "2023-09-25"},
    {"id": "oct2023", "start": "2023-10-15", "end": "2023-10-25"},
    # Event verified: tidal surge in the early hours of 2024-02-15 at Keta.
    # End extended to Feb 25: GEE coverage audit found no S1 pass in Feb 10-20.
    {"id": "feb2024", "start": "2024-02-10", "end": "2024-02-25"},
    # Onset 2025-01-16 (Salakope/Agavedzi, news-verified); old window
    # Jan 10-20 started six days before the event.
    {"id": "jan2025", "start": "2025-01-16", "end": "2025-01-26"},
    # Onset 2025-02-01 (Agavedzi/Amutsinu, 447 displaced, IOM RNA); old
    # window Feb 5-15 missed the onset entirely.
    {"id": "feb2025", "start": "2025-02-01", "end": "2025-02-11"},
    # Second surge 2025-03-01 flooded the Denu-Keta road (news-verified).
    # End extended to Mar 15: GEE coverage audit found no S1 pass in Mar 1-10.
    {"id": "mar2025", "start": "2025-03-01", "end": "2025-03-15"},
    # Renamed from may2024: no documented May 2024 event exists; the
    # documented event is 2025-05-26 (Ketu South, road submerged).
    {"id": "may2025", "start": "2025-05-24", "end": "2025-06-03"},
    {"id": "sep2025_market", "start": "2025-09-12", "end": "2025-09-15"},
    {"id": "jun2025_lawoshime", "start": "2025-06-25", "end": "2025-07-05"},
    {"id": "may2026_downpour", "start": "2026-05-06", "end": "2026-05-18"},
    {"id": "jun2026_floodgates", "start": "2026-06-15", "end": "2026-07-03"},
]


# ---------------------------------------------------------------------------
# STEP 1 — auto-discover constituents from the folders and build a config
# ---------------------------------------------------------------------------
def discover_constituents(folder):
    """Return {constituent_name: filepath} from all .nc files in a folder."""
    paths = {}
    for fp in glob.glob(os.path.join(folder, "*.nc")):
        name = os.path.splitext(os.path.basename(fp))[0].lower()
        if "mask" in name:
            continue
        if name.endswith("_fes2022"):
            name = name[:-8]
        elif name.endswith("_fes2022b"):
            name = name[:-9]
        paths[name] = fp
    if not paths:
        raise FileNotFoundError(
            f"No .nc files found in '{folder}'. Check OCEAN_TIDE_DIR/LOAD_TIDE_DIR paths."
        )
    return paths


def load_models(ocean_paths, load_paths, bbox):
    """
    Load the FES2022 Cartesian tide and load-tide models using the modern
    pyfes.config.Cartesian API (pyfes ≥ 2025).

    bbox = (lon_min, lat_min, lon_max, lat_max) clips each constituent grid
    to the region of interest, keeping RAM to ~50 MB for a 2°×2° window.
    """
    tide_model = Cartesian(
        latitude="lat",
        longitude="lon",
        amplitude="amplitude",
        phase="phase",
        paths=ocean_paths,
        tidal_type="tide",
        bbox=bbox,
    ).load()

    radial_model = Cartesian(
        latitude="lat",
        longitude="lon",
        amplitude="amplitude",
        phase="phase",
        paths=load_paths,
        tidal_type="radial",
        bbox=bbox,
    ).load()

    return tide_model, radial_model


# ---------------------------------------------------------------------------
# STEP 2 — compute tide time series for one event window
# ---------------------------------------------------------------------------
def compute_event_tide(tide_model, radial_model, start, end, lon, lat):
    """
    Returns a DataFrame with hourly geocentric tide (ocean + load) in metres
    across the half-open window [start, end).

    pyfes.evaluate_tide returns (short_period_cm, long_period_cm, quality_flags).
    All model values are in centimetres; we convert to metres here.
    Quality flag > 0 → interpolated; < 0 → extrapolated; 0 → undefined.
    Keta sits at the edge of the continental shelf so a few extrapolated
    points are expected — the ocean_tide_extrapolated grids are already
    extended to the coast, so this is the intended behaviour.
    """
    dates = pd.date_range(start=start, end=end, freq="1h", inclusive="left")
    dates_np = dates.values.astype("datetime64[us]")

    lons = np.full(dates_np.shape, lon)
    lats = np.full(dates_np.shape, lat)

    # pyfes.evaluate_tide → (short_period_cm, long_period_cm, quality_flags)
    ocean_short, ocean_long, _ = pyfes.evaluate_tide(tide_model, dates_np, lons, lats)
    load_short,  load_long,  _ = pyfes.evaluate_tide(radial_model, dates_np, lons, lats)

    ocean_total_cm = ocean_short + ocean_long
    load_total_cm  = load_short  + load_long

    df = pd.DataFrame({
        "datetime":     dates,
        "ocean_tide_m": ocean_total_cm / 100.0,
        "load_tide_m":  load_total_cm  / 100.0,
    })
    df["combined_m"] = df["ocean_tide_m"] + df["load_tide_m"]
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Discovering constituents...")
    ocean_paths = discover_constituents(OCEAN_TIDE_DIR)
    load_paths  = discover_constituents(LOAD_TIDE_DIR)
    print(f"  Ocean tide constituents found: {len(ocean_paths)}")
    print(f"  Load tide constituents found:  {len(load_paths)}")

    print(f"  Loading models (bbox {SITE_BBOX})...")
    tide_model, radial_model = load_models(ocean_paths, load_paths, SITE_BBOX)
    print("  Models loaded.")

    results = []
    for event in ALL_EVENTS:
        df = compute_event_tide(
            tide_model, radial_model,
            event["start"], event["end"],
            SITE_LON, SITE_LAT,
        )

        max_val = float(df["combined_m"].max())      # geocentric tide (ocean + load)
        min_val = float(df["combined_m"].min())
        range_val = max_val - min_val

        results.append({
            "id": event["id"],
            "max": round(max_val, 3),
            "range": round(range_val, 3),
            "ocean_only_max": round(float(df["ocean_tide_m"].max()), 3),  # reference only
            "load_only_max_mm": round(float(df["load_tide_m"].max()) * 1000, 2),  # reference only
        })

        print(f"{event['id']:>10}: max={max_val:.3f} m  range={range_val:.3f} m")

    results_df = pd.DataFrame(results).set_index("id")

    # Spring/neap flag: relative to the median range across ALL 19 events.
    # NOTE: this is a proxy rule. If your 2019-2023 spring flags were set
    # by a different method (e.g. lunar phase / syzygy proximity), replace
    # this block so the new 5 events stay consistent with the old 14.
    median_range = results_df["range"].median()
    results_df["spring"] = (results_df["range"] >= median_range).astype(int)

    print(f"\n=== FULL RESULTS (all {len(ALL_EVENTS)} events, for consistency check) ===")
    print(results_df)

    # ---- Emit the JS snippet for ALL 19 events ----
    # Regenerating all of them (not just the 5 placeholders) ensures every
    # row in FES2022_DATA comes from the same consistent method (geocentric
    # tide, hourly resolution, this script) rather than mixing old values
    # of unknown provenance with newly-computed ones.
    print("\n=== Paste this whole block in as the new FES2022_DATA object ===\n")
    print("var FES2022_DATA = {")
    for i, event in enumerate(ALL_EVENTS):
        row = results_df.loc[event["id"]]
        comma = "," if i < len(ALL_EVENTS) - 1 else ""
        print(
            f'  "{event["id"]}":{{ max:{row["max"]:.3f}, '
            f'range:{row["range"]:.3f}, spring:{int(row["spring"])} }}{comma}'
        )
    print("};")

    results_df.to_csv("fes2022_extracted_tides.csv")
    print("\nFull results also saved to fes2022_extracted_tides.csv")


if __name__ == "__main__":
    main()
