import numpy as np
import pandas as pd
import pyfes
from pyfes.config import Cartesian

OCEAN_TIDE_DIR = "../ocean_tide_extrapolated"
LOAD_TIDE_DIR = "../load_tide"

SITE_LON = 0.97
SITE_LAT = 5.90

import glob
import os

def discover_constituents(folder):
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
    return paths

def main():
    print("Discovering constituents...")
    ocean_paths = discover_constituents(OCEAN_TIDE_DIR)
    load_paths = discover_constituents(LOAD_TIDE_DIR)
    
    print(f"Ocean constituents: {len(ocean_paths)}")
    print(f"Load constituents: {len(load_paths)}")
    
    print("Loading tide model...")
    tide_model = Cartesian(
        latitude="lat",
        longitude="lon",
        amplitude="amplitude",
        phase="phase",
        paths=ocean_paths,
        tidal_type="tide",
        bbox=(0.0, 4.9, 2.0, 6.9)
    ).load()
    
    print("Loading radial model...")
    radial_model = Cartesian(
        latitude="lat",
        longitude="lon",
        amplitude="amplitude",
        phase="phase",
        paths=load_paths,
        tidal_type="radial",
        bbox=(0.0, 4.9, 2.0, 6.9)
    ).load()
    
    # Test dates: 2024-02-10 to 2024-02-20
    dates = pd.date_range(start="2024-02-10", end="2024-02-20", freq="1h", inclusive="left")
    dates_np = dates.values.astype("datetime64[us]")
    
    lons = np.full(dates_np.shape, SITE_LON)
    lats = np.full(dates_np.shape, SITE_LAT)
    
    print("Evaluating ocean tide...")
    ocean_short, ocean_long, flags_ocean = pyfes.evaluate_tide(tide_model, dates_np, lons, lats)
    print("Evaluating radial tide...")
    load_short, load_long, flags_load = pyfes.evaluate_tide(radial_model, dates_np, lons, lats)
    
    ocean_total_cm = ocean_short + ocean_long
    load_total_cm = load_short + load_long
    
    df = pd.DataFrame({
        "datetime": dates,
        "ocean_tide_m": ocean_total_cm / 100.0,
        "load_tide_m": load_total_cm / 100.0,
    })
    df["combined_m"] = df["ocean_tide_m"] + df["load_tide_m"]
    
    print("Test successful!")
    print(df.head())
    print("Max combined tide:", df["combined_m"].max())
    print("Min combined tide:", df["combined_m"].min())

if __name__ == "__main__":
    main()
