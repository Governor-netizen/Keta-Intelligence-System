#!/usr/bin/env python3
"""
Study-area map (Figure 1) for the Keta manuscript.
OpenStreetMap tile mosaic (z11) with the model domain, settlements
(approximate locations), and data-extraction points overlaid.
Basemap (c) OpenStreetMap contributors.
"""
import io
import math
import os
import requests
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.abspath(__file__))
Z = 11
X0, X1 = 1028, 1030   # tile x range (inclusive)
Y0, Y1 = 988, 991     # tile y range (inclusive)
UA = {"User-Agent": "keta-flood-research/1.0 (academic use)"}


def tile_lon(x):
    return x / 2 ** Z * 360.0 - 180.0


def tile_lat(y):
    n = math.pi * (1 - 2 * y / 2 ** Z)
    return math.degrees(math.atan(math.sinh(n)))


def main():
    cols = X1 - X0 + 1
    rows = Y1 - Y0 + 1
    mosaic = Image.new("RGB", (256 * cols, 256 * rows))
    for xi in range(X0, X1 + 1):
        for yi in range(Y0, Y1 + 1):
            url = f"https://tile.openstreetmap.org/{Z}/{xi}/{yi}.png"
            r = requests.get(url, headers=UA, timeout=60)
            r.raise_for_status()
            mosaic.paste(Image.open(io.BytesIO(r.content)).convert("RGB"),
                         ((xi - X0) * 256, (yi - Y0) * 256))

    extent = [tile_lon(X0), tile_lon(X1 + 1), tile_lat(Y1 + 1), tile_lat(Y0)]
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.imshow(np.asarray(mosaic), extent=extent, aspect="auto")

    # Model domain
    ax.add_patch(mpatches.Rectangle((0.80, 5.75), 0.35, 0.35, fill=False,
                                    edgecolor="#d62728", lw=2))
    ax.annotate("model domain  0.80-1.15 E, 5.75-6.10 N", (0.805, 5.762),
                fontsize=8, color="#d62728", fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                          pad=1.5))

    # Settlements (approximate, nudged onto the barrier)
    towns = {
        "Anloga": (0.897, 5.792), "Tegbi": (0.938, 5.852),
        "Woe": (0.955, 5.872), "Keta": (0.987, 5.918),
        "Kedzi/Havedzi": (1.003, 5.942), "Blekusu": (1.040, 5.975),
        "Agavedzi/Salakope": (1.105, 6.030), "Denu": (1.133, 6.086),
    }
    for name, (lo, la) in towns.items():
        ax.plot(lo, la, "o", color="#1f77b4", ms=5)
        ax.annotate(name, (lo, la), xytext=(5, 4),
                    textcoords="offset points", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none",
                              pad=0.8))

    # Extraction points
    ax.plot(0.97, 5.90, "*", color="#d62728", ms=14)
    ax.annotate("FES2022 tide point\n(0.97 E, 5.90 N)", (0.97, 5.90),
                xytext=(-95, -18), textcoords="offset points",
                fontsize=8, color="#d62728")
    ax.annotate("WW3 wave point (1.0 E, 5.5 N), off map to the south",
                (1.0, 5.63), xytext=(0, 0), textcoords="offset points",
                fontsize=8, color="#d62728", ha="center")
    ax.annotate("", xy=(1.0, 5.62), xytext=(1.0, 5.66),
                arrowprops=dict(arrowstyle="->", color="#d62728"))

    ax.annotate("KETA LAGOON", (0.935, 5.965), fontsize=11, color="#1f5f8b",
                fontstyle="italic", ha="center")
    ax.annotate("GULF OF GUINEA", (1.02, 5.72), fontsize=11, color="#1f5f8b",
                fontstyle="italic", ha="center")

    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    ax.set_title("Study area: the Keta barrier and lagoon, Volta Region, Ghana",
                 fontsize=11, fontweight="bold")
    ax.text(0.995, 0.005, "basemap (c) OpenStreetMap contributors",
            transform=ax.transAxes, fontsize=6, ha="right", va="bottom")
    plt.tight_layout()
    fig.savefig(os.path.join(BASE, "keta_study_area_map.png"), dpi=200)
    plt.close(fig)
    print("Saved keta_study_area_map.png")


if __name__ == "__main__":
    main()
