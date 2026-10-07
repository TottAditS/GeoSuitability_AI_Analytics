"""Helper script (not a notebook) for notebook 10.

Zonal-stats the 6 Banten-clipped Global Crop Suitability 2024 rasters onto the 1 km grid and
writes the result to a small CSV. Notebook 10 runs it as a plain Python process. It was split out
when the same zonal_stats calls kept killing the Jupyter kernel; the cause turned out to be a kernel
started without the conda environment activated (missing Library/bin DLL folders and GDAL_DATA /
PROJ_LIB), not GDAL. The calls also work as notebook code in an activated environment; the separate
process is kept because it is harmless and gives the same output.
"""
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

logging.getLogger("rasterio._env").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRID_PATH = PROJECT_ROOT / "datasets" / "processed" / "grid" / "grid_1km.gpkg"
CLIP_DIR = PROJECT_ROOT / "datasets" / "raw" / "global_crop_suitability_2024" / "clipped"
OUT_PATH = PROJECT_ROOT / "datasets" / "processed" / "features" / "gcs2024_zonal_stats.csv"

CROPS = ["rice", "maize", "cassava", "soybean", "oilpalm", "sugarcane"]


def main() -> None:
    grid = gpd.read_file(GRID_PATH)[["grid_id"]].copy()
    grid_geom = gpd.read_file(GRID_PATH).to_crs("EPSG:4326").geometry
    print(f"{len(grid)} grid cells loaded", flush=True)

    for crop in CROPS:
        path = CLIP_DIR / f"{crop}_banten.tif"
        stats = zonal_stats(grid_geom, str(path), stats=["mean"], nodata=-1.0, all_touched=True)
        grid[f"gcs2024_{crop}"] = [s["mean"] for s in stats]
        n_valid = grid[f"gcs2024_{crop}"].notna().sum()
        print(f"{crop}: {n_valid}/{len(grid)} cells got a value", flush=True)

    grid.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
