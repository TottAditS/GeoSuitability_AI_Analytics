"""Helper script (not a notebook) for notebook 10.

Zonal-stats the 6 Banten-clipped Global Crop Suitability 2024 rasters onto the 1 km grid and
writes the result to a small CSV. Runs as a plain Python process, invoked via subprocess from
notebook 10, rather than as notebook code directly - the exact same zonal_stats calls reliably
kill the Jupyter kernel on this machine (reproducible, cause not identified - looks like a
GDAL/ipykernel interaction, not a memory or logic problem, since this script works fine standalone
every time). Routing it through a real subprocess avoids the kernel entirely.
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
