# Data storage layout

## `raw/` — data lake

Everything landed here exactly as ingested by `scripts/download/`: original formats (GeoTIFF, GeoJSON, SAFE imagery bundles), native resolutions and coordinate systems, no shared schema across sources. Schema-on-read: each notebook in `scripts/preprocessing/` interprets a subset of this on demand. See `data_dictionary.csv` for the full source-by-source breakdown (format, resolution, size, license, structure type).

## `processed/` — data warehouse / mart

Output of the ETL pipeline (`scripts/preprocessing/01`-`04`): reprojected, resampled, and aggregated onto a single 1km analysis grid over Banten (EPSG:32748). Fixed schema, analysis-ready.

- `processed/grid/grid_1km.gpkg` — the 9,333-cell analysis grid (`grid_id` primary key)
- `processed/sentinel2/` — mosaicked Sentinel-2 composites per year
- `processed/features/sentinel2_cell_stats.csv` — Sentinel-2 statistics per grid cell (2020-2024: band means and
  standard deviations, NDVI percentiles, year-to-year variability), a 1:1 extension of the fact table on `grid_id`, and
  `processed/sentinel2_patches/` — one 25 x 25 image patch per cell and year for convolutional models (both written by
  `scripts/preprocessing/11_sentinel2_cell_features.ipynb`)
- `processed/features/tabular_features.csv` — the single fact table: one row per grid cell, one column per feature (climate, soil, terrain, land cover, GAEZ suitability at ~9 km and 1 km, GCS 2024), used directly for EDA, modeling, and the Sedona/Spark processing demo (`scripts/preprocessing/06_sedona_processing.ipynb`)
- `processed/spark_output/` — partitioned Parquet written by the Sedona job (notebook 06); `checkpoints/` is the scratch
  space of GraphFrames (`scripts/modeling/g01_graph_analytics.ipynb`), emptied when its Spark session stops
- `processed/experiments/` — per-cell outputs of `scripts/modeling`, keyed by `grid_id`: out-of-fold predictions, the
  intervals and label probabilities of m15 and the graph attributes of g01 (`g01_nodes.parquet`: community, recommended
  crop, patch id and patch size per crop), read by the dashboard

### Note: GAEZ suitability now covers 10 crops, all on the irrigated scenario

The original scope (`scripts/preprocessing/04_zonal_statistics.ipynb`) was 4 crops - rice, maize, cassava,
soybean - on GAEZ v5's `HRLM` (rain-fed, high input) scenario. Two follow-up notebooks changed this:

- `07_fix_gaez_irrigated_scenario.ipynb` - switched to `HILM` (irrigated, high input). Started as a rice-only
  fix (wetland/paddy rice is conventionally irrigated; the rain-fed-only version understated it almost
  everywhere in Banten), then extended to all crops to keep the "same scenario for every crop" comparability
  the original download notebook intended. Original `HRLM` values are preserved as `gaez_class_<crop>_rainfed`
  / `gaez_index_<crop>_rainfed`, not used by the labeling rule.
- `08_expand_gaez_crops.ipynb` - added 6 more crops (oil palm, rubber, sugarcane, coffee, cacao, coconut),
  chosen to span Banten's coastal/lowland/upland niches. With only 4 crops, maize and soybean never came
  close to winning anywhere (see notebook 07's validation) - the result was really a 2-crop story. With all
  10, cassava, coconut, and oil palm end up as the real competitors.

### Note: GAEZ v5 is also used at 1 km (30 arc-seconds)

The files in `raw/gaez_v5/` are GAEZ v5's 5 arc-minute version (~9 km): only 164 GAEZ pixels cover Banten, so every
1 km cell inside one of them shares the same values. GAEZ v5 also publishes the suitability index and class at
30 arc-seconds (`MAPSET/RES05-SXX30AS`, `MAPSET/RES05-SIX30AS`), with the same scenario (HP0120, AgERA5, HIST,
HILM). `scripts/preprocessing/12_gaez_v5_1km.ipynb` reads only the Banten window of these Cloud-Optimized GeoTIFFs
over HTTP, keeps the clips in `raw/gaez_v5_30as/` and adds `gaez1k_index_<crop>` / `gaez1k_class_<crop>` to the fact
table, plus rain-fed audit columns (`*_rainfed`) for every crop, as at ~9 km (notebooks 07 and 13). The 1 km class legend differs
from the ~9 km one: 1-8 are the suitability classes, 9 = no cropland (land GAEZ leaves out for that crop and scenario)
and 10 = water. Cells left without a value at 1 km stay empty: 204 for most irrigated crops (181 no cropland, mostly
under tree cover, and 23 open water); coffee, cacao and the rain-fed columns only miss the 23. The dashboard uses the
1 km version by default and can switch back to ~9 km; the modeling notebooks learn the 1 km target and m13 compares
it with the ~9 km one.

### Note: Global Crop Suitability 2024 is now used as an independent cross-validation source

Previously downloaded but never processed (CropSuite, the other suitability dataset, turned out to be
PNG map renders with no georeferencing at all - not usable quantitatively, see `data_dictionary.csv`).
Global Crop Suitability 2024 *is* a real GeoTIFF source (Zenodo/Nature Scientific Data, independent of GAEZ -
different authors, different methodology, newer baseline). `09_crossvalidate_global_crop_suitability.ipynb`
and `10_gcs2024_zonal_stats_and_compare.ipynb` clip its 6 overlapping crops to Banten and add them as
`gcs2024_<crop>` columns (own 0-1 scale, kept separate from `gaez_index_<crop>`, not used by the labeling
rule) plus a Spearman correlation against GAEZ per crop - agreement ranges from strong (sugarcane 0.80, rice
0.78) to actively negative (soybean -0.17), which is itself a finding, not a bug.

**Engineering note:** while notebooks 09/10 were built, several operations (large windowed raster reads,
Spearman correlations on thousands of points, zonal statistics) kept killing Jupyter kernels. The cause was the
environment, not the libraries: the notebooks were being executed from a shell where the conda environment was
not activated, so its DLL folders (`Library/bin`: MKL and friends) and `GDAL_DATA` / `PROJ_LIB` were missing.
With the environment activated, as PyCharm does, all of these work directly in a notebook. The workarounds left in
notebooks 09/10 and `_gcs2024_zonal_stats.py` (the `gdal_translate` CLI for clipping, a subprocess for the zonal
statistics, a hand-written Spearman formula that matches `scipy.stats.spearmanr`) give the same results and were kept.

## Why this split

Raw geospatial sources arrive in incompatible formats/resolutions/CRS and can't be queried together directly — they're stored as-is (lake) and only reconciled into a shared schema (warehouse/mart) during the ETL pass, once, so every downstream step (EDA, modeling, processing frameworks) reads from one consistent table instead of re-deriving it.
