# Data storage layout

## `raw/` — data lake

Everything landed here exactly as ingested by `scripts/download/`: original formats (GeoTIFF, GeoJSON, SAFE imagery bundles), native resolutions and coordinate systems, no shared schema across sources. Schema-on-read: each notebook in `scripts/preprocessing/` interprets a subset of this on demand. See `data_dictionary.csv` for the full source-by-source breakdown (format, resolution, size, license, structure type).

## `processed/` — data warehouse / mart

Output of the ETL pipeline (`scripts/preprocessing/01`-`04`): reprojected, resampled, and aggregated onto a single 1km analysis grid over Banten (EPSG:32748). Fixed schema, analysis-ready.

- `processed/grid/grid_1km.gpkg` — the 9,333-cell analysis grid (`grid_id` primary key)
- `processed/sentinel2/` — mosaicked Sentinel-2 composites per year
- `processed/features/tabular_features.csv` — the single fact table: one row per grid cell, one column per feature (climate, soil, terrain, land cover, GAEZ suitability), used directly for EDA, modeling, and the Sedona/Spark processing demo (`scripts/preprocessing/06_sedona_processing.ipynb`)

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

### Note: Global Crop Suitability 2024 is now used as an independent cross-validation source

Previously downloaded but never processed (CropSuite, the other suitability dataset, turned out to be
PNG map renders with no georeferencing at all - not usable quantitatively, see `data_dictionary.csv`).
Global Crop Suitability 2024 *is* a real GeoTIFF source (Zenodo/Nature Scientific Data, independent of GAEZ -
different authors, different methodology, newer baseline). `09_crossvalidate_global_crop_suitability.ipynb`
and `10_gcs2024_zonal_stats_and_compare.ipynb` clip its 6 overlapping crops to Banten and add them as
`gcs2024_<crop>` columns (own 0-1 scale, kept separate from `gaez_index_<crop>`, not used by the labeling
rule) plus a Spearman correlation against GAEZ per crop - agreement ranges from strong (sugarcane 0.80, rice
0.78) to actively negative (soybean -0.17), which is itself a finding, not a bug.

**Engineering note:** on this machine, several GDAL/scipy operations (large windowed raster reads, and
separately `scipy.stats.spearmanr`/`pandas.corr(method="spearman")`/`np.corrcoef` on several-thousand-element
arrays) reliably or intermittently kill the Jupyter kernel when run as notebook code directly, while the
exact same logic runs fine as a plain Python process. Notebooks 09/10 and `_gcs2024_zonal_stats.py` work
around this (clip via the `gdal_translate` CLI instead of a rasterio windowed read; run zonal statistics in a
subprocess; compute Spearman's rho from its rank-difference formula by hand instead of calling into scipy).

## Why this split

Raw geospatial sources arrive in incompatible formats/resolutions/CRS and can't be queried together directly — they're stored as-is (lake) and only reconciled into a shared schema (warehouse/mart) during the ETL pass, once, so every downstream step (EDA, modeling, processing frameworks) reads from one consistent table instead of re-deriving it.
