# Data storage layout

## `raw/` — data lake

Everything landed here exactly as ingested by `scripts/download/`: original formats (GeoTIFF, GeoJSON, SAFE imagery bundles), native resolutions and coordinate systems, no shared schema across sources. Schema-on-read: each notebook in `scripts/preprocessing/` interprets a subset of this on demand. See `data_dictionary.csv` for the full source-by-source breakdown (format, resolution, size, license, structure type).

## `processed/` — data warehouse / mart

Output of the ETL pipeline (`scripts/preprocessing/01`-`04`): reprojected, resampled, and aggregated onto a single 1km analysis grid over Banten (EPSG:32748). Fixed schema, analysis-ready.

- `processed/grid/grid_1km.gpkg` — the 9,333-cell analysis grid (`grid_id` primary key)
- `processed/sentinel2/` — mosaicked Sentinel-2 composites per year
- `processed/features/tabular_features.csv` — the single fact table: one row per grid cell, one column per feature (climate, soil, terrain, land cover, GAEZ suitability), used directly for EDA, modeling, and the Sedona/Spark processing demo (`scripts/preprocessing/06_sedona_processing.ipynb`)

## Why this split

Raw geospatial sources arrive in incompatible formats/resolutions/CRS and can't be queried together directly — they're stored as-is (lake) and only reconciled into a shared schema (warehouse/mart) during the ETL pass, once, so every downstream step (EDA, modeling, processing frameworks) reads from one consistent table instead of re-deriving it.
