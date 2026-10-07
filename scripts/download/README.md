# Dataset download notebooks

Everything here is a self-contained **Jupyter Notebook** (`.ipynb`), one file per dataset, meant to
be run cell by cell so you can watch progress as it happens rather than staring at a blank terminal.

## Setup

This project uses a dedicated conda env, `geoai-suitability` (Python 3.11, PyTorch 2.11 + CUDA 12.8
for an RTX 5060, plus GDAL/rasterio/geopandas/scikit-learn/XGBoost/SHAP from conda-forge). The full
spec lives in `environment.yml` at the project root.

```bash
# Rebuild the env from scratch (new machine, or if it got deleted):
conda env create -f environment.yml
conda activate geoai-suitability

# Set up credentials for the notebooks that need an account:
cp .env.example .env
# fill in OPENTOPOGRAPHY_API_KEY (for the DEM) in .env
# (Sentinel-2 uses an interactive GEE browser login instead, not .env - see that notebook)
```

The `geoai-suitability` Jupyter kernel is already registered. If it doesn't show up in PyCharm/Jupyter:

```bash
conda run -n geoai-suitability python -m ipykernel install --user --name geoai-suitability
```

In PyCharm: Settings > Project > Python Interpreter > Add Interpreter > Conda Environment >
Existing environment, point it at `<conda_install_path>\envs\geoai-suitability\python.exe`.

## Notebooks

| Notebook | Dataset | Needs credentials? |
|---|---|---|
| `download_geoboundaries.ipynb` | Admin boundaries (Banten AOI) | No |
| `download_worldclim.ipynb` | WorldClim (BIO1/12/15) + raster preview | No |
| `download_chirps.ipynb` | CHIRPS monthly rainfall, 2020-2024 | No |
| `download_worldcover.ipynb` | ESA WorldCover 10m | No |
| — (manual download) | Global Crop Suitability 2024 (https://zenodo.org/records/17666722) and CropSuite, both from Zenodo, saved by hand into `datasets/raw/global_crop_suitability_2024/` and `datasets/raw/cropsuite/` | No |
| `download_soilgrids.ipynb` | SoilGrids (pH, SOC, N, clay, sand, CEC, bdod) | No |
| `download_opentopography_dem.ipynb` | Copernicus DEM 30m | Yes — `OPENTOPOGRAPHY_API_KEY` |
| `download_sentinel2.ipynb` | Sentinel-2 L2A via Google Earth Engine (annual dry-season composites, 2020-2024, + NDVI/EVI/NDMI) | Yes — a GEE account (interactive browser login on first run) |
| `download_gaez.ipynb` | GAEZ v5 Suitability Class + Index at ~9 km (5 arc-min) for the first four crops — straight from a public Google Cloud Storage bucket. The irrigated scenario and six more crops are fetched by `scripts/preprocessing/07`/`08`; the 1 km (30 arc-sec) version is read for Banten by `scripts/preprocessing/12_gaez_v5_1km.ipynb` | No |

## Usage

1. Open the `.ipynb` file you want in PyCharm/Jupyter
2. Make sure the kernel (top right) is set to **geoai-suitability**
3. Run cells top to bottom (Shift+Enter) — `tqdm` progress bars and log output show up right under
   each cell
4. Every notebook skips files that already exist, so it's safe to re-run or resume after an
   interruption (e.g. a dropped connection)

Downloads land in `datasets/raw/<dataset_name>/`. Raw data is gitignored on purpose - it's large and
easy to regenerate from these notebooks, no need to version it.

`common.py` stays a plain `.py` file (not a notebook) since it's just shared helper code (the Banten
bbox, a download-with-progress helper, logging setup) imported by every notebook via
`from common import ...` — it isn't a download step on its own.

## What isn't used

- **National government statistics and portals** (e.g. BPS crop production statistics) are deliberately
  left out: the project relies only on open international products. GAEZ is cross-checked against the
  independent Global Crop Suitability 2024 product instead (`scripts/preprocessing/09`/`10`).

## Sentinel-2 / GEE notes

`download_sentinel2.ipynb` uses `Export.image.toDrive` — the compositing and export run **async on
Google's servers**, not in the notebook itself. After running all the cells:
1. Check progress with the status cell in the notebook (safe to re-run), or at
   https://code.earthengine.google.com/tasks
2. Once a task shows `COMPLETED`, download the `.tif` from the `GeoSuitability_Sentinel2` folder in
   Google Drive
3. Move it into `datasets/raw/sentinel2gee/`

Google Drive's free tier is only 15GB, and a handful of these composites can fill that up fast — if
an export fails with "not enough space", download and delete the completed years from Drive first,
then re-run just the years that failed.

## Adjusting area / dates

- The Banten bounding box lives in `common.py` (`BANTEN_BBOX`) — swap it for the filtered AOI from
  `download_geoboundaries.ipynb` if you need more precision.
- CHIRPS year range: edit the `YEAR_START`/`YEAR_END` constants in `download_chirps.ipynb`.
- Sentinel-2 year range, season, cloud cover: edit `YEAR_START`/`YEAR_END`/`DRY_SEASON`/`CLOUD_COVER_MAX`
  in `download_sentinel2.ipynb`.
- GAEZ v5 crops: edit the `CROPS` dict in `download_gaez.ipynb` (more crop codes are listed in the
  Excel file linked from that notebook).
