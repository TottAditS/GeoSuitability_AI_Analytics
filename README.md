# GeoSuitability AI Analytics

Which crop fits which piece of land in Banten Province, Indonesia? This project brings open geodata into a local data lake
and warehouse on a 1 km analysis grid (9,333 cells), scores ten crops with the FAO/IIASA GAEZ v5 suitability index at 1 km,
describes every cell with Sentinel-2 imagery, and tests with fifteen machine-learning / deep-learning experiments how far
those suitability maps can be reproduced, explained and trusted. A Streamlit dashboard shows all of it: recommendation
maps, per-crop suitability, uncertainty, the data lake and warehouse, exploratory analysis and an interactive AI lab.

## Highlights

- **Per-pixel suitability.** GAEZ v5 at 30 arc-seconds (~1 km) for rice, maize, cassava, soybean, oil palm, rubber,
  sugarcane, coffee, cacao and coconut, with the ~9 km product kept for comparison and the independent Global Crop
  Suitability 2024 product as a cross-check.
- **Honest evaluation.** Every headline score holds out whole ~9 km blocks (pixel-grouped cross-validation): the 14
  physical drivers (climate, soil, terrain) reproduce the 1 km GAEZ index with R² 0.82 (ExtraTrees), while a naive random
  K-fold would report 0.91.
- **Multimodal deep learning.** A fusion network (CNN on the 25 x 25 Sentinel-2 image patch of each cell + MLP on the
  drivers) reaches R² 0.87, the best model of the project; Sentinel-2 helps every crop, most of all on built-up land and
  cropland.
- **Uncertainty and class probabilities.** Locally adaptive conformal 90 % prediction intervals, P(suitable) per crop and
  the probability of every recommendation label for every cell, calibrated on spatially held-out errors (m15).
- **Explainability.** SHAP contributions, feature-group ablations and response curves (m03, AI Lab).

## Data sources

Open products from international providers only (no national statistics). Formats, resolutions, sizes and licence notes
of every source are in [datasets/data_dictionary.csv](datasets/data_dictionary.csv).

| Source | Provider | Native resolution | Used for |
| --- | --- | --- | --- |
| GAEZ v5 suitability index and class | FAO / IIASA | 30 arc-sec (~1 km), 5 arc-min (~9 km) | target, irrigated and rain-fed scenarios |
| Global Crop Suitability 2024 | Zenodo / Scientific Data | ~1 km | independent cross-check (6 crops) |
| SoilGrids 2.0 | ISRIC | 250 m | soil drivers |
| WorldClim v2.1 | WorldClim | ~1 km | climate drivers |
| CHIRPS v2.0 | UCSB Climate Hazards Center | ~5 km | recent rainfall |
| Copernicus DEM | Copernicus / ESA | 30 m | elevation, slope |
| ESA WorldCover | ESA | 10 m | land cover |
| Sentinel-2 L2A | Copernicus (via Google Earth Engine) | 10 m | image features and patches |
| geoBoundaries | geoBoundaries | vector | province boundary |

## Repository layout

```
datasets/                 data lake (raw/) and warehouse (processed/); data are git-ignored, README + data dictionary tracked
scripts/download/         ingestion, one notebook per source
scripts/preprocessing/    ETL notebooks 01-13 (validation, mosaics, 1 km grid, zonal statistics, Apache Sedona, GAEZ
                          scenario and crops, GCS 2024, Sentinel-2 cell features, GAEZ v5 at 1 km, rain-fed audit)
scripts/modeling/         experiments m01-m15, shared libraries exp_lib.py / exp_torch.py, results/ and figures/
scripts/dashboard/        Streamlit dashboard (app.py, one file per page) and its launcher notebook
tests/                    automated tests
tools/                    winutils binaries for Spark on Windows (local mode, no Hadoop install)
```

## Setup

```bash
conda env create -f environment.yml          # Python 3.11, GDAL stack, PyTorch with CUDA 12.8, Spark/Sedona, Streamlit
conda activate geoai-suitability
cp .env.example .env                         # add OPENTOPOGRAPHY_API_KEY (Copernicus DEM download)
```

`requirements.txt` lists the same packages for pip. Sentinel-2 export needs a Google Earth Engine account (interactive
login in `scripts/download/download_sentinel2.ipynb`); Apache Sedona needs Java 17 (JDK) and fetches its JARs once.

## Pipeline

Run the notebooks in order, each on its own (they skip work that is already done):

1. **Ingestion** - `scripts/download/*.ipynb` into `datasets/raw/` (Global Crop Suitability 2024 and CropSuite are
   downloaded by hand from Zenodo; see [scripts/download/README.md](scripts/download/README.md)).
2. **Preprocessing** - `scripts/preprocessing/01` ... `13`: validate the raw data, mosaic Sentinel-2, build the 1 km grid,
   zonal statistics into the fact table `datasets/processed/features/tabular_features.csv`, a first EDA (05), the Apache
   Sedona / Spark job (06), the irrigated scenario and ten crops (07, 08), GCS 2024 (09, 10), Sentinel-2 cell features and
   image patches (11), GAEZ v5 at 1 km (12) and the rain-fed audit columns of every crop (13).
3. **Modeling** - `scripts/modeling/m01` ... `m15`, a few minutes each with at most 6 CPU workers; m09, m10, m14 use the
   GPU when there is one. What each experiment asks is listed in [scripts/modeling/README.md](scripts/modeling/README.md).

## Dashboard

Open `scripts/dashboard/dashboard.ipynb` in PyCharm or Jupyter and run its first cell (a local link appears), or:

```bash
cd scripts/dashboard
streamlit run app.py
```

Pages: Overview, Suitability Map (recommendation, every crop, uncertainty, raw layers), GAEZ x GCS 2024, Data Lake &
Warehouse, Exploratory Analysis, Data Explorer, ML/DL Experiments and the AI Lab (predict any cell with intervals and label
probabilities, what-if sliders, a live model arena and a deep-learning playground with the fusion network). The sidebar sets
the recommendation rule (crops considered, relative or absolute ranking) and the GAEZ resolution (1 km or ~9 km).

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests of the recommendation rule and the experiment library, and headless smoke tests of every dashboard page, view,
section and experiment. Tests that need the data skip themselves when the pipeline has not been run.

## Limitations

- GAEZ is the reference: the models are surrogates of GAEZ built from open drivers, not predictions validated against
  field yields.
- CHIRPS rainfall is ~5 km at the source; the 1 km rainfall climate comes from WorldClim.
- Global Crop Suitability 2024 covers six of the ten crops; the CropSuite archive holds rendered maps only and is not used
  quantitatively.
- The uncertainty describes the model estimate; it is calibrated on spatially held-out blocks, and spatial
  autocorrelation makes its coverage approximate rather than guaranteed.
