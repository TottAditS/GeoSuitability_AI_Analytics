"""Shared pieces of the dashboard: paths, constants, the column catalogue, cached loaders and plot helpers.

Every page imports this module as `C`. Nothing here trains a model or touches the large rasters, so
importing it is cheap; the heavy work lives in ai_models.py and only runs when the AI Lab asks for it.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
RAW_DIR = DATASETS / "raw"
PROCESSED_DIR = DATASETS / "processed"
FEATURES_CSV = PROCESSED_DIR / "features" / "tabular_features.csv"
GCS_CSV = PROCESSED_DIR / "features" / "gcs2024_zonal_stats.csv"
S2_STATS_CSV = PROCESSED_DIR / "features" / "sentinel2_cell_stats.csv"
GRID_GPKG = PROCESSED_DIR / "grid" / "grid_1km.gpkg"
SPARK_DIR = PROCESSED_DIR / "spark_output"
DATA_DICT = DATASETS / "data_dictionary.csv"
VALIDATION_REPORT = RAW_DIR / "_validation_report.csv"
MODELING_DIR = ROOT / "scripts" / "modeling"
RESULTS_DIR = MODELING_DIR / "results"
FIGURES_DIR = MODELING_DIR / "figures"
OOF_DIR = PROCESSED_DIR / "experiments"
MODELS_DIR = PROCESSED_DIR / "models"

CROPS = ["rice", "maize", "cassava", "soybean", "oilpalm", "rubber", "sugarcane", "coffee", "cacao", "coconut"]
GCS_CROPS = ["rice", "maize", "cassava", "soybean", "oilpalm", "sugarcane"]  # crops covered by both products
CROP_LABELS = {
    "rice": "Rice", "maize": "Maize", "cassava": "Cassava", "soybean": "Soybean",
    "oilpalm": "Oil palm", "rubber": "Rubber", "sugarcane": "Sugarcane",
    "coffee": "Coffee", "cacao": "Cacao", "coconut": "Coconut",
}
LABEL_TO_CROP = {v: k for k, v in CROP_LABELS.items()}

# Thresholds match the rule-based labeling logic in the written report (Section 4.2.3) exactly.
NOT_RECOMMENDED_THRESHOLD = 5000
TIE_THRESHOLD = 500

# --------------------------------------------------------------------------------------------
# Visual identity: natural greens on paper white, earth tones for contrast (mirrors .streamlit/config.toml)
# --------------------------------------------------------------------------------------------
FOREST, PINE, FERN, MOSS, SAGE, MIST, PAPER = "#14281D", "#2F5D3A", "#4A7D42", "#7FA35E", "#A9BF92", "#E3EDD9", "#F7F8F3"
OCHRE, CLAY, BARK, TEAL, STONE, INK, MUTED = "#C49A3A", "#B8653F", "#6B4F3A", "#3E7F7A", "#DAD7CC", "#1D2B22", "#5E6E62"


def _scale(colors):
    return [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]


SEQ = _scale(["#F4F7EF", "#DDE8CF", "#BFD4A7", "#97BB7E", "#6E9E5A", "#4A7D42", "#2F5D3A", "#1B3A2A"])    # low -> high
DIV = _scale(["#8E3B24", "#B8653F", "#D99A72", "#EFDCC5", "#F4F1E8", "#C9DDB5", "#8DB474", "#4F8A45", "#23492F"])  # - clay / + green
DIV_ERR = _scale(["#245F66", "#73B0A6", "#CFE3DD", "#F4F1E8", "#EBC9A9", "#C4825A", "#8E3B24"])  # under teal / over clay
SEQ_CLAY = _scale(["#F6F1E7", "#EBD6BC", "#D9AE87", "#C4825A", "#A65A39", "#7F3B22"])   # losses
SEQ_EARTH = _scale(["#F3EFE2", "#E0D3AE", "#C5B07A", "#9C8B55", "#6F6A3E", "#3F4A2B"])  # soil, terrain
SEQ_WATER = _scale(["#F2F6F2", "#D3E6E0", "#A6CFC5", "#73B0A6", "#46898A", "#245F66"])  # rainfall
SEQ_WARM = _scale(["#F7F1E1", "#EBD9A8", "#D9B76E", "#C4924A", "#A86A35", "#7F4A24"])   # temperature
EXTRA = ["#8FA98A", "#C7B98F", "#7C9C9A", "#B5A27A", "#A4B4A0", "#9C8E7A", "#C9C3A6", "#6F8F7F", "#B9A58B", "#869C75",
         "#D2C6A5", "#93A7A0"]
MAP_STYLE = "carto-positron"

# Crop colours: each crop gets a tone that recalls the plant (paddy gold, palm green, coffee brown, coastal teal ...).
LABEL_COLORS = {
    "Rice": "#D6B13F", "Maize": "#E8D28A", "Cassava": "#B88A55", "Soybean": "#9DBA6A",
    "Oil palm": "#1F4D2E", "Rubber": "#7A8B4E", "Sugarcane": "#5E9A55", "Coffee": "#5B3A29",
    "Cacao": "#9A4F33", "Coconut": "#3E7F7A", "Not recommended": "#DAD7CC",
    "No cropland / water": "#D5DEDD",
}

# ESA WorldCover class codes (fixed codebook, https://esa-worldcover.org/).
WORLDCOVER_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
    80: "Permanent water bodies", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss and lichen",
}
# GAEZ v5 suitability class codes, from FAO's own legends (map styles SIX and SCX): the ~9 km product stops at
# 9 = water; the 1 km product adds 9 = no cropland (land GAEZ leaves out for that crop and scenario) and 10 = water.
GAEZ_CLASSES = {
    1: "Very high", 2: "High", 3: "Good", 4: "Medium", 5: "Moderate",
    6: "Marginal", 7: "Very marginal", 8: "Not suitable", 9: "Water",
}
GAEZ_CLASSES_1KM = {**{k: v for k, v in GAEZ_CLASSES.items() if k <= 8}, 9: "No cropland", 10: "Water"}

# Variable groups used across the EDA, the AI Lab and the column catalogue.
FEATURE_GROUPS = {
    "Climate": ["bio1_temp_mean_c", "bio12_precip_annual_mm", "bio15_precip_seasonality",
                "chirps_annual_mean_mm", "chirps_annual_std_mm"],
    "Soil": ["soil_ph", "soil_soc", "soil_nitrogen", "soil_clay_pct", "soil_sand_pct", "soil_cec", "soil_bulk_density"],
    "Terrain": ["elevation_mean_m", "slope_mean_deg"],
    "Land cover": ["cropland_pct"],
    "GAEZ index (1 km)": [f"gaez1k_index_{c}" for c in CROPS],
    "GAEZ index (~9 km)": [f"gaez_index_{c}" for c in CROPS],
    "GCS 2024": [f"gcs2024_{c}" for c in GCS_CROPS],
}
BASE_FEATURES = FEATURE_GROUPS["Climate"] + FEATURE_GROUPS["Soil"] + FEATURE_GROUPS["Terrain"]

_SOIL = {  # SoilGrids 2.0, 0-5 cm, values kept in the native integer units of the product
    "soil_ph": ("Soil pH in water", "pH x 10", "phh2o"),
    "soil_soc": ("Soil organic carbon", "dg/kg", "soc"),
    "soil_nitrogen": ("Total nitrogen", "cg/kg", "nitrogen"),
    "soil_clay_pct": ("Clay content (column name says pct; values are g/kg)", "g/kg", "clay"),
    "soil_sand_pct": ("Sand content (column name says pct; values are g/kg)", "g/kg", "sand"),
    "soil_cec": ("Cation exchange capacity", "mmol(c)/kg", "cec"),
    "soil_bulk_density": ("Bulk density of the fine earth", "cg/cm3", "bdod"),
}
_FIXED_META = {
    "grid_id": ("Key", "1 km grid (this project)", "", "Primary key of the 1 km analysis grid"),
    "row": ("Grid index", "1 km grid (this project)", "", "Grid row, 0 = southern edge"),
    "col": ("Grid index", "1 km grid (this project)", "", "Grid column, 0 = western edge"),
    "lon": ("Location", "1 km grid (this project)", "degrees", "Cell centroid longitude (WGS84)"),
    "lat": ("Location", "1 km grid (this project)", "degrees", "Cell centroid latitude (WGS84)"),
    "bio1_temp_mean_c": ("Climate", "WorldClim v2.1", "deg C", "Annual mean temperature (BIO1, 1970-2000)"),
    "bio12_precip_annual_mm": ("Climate", "WorldClim v2.1", "mm", "Annual precipitation (BIO12, 1970-2000)"),
    "bio15_precip_seasonality": ("Climate", "WorldClim v2.1", "CV %", "Precipitation seasonality (BIO15, coefficient of variation)"),
    "chirps_annual_mean_mm": ("Climate", "CHIRPS v2.0", "mm", "Mean annual precipitation 2020-2024"),
    "chirps_annual_std_mm": ("Climate", "CHIRPS v2.0", "mm", "Year-to-year standard deviation of annual precipitation 2020-2024"),
    "elevation_mean_m": ("Terrain", "Copernicus DEM", "m", "Mean elevation of the cell"),
    "slope_mean_deg": ("Terrain", "Copernicus DEM", "degrees", "Mean slope of the cell (derived from the DEM)"),
    "landcover_majority": ("Land cover", "ESA WorldCover", "class code", "Most frequent WorldCover class in the cell"),
    "cropland_pct": ("Land cover", "ESA WorldCover", "%", "Share of the cell mapped as cropland (class 40)"),
    "soil_data_interpolated": ("Data quality flag", "This project (ETL)", "bool", "True where SoilGrids had no value and soil_* was interpolated"),
    "recommended_crop": ("Derived (dashboard)", "Rule on GAEZ indices", "label",
                         "Recommended crop under the sidebar rule (crops considered, ranking, GAEZ resolution)"),
    "landcover_label": ("Derived (dashboard)", "ESA WorldCover", "label", "Name of the majority land-cover class"),
}


_S2_BANDS = {"B2": "Blue (B2)", "B3": "Green (B3)", "B4": "Red (B4)", "B8": "Near-infrared (B8)",
             "B11": "Short-wave infrared 1 (B11)", "B12": "Short-wave infrared 2 (B12)",
             "NDVI": "NDVI (vegetation greenness)", "EVI": "EVI (enhanced vegetation index)", "NDMI": "NDMI (vegetation moisture)"}


def _s2_meta(name: str) -> tuple[str, str]:
    """Unit and description of one Sentinel-2 cell statistic (notebook 11; every value is the mean of the five years)."""
    if name == "valid_frac":
        return "share", "Share of the cell's 10 m pixels with a cloud-free value"
    if name.endswith("_iav"):
        band = name.split("_")[0]
        return "index", f"Year-to-year standard deviation of the cell's mean {band}, 2020-2024"
    band, stat = name.rsplit("_", 1)
    unit = "reflectance 0-1" if band.startswith("B") else "index"
    if stat in ("p10", "p50", "p90"):
        return unit, f"{stat[1:]}th percentile of {_S2_BANDS.get(band, band)} within the cell"
    what = "mean" if stat == "mean" else "standard deviation"
    return unit, f"{_S2_BANDS.get(band, band)}, {what} over the cell's pixels"


def column_catalogue(columns) -> pd.DataFrame:
    """One row per column: group, source dataset, unit and a plain-language description."""
    rows = []
    for col in columns:
        if col in _FIXED_META:
            group, source, unit, desc = _FIXED_META[col]
        elif col.endswith("_raw") and col.removesuffix("_raw") in _SOIL:
            name, unit, var = _SOIL[col.removesuffix("_raw")]
            group, source, desc = "Soil (raw, audit)", "SoilGrids 2.0", f"{name}, before gap-filling (NaN where SoilGrids had no value)"
        elif col in _SOIL:
            name, unit, var = _SOIL[col]
            group, source, desc = "Soil", "SoilGrids 2.0", f"{name} (SoilGrids '{var}', 0-5 cm, gap-filled)"
        elif col.startswith("gaez1k_index_"):
            crop = col.removeprefix("gaez1k_index_").removesuffix("_rainfed")
            rainfed = col.endswith("_rainfed")
            group = "GAEZ index (rain-fed 1 km, audit)" if rainfed else "GAEZ index (1 km)"
            source, unit = "GAEZ v5 (FAO / IIASA), 30 arc-sec", "0-10,000"
            desc = (f"{CROP_LABELS.get(crop, crop)} suitability index at ~1 km, {'rain-fed' if rainfed else 'irrigated'} "
                    "high-input scenario (mean of the cell's land pixels; empty where GAEZ marks no cropland or water)")
        elif col.startswith("gaez1k_class_"):
            crop = col.removeprefix("gaez1k_class_").removesuffix("_rainfed")
            rainfed = col.endswith("_rainfed")
            group = "GAEZ class (rain-fed 1 km, audit)" if rainfed else "GAEZ class (1 km)"
            source, unit = "GAEZ v5 (FAO / IIASA), 30 arc-sec", "class 1-10"
            desc = (f"{CROP_LABELS.get(crop, crop)} suitability class at ~1 km, {'rain-fed' if rainfed else 'irrigated'} "
                    "(1 = very high ... 8 = not suitable, 9 = no cropland, 10 = water)")
        elif col.startswith("gaez_index_"):
            crop = col.removeprefix("gaez_index_").removesuffix("_rainfed")
            rainfed = col.endswith("_rainfed")
            group = "GAEZ index (rain-fed ~9 km, audit)" if rainfed else "GAEZ index (~9 km)"
            source, unit = "GAEZ v5 (FAO / IIASA), 5 arc-min", "0-10,000"
            desc = f"{CROP_LABELS.get(crop, crop)} suitability index at ~9 km, {'rain-fed' if rainfed else 'irrigated'} high-input scenario"
        elif col.startswith("gaez_class_"):
            crop = col.removeprefix("gaez_class_").removesuffix("_rainfed")
            rainfed = col.endswith("_rainfed")
            group = "GAEZ class (rain-fed ~9 km, audit)" if rainfed else "GAEZ class (~9 km)"
            source, unit = "GAEZ v5 (FAO / IIASA), 5 arc-min", "class 1-9"
            desc = f"{CROP_LABELS.get(crop, crop)} suitability class at ~9 km (1 = very high ... 8 = not suitable, 9 = water)"
        elif col.startswith("gcs2024_"):
            crop = col.removeprefix("gcs2024_")
            group, source, unit = "GCS 2024", "Global Crop Suitability 2024", "0-1"
            desc = f"{CROP_LABELS.get(crop, crop)} suitability from an independent model (1 km)"
        elif col.startswith("s2_"):
            group, source = "Sentinel-2", "Sentinel-2 L2A (Copernicus), 2020-2024"
            unit, desc = _s2_meta(col.removeprefix("s2_"))
        else:
            group, source, unit, desc = "Other", "", "", ""
        rows.append({"column": col, "group": group, "source": source, "unit": unit, "description": desc})
    return pd.DataFrame(rows)


def nice(col: str) -> str:
    """Short human-readable name for axis titles."""
    cat = column_catalogue([col]).iloc[0]
    unit = f" ({cat['unit']})" if cat["unit"] and cat["unit"] not in ("label", "bool") else ""
    if cat["group"].startswith("GAEZ index") or cat["group"] == "GCS 2024":
        return f"{cat['description'].split(' suitability')[0]} - {cat['group']}{unit}"
    if cat["group"] in ("Climate", "Terrain", "Land cover", "Soil"):
        return cat["description"].split(" (")[0] + unit
    return col


# --------------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------------
CROP_GROUPS = {
    "All 10 crops": CROPS,
    "Food crops": ["rice", "maize", "cassava", "soybean"],
    "Tree & estate crops": ["oilpalm", "rubber", "sugarcane", "coffee", "cacao", "coconut"],
}
RANKINGS = ["Relative", "Absolute"]
DEFAULT_GROUP, DEFAULT_RANKING = "All 10 crops", "Relative"


def recommend(df: pd.DataFrame, crops=CROPS, ranking: str = "Absolute", reference: pd.DataFrame | None = None,
              res: str = "~9 km") -> pd.DataFrame:
    """Recommended crop per cell, plus what the maps need to shade it.

    Absolute (the rule of the written report): the highest GAEZ index wins; if the runner-up is within 500 points the
        label names both ("A / B (tied)"); 'Not recommended' when even the best index is below 5,000.
    Relative: among the crops whose index reaches 5,000, the one that is strongest relative to its own range in
        Banten (percentile of its index across all cells) wins - comparative advantage. Crops with generally lower
        indices, such as rice, then show where they are at their best instead of always losing to coconut or oil palm.
    `reference` is the table the percentiles come from (default: df itself); `res` picks the GAEZ product (~9 km or
    1 km). At 1 km GAEZ leaves no value where it marks no cropland or water: such a crop cannot win there, and a cell
    where no crop has a value is labelled "No cropland / water".
    Returns label, winner (crop key or None), winner_index (0-10,000) and margin (0-1, how clear-cut the win is).
    """
    cols = [gaez_col(c, "index", res) for c in crops]
    raw = df[cols].to_numpy(float)
    missing = np.isnan(raw)
    idx = np.where(missing, -1.0, raw)
    rows = np.arange(len(df))
    if ranking == "Relative":
        ref = df if reference is None else reference
        refs = [np.sort(ref[c].dropna().to_numpy(float)) for c in cols]
        score = np.column_stack([np.searchsorted(r_, idx[:, j], side="right") / max(len(r_), 1) for j, r_ in enumerate(refs)])
        score = np.where(idx >= NOT_RECOMMENDED_THRESHOLD, score, -1.0)
    else:
        score = idx
    order = np.argsort(score, axis=1)
    best = order[:, -1]
    second = order[:, -2] if len(crops) > 1 else best
    b_score, s_score = score[rows, best], score[rows, second]
    ok = idx[rows, best] >= NOT_RECOMMENDED_THRESHOLD
    labels = np.array([CROP_LABELS[crops[i]] for i in best], dtype=object)
    if ranking == "Absolute" and len(crops) > 1:
        tie = (b_score - s_score) <= TIE_THRESHOLD
        labels[tie] = [" / ".join(sorted((CROP_LABELS[crops[i]], CROP_LABELS[crops[j]]))) + " (tied)"
                       for i, j in zip(best[tie], second[tie])]
        margin = (b_score - s_score) / 2500
    else:
        margin = np.where(s_score >= 0, b_score - s_score, 1.0) / 0.3
    labels[~ok] = "Not recommended"  # highest priority
    labels[missing.all(axis=1)] = "No cropland / water"
    return pd.DataFrame({"label": labels, "winner": np.where(ok, np.array(crops, dtype=object)[best], None),
                         "winner_index": raw[rows, best], "margin": np.clip(margin, 0, 1)}, index=df.index)


def label_cells(df: pd.DataFrame) -> pd.Series:
    """The labelling rule of the written report (absolute, all ten crops)."""
    return recommend(df)["label"].rename("recommended_crop")


GAEZ_RES = ["1 km", "~9 km"]
DEFAULT_RES = "1 km"


def gaez_res() -> str:
    """GAEZ product chosen in the sidebar: '1 km' (30 arc-sec, one value per pixel) or '~9 km' (5 arc-min)."""
    return st.session_state.get("gaez_res") or DEFAULT_RES


def gaez_col(crop: str, kind: str = "index", res: str | None = None) -> str:
    """Warehouse column of one crop's GAEZ index or class at the chosen (or given) resolution."""
    return f"gaez1k_{kind}_{crop}" if (res or gaez_res()) == "1 km" else f"gaez_{kind}_{crop}"


def gaez_classes(res: str | None = None) -> dict:
    return GAEZ_CLASSES_1KM if (res or gaez_res()) == "1 km" else GAEZ_CLASSES


def fmt_index(values: pd.Series) -> pd.Series:
    return values.map(lambda v: "-" if pd.isna(v) else f"{v:,.0f}")


def rule() -> tuple[str, str]:
    """The recommendation rule chosen in the sidebar (crop group, ranking)."""
    return (st.session_state.get("rule_group") or DEFAULT_GROUP, st.session_state.get("rule_ranking") or DEFAULT_RANKING)


def rule_controls():
    st.sidebar.markdown('<div class="gs-side-label">Recommendation rule</div>', unsafe_allow_html=True)
    st.sidebar.selectbox("Crops considered", list(CROP_GROUPS), key="rule_group")
    st.sidebar.radio("Ranking", RANKINGS, key="rule_ranking", horizontal=True,
                     help="Relative: each crop is compared with its own range in Banten, so every crop that is suitable "
                          "somewhere shows where it is at its best. Absolute: the highest GAEZ index wins (the rule of "
                          "the written report).")
    st.sidebar.radio("GAEZ resolution", GAEZ_RES, key="gaez_res", horizontal=True,
                     help="1 km: GAEZ v5 published at 30 arc-seconds, one value per pixel. ~9 km: the 5 arc-minute "
                          "version used by the first analyses (experiment m13 compares the two).")


def rule_description() -> str:
    group, ranking = rule()
    res = gaez_res()
    extra = (" Cells GAEZ marks as no cropland or water are labelled so." if res == "1 km" else "")
    if ranking == "Relative":
        return (f"**{group}, relative ranking, GAEZ {res}.** Among the crops whose GAEZ index reaches 5,000 in a cell, the one that is "
                "strongest compared with its own range across Banten wins (comparative advantage). Cells where no crop "
                "reaches 5,000 are 'Not recommended'." + extra)
    return (f"**{group}, absolute ranking, GAEZ {res}** (the rule of the written report). The crop with the highest GAEZ index wins; "
            "if the runner-up is within 500 points both are named ('A / B (tied)'); 'Not recommended' when even the best "
            "index is below 5,000." + extra)


@st.cache_data(show_spinner=False)
def fact_columns() -> list[str]:
    """Columns of the fact table itself (tabular_features.csv), without the joined Sentinel-2 table."""
    return list(pd.read_csv(FEATURES_CSV, nrows=0).columns)


@st.cache_data(show_spinner=False)
def _load_base() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    if S2_STATS_CSV.exists():   # Sentinel-2 cell statistics (notebook 11): a separate 1:1 table keyed by grid_id
        df = df.merge(pd.read_csv(S2_STATS_CSV), on="grid_id", how="left").copy()   # copy() consolidates the merged frame
    df["landcover_label"] = df["landcover_majority"].round().astype("Int64").map(WORLDCOVER_CLASSES).fillna("Unknown")
    return df


@st.cache_data(show_spinner=False)
def _recommendation(group: str, ranking: str, res: str) -> pd.DataFrame:
    return recommend(_load_base(), CROP_GROUPS[group], ranking, res=res)


def recommendation() -> pd.DataFrame:
    """Recommendation under the sidebar rule, aligned with load_data()."""
    return _recommendation(*rule(), gaez_res())


def load_data() -> pd.DataFrame:
    """The warehouse table plus two dashboard labels: the recommended crop (sidebar rule) and the land-cover name.
    Also gives every tie label a fixed colour, so a label looks the same on every map and chart."""
    df = _load_base()
    df["recommended_crop"] = recommendation()["label"].to_numpy()
    for i, label in enumerate(sorted(set(df["recommended_crop"]) - set(LABEL_COLORS))):
        LABEL_COLORS[label] = EXTRA[i % len(EXTRA)]
    return df


def recommendation_options(df: pd.DataFrame) -> list[str]:
    """Labels that actually occur in the data, 'Not recommended' last."""
    return sorted(df["recommended_crop"].unique(), key=lambda x: (x in ("Not recommended", "No cropland / water"), x))


@st.cache_data(show_spinner=False)
def load_grid_geojson() -> dict:
    """1 km grid cell polygons, reprojected to WGS84, for the basemap choropleths."""
    grid = gpd.read_file(GRID_GPKG)[["grid_id", "geometry"]].to_crs("EPSG:4326")
    return json.loads(grid.to_json())


@st.cache_data(show_spinner=False)
def load_data_dictionary() -> pd.DataFrame:
    return pd.read_csv(DATA_DICT)


@st.cache_data(show_spinner=False)
def load_result(name: str) -> pd.DataFrame | None:
    """A results table written by scripts/modeling (None if that notebook has not been run)."""
    path = RESULTS_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def figure_path(name: str) -> pathlib.Path | None:
    path = FIGURES_DIR / f"{name}.png"
    return path if path.exists() else None


def _scan(folder: pathlib.Path) -> dict:
    files = [f for f in folder.rglob("*") if f.is_file()]
    exts = Counter(f.suffix.lower() or "(none)" for f in files)
    return {"files": len(files), "size_bytes": sum(f.stat().st_size for f in files),
            "file_types": ", ".join(f"{e} x{n}" for e, n in exts.most_common(4))}


@st.cache_data(ttl=3600, show_spinner="Scanning the data lake...")
def lake_inventory() -> pd.DataFrame:
    """What is physically on disk under datasets/raw (folders and loose files)."""
    rows = []
    for p in sorted(RAW_DIR.iterdir()):
        info = _scan(p) if p.is_dir() else {"files": 1, "size_bytes": p.stat().st_size, "file_types": p.suffix.lower()}
        rows.append({"item": p.name, "kind": "folder" if p.is_dir() else "file",
                     "local_path": f"datasets/raw/{p.name}/" if p.is_dir() else f"datasets/raw/{p.name}", **info})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner="Scanning the processed layer...")
def processed_inventory() -> pd.DataFrame:
    rows = []
    for p in sorted(PROCESSED_DIR.iterdir()):
        if p.is_dir():
            rows.append({"item": p.name, "local_path": f"datasets/processed/{p.name}/", **_scan(p)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------------------------
MAP_BOUNDS = dict(west=104.95, east=106.93, south=-7.18, north=-5.66)
MAP_CENTER = dict(lat=-6.42, lon=105.94)


def lock_banten_view(fig, zoom: float = 8.2):
    """Lock a basemap figure to the Banten grid extent (never a default world view)."""
    fig.update_maps(center=MAP_CENTER, zoom=zoom, bounds=MAP_BOUNDS)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))


@st.cache_data(show_spinner=False)
def grid_shape() -> tuple[int, int]:
    d = pd.read_csv(FEATURES_CSV, usecols=["row", "col"])
    return int(d["row"].max()) + 1, int(d["col"].max()) + 1


def to_grid(df: pd.DataFrame, values) -> np.ndarray:
    """Values of the given cells placed on the full 1 km grid (NaN elsewhere); row 0 is the south."""
    z = np.full(grid_shape(), np.nan)
    z[df["row"].to_numpy(), df["col"].to_numpy()] = np.asarray(values, dtype=float)
    return z


def _grid_layout(fig, title, height):
    fig.update_xaxes(visible=False, constrain="domain")
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_layout(title=title, height=height, margin=dict(l=0, r=0, t=40 if title else 10, b=0),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def grid_map(df: pd.DataFrame, values, title: str | None = None, colorscale=None, zmin=None, zmax=None,
             height: int = 420, colorbar_title: str = "", fmt: str = ".2f"):
    """Lightweight raster-style map of one value per cell (no basemap, ~100x lighter than a choropleth)."""
    z = to_grid(df, values)
    lon, lat = to_grid(df, df["lon"]), to_grid(df, df["lat"])
    fig = go.Figure(go.Heatmap(
        z=z, customdata=np.dstack([lon, lat]), colorscale=colorscale or SEQ, zmin=zmin, zmax=zmax,
        colorbar=dict(title=colorbar_title, thickness=10, outlinewidth=0), hoverongaps=False,
        hovertemplate=f"lon %{{customdata[0]:.3f}}, lat %{{customdata[1]:.3f}}<br>value %{{z:{fmt}}}<extra></extra>"))
    return _grid_layout(fig, title, height)


def _mix(color: str, toward: str = "#FFFFFF", t: float = 0.75) -> str:
    """Blend a hex colour toward another one (t = 0 keeps it, t = 1 replaces it)."""
    a = [int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(toward.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


FLAT_LABELS = {"Not recommended", "No GCS value", "No cropland / water"}


def category_scale(labels: pd.Series, colors: dict | None = None, strength=None):
    """Categories, per-cell z values and a stepped colour scale for a categorical map.

    Without `strength` every category is one flat colour. With `strength` (0-1 per cell) every category becomes its
    own light-to-full ramp: the hue says which label, the depth says how strongly (labels in FLAT_LABELS stay flat)."""
    cats = [c for c in (colors or {}) if c in set(labels)] + sorted(set(labels) - set(colors or {}))
    extra = iter(EXTRA * 3)
    col_of = {c: (colors or {}).get(c) or next(extra) for c in cats}
    code = labels.map({c: i for i, c in enumerate(cats)}).to_numpy(dtype=float)
    if strength is None:
        z = code + 0.5
    else:
        st_ = np.clip(np.nan_to_num(np.asarray(strength, dtype=float), nan=0.6), 0.0, 0.999)
        z = code + np.where(labels.isin(FLAT_LABELS).to_numpy(), 0.5, st_)
    scale = []
    n = max(len(cats), 1)
    for i, c in enumerate(cats):
        light = col_of[c] if strength is None or c in FLAT_LABELS else _mix(col_of[c], "#FFFFFF", 0.78)
        scale += [[i / n, light], [(i + 1) / n, col_of[c]]]
    return cats, z, scale or [[0, "#ccc"], [1, "#ccc"]], [i + 0.5 for i in range(len(cats))], n


def category_map(df: pd.DataFrame, labels: pd.Series, colors: dict | None = None, title: str | None = None,
                 height: int = 420, strength=None, hover=None):
    """Raster-style map of a categorical label per cell (optionally with colour depth), legend as a colour bar."""
    cats, zc, scale, ticks, n = category_scale(labels, colors, strength)
    z = to_grid(df, zc)
    text = np.full(z.shape, "", dtype=object)
    text[df["row"].to_numpy(), df["col"].to_numpy()] = np.asarray(labels if hover is None else hover, dtype=object)
    fig = go.Figure(go.Heatmap(
        z=z, text=text, colorscale=scale, zmin=0, zmax=n, hoverongaps=False,
        colorbar=dict(tickvals=ticks, ticktext=cats, title="", thickness=10, outlinewidth=0),
        hovertemplate="%{text}<extra></extra>"))
    return _grid_layout(fig, title, height)


def category_choropleth(df: pd.DataFrame, labels: pd.Series, geojson: dict, colors: dict | None = None, strength=None,
                        hover=None, height: int = 650):
    """The same categorical map on a basemap: one 1 km square per cell, hue = label, depth = strength."""
    cats, z, scale, ticks, n = category_scale(labels, colors, strength)
    fig = go.Figure(go.Choroplethmap(
        geojson=geojson, locations=df["grid_id"], featureidkey="properties.grid_id", z=z, colorscale=scale, zmin=0,
        zmax=n, marker_line_width=0, marker_opacity=0.88, text=np.asarray(labels if hover is None else hover, dtype=object),
        hovertemplate="%{text}<extra></extra>",
        colorbar=dict(tickvals=ticks, ticktext=cats, title="", thickness=10, outlinewidth=0)))
    fig.update_layout(height=height, map=dict(style=MAP_STYLE))
    lock_banten_view(fig)
    return fig


# ---------------------------------------------------------------------------------------------- per-crop values
SOURCES = ["GAEZ 1 km (30 arc-sec)", "GAEZ ~9 km (as published)", "GAEZ ~9 km, smoothed (display only)",
           "Model estimate (out-of-fold)", "GCS 2024 (1 km, 6 crops)"]
SOURCE_NOTES = {
    SOURCES[0]: "GAEZ v5 suitability index published at 30 arc-seconds: a real value for every 1 km pixel (empty where "
                "GAEZ marks no cropland or water).",
    SOURCES[1]: "GAEZ v5 index at 5 arc-minutes, as used by the first analyses: one value per ~9 km GAEZ pixel, hence the blocks.",
    SOURCES[2]: "The ~9 km values with a Gaussian blur (sigma 3 km) so block edges fade; a display aid only, nothing new is added.",
    SOURCES[3]: "What the 14 physical drivers alone predict for each cell: ExtraTrees trained on the 1 km GAEZ, out-of-fold, "
                "never having seen the cell's ~9 km block (notebook m01). Where it departs from GAEZ, GAEZ draws on "
                "information the drivers do not carry.",
    SOURCES[4]: "Global Crop Suitability 2024, an independent 1 km product (shown x 10,000); it covers rice, maize, "
                "cassava, soybean, oil palm and sugarcane.",
}


@st.cache_data(show_spinner=False)
def _smoothed_column(col: str, sigma: float = 3.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    d = _load_base()
    v = d[col].to_numpy(float)
    ok = ~np.isnan(v)
    r, c = d["row"].to_numpy(), d["col"].to_numpy()
    grid, w = np.zeros(grid_shape()), np.zeros(grid_shape())
    grid[r[ok], c[ok]] = v[ok]
    w[r[ok], c[ok]] = 1.0
    num, den = gaussian_filter(grid, sigma, mode="constant"), gaussian_filter(w, sigma, mode="constant")
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)[r, c]


@st.cache_data(show_spinner=False)
def load_oof() -> pd.DataFrame | None:
    """Out-of-fold predictions of the m01 tree models, aligned with the warehouse rows (None if m01 was not run)."""
    path = OOF_DIR / "m01_oof_pixel.parquet"
    if not path.exists():
        return None
    return _load_base()[["grid_id"]].merge(pd.read_parquet(path), on="grid_id", how="left")


def crop_values(df: pd.DataFrame, crop: str, source: str) -> pd.Series | None:
    """One crop's suitability (0-10,000 scale) for the rows of df from the chosen source; None if not available."""
    base_index = _load_base().index
    if source == SOURCES[0]:
        return df[f"gaez1k_index_{crop}"]
    if source == SOURCES[1]:
        return df[f"gaez_index_{crop}"]
    if source == SOURCES[2]:
        return pd.Series(_smoothed_column(f"gaez_index_{crop}"), index=base_index).reindex(df.index)
    if source == SOURCES[3]:
        oof = load_oof()
        return None if oof is None else pd.Series(oof[f"et__{crop}"].to_numpy(), index=base_index).reindex(df.index)
    if crop not in GCS_CROPS:
        return None
    return df[f"gcs2024_{crop}"] * 10000


def crop_scale(crop: str):
    """A ramp in the crop's own colour: pale tint (unsuitable) -> full colour -> deep shade (very suitable)."""
    base = LABEL_COLORS[CROP_LABELS[crop]]
    return _scale([_mix(base, "#FFFFFF", 0.92), _mix(base, "#FFFFFF", 0.55), base, _mix(base, FOREST, 0.45)])


SHADINGS = ["Suitability", "Suitability, smoothed", "Model estimate", "Clear-cut win", "Flat"]
SHADING_NOTES = {
    SHADINGS[0]: "Colour depth = the winning crop's GAEZ index at the chosen resolution (pale = barely suitable, deep = "
                 "very suitable).",
    SHADINGS[1]: "Colour depth = the winning crop's GAEZ index blurred over ~3 km (mostly useful with the ~9 km product).",
    SHADINGS[2]: "Colour depth = the out-of-fold model estimate for the winning crop (ExtraTrees on the 14 drivers, m01).",
    SHADINGS[3]: "Colour depth = how clearly the winner beats the runner-up (pale = close call, deep = clear winner).",
    SHADINGS[4]: "One flat colour per label.",
}


def shading(df: pd.DataFrame, how: str):
    """0-1 colour depth per row of df for the recommendation maps (None = flat)."""
    if how == "Flat":
        return None
    rec = recommendation().reindex(df.index)
    if how == "Clear-cut win":
        return rec["margin"].to_numpy()
    base_index = _load_base().index
    out = np.full(len(df), np.nan)
    winners = rec["winner"].to_numpy()
    for crop in {w for w in winners if isinstance(w, str)}:  # pandas 3 turns None into NaN in text columns
        if how == SHADINGS[0]:
            values = df[gaez_col(crop)]
        elif how == SHADINGS[1]:
            values = pd.Series(_smoothed_column(gaez_col(crop)), index=base_index).reindex(df.index)
        else:
            values = crop_values(df, crop, SOURCES[3])
        if values is not None:
            m = winners == crop
            out[m] = values.to_numpy()[m]
    return np.clip(out / 10000, 0, 1)


def no_results_note(notebook: str):
    st.info(f"No results yet: run `scripts/modeling/{notebook}` first (the tables land in `scripts/modeling/results/`).")


def scale_for(col: str):
    """A colour ramp that suits the variable: green for suitability, water for rain, earth for soil/terrain ..."""
    if col.startswith(("gaez_", "gaez1k_", "gcs2024_")) or col == "cropland_pct":
        return SEQ
    if "precip" in col or "chirps" in col:
        return SEQ_WATER
    if "temp" in col:
        return SEQ_WARM
    if col.startswith("soil_") or col in ("elevation_mean_m", "slope_mean_deg"):
        return SEQ_EARTH
    return SEQ


# --------------------------------------------------------------------------------------------
# Page furniture: one header style, one section style, one footer, minimal CSS
# --------------------------------------------------------------------------------------------
_CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
.gs-eyebrow {font-size: .72rem !important; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; color: #6E9E5A;
             margin: 0 0 .35rem 0; line-height: 1.2;}
.gs-title {font-size: 2.15rem !important; font-weight: 700; color: #14281D; line-height: 1.12 !important; letter-spacing: -.02em; margin: 0;}
.gs-rule {width: 42px; height: 3px; border-radius: 3px; background: #2F5D3A; margin: .8rem 0 .75rem 0;}
.gs-sub {font-size: .97rem !important; color: #5E6E62; line-height: 1.65 !important; max-width: 64rem; margin: 0 0 1.5rem 0;}
.gs-sub b {color: #23492F; font-weight: 600;}
.gs-sub code, .gs-note code {background: #EDF1E6; color: #23492F; padding: .05rem .3rem; border-radius: .3rem; font-size: .85em;}
.gs-section {display: flex; align-items: baseline; gap: .75rem; margin: 1.9rem 0 .6rem 0; padding-bottom: .45rem;
             border-bottom: 1px solid #DCE3D3;}
.gs-section .gs-h {font-size: 1.05rem; font-weight: 650; color: #1F4D2E; letter-spacing: -.01em;}
.gs-section .gs-n {font-size: .84rem; color: #7A8A7E;}
.gs-note {font-size: .86rem; color: #5E6E62; line-height: 1.55; border-left: 3px solid #A9BF92; padding: .1rem 0 .1rem .8rem; margin: .4rem 0 1rem 0;}
.gs-foot {margin-top: 3rem; padding-top: .9rem; border-top: 1px solid #DCE3D3; font-size: .76rem; color: #8A978D; line-height: 1.6;}
[data-testid="stMetric"] {background: #FFFFFF; padding: .85rem 1rem;}
[data-testid="stMetricLabel"] p {font-size: .7rem !important; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: #6A7A6E;}
[data-testid="stMetricValue"] {color: #1F4D2E;}
[data-testid="stCaptionContainer"] {color: #6A7A6E;}
[data-testid="stExpander"] details {background: #FFFFFF;}
.gs-side-label {font-size: .68rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
                color: #A9C98E; margin: 1.1rem 0 .1rem 0;}
.gs-stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(112px, 1fr)); gap: .6rem; margin: .5rem 0 1.1rem 0;}
.gs-stat {background: #FFFFFF; border: 1px solid #DCE3D3; border-radius: .6rem; padding: .6rem .85rem;}
.gs-stat-l {font-size: .64rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: #6A7A6E;}
.gs-stat-v {font-size: 1.2rem; font-weight: 650; color: #1F4D2E; margin-top: .15rem; white-space: nowrap;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: #9DB39A !important;}
[data-testid="stSidebarNavSeparator"] {border-color: #2C4D37;}
</style>
"""


def apply_style():
    st.markdown(_CSS, unsafe_allow_html=True)


def _inline_md(text: str) -> str:
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    st.markdown(f'<div class="gs-eyebrow">{eyebrow}</div><div class="gs-title">{title}</div><div class="gs-rule"></div>'
                + (f'<div class="gs-sub">{_inline_md(subtitle)}</div>' if subtitle else ""), unsafe_allow_html=True)


def section(title: str, note: str = ""):
    st.markdown(f'<div class="gs-section"><span class="gs-h">{title}</span>'
                + (f'<span class="gs-n">{_inline_md(note)}</span>' if note else "") + "</div>", unsafe_allow_html=True)


MODEL_NAMES = {"et": "ExtraTrees", "hgb": "HistGB", "xgb": "XGBoost", "knn": "kNN", "ridge": "Ridge",
               "dummy": "Mean baseline", "logreg": "Logistic regression", "mlp3": "MLP (3-seed, GPU)", "mlp": "MLP"}


def model_name(code) -> str:
    """Display name for a model key of the result tables: 'hgb + rule' -> 'HistGB + rule', 'et (m01)' -> 'ExtraTrees (m01)'."""
    import re
    return re.sub(r"^(et|hgb|xgb|knn|ridge|dummy|logreg|mlp3|mlp)\b", lambda m: MODEL_NAMES[m.group(1)], str(code))


def stat_strip(items):
    """A row of small label / value tiles; fits eight or more values where st.metric would truncate."""
    cells = "".join(f'<div class="gs-stat"><div class="gs-stat-l">{k}</div><div class="gs-stat-v">{v}</div></div>' for k, v in items)
    st.markdown(f'<div class="gs-stats">{cells}</div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="gs-note">{_inline_md(text)}</div>', unsafe_allow_html=True)


def footer():
    st.markdown('<div class="gs-foot">Data: SoilGrids 2.0 (ISRIC) · WorldClim v2.1 · CHIRPS v2.0 · Copernicus DEM · '
                "ESA WorldCover · GAEZ v5 (FAO / IIASA) · Global Crop Suitability 2024 · Sentinel-2 (Copernicus) · "
                "geoBoundaries.<br>Analysis grid: 9,333 cells of 1 km x 1 km over Banten Province, UTM zone 48S.</div>",
                unsafe_allow_html=True)
