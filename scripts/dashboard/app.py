"""BI/analytics dashboard for the GeoSuitability AI Analytics data warehouse.

Reads directly from the processed data warehouse/mart layer
(datasets/processed/features/tabular_features.csv) built by the preprocessing
pipeline (scripts/preprocessing/01-10). No separate database or backend.

Run with:
    streamlit run scripts/dashboard/app.py
"""

import json
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
FEATURES_CSV = ROOT / "datasets" / "processed" / "features" / "tabular_features.csv"
GRID_GPKG = ROOT / "datasets" / "processed" / "grid" / "grid_1km.gpkg"

CROPS = [
    "rice", "maize", "cassava", "soybean",
    "oilpalm", "rubber", "sugarcane", "coffee", "cacao", "coconut",
]
CROP_LABELS = {
    "rice": "Rice", "maize": "Maize", "cassava": "Cassava", "soybean": "Soybean",
    "oilpalm": "Oil palm", "rubber": "Rubber", "sugarcane": "Sugarcane",
    "coffee": "Coffee", "cacao": "Cacao", "coconut": "Coconut",
}

# Thresholds match the rule-based labeling logic in the written report (Section 4.2.3) exactly.
NOT_RECOMMENDED_THRESHOLD = 5000
TIE_THRESHOLD = 500

# Fixed colors for solo winners / not-recommended. Tie-pair labels (e.g. "Cassava / Rice (tied)")
# are generated dynamically in label_cells() and get auto-assigned a color by plotly - there's no
# fixed list of possible pairs to hardcode here, it depends on which crops are actually close.
LABEL_COLORS = {
    "Rice": "#E53935",
    "Maize": "#FDD835",
    "Cassava": "#C87533",
    "Soybean": "#43A047",
    "Oil palm": "#1B5E20",
    "Rubber": "#5D4037",
    "Sugarcane": "#8BC34A",
    "Coffee": "#4E342E",
    "Cacao": "#6D4C41",
    "Coconut": "#00ACC1",
    "Not recommended": "#BDBDBD",
}

# ESA WorldCover class codes (fixed codebook, https://esa-worldcover.org/).
WORLDCOVER_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
    80: "Permanent water bodies", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss and lichen",
}

NUMERIC_COLS = [
    "soil_ph", "soil_soc", "soil_nitrogen", "soil_clay_pct", "soil_sand_pct",
    "soil_cec", "soil_bulk_density", "elevation_mean_m", "slope_mean_deg",
    "bio1_temp_mean_c", "bio12_precip_annual_mm", "chirps_annual_mean_mm",
    "cropland_pct",
]

# Every non-GAEZ raw layer that made it into the feature table, grouped by source dataset - so the
# map can show each source dataset's actual contribution, not just the final crop recommendation.
RAW_LAYERS = {
    "SoilGrids 2.0": {
        "Soil pH": ("soil_ph", None),
        "Soil organic carbon (g/kg)": ("soil_soc", None),
        "Soil nitrogen (g/kg)": ("soil_nitrogen", None),
        "Clay fraction (%)": ("soil_clay_pct", None),
        "Sand fraction (%)": ("soil_sand_pct", None),
        "Cation exchange capacity (cmol/kg)": ("soil_cec", None),
        "Bulk density (kg/dm3)": ("soil_bulk_density", None),
    },
    "WorldClim v2.1": {
        "Mean annual temperature (C)": ("bio1_temp_mean_c", None),
        "Annual precipitation 1970-2000 (mm)": ("bio12_precip_annual_mm", None),
        "Precipitation seasonality (index)": ("bio15_precip_seasonality", None),
    },
    "CHIRPS": {
        "Annual precipitation 2020-2024 (mm)": ("chirps_annual_mean_mm", None),
        "Std. dev. of annual precipitation (mm)": ("chirps_annual_std_mm", None),
    },
    "Copernicus DEM": {
        "Elevation (m)": ("elevation_mean_m", None),
        "Slope (degrees)": ("slope_mean_deg", None),
    },
    "ESA WorldCover": {
        "Cropland share (%)": ("cropland_pct", None),
        "Majority land-cover class": ("landcover_label", "categorical"),
    },
}


def label_cells(df: pd.DataFrame) -> pd.Series:
    """argmax(gaez_index_*) with not-recommended / tie thresholds.

    Ties are labeled with the actual competing crop pair (e.g. "Cassava / Rice (tied)") instead of
    a generic "multi-commodity zone" bucket, so every crop's footprint stays visible on the map -
    which pair(s) occur depends entirely on the data, so this is computed, not hardcoded.
    """
    idx_cols = [f"gaez_index_{c}" for c in CROPS]
    scores = df[idx_cols].to_numpy()
    order = np.argsort(scores, axis=1)
    best_i = order[:, -1]
    second_i = order[:, -2]
    best_score = scores[np.arange(len(df)), best_i]
    second_score = scores[np.arange(len(df)), second_i]

    labels = np.array([CROP_LABELS[CROPS[i]] for i in best_i], dtype=object)

    tie_mask = (best_score - second_score) <= TIE_THRESHOLD
    tie_labels = [
        " / ".join(sorted((CROP_LABELS[CROPS[b]], CROP_LABELS[CROPS[s]]))) + " (tied)"
        for b, s in zip(best_i[tie_mask], second_i[tie_mask])
    ]
    labels[tie_mask] = tie_labels
    labels[best_score < NOT_RECOMMENDED_THRESHOLD] = "Not recommended"  # highest priority
    return pd.Series(labels, index=df.index, name="recommended_crop")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    df["recommended_crop"] = label_cells(df)
    df["landcover_label"] = (
        df["landcover_majority"].round().astype("Int64").map(WORLDCOVER_CLASSES).fillna("Unknown")
    )
    return df


@st.cache_data
def load_grid_geojson() -> dict:
    """1 km grid cell polygons, reprojected to WGS84, for the choropleth map."""
    grid = gpd.read_file(GRID_GPKG)[["grid_id", "geometry"]].to_crs("EPSG:4326")
    return json.loads(grid.to_json())


st.set_page_config(page_title="GeoSuitability AI - Dashboard", layout="wide")

df = load_data()
# whatever labels actually occur in the data, not a hardcoded guess - "Not recommended" last
CROP_OPTIONS = sorted(df["recommended_crop"].unique(), key=lambda x: (x == "Not recommended", x))

st.sidebar.title("GeoSuitability AI")
st.sidebar.caption("Land suitability data warehouse - Banten Province")

st.sidebar.divider()
st.sidebar.subheader("Filter")
selected_crops = st.sidebar.multiselect(
    "Crop recommendation", options=CROP_OPTIONS, default=CROP_OPTIONS
)
filtered = df[df["recommended_crop"].isin(selected_crops)]
st.sidebar.caption(f"{len(filtered):,} / {len(df):,} grid cells shown")

st.sidebar.divider()
st.sidebar.caption(
    "Read directly from the data warehouse/mart produced by the Big Data pipeline "
    f"(`tabular_features.csv`, {len(df):,} grid cells of 1 km x 1 km, UTM 48S)."
)

st.title("GeoSuitability AI")
st.caption("Agricultural Land Suitability - Banten Province")

tab_overview, tab_map, tab_explorer = st.tabs(
    ["Overview", "Suitability Map", "Data Explorer"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total grid cells", f"{len(df):,}")
    c2.metric("Area covered", f"{len(df):,} km²")
    c3.metric("Data sources", "10 open sources")
    c4.metric("Dominant recommendation", df["recommended_crop"].value_counts().idxmax())

    st.subheader("Crop Recommendation Distribution")
    dist = df["recommended_crop"].value_counts().reindex(CROP_OPTIONS).reset_index()
    dist.columns = ["Recommendation", "Cells"]
    dist["Percentage"] = (dist["Cells"] / len(df) * 100).round(1)
    fig_dist = px.bar(
        dist, x="Recommendation", y="Cells", color="Recommendation",
        color_discrete_map=LABEL_COLORS, text="Percentage",
    )
    fig_dist.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_dist.update_layout(showlegend=False)
    st.plotly_chart(fig_dist, width="stretch")

    st.caption(
        "Labels come from the argmax of the GAEZ v5 suitability index over 10 crops (rice, maize, "
        "cassava, soybean, oil palm, rubber, sugarcane, coffee, cacao, coconut - scale 0-10,000): "
        "'Not recommended' if the top score is < 5,000; if the top two scores differ by <= 500, the "
        "label names both competing crops (e.g. 'Cassava / Rice (tied)'). The written Assignment I "
        "report used only the first 4 crops; this was extended to 10 because a 4-crop result was "
        "too narrow for real analysis."
    )

MAP_BOUNDS = dict(west=104.95, east=106.93, south=-7.18, north=-5.66)
MAP_CENTER = dict(lat=-6.42, lon=105.94)


def _lock_banten_view(fig):
    # lock the viewport to the Banten grid extent (+ small padding) - never a default world view
    fig.update_maps(center=MAP_CENTER, zoom=8.2, bounds=MAP_BOUNDS)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))


VIEW_FINAL = "Final recommendation (who wins)"
VIEW_GAEZ = "Per-crop suitability score (GAEZ)"
VIEW_RAW = "Raw data layers (climate / soil / terrain / land cover)"

with tab_map:
    st.header("Interactive Map")
    view_mode = st.radio("View", [VIEW_FINAL, VIEW_GAEZ, VIEW_RAW], horizontal=True)

    if view_mode == VIEW_FINAL:
        if filtered.empty:
            st.warning("No grid cells match the current filter.")
        else:
            grid_geojson = load_grid_geojson()
            fig_map = px.choropleth_map(
                filtered, geojson=grid_geojson, locations="grid_id", featureidkey="properties.grid_id",
                color="recommended_crop", color_discrete_map=LABEL_COLORS,
                hover_data={
                    "grid_id": True,
                    "soil_ph": ":.2f", "chirps_annual_mean_mm": ":.0f",
                    "elevation_mean_m": ":.0f",
                },
                height=650, map_style="open-street-map", opacity=0.75,
            )
            fig_map.update_traces(marker_line_width=0)
            fig_map.update_layout(legend_title_text="Recommendation")
            _lock_banten_view(fig_map)
            st.plotly_chart(fig_map, width="stretch")
    elif view_mode == VIEW_GAEZ:
        st.caption(
            "This view ignores the sidebar filter and shows the raw GAEZ v5 suitability score "
            "(0-10,000, irrigated scenario) for a single crop - including crops that never win "
            "in the 'Final recommendation' view."
        )
        crop_choice = st.selectbox("Select crop", [CROP_LABELS[c] for c in CROPS])
        crop_key = {v: k for k, v in CROP_LABELS.items()}[crop_choice]
        score_col = f"gaez_index_{crop_key}"
        grid_geojson = load_grid_geojson()
        fig_score = px.choropleth_map(
            df, geojson=grid_geojson, locations="grid_id", featureidkey="properties.grid_id",
            color=score_col, color_continuous_scale="YlGn", range_color=(0, 10000),
            hover_data={"grid_id": True, score_col: ":.0f"},
            height=650, map_style="open-street-map", opacity=0.75,
        )
        fig_score.update_traces(marker_line_width=0)
        fig_score.update_layout(coloraxis_colorbar_title=f"Index<br>{crop_choice}")
        _lock_banten_view(fig_score)
        st.plotly_chart(fig_score, width="stretch")
        st.caption(
            f"{crop_choice} suitability index - darker green means more suitable. The "
            "'Not recommended' threshold in the final recommendation view is a score < 5,000."
        )
    else:
        st.caption(
            "This view ignores the sidebar filter and shows the raw variables from each source "
            "dataset (SoilGrids, WorldClim, CHIRPS, Copernicus DEM, ESA WorldCover) - these "
            "datasets contribute real information of their own, not just the final recommendation."
        )
        col_source, col_layer = st.columns(2)
        with col_source:
            source_choice = st.selectbox("Select source dataset", list(RAW_LAYERS.keys()))
        with col_layer:
            layer_choice = st.selectbox("Select variable", list(RAW_LAYERS[source_choice].keys()))
        layer_col, layer_kind = RAW_LAYERS[source_choice][layer_choice]
        grid_geojson = load_grid_geojson()

        if layer_kind == "categorical":
            fig_layer = px.choropleth_map(
                df, geojson=grid_geojson, locations="grid_id", featureidkey="properties.grid_id",
                color=layer_col, hover_data={"grid_id": True, layer_col: True},
                height=650, map_style="open-street-map", opacity=0.75,
            )
            fig_layer.update_layout(legend_title_text=layer_choice)
        else:
            fig_layer = px.choropleth_map(
                df, geojson=grid_geojson, locations="grid_id", featureidkey="properties.grid_id",
                color=layer_col, color_continuous_scale="YlGnBu",
                hover_data={"grid_id": True, layer_col: ":.2f"},
                height=650, map_style="open-street-map", opacity=0.75,
            )
            fig_layer.update_layout(coloraxis_colorbar_title=layer_choice)
        fig_layer.update_traces(marker_line_width=0)
        _lock_banten_view(fig_layer)
        st.plotly_chart(fig_layer, width="stretch")
        st.caption(f"Source: {source_choice} - see `datasets/data_dictionary.csv` for the original license/resolution details.")

with tab_explorer:
    st.header("Data Explorer")
    st.write(f"Showing **{len(filtered):,}** of {len(df):,} grid cells (per the sidebar filter).")

    col_select, col_hist = st.columns([1, 2])
    with col_select:
        chosen_col = st.selectbox("Select a variable for the histogram", NUMERIC_COLS, index=0)
        st.dataframe(filtered[chosen_col].describe().round(2), width="stretch")
    with col_hist:
        fig_hist = px.histogram(filtered, x=chosen_col, nbins=40, color_discrete_sequence=["#2C5F2D"])
        st.plotly_chart(fig_hist, width="stretch")

    st.subheader("Data Table")
    display_cols = ["grid_id", "lon", "lat", "recommended_crop", "landcover_label"] + NUMERIC_COLS
    st.dataframe(filtered[display_cols], width="stretch", height=400)

    csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data (CSV)", data=csv_bytes,
        file_name="tabular_features_filtered.csv", mime="text/csv",
    )
