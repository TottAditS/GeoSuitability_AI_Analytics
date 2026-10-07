import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import common as C

df = C.load_data()
lake = C.lake_inventory()
proc = C.processed_inventory()
dd = C.load_data_dictionary()
STRUCTURE_COLORS = {"structured": C.PINE, "semi-structured": C.MOSS, "unstructured": C.OCHRE,
                    "unstructured (pixel content)": C.OCHRE, "other": C.STONE, "(?)": C.STONE}

C.page_header("Data platform", "Data Lake & Warehouse", "How the data is stored: raw sources land untouched in a file-based **data lake** (`datasets/raw`, "
           "schema-on-read); the ETL pipeline reconciles them onto one 1 km grid in the **warehouse / mart** layer "
           "(`datasets/processed`, schema-on-write) that the dashboard and the experiments read.")

k = st.columns(5)
k[0].metric("Lake volume", f"{lake['size_bytes'].sum() / 1e9:.1f} GB", border=True)
k[1].metric("Lake files", f"{int(lake['files'].sum()):,}", border=True)
k[2].metric("Processed layer", f"{proc['size_bytes'].sum() / 1e9:.1f} GB", border=True)
k[3].metric("Fact table", f"{C.FEATURES_CSV.stat().st_size / 1e6:.1f} MB", border=True)
k[4].metric("Raw : fact table", f"{lake['size_bytes'].sum() / C.FEATURES_CSV.stat().st_size:,.0f} : 1", border=True,
            help="How many bytes of raw data stand behind each byte of the analysis-ready table")

tabs = st.tabs(["Architecture", "Data lake", "Warehouse", "Relations & lineage", "Spark / Sedona", "Data quality"])

# ----------------------------------------------------------------------------------------- architecture
with tabs[0]:
    st.graphviz_chart("""
digraph G {
  rankdir=LR; nodesep=0.25; ranksep=0.45; bgcolor="transparent"; fontname="Helvetica"; fontcolor="#5E6E62";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#9FB08F", fontcolor="#1D2B22", margin="0.12,0.06"];
  edge [color="#8A9A86", arrowsize=0.6];
  subgraph cluster_src { label="Open sources"; style=dashed; color=grey;
    node [fillcolor="#EEF2E8"];
    s1 [label="SoilGrids 2.0"]; s2 [label="WorldClim v2.1"]; s3 [label="CHIRPS v2.0"]; s4 [label="Copernicus DEM"];
    s5 [label="ESA WorldCover"]; s6 [label="GAEZ v5"]; s7 [label="Global Crop\\nSuitability 2024"];
    s8 [label="Sentinel-2 L2A\\n(Earth Engine)"]; s9 [label="geoBoundaries"]; s10 [label="CropSuite (PNG)"]; }
  ing [label="Ingestion\\nscripts/download/*.ipynb\\nbatch HTTP / EE export", fillcolor="#F3ECDA"];
  lake [label="DATA LAKE\\ndatasets/raw\\nGeoTIFF, GeoJSON, ZIP, CSV\\nschema-on-read", fillcolor="#E9D9B4", shape=cylinder];
  subgraph cluster_etl { label="Processing (scripts/preprocessing)"; style=dashed; color=grey; node [fillcolor="#E2EBDA"];
    p1 [label="01 validate"]; p2 [label="02 mosaic Sentinel-2"]; p3 [label="03 build 1 km grid"];
    p4 [label="04 zonal statistics\\n(rasterstats)"]; p6 [label="06 Apache Sedona\\non Spark"]; p7 [label="07-08 GAEZ scenario\\n+ 10 crops"];
    p9 [label="09-10 GCS 2024\\ncross-check"]; p11 [label="11 Sentinel-2\\ncell features"]; p12 [label="12 GAEZ v5 at 1 km\\n(remote COG window)"]; }
  subgraph cluster_wh { label="WAREHOUSE / MART (datasets/processed)"; style=filled; color="#F1F5EC"; node [fillcolor="#CFE0C3"];
    grid [label="grid_1km.gpkg\\n(dimension)"]; fact [label="tabular_features.csv\\n(fact table)"];
    gcs [label="gcs2024_zonal_stats.csv"]; s2m [label="Sentinel-2 mosaics\\n(2020-2024)"]; spark [label="spark_output\\n(Parquet)"];
    s2c [label="sentinel2_cell_stats.csv\\n(1:1 with the fact table)"]; s2p [label="sentinel2_patches/*.npy\\n(25 x 25 px per cell)"]; }
  subgraph cluster_use { label="Consumers"; style=dashed; color=grey; node [fillcolor="#F5F2E6"];
    dash [label="Streamlit dashboard"]; ml [label="ML / DL experiments\\nscripts/modeling m01-m14"]; oof [label="experiments/*.parquet\\n(out-of-fold predictions)"]; }
  {s1 s2 s3 s4 s5 s6 s7 s8 s9 s10} -> ing -> lake;
  lake -> p1; lake -> p2 -> s2m; lake -> p3 -> grid; lake -> p4; grid -> p4 -> fact;
  lake -> p7 -> fact; lake -> p12 -> fact; lake -> p9 -> gcs -> fact; fact -> p6 -> spark; grid -> p6; s2m -> p11;
  p11 -> s2c; p11 -> s2p; s2c -> dash; s2c -> ml; s2p -> ml;
  fact -> dash; spark -> dash; fact -> ml -> oof -> dash;
}
""", width="stretch")
    vol = pd.concat([
        lake.assign(layer="Data lake (raw)", name=lake["item"]),
        proc.assign(layer="Processed layer", name=proc["item"]),
    ])[["layer", "name", "size_bytes"]]
    vol.loc[len(vol)] = ["Fact table", "tabular_features.csv", C.FEATURES_CSV.stat().st_size]
    vol["MB"] = vol["size_bytes"] / 1e6
    fig = px.bar(vol.groupby("layer", sort=False)["MB"].sum().reset_index(), x="layer", y="MB", log_y=True, text_auto=".3s",
                 color="layer", color_discrete_sequence=[C.OCHRE, C.MOSS, C.PINE], height=330, title="The Big Data funnel: volume per layer (log scale)")
    fig.update_layout(showlegend=False, xaxis_title="", margin=dict(t=40))
    fig.update_yaxes(dtick=1, title="MB (log)")
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        "- **Lake**: every source kept in its native format, resolution and CRS; nothing is joined there.\n"
        "- **Processing**: notebooks validate, mosaic, grid and aggregate (zonal statistics), and Apache Sedona on "
        "Spark (local mode) runs the distributed spatial SQL / aggregation step.\n"
        "- **Warehouse / mart**: one fact table keyed by `grid_id` (one row per 1 km cell) plus the grid geometry; "
        "fixed schema, ready for BI and ML.\n"
        "- **Consumers**: this dashboard and the modeling notebooks read only the warehouse layer."
    )

# ----------------------------------------------------------------------------------------- lake
with tabs[1]:
    inv = lake.merge(dd, on="local_path", how="left")
    inv["dataset"] = inv["dataset"].fillna(inv["item"])
    inv["structure_type"] = inv["structure_type"].fillna(
        pd.Series(np.where(inv["item"].str.endswith(".csv"), "structured", "other"), index=inv.index))
    inv["GB"] = inv["size_bytes"] / 1e9
    l, r = st.columns([1, 1])
    fig = px.treemap(inv, path=["structure_type", "dataset"], values="size_bytes", color="structure_type", height=420,
                     color_discrete_map=STRUCTURE_COLORS,
                     title="Lake volume by structure type and dataset")
    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
    l.plotly_chart(fig, width="stretch")
    fig = px.bar(inv.sort_values("GB"), x="GB", y="dataset", color="structure_type", orientation="h", log_x=True,
                 color_discrete_map=STRUCTURE_COLORS,
                 height=420, title="Size on disk (GB, log scale)")
    fig.update_layout(margin=dict(t=40), yaxis_title="")
    r.plotly_chart(fig, width="stretch")
    st.dataframe(inv[["dataset", "provider", "format", "structure_type", "native_resolution", "crs", "files", "GB",
                      "file_types", "license_notes", "local_path"]].round(3),
                 hide_index=True, width="stretch")
    st.caption("Sizes and file counts are measured on disk; descriptive columns come from `datasets/data_dictionary.csv`. "
               "Structured = fixed tabular schema, semi-structured = self-describing rasters/vectors (CRS, transform, "
               "nodata), unstructured = imagery whose pixels mean nothing until an algorithm processes them.")

    C.section("Derived products in the processed layer")
    proc_d = proc.merge(dd, on="local_path", how="left")
    proc_d["MB"] = proc_d["size_bytes"] / 1e6
    st.dataframe(proc_d[["item", "files", "MB", "file_types", "format", "structure_type", "local_path"]].round(2),
                 hide_index=True, width="stretch")

    if C.VALIDATION_REPORT.exists():
        C.section("Raw file validation report")
        vr = pd.read_csv(C.VALIDATION_REPORT)
        pick = st.multiselect("Datasets", sorted(vr["dataset"].unique()), default=sorted(vr["dataset"].unique()))
        vr = vr[vr["dataset"].isin(pick)]
        st.dataframe(vr, hide_index=True, width="stretch", height=300)
        st.caption("Written by `scripts/preprocessing/01_validate_datasets.ipynb`: dimensions, CRS, data type, nodata, "
                   "resolution, bounds and value range of every raw raster, checked before any processing.")

# ----------------------------------------------------------------------------------------- warehouse
with tabs[2]:
    C.section("Fact table: `tabular_features.csv`")
    cat = C.column_catalogue(df.columns)
    num = df.select_dtypes("number")
    prof = pd.DataFrame({
        "dtype": df.dtypes.astype(str), "non_null": df.notna().sum(), "missing_%": df.isna().mean() * 100,
        "unique": df.nunique(), "min": num.min(), "mean": num.mean(), "max": num.max(),
    })
    cat = cat.merge(prof, left_on="column", right_index=True, how="left")
    a, b = st.columns([2, 1])
    groups = a.multiselect("Column groups", sorted(cat["group"].unique()), default=sorted(cat["group"].unique()))
    q = b.text_input("Search columns", "")
    view = cat[cat["group"].isin(groups)]
    if q:
        view = view[view["column"].str.contains(q, case=False) | view["description"].str.contains(q, case=False)]
    st.dataframe(view.round(3), hide_index=True, width="stretch", height=420)
    st.caption(f"{len(df.columns) - 2} stored columns (+2 labels derived by the dashboard: `recommended_crop`, "
               "`landcover_label`). Grain: one row per 1 km grid cell; primary key `grid_id`.")
    fig = px.bar(cat.groupby("group").size().reset_index(name="columns").sort_values("columns"), x="columns", y="group",
                 orientation="h", height=360, title="Columns per group")
    fig.update_layout(yaxis_title="", margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")

    C.section("Dimension: `grid_1km.gpkg`")
    grid = gpd.read_file(C.GRID_GPKG, rows=5)
    gl, gr = st.columns([1, 1])
    gl.dataframe(pd.DataFrame({"column": grid.columns, "dtype": grid.dtypes.astype(str)}), hide_index=True, width="stretch")
    gr.markdown(f"- CRS: `{grid.crs}`\n- cells: {len(df):,} squares of 1,000 m\n- key: `grid_id` (shared with the fact table)\n"
                "- geometry: polygons, used for the basemap and for every zonal statistic")
    C.section("Preview")
    st.dataframe(df.head(50), width="stretch", height=300)

# ----------------------------------------------------------------------------------------- relations
with tabs[3]:
    s2_state = "built" if C.S2_STATS_CSV.exists() else "planned (notebook 11 not run yet)"
    st.graphviz_chart(f"""
digraph ER {{
  rankdir=LR; bgcolor="transparent"; node [shape=plaintext, fontname="Helvetica", fontsize=11, fontcolor="#1D2B22"];
  edge [color="#8A9A86", fontcolor="#5E6E62", fontname="Helvetica", fontsize=9, arrowsize=0.6];
  grid [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#CFE0C3"><b>grid_1km.gpkg</b></td></tr>
    <tr><td align="left">PK grid_id</td></tr><tr><td align="left">row, col</td></tr><tr><td align="left">lon, lat</td></tr><tr><td align="left">geometry (EPSG:32748)</td></tr></table>>];
  fact [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#B5D19F"><b>tabular_features.csv</b> (fact)</td></tr>
    <tr><td align="left">PK/FK grid_id</td></tr><tr><td align="left">climate (5), soil (7 + 7 raw), terrain (2)</td></tr>
    <tr><td align="left">land cover (2), soil flag</td></tr><tr><td align="left">GAEZ ~9 km index + class (20 + 8 rain-fed)</td></tr><tr><td align="left">GAEZ 1 km index + class (20 + 8 rain-fed)</td></tr><tr><td align="left">GCS 2024 (6)</td></tr></table>>];
  gcs [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#EEF2E8"><b>gcs2024_zonal_stats.csv</b></td></tr>
    <tr><td align="left">FK grid_id</td></tr><tr><td align="left">gcs2024_* (6 crops)</td></tr></table>>];
  g1k [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#EEF2E8"><b>gaez1k_zonal_stats.csv</b></td></tr>
    <tr><td align="left">FK grid_id</td></tr><tr><td align="left">gaez1k_index_* / gaez1k_class_* (10 crops)</td></tr></table>>];
  s2 [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#F1E9C8"><b>sentinel2_cell_stats.csv</b></td></tr>
    <tr><td align="left">FK grid_id</td></tr><tr><td align="left">band means / std, NDVI pct</td></tr><tr><td align="left"><i>{s2_state}</i></td></tr></table>>];
  spark [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#E2EBDA"><b>spark_output/*.parquet</b></td></tr>
    <tr><td align="left">features_by_best_crop (partitioned)</td></tr><tr><td align="left">grid_by_province</td></tr></table>>];
  oof [label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="#E9D9B4"><b>experiments/*.parquet</b></td></tr>
    <tr><td align="left">FK grid_id</td></tr><tr><td align="left">out-of-fold predictions per model x crop</td></tr></table>>];
  grid -> fact [label="1 : 1  (grid_id)"]; gcs -> fact [label="merged into (grid_id)"]; g1k -> fact [label="merged into (grid_id)"];
  s2 -> fact [label="1 : 1 (grid_id)", style=dashed];
  fact -> spark [label="aggregated by Sedona"]; fact -> oof [label="1 : n  (grid_id)"];
}}
""", width="stretch")
    st.markdown("**Column lineage**: which source dataset feeds which part of the fact table (width = number of columns).")
    cat = C.column_catalogue([c for c in df.columns if c not in ("recommended_crop", "landcover_label")])
    flows = cat.groupby(["source", "group"]).size().reset_index(name="n")
    nodes = list(dict.fromkeys(list(flows["source"]) + list(flows["group"]) + ["tabular_features.csv"]))
    idx = {n: i for i, n in enumerate(nodes)}
    src = [idx[s] for s in flows["source"]] + [idx[g] for g in flows.groupby("group")["n"].sum().index]
    dst = [idx[g] for g in flows["group"]] + [idx["tabular_features.csv"]] * flows["group"].nunique()
    val = list(flows["n"]) + list(flows.groupby("group")["n"].sum().values)
    fig = go.Figure(go.Sankey(node=dict(label=nodes, pad=14, thickness=12, color=C.MOSS, line=dict(width=0)),
                               link=dict(source=src, target=dst, value=val, color="rgba(127,163,94,0.28)")))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Every table joins on `grid_id`, the primary key of the 1 km grid: the grid is the single dimension "
               "that makes ten incompatible sources comparable.")

# ----------------------------------------------------------------------------------------- spark
with tabs[4]:
    st.caption("Output of `scripts/preprocessing/06_sedona_processing.ipynb`: Apache Sedona (spatial SQL on Apache Spark, "
               "local mode) loads the grid and the fact table, runs the spatial join / aggregation and writes Parquet.")
    found = False
    for name in ["features_by_best_crop", "grid_by_province"]:
        path = C.SPARK_DIR / name
        if not path.exists() or not any(path.rglob("*.parquet")):
            continue
        found = True
        part = pd.read_parquet(path)
        C.section(f"`{name}`")
        st.write(f"{len(part):,} rows x {part.shape[1]} columns, {len(list(path.rglob('*.parquet')))} Parquet part files")
        l, r = st.columns([1, 1])
        l.dataframe(pd.DataFrame({"column": part.columns, "dtype": part.dtypes.astype(str)}), hide_index=True,
                    width="stretch", height=260)
        r.dataframe(part.head(20), width="stretch", height=260)
        if "best_crop" in part.columns:
            cnt = part["best_crop"].astype(str).value_counts().reset_index()
            cnt.columns = ["best_crop (partition)", "cells"]
            fig = px.bar(cnt, x="best_crop (partition)", y="cells", height=300)
            fig.update_layout(margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Partitioned by the plain argmax crop (no 5,000 threshold), as the Sedona job writes it.")
    if not found:
        st.info("No Spark output found: run `scripts/preprocessing/06_sedona_processing.ipynb`.")

# ----------------------------------------------------------------------------------------- quality
with tabs[5]:
    stored = df.drop(columns=["recommended_crop", "landcover_label"])
    checks = pd.DataFrame([
        ("grid_id is unique", stored["grid_id"].is_unique),
        ("(row, col) is unique", not stored.duplicated(["row", "col"]).any()),
        ("no fully duplicated rows", not stored.duplicated().any()),
        ("GAEZ indices within 0-10,000", stored[[c for c in stored if c.startswith(("gaez_index_", "gaez1k_index_"))]].stack().between(0, 10000).all()),
        ("GCS values within 0-1", stored[[c for c in stored if c.startswith("gcs2024_")]].stack().between(0, 1).all()),
        ("cropland share within 0-100 %", stored["cropland_pct"].between(0, 100).all()),
        ("all cells inside the Banten bounding box", stored["lon"].between(105.0, 106.9).all() and stored["lat"].between(-7.1, -5.7).all()),
        ("no missing climate values", stored[C.FEATURE_GROUPS["Climate"]].notna().all().all()),
        ("no missing (gap-filled) soil values", stored[C.FEATURE_GROUPS["Soil"]].notna().all().all()),
    ], columns=["check", "passed"])
    st.dataframe(checks, hide_index=True, width="stretch")
    miss = stored.isna().mean().mul(100).sort_values(ascending=False)
    miss = miss[miss > 0].reset_index()
    miss.columns = ["column", "missing_%"]
    fig = px.bar(miss, x="column", y="missing_%", height=320, title="Missing values per column (only columns with gaps)")
    fig.update_layout(margin=dict(t=40), xaxis_title="")
    st.plotly_chart(fig, width="stretch")
    l, m, r = st.columns(3)
    l.plotly_chart(C.category_map(df, df["soil_data_interpolated"].map({True: "interpolated", False: "SoilGrids value"}),
                                  {"SoilGrids value": "#CFE0C3", "interpolated": "#B8653F"}, "Soil gap-fill", 380), width="stretch")
    gcs_ok = df[[f"gcs2024_{c}" for c in C.GCS_CROPS]].notna().all(axis=1)
    m.plotly_chart(C.category_map(df, gcs_ok.map({True: "GCS value", False: "no GCS value"}),
                                  {"GCS value": "#CFE0C3", "no GCS value": "#B8653F"}, "GCS 2024 coverage", 380), width="stretch")
    g1k = df[[f"gaez1k_index_{c}" for c in C.CROPS]].notna()
    g1k_state = pd.Series(np.select([g1k.all(axis=1), g1k.any(axis=1)], ["all 10 crops", "no cropland for some crops"],
                                    "water (no value)"), index=df.index)
    r.plotly_chart(C.category_map(df, g1k_state, {"all 10 crops": "#CFE0C3", "no cropland for some crops": C.OCHRE,
                                                  "water (no value)": C.TEAL}, "GAEZ 1 km coverage", 380), width="stretch")
    partial = g1k_state == "no cropland for some crops"
    tree = (df.loc[partial, "landcover_label"] == "Tree cover").mean() if partial.any() else 0.0
    st.caption(f"{int(df['soil_data_interpolated'].sum()):,} cells ({df['soil_data_interpolated'].mean():.1%}) had no SoilGrids "
               "value (built-up land, water, coast) and were filled by spatial interpolation; the originals are kept as "
               "`soil_*_raw`. The 18 cells without elevation/slope lie just east of 106.75° E, the eastern edge of the "
               "downloaded DEM tile (mostly built-up land on the Jakarta border). GAEZ at 1 km leaves "
               f"{int(partial.sum()):,} cells without a value for some crops, land it classes as 'no cropland' for them "
               f"({tree:.0%} under tree cover; coffee and cacao keep a value there), and "
               f"{int((g1k_state == 'water (no value)').sum()):,} open-water cells without any value.")
