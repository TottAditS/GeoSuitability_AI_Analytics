import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import common as C

df = C.load_data()
rec = C.recommendation()
options = C.recommendation_options(df)
group, ranking = C.rule()

# Every raw layer that made it into the warehouse, grouped by source dataset, in the product's own units.
RAW_LAYERS = {
    "SoilGrids 2.0 (0-5 cm)": {
        "Soil pH in water (pH x 10)": "soil_ph",
        "Soil organic carbon (dg/kg)": "soil_soc",
        "Total nitrogen (cg/kg)": "soil_nitrogen",
        "Clay content (g/kg)": "soil_clay_pct",
        "Sand content (g/kg)": "soil_sand_pct",
        "Cation exchange capacity (mmol(c)/kg)": "soil_cec",
        "Bulk density (cg/cm3)": "soil_bulk_density",
    },
    "WorldClim v2.1": {
        "Annual mean temperature (deg C)": "bio1_temp_mean_c",
        "Annual precipitation 1970-2000 (mm)": "bio12_precip_annual_mm",
        "Precipitation seasonality (CV %)": "bio15_precip_seasonality",
    },
    "CHIRPS v2.0": {
        "Mean annual precipitation 2020-2024 (mm)": "chirps_annual_mean_mm",
        "Year-to-year std of annual precipitation (mm)": "chirps_annual_std_mm",
    },
    "Copernicus DEM": {"Elevation (m)": "elevation_mean_m", "Slope (degrees)": "slope_mean_deg"},
    "ESA WorldCover": {"Cropland share (%)": "cropland_pct", "Majority land-cover class": "landcover_label"},
    "Sentinel-2 L2A (2020-2024 mean)": {
        "NDVI, vegetation greenness": "s2_NDVI_mean",
        "EVI, enhanced vegetation index": "s2_EVI_mean",
        "NDMI, vegetation moisture": "s2_NDMI_mean",
        "Near-infrared reflectance (B8)": "s2_B8_mean",
        "NDVI year-to-year variability": "s2_NDVI_mean_iav",
        "Cloud-free pixel share": "s2_valid_frac",
    },
}
RAW_LAYERS = {src: layers for src, layers in RAW_LAYERS.items() if all(c in df.columns for c in layers.values())}

VIEWS = ["Recommendation", "Every crop", "One crop on the map", "Uncertainty", "Raw data layers"]

C.page_header("Suitability", "Suitability Map",
              "Where each crop fits in Banten. The hue says **which** crop, the depth of the colour says **how suitable**: "
              "pale is barely suitable, deep is very suitable. Every crop also has its own map.")
wanted = st.query_params.get("view")  # deep link, e.g. ?view=Every crop
view = st.segmented_control("View", VIEWS, default=wanted if wanted in VIEWS else VIEWS[0], key="map_view") or VIEWS[0]
geo = C.load_grid_geojson()


def show(fig, colorbar_title=None):
    fig.update_traces(marker_line_width=0)
    if colorbar_title:
        fig.update_layout(coloraxis_colorbar=dict(title=colorbar_title, thickness=10, outlinewidth=0))
    C.lock_banten_view(fig)
    st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------------------------------------------ recommendation
if view == "Recommendation":
    a, b = st.columns([3, 1.1], vertical_alignment="bottom")
    selected = a.pills("Show", options, selection_mode="multi", default=options, key=f"map_filter_{group}_{ranking}") or []
    shade = b.selectbox("Colour depth", C.SHADINGS, key="map_shade")
    shown = df[df["recommended_crop"].isin(selected)]
    if shown.empty:
        st.warning("No grid cells match the current filter.")
    else:
        r = rec.loc[shown.index]
        elev = shown["elevation_mean_m"].map(lambda v: "-" if pd.isna(v) else f"{v:,.0f} m")
        hover = (shown["recommended_crop"] + f"<br>GAEZ index ({C.gaez_res()}) of the winner: " + C.fmt_index(r["winner_index"])
                 + "<br>" + shown["landcover_label"] + " · " + elev)
        fig = C.category_choropleth(shown, shown["recommended_crop"], geo, C.LABEL_COLORS, C.shading(shown, shade), hover)
        st.plotly_chart(fig, width="stretch")
    st.caption(f"{len(shown):,} of {len(df):,} cells shown · click a label above to hide or show it.")
    C.note(C.rule_description() + " " + C.SHADING_NOTES[shade] + " Change the rule in the sidebar.")

    C.section("Every crop: suitable area vs. area where it wins",
              "a crop can be suitable in many cells and still rarely be the top choice")
    crops = C.CROP_GROUPS[group]
    rows = []
    for c in crops:
        rows.append({"crop": C.CROP_LABELS[c], "measure": "suitable (index >= 5,000)",
                     "cells": int((df[C.gaez_col(c)] >= C.NOT_RECOMMENDED_THRESHOLD).sum())})
        rows.append({"crop": C.CROP_LABELS[c], "measure": "recommended under this rule",
                     "cells": int(df["recommended_crop"].str.contains(C.CROP_LABELS[c], regex=False).sum())})
    fig = px.bar(pd.DataFrame(rows), x="crop", y="cells", color="measure", barmode="group", height=340,
                 color_discrete_map={"suitable (index >= 5,000)": C.SAGE, "recommended under this rule": C.PINE})
    fig.update_layout(margin=dict(t=10), xaxis_title="", legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    never = [C.CROP_LABELS[c] for c in crops if df[C.gaez_col(c)].max() < C.NOT_RECOMMENDED_THRESHOLD]
    if never:
        st.caption(f"{', '.join(never)} never reach the 5,000 'suitable' line anywhere in Banten under GAEZ v5 (irrigated, "
                   "high input) at this resolution, so no rule can recommend them; their own maps are under 'Every crop'.")

# ------------------------------------------------------------------------------------------------ every crop
elif view == "Every crop":
    src = st.segmented_control("Source", C.SOURCES, default=C.SOURCES[0], key="every_src") or C.SOURCES[0]
    C.note(C.SOURCE_NOTES[src] + " Each crop in its own colour; lighter to deeper = suitability index 0 to 10,000.")
    for start in (0, 5):
        cols = st.columns(5)
        for col, crop in zip(cols, C.CROPS[start:start + 5]):
            with col:
                values = C.crop_values(df, crop, src)
                if values is None:
                    st.markdown(f"**{C.CROP_LABELS[crop]}**")
                    st.caption("Not covered by this source.")
                    continue
                fig = C.grid_map(df, values, C.CROP_LABELS[crop], C.crop_scale(crop), 0, 10000, height=250, fmt=",.0f")
                fig.update_traces(showscale=False)
                fig.update_layout(title=dict(font=dict(size=13)), margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, width="stretch", key=f"every_{crop}")
                st.caption(f"suitable: {int((values >= C.NOT_RECOMMENDED_THRESHOLD).sum()):,} cells · max {np.nanmax(values):,.0f}")

# ------------------------------------------------------------------------------------------------ one crop on the map
elif view == "One crop on the map":
    a, b = st.columns(2)
    crop = a.selectbox("Crop", C.CROPS, format_func=C.CROP_LABELS.get, key="one_crop")
    src = b.selectbox("Source", C.SOURCES, index=0, key="one_src")
    values = C.crop_values(df, crop, src)
    if values is None:
        st.info("GCS 2024 covers rice, maize, cassava, soybean, oil palm and sugarcane only; pick another source or crop.")
    else:
        d = df[["grid_id", "landcover_label"]].assign(suitability=values)
        fig = px.choropleth_map(d.dropna(subset=["suitability"]), geojson=geo, locations="grid_id",
                                featureidkey="properties.grid_id", color="suitability", color_continuous_scale=C.crop_scale(crop),
                                range_color=(0, 10000), height=650, map_style=C.MAP_STYLE, opacity=0.88,
                                hover_data={"grid_id": True, "suitability": ":,.0f", "landcover_label": True})
        show(fig, colorbar_title=f"{C.CROP_LABELS[crop]}<br>0-10,000")
        st.caption(f"{C.CROP_LABELS[crop]} · {C.SOURCE_NOTES[src]} The 'recommended' line is 5,000.")

# ------------------------------------------------------------------------------------------------ uncertainty
elif view == "Uncertainty":
    unc = C.load_uncertainty()
    if unc is None:
        C.no_results_note("m15_uncertainty.ipynb")
    else:
        a, b, c = st.columns([1.4, 1, 1.3])
        choices = list(C.UNCERTAINTY_INPUTS)
        inputs = a.segmented_control("Model inputs", choices, default=choices[0], key="unc_inputs") or choices[0]
        crop = b.selectbox("Crop", C.CROPS, format_func=C.CROP_LABELS.get, key="unc_crop")
        SHOW = ["Width of the 90 % interval", "P(index >= 5,000)", "Model estimate", "Lower bound of the interval",
                "Upper bound of the interval"]
        what = c.selectbox("Show", SHOW, key="unc_show")
        tag = C.UNCERTAINTY_INPUTS[inputs]
        lo, hi = unc[f"{tag}__lo__{crop}"], unc[f"{tag}__hi__{crop}"]
        values, scale, zmax, fmt = {
            SHOW[0]: (hi - lo, C.SEQ_CLAY, 6000, ",.0f"),
            SHOW[1]: (unc[f"{tag}__psuit__{crop}"], C.SEQ, 1, ".2f"),
            SHOW[2]: (unc[f"{tag}__pred__{crop}"], C.crop_scale(crop), 10000, ",.0f"),
            SHOW[3]: (lo, C.crop_scale(crop), 10000, ",.0f"),
            SHOW[4]: (hi, C.crop_scale(crop), 10000, ",.0f"),
        }[what]
        l, r = st.columns(2)
        l.plotly_chart(C.grid_map(df, values, f"{C.CROP_LABELS[crop]}: {what[0].lower() + what[1:]}", scale, 0, zmax, 480,
                                  fmt=fmt), width="stretch")
        labels, pmax = C.uncertainty_label(unc, inputs)
        r.plotly_chart(C.category_map(df, labels, C.LABEL_COLORS, "Most likely recommendation (depth = its probability)",
                                      480, strength=pmax), width="stretch")
        cov = C.load_result("m15_coverage")
        m15_name = {"14 drivers + Sentinel-2": "drivers + Sentinel-2", "14 drivers": "drivers"}[inputs]
        if cov is not None:
            ad = cov[(cov["inputs"] == m15_name) & (cov["interval"] == "adaptive")]
            got = ad.set_index("crop").loc[crop]
            C.note(f"Uncertainty of the **model estimate** from the {inputs.replace('14 ', '')} (ExtraTrees, notebook m15), "
                   f"not of GAEZ itself. The 90 % intervals are calibrated on errors in held-out ~9 km blocks: for "
                   f"{C.CROP_LABELS[crop].lower()} they contain the true 1 km GAEZ index in {got['coverage']:.0%} of the cells "
                   f"(mean width {got['mean width']:,.0f} index points; all crops {ad['coverage'].mean():.0%}). The right "
                   "map applies the absolute rule (highest index wins, 'Not recommended' below 5,000) to 200 plausible "
                   "versions of every cell: pale = the recommendation is uncertain, deep = it is almost sure.")

# ------------------------------------------------------------------------------------------------ raw layers
else:
    a, b = st.columns(2)
    source = a.selectbox("Source dataset", list(RAW_LAYERS))
    layer = b.selectbox("Variable", list(RAW_LAYERS[source]))
    col = RAW_LAYERS[source][layer]
    if col == "landcover_label":
        fig = px.choropleth_map(df, geojson=geo, locations="grid_id", featureidkey="properties.grid_id", color=col,
                                height=650, map_style=C.MAP_STYLE, opacity=0.85,
                                color_discrete_sequence=C.EXTRA, hover_data={"grid_id": True, col: True, "cropland_pct": ":.0f"})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(legend_title_text=layer)
        C.lock_banten_view(fig)
        st.plotly_chart(fig, width="stretch")
    else:
        fig = px.choropleth_map(df, geojson=geo, locations="grid_id", featureidkey="properties.grid_id", color=col,
                                color_continuous_scale=C.scale_for(col), height=650, map_style=C.MAP_STYLE, opacity=0.85,
                                hover_data={"grid_id": True, col: ":.2f"})
        show(fig, colorbar_title=layer)
    st.caption(f"Source: {source}. Units are the product's own (SoilGrids stores scaled integers, e.g. pH x 10). "
               "Origin, licence and resolution of every source are on the 'Data Lake & Warehouse' page.")
