import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

import common as C

df = C.load_data()
CAT = C.column_catalogue(df.columns).set_index("column")
NUMERIC = [c for c in df.select_dtypes("number").columns if c not in ("row", "col", "landcover_majority")]
GROUP_OF = CAT["group"].to_dict()
ORDER = ["Climate", "Soil", "Terrain", "Land cover", "GAEZ index (1 km)", "GAEZ class (1 km)", "GAEZ index (~9 km)",
         "GAEZ class (~9 km)", "GCS 2024", "Sentinel-2", "Location", "Soil (raw, audit)", "GAEZ index (rain-fed 1 km, audit)",
         "GAEZ class (rain-fed 1 km, audit)", "GAEZ index (rain-fed ~9 km, audit)", "GAEZ class (rain-fed ~9 km, audit)"]
NUMERIC = sorted(NUMERIC, key=lambda c: (ORDER.index(GROUP_OF[c]) if GROUP_OF[c] in ORDER else 99, c))


def fmt(col):
    return f"{GROUP_OF.get(col, '')} · {col}"


C.page_header("Data · analysis", "Exploratory Data Analysis", f"Every column of the warehouse: the fact table and the Sentinel-2 cell statistics ({len(df):,} cells x {len(df.columns) - 2} stored columns). "
           "Units are the products' own; see the column catalogue on the 'Data Lake & Warehouse' page.")

SECTIONS = ["Column profiles", "Distributions by group", "Correlations", "Relationships", "Spatial patterns",
            "Categories", "Irrigated vs rain-fed", "PCA"]
section = st.segmented_control("Section", SECTIONS, default=SECTIONS[0], key="eda_section") or SECTIONS[0]

# ----------------------------------------------------------------------------------------------------------
if section == "Column profiles":
    col = st.selectbox("Column", NUMERIC, format_func=fmt)
    s = df[col]
    st.caption(f"**{col}**: {CAT.loc[col, 'description']} · source: {CAT.loc[col, 'source']} · unit: {CAT.loc[col, 'unit'] or '-'}")
    C.stat_strip([("count", f"{s.notna().sum():,}"), ("missing", f"{s.isna().sum():,}"), ("mean", f"{s.mean():,.4g}"),
                  ("std", f"{s.std():,.4g}"), ("min", f"{s.min():,.4g}"), ("median", f"{s.median():,.4g}"),
                  ("max", f"{s.max():,.4g}"), ("skew", f"{s.skew():.2f}"), ("kurtosis", f"{s.kurtosis():.2f}")])
    l, r = st.columns(2)
    fig = px.histogram(df, x=col, nbins=50, marginal="box", height=420, color_discrete_sequence=[C.PINE])
    fig.update_layout(margin=dict(t=10))
    l.plotly_chart(fig, width="stretch")
    r.plotly_chart(C.grid_map(df, s, None, C.scale_for(col), height=420, colorbar_title=col[:14]), width="stretch")
    corr = df[NUMERIC].corrwith(s, method="spearman").drop(col).dropna()
    top = corr.reindex(corr.abs().sort_values(ascending=False).index[:12]).reset_index()
    top.columns = ["column", "spearman"]
    fig = px.bar(top, x="spearman", y="column", orientation="h", color="spearman", color_continuous_scale=C.DIV,
                 range_color=(-1, 1), height=380, title=f"Strongest rank correlations with {col}")
    fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=40), yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    with st.expander("Summary of every numeric column"):
        desc = df[NUMERIC].describe().T
        desc["missing"] = df[NUMERIC].isna().sum()
        desc["skew"] = df[NUMERIC].skew()
        desc["kurtosis"] = df[NUMERIC].kurtosis()
        desc.insert(0, "group", [GROUP_OF[c] for c in desc.index])
        st.dataframe(desc.round(3), width="stretch", height=420)

# ----------------------------------------------------------------------------------------------------------
elif section == "Distributions by group":
    a, b, c = st.columns(3)
    col = a.selectbox("Variable", NUMERIC, format_func=fmt)
    by = b.selectbox("Group by", ["Recommendation", "Land cover", "GAEZ class of a crop", "Elevation band", "Soil gap-filled"])
    kind = c.segmented_control("Plot", ["Box", "Violin"], default="Box", key="eda_plot") or "Box"
    d = df.copy()
    if by == "Recommendation":
        d["group"] = d["recommended_crop"]
    elif by == "Land cover":
        d["group"] = d["landcover_label"]
    elif by == "GAEZ class of a crop":
        crop = st.selectbox("Crop", C.CROPS, format_func=C.CROP_LABELS.get)
        classes = C.gaez_classes()
        d["group"] = d[C.gaez_col(crop, "class")].round().astype("Int64").map(
            lambda v: "no value (open water)" if pd.isna(v) else f"{v} {classes.get(v, '')}")
    elif by == "Elevation band":
        d["group"] = pd.cut(d["elevation_mean_m"], [-1, 25, 100, 250, 500, 1000, 3000],
                            labels=["0-25 m", "25-100 m", "100-250 m", "250-500 m", "500-1000 m", ">1000 m"]).astype(str)
    else:
        d["group"] = d["soil_data_interpolated"].map({True: "interpolated", False: "SoilGrids value"})
    order = d.groupby("group")[col].median().sort_values().index.tolist()
    plot = px.box if kind == "Box" else px.violin
    fig = plot(d, x="group", y=col, color="group", category_orders={"group": order}, height=480,
               color_discrete_map=C.LABEL_COLORS if by == "Recommendation" else None)
    fig.update_layout(showlegend=False, margin=dict(t=10), xaxis_title="")
    st.plotly_chart(fig, width="stretch")
    g = d.groupby("group")[col].agg(["count", "mean", "median", "std", "min", "max"]).reindex(order)
    st.dataframe(g.round(3), width="stretch")
    samples = [v.dropna().to_numpy() for _, v in d.groupby("group")[col] if v.notna().sum() > 1]
    if len(samples) > 1:
        h, p = stats.kruskal(*samples)
        st.caption(f"Kruskal-Wallis test across groups: H = {h:,.1f}, p = {p:.2e} (with 9,333 spatially autocorrelated "
                   "cells almost any difference is 'significant'; look at the effect size in the table).")

# ----------------------------------------------------------------------------------------------------------
elif section == "Correlations":
    a, b = st.columns([3, 1])
    groups = a.multiselect("Column groups", ORDER, default=["Climate", "Soil", "Terrain", "Land cover", "GAEZ index (1 km)", "GCS 2024"])
    method = b.segmented_control("Method", ["spearman", "pearson"], default="spearman", key="eda_method") or "spearman"
    cols = [c for c in NUMERIC if GROUP_OF[c] in groups]
    if len(cols) < 2:
        st.info("Pick at least one group with two or more columns.")
    else:
        cm = df[cols].corr(method=method)
        fig = px.imshow(cm, color_continuous_scale=C.DIV, zmin=-1, zmax=1, aspect="auto",
                        height=max(450, 18 * len(cols)), text_auto=".2f" if len(cols) <= 30 else False)
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        tri = cm.where(np.triu(np.ones(cm.shape, dtype=bool), 1)).stack().rename("r").reset_index()
        tri.columns = ["column A", "column B", method]
        tri["group A"] = tri["column A"].map(GROUP_OF)
        tri["group B"] = tri["column B"].map(GROUP_OF)
        st.markdown("**Strongest pairs**")
        st.dataframe(tri.reindex(tri[method].abs().sort_values(ascending=False).index).head(40).round(3),
                     hide_index=True, width="stretch", height=360)
    st.markdown("**Physical drivers vs crop suitability** (which conditions go with high suitability for each crop)")
    drivers = C.BASE_FEATURES + ["cropland_pct"]
    targets = [C.gaez_col(c) for c in C.CROPS] + [f"gcs2024_{c}" for c in C.GCS_CROPS]
    dm = df[drivers + targets].corr(method=method).loc[drivers, targets]
    fig = px.imshow(dm, color_continuous_scale=C.DIV, zmin=-1, zmax=1, text_auto=".2f", aspect="auto", height=560)
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------------------------------------
elif section == "Relationships":
    a, b, c, d = st.columns(4)
    x = a.selectbox("X", NUMERIC, index=NUMERIC.index("elevation_mean_m"), format_func=fmt)
    y = b.selectbox("Y", NUMERIC, index=NUMERIC.index("bio1_temp_mean_c"), format_func=fmt)
    color = c.selectbox("Colour", ["Recommendation", "Land cover", "None"] + NUMERIC)
    trend = d.segmented_control("Trend line", ["none", "ols", "lowess"], default="ols", key="eda_trend") or "none"
    kw = {}
    if color == "Recommendation":
        kw = dict(color="recommended_crop", color_discrete_map=C.LABEL_COLORS)
    elif color == "Land cover":
        kw = dict(color="landcover_label")
    elif color != "None":
        kw = dict(color=color, color_continuous_scale=C.SEQ)
    density = st.toggle("Density view (no overplotting)", value=False)
    if density:
        fig = px.density_heatmap(df, x=x, y=y, nbinsx=60, nbinsy=60, color_continuous_scale=C.SEQ, height=520)
    else:
        fig = px.scatter(df, x=x, y=y, opacity=0.5, render_mode="webgl", height=520,
                         trendline=None if trend == "none" else trend,
                         trendline_scope="overall", trendline_color_override="black", **kw)
        fig.update_traces(marker=dict(size=4))
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    ok = df[[x, y]].dropna()
    st.caption(f"Pearson r = {ok[x].corr(ok[y]):.3f} · Spearman rho = {ok[x].corr(ok[y], method='spearman'):.3f} · n = {len(ok):,}")
    st.markdown("**Scatter matrix**")
    pick = st.multiselect("Variables (up to 6)", NUMERIC, default=["bio1_temp_mean_c", "chirps_annual_mean_mm",
                                                                   "elevation_mean_m", "soil_ph", C.gaez_col("rice")],
                          max_selections=6, format_func=fmt)
    if len(pick) >= 2:
        sample = df.sample(min(3000, len(df)), random_state=0)
        fig = px.scatter_matrix(sample, dimensions=pick, color="recommended_crop", color_discrete_map=C.LABEL_COLORS,
                                height=150 * len(pick) + 120, opacity=0.5)
        fig.update_traces(diagonal_visible=False, marker=dict(size=3))
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("3,000 randomly sampled cells, coloured by recommendation.")

# ----------------------------------------------------------------------------------------------------------
elif section == "Spatial patterns":
    pick = st.multiselect("Variables", NUMERIC, default=["bio1_temp_mean_c", "chirps_annual_mean_mm", "elevation_mean_m",
                                                         C.gaez_col("rice"), C.gaez_col("coconut")], format_func=fmt)
    zs = st.toggle("Standardise (z-scores) so the lines share one axis", value=True)
    if pick:
        d = df[pick].copy()
        if zs:
            d = (d - d.mean()) / d.std()
        l, r = st.columns(2)
        ns = d.groupby(df["lat"].round(2)).mean().reset_index().melt(id_vars="lat", var_name="variable", value_name="value")
        fig = px.line(ns, x="lat", y="value", color="variable", height=380, title="South -> north profile (mean per latitude)")
        fig.update_layout(margin=dict(t=40))
        l.plotly_chart(fig, width="stretch")
        we = d.groupby(df["lon"].round(2)).mean().reset_index().melt(id_vars="lon", var_name="variable", value_name="value")
        fig = px.line(we, x="lon", y="value", color="variable", height=380, title="West -> east profile (mean per longitude)")
        fig.update_layout(margin=dict(t=40))
        r.plotly_chart(fig, width="stretch")

    st.markdown("**Spatial autocorrelation** (Moran's I with rook neighbours on the 1 km grid; 1 = smooth field, 0 = random)")

    def morans_i(values):
        z = C.to_grid(df, values)
        z = z - np.nanmean(z)
        num, w = 0.0, 0
        for a, b in [(z[:, :-1], z[:, 1:]), (z[:-1, :], z[1:, :])]:
            ok = ~np.isnan(a) & ~np.isnan(b)
            num += 2 * np.sum(a[ok] * b[ok])
            w += 2 * ok.sum()
        n = np.sum(~np.isnan(z))
        return (n / w) * num / np.nansum(z ** 2)

    mi = pd.Series({c: morans_i(df[c]) for c in C.BASE_FEATURES + ["cropland_pct"] + [C.gaez_col(c) for c in C.CROPS]
                    + [f"gcs2024_{c}" for c in C.GCS_CROPS]}).rename("Moran's I").reset_index()
    mi.columns = ["column", "Moran's I"]
    mi["group"] = mi["column"].map(GROUP_OF)
    fig = px.bar(mi.sort_values("Moran's I"), x="Moran's I", y="column", color="group", orientation="h", height=700)
    fig.update_layout(margin=dict(t=10), yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.caption("Almost everything is strongly autocorrelated in space. That is why random train/test splits overstate "
               "model skill (see the ML/DL Experiments page).")

# ----------------------------------------------------------------------------------------------------------
elif section == "Categories":
    l, r = st.columns(2)
    lc = df["landcover_label"].value_counts().reset_index()
    lc.columns = ["land cover", "cells"]
    fig = px.bar(lc, x="cells", y="land cover", orientation="h", height=360, title="Majority land cover (ESA WorldCover)")
    fig.update_layout(margin=dict(t=40), yaxis_title="", yaxis=dict(autorange="reversed"))
    l.plotly_chart(fig, width="stretch")
    fig = px.box(df, x="recommended_crop", y="cropland_pct", color="recommended_crop", color_discrete_map=C.LABEL_COLORS,
                 height=360, title="Cropland share by recommendation")
    fig.update_layout(showlegend=False, margin=dict(t=40), xaxis_title="")
    r.plotly_chart(fig, width="stretch")
    norm = st.toggle("Show shares within each land-cover class", value=True)
    ct = pd.crosstab(df["landcover_label"], df["recommended_crop"], normalize="index" if norm else False)
    fig = px.imshow(ct.round(3 if norm else 0), text_auto=".0%" if norm else True, color_continuous_scale=C.SEQ,
                    aspect="auto", height=420, title="Land cover x recommendation")
    fig.update_layout(margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")
    classes = C.gaez_classes()
    cls = pd.DataFrame({C.CROP_LABELS[c]: df[C.gaez_col(c, "class")].round().value_counts(normalize=True) for c in C.CROPS}).T.fillna(0)
    cls.columns = [f"{int(k)} {classes.get(int(k), '')}" for k in cls.columns]
    shades = {"Very high": "#1F4D2E", "High": "#3F7A3E", "Good": "#6E9E5A", "Medium": "#A3C383", "Moderate": "#D8D59A",
              "Marginal": "#E2B97A", "Very marginal": "#CF8A5C", "Not suitable": "#A65A39", "No cropland": "#CFCAC0",
              "Water": C.TEAL}
    class_colors = {f"{k} {v}": shades[v] for k, v in classes.items()}
    fig = px.bar(cls.reset_index(names="crop").melt(id_vars="crop", var_name="GAEZ class", value_name="share"), x="crop",
                 y="share", color="GAEZ class", height=420, title=f"GAEZ suitability classes per crop at {C.gaez_res()} (share of cells)",
                 color_discrete_map=class_colors, category_orders={"GAEZ class": list(class_colors)})
    fig.update_layout(margin=dict(t=40), xaxis_title="", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------------------------------------
elif section == "Irrigated vs rain-fed":
    st.caption("The warehouse keeps GAEZ's rain-fed scenario (HRLM) for the first four crops as audit columns at both "
               "resolutions; the dashboard and all models use the irrigated high-input scenario (HILM). Shown at GAEZ "
               f"{C.gaez_res()} (switch in the sidebar).")
    four = ["rice", "maize", "cassava", "soybean"]
    irr = {c: C.gaez_col(c) for c in four}
    rf = {c: f"{C.gaez_col(c)}_rainfed" for c in four}
    means = pd.DataFrame({"irrigated (used)": [df[irr[c]].mean() for c in four],
                          "rain-fed (audit)": [df[rf[c]].mean() for c in four]},
                         index=[C.CROP_LABELS[c] for c in four])
    l, r = st.columns(2)
    fig = px.bar(means.reset_index(names="crop").melt(id_vars="crop", var_name="scenario", value_name="mean index"),
                 x="crop", y="mean index", color="scenario", barmode="group", height=380)
    fig.update_layout(margin=dict(t=10), xaxis_title="")
    l.plotly_chart(fig, width="stretch")
    crop = r.selectbox("Crop", four, format_func=C.CROP_LABELS.get)
    fig = px.density_heatmap(df, x=rf[crop], y=irr[crop], nbinsx=40, nbinsy=40, height=330,
                             color_continuous_scale=C.SEQ, labels={rf[crop]: "rain-fed", irr[crop]: "irrigated"})
    fig.update_layout(margin=dict(t=10))
    r.plotly_chart(fig, width="stretch")
    st.plotly_chart(C.grid_map(df, df[irr[crop]] - df[rf[crop]],
                               f"{C.CROP_LABELS[crop]}: irrigated minus rain-fed index", C.DIV, -6000, 6000, 430, "difference"),
                    width="stretch")
    win = {}
    for scen, cols in [("irrigated", irr), ("rain-fed", rf)]:
        sc = df[[cols[c] for c in four]].to_numpy(float)
        empty = np.isnan(sc).all(axis=1)
        sc = np.nan_to_num(sc, nan=-1.0)  # a crop without a value cannot win the cell
        lab = np.array([C.CROP_LABELS[four[i]] for i in sc.argmax(axis=1)], dtype=object)
        lab[sc.max(axis=1) < C.NOT_RECOMMENDED_THRESHOLD] = "Not recommended"
        lab[empty] = "No cropland / water"
        win[scen] = pd.Series(lab).value_counts()
    st.markdown("**Four-crop recommendation under each scenario** (cells)")
    st.dataframe(pd.DataFrame(win).fillna(0).astype(int), width="stretch")

# ----------------------------------------------------------------------------------------------------------
else:  # PCA
    X = df[C.BASE_FEATURES].fillna(df[C.BASE_FEATURES].median())
    Z = ((X - X.mean()) / X.std()).to_numpy()
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    ev = S ** 2 / np.sum(S ** 2)
    pcs = Z @ Vt.T
    l, r = st.columns([1, 1.4])
    evd = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(ev))], "explained": ev, "cumulative": np.cumsum(ev)})
    fig = px.bar(evd, x="component", y="explained", height=380, title="Explained variance (14 standardised drivers)")
    fig.add_scatter(x=evd["component"], y=evd["cumulative"], mode="lines+markers", name="cumulative")
    fig.update_layout(margin=dict(t=40), yaxis_tickformat=".0%", showlegend=False)
    l.plotly_chart(fig, width="stretch")
    load = pd.DataFrame(Vt[:4].T, index=C.BASE_FEATURES, columns=["PC1", "PC2", "PC3", "PC4"])
    fig = px.imshow(load, color_continuous_scale=C.DIV, zmin=-0.7, zmax=0.7, text_auto=".2f", aspect="auto", height=380,
                    title="Loadings")
    fig.update_layout(margin=dict(t=40))
    r.plotly_chart(fig, width="stretch")
    color = st.selectbox("Colour the cells by", ["Recommendation"] + NUMERIC, format_func=lambda c: c if c == "Recommendation" else fmt(c))
    pdf = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "recommended_crop": df["recommended_crop"]})
    if color == "Recommendation":
        fig = px.scatter(pdf, x="PC1", y="PC2", color="recommended_crop", color_discrete_map=C.LABEL_COLORS, opacity=0.5,
                         render_mode="webgl", height=480)
    else:
        pdf[color] = df[color]
        fig = px.scatter(pdf, x="PC1", y="PC2", color=color, color_continuous_scale=C.SEQ, opacity=0.6,
                         render_mode="webgl", height=480)
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    m1, m2 = st.columns(2)
    m1.plotly_chart(C.grid_map(df, pcs[:, 0], "PC1 on the map", C.DIV, height=380), width="stretch")
    m2.plotly_chart(C.grid_map(df, pcs[:, 1], "PC2 on the map", C.DIV, height=380), width="stretch")
