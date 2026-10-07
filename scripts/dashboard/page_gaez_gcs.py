import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import common as C

df = C.load_data()
GC = C.GCS_CROPS
has_gcs = df[[f"gcs2024_{c}" for c in GC]].notna().all(axis=1)

C.page_header(
    "Suitability · cross-check", "GAEZ x Global Crop Suitability 2024",
    f"Two independent suitability products for the six crops they share: **GAEZ v5** (FAO / IIASA, 0-10,000 scale, "
    f"shown here at {C.gaez_res()}; switch in the sidebar) and **GCS 2024** (Zenodo / Scientific Data, a different "
    f"model at ~1 km on a 0-1 scale). {int(has_gcs.sum()):,} of {len(df):,} cells have a GCS value (the rest are "
    "coastal cells outside its land mask)."
)
no_gaez = df[[C.gaez_col(c) for c in GC]].isna().all(axis=1)  # 1 km: GAEZ marks the cell as no cropland or water

with st.container(border=True):
    a, b, c, d = st.columns([1.2, 1.2, 1, 1])
    method = a.selectbox("How to combine", ["Weighted mean", "Conservative (minimum)", "Rank average"],
                         help="Weighted mean of the two 0-1 scores; minimum = a cell is only as good as the less "
                              "optimistic product; rank average = mean of the two percentile ranks (ignores calibration).")
    w = b.slider("Weight on GAEZ", 0.0, 1.0, 0.5, 0.05, disabled=method != "Weighted mean")
    t_gaez = c.slider("GAEZ 'suitable' from", 0, 10000, C.NOT_RECOMMENDED_THRESHOLD, 250)
    t_gcs = d.slider("GCS 'suitable' from", 0.0, 1.0, 0.5, 0.05)
    t_comb = st.slider("Combined score needed for a recommendation", 0.0, 1.0, 0.5, 0.05,
                       help="A crop is eligible in a cell if its combined score reaches this value; the eligible crop "
                            "with the highest combined score wins, otherwise the cell is 'Not recommended'.")


def scaled(crop):
    return df[C.gaez_col(crop)] / 10000.0, df[f"gcs2024_{crop}"]


def combine(g, s):
    if method == "Weighted mean":
        return w * g + (1 - w) * s
    if method == "Conservative (minimum)":
        return np.minimum(g, s)
    return (g.rank(pct=True) + s.rank(pct=True)) / 2


combined = pd.DataFrame({c: combine(*scaled(c)) for c in GC}).where(has_gcs)

tab_crop, tab_rec, tab_stats, tab_dis = st.tabs(["Per crop", "Combined recommendation", "Agreement statistics", "Disagreement"])

with tab_crop:
    crop = st.selectbox("Crop", GC, format_func=C.CROP_LABELS.get)
    g, s = scaled(crop)
    m1, m2, m3 = st.columns(3)
    m1.plotly_chart(C.grid_map(df, g, f"GAEZ {C.gaez_res()} (index / 10,000)", C.crop_scale(crop), 0, 1, 360), width="stretch")
    m2.plotly_chart(C.grid_map(df, s, "GCS 2024 (0-1)", C.crop_scale(crop), 0, 1, 360), width="stretch")
    m3.plotly_chart(C.grid_map(df, combined[crop], f"Combined ({method.lower()})", C.crop_scale(crop), 0, 1, 360), width="stretch")

    g_ok, s_ok = df[C.gaez_col(crop)] >= t_gaez, df[f"gcs2024_{crop}"] >= t_gcs
    agree = (pd.Series(np.select([g_ok & s_ok, g_ok & ~s_ok, ~g_ok & s_ok], ["Both suitable", "GAEZ only", "GCS only"],
                                 "Neither"), index=df.index)
             .where(df[C.gaez_col(crop)].notna(), "No GAEZ value").where(has_gcs, "No GCS value"))
    colors = {"Both suitable": C.PINE, "GAEZ only": C.OCHRE, "GCS only": C.TEAL, "Neither": "#E6E3DA",
              "No GAEZ value": C.LABEL_COLORS["No cropland / water"], "No GCS value": "#FFFFFF"}
    left, right = st.columns([1.3, 1])
    left.plotly_chart(C.category_map(df, agree, colors, "Where the products agree", 430), width="stretch")
    share = agree.value_counts(normalize=True).reindex(list(colors)).fillna(0).mul(100).round(1).rename("% of cells")
    right.dataframe(share.to_frame(), width="stretch")
    sub = df[has_gcs]
    fig = px.density_heatmap(sub, x=C.gaez_col(crop), y=f"gcs2024_{crop}", nbinsx=40, nbinsy=40,
                             color_continuous_scale=C.SEQ, height=330,
                             labels={C.gaez_col(crop): "GAEZ index", f"gcs2024_{crop}": "GCS 2024"})
    fig.add_vline(x=t_gaez, line_dash="dash", line_color="grey")
    fig.add_hline(y=t_gcs, line_dash="dash", line_color="grey")
    fig.update_layout(margin=dict(t=10))
    right.plotly_chart(fig, width="stretch")
    rho = sub[C.gaez_col(crop)].rank().corr(sub[f"gcs2024_{crop}"].rank())
    right.caption(f"Spearman rank correlation between the two products for {C.CROP_LABELS[crop]}: **{rho:.2f}**.")

with tab_rec:
    scored = combined.notna().any(axis=1)  # pandas 3: idxmax refuses rows that are all NaN
    best = combined[scored].idxmax(axis=1).reindex(df.index)
    best_score = combined.max(axis=1)
    rec = (best.map(C.CROP_LABELS).where(best_score >= t_comb, "Not recommended")
           .where(~no_gaez, "No cropland / water").where(has_gcs, "No GCS value"))
    g6 = df[[C.gaez_col(c) for c in GC]].fillna(-1)
    gaez_rec = pd.Series(np.array([C.CROP_LABELS[GC[i]] for i in g6.to_numpy().argmax(axis=1)], dtype=object), index=df.index)
    gaez_rec = gaez_rec.where(g6.max(axis=1) >= C.NOT_RECOMMENDED_THRESHOLD, "Not recommended").where(~no_gaez, "No cropland / water")
    colors = {**{C.CROP_LABELS[c]: C.LABEL_COLORS[C.CROP_LABELS[c]] for c in GC}, "Not recommended": C.STONE,
              "No cropland / water": C.LABEL_COLORS["No cropland / water"], "No GCS value": "#FFFFFF"}
    l, r = st.columns(2)
    l.plotly_chart(C.category_map(df, gaez_rec, colors, "GAEZ only (same 6 crops, 5,000 rule)", 430,
                                  strength=g6.max(axis=1) / 10000), width="stretch")
    r.plotly_chart(C.category_map(df, rec, colors, f"GAEZ + GCS combined ({method.lower()})", 430,
                                  strength=best_score.where(has_gcs)), width="stretch")
    st.caption("Colour depth: left = the winning crop's GAEZ index, right = its combined GAEZ + GCS score "
               "(pale = barely suitable, deep = very suitable)."
               + (" The right map is sharper because GCS is a 1 km product; switch GAEZ to 1 km in the sidebar to "
                  "compare like with like." if C.gaez_res() != "1 km" else ""))
    counts = pd.DataFrame({"GAEZ only": gaez_rec.value_counts(), "Combined": rec.value_counts()}).fillna(0).astype(int)
    fig = px.bar(counts.reset_index(names="label").melt(id_vars="label", var_name="source", value_name="cells"),
                 x="label", y="cells", color="source", barmode="group", height=330)
    fig.update_layout(margin=dict(t=10), xaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.markdown("**How the combination changes the answer** (rows: GAEZ-only label, columns: combined label, cells)")
    ct = pd.crosstab(gaez_rec.where(has_gcs, "No GCS value"), rec)
    fig = px.imshow(ct, text_auto=True, color_continuous_scale=C.SEQ, aspect="auto", height=380)
    fig.update_layout(margin=dict(t=10), xaxis_title="combined", yaxis_title="GAEZ only")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{(gaez_rec[has_gcs] == rec[has_gcs]).mean():.0%} of the cells with a GCS value keep the same label after "
               "adding GCS 2024. Labels change most where the two products disagree on a crop (see the next tabs).")

with tab_stats:
    rows = []
    for c in GC:
        both = has_gcs & df[C.gaez_col(c)].notna()
        g, s = df.loc[both, C.gaez_col(c)], df.loc[both, f"gcs2024_{c}"]
        g_ok, s_ok = g >= t_gaez, s >= t_gcs
        rows.append({"crop": C.CROP_LABELS[c], "Spearman": g.rank().corr(s.rank()), "Pearson": g.corr(s),
                     "mean GAEZ (0-1)": (g / 10000).mean(), "mean GCS (0-1)": s.mean(),
                     "% both suitable": (g_ok & s_ok).mean() * 100, "% GAEZ only": (g_ok & ~s_ok).mean() * 100,
                     "% GCS only": (~g_ok & s_ok).mean() * 100, "% neither": (~g_ok & ~s_ok).mean() * 100})
    stats = pd.DataFrame(rows).set_index("crop")
    st.dataframe(stats.round(2), width="stretch")
    fig = px.bar(stats.reset_index(), x="crop", y="Spearman", color="Spearman", color_continuous_scale=C.DIV,
                 range_color=(-1, 1), height=320)
    fig.update_layout(margin=dict(t=10), xaxis_title="")
    st.plotly_chart(fig, width="stretch")
    pixel = np.floor((90 - df["lat"]) * 12) * 100000 + np.floor((df["lon"] + 180) * 12)  # the 5 arc-min GAEZ pixel
    pix_var = []
    for c in GC:
        for name, v in [("GAEZ ~9 km", df[f"gaez_index_{c}"] / 10000), ("GAEZ 1 km", df[f"gaez1k_index_{c}"] / 10000),
                        ("GCS 2024", df[f"gcs2024_{c}"])]:
            v = v[has_gcs]
            within = (v - v.groupby(pixel[has_gcs]).transform("mean")).var()
            pix_var.append({"crop": C.CROP_LABELS[c], "product": name, "share of variance inside ~9 km pixels": within / v.var()})
    fig = px.bar(pd.DataFrame(pix_var), x="crop", y="share of variance inside ~9 km pixels", color="product",
                 barmode="group", height=300, color_discrete_map={"GAEZ ~9 km": C.SAGE, "GAEZ 1 km": C.PINE, "GCS 2024": C.TEAL})
    fig.update_layout(margin=dict(t=10), xaxis_title="", legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    st.caption("The second chart shows resolution: how much of each product's variation lies inside the ~9 km GAEZ "
               "pixels. Almost none for GAEZ at ~9 km (one value per pixel; the small rest comes from cells straddling a "
               "pixel edge), a fifth to a quarter for GAEZ at 1 km and up to a third for GCS 2024: local detail the "
               "~9 km product cannot show. Agreement ranges from strong (rice, sugarcane) to negative (soybean) at both "
               "GAEZ resolutions: the two models disagree on what limits soybean in Banten, which is itself a finding.")

with tab_dis:
    crop = st.selectbox("Crop ", GC, format_func=C.CROP_LABELS.get, key="dis_crop")
    g, s = df[C.gaez_col(crop)], df[f"gcs2024_{crop}"]
    diff = (g.rank(pct=True) - s.rank(pct=True)).where(has_gcs)
    l, r = st.columns([1.3, 1])
    l.plotly_chart(C.grid_map(df, diff, "GAEZ rank minus GCS rank (green: GAEZ more optimistic, clay: GCS more optimistic)", C.DIV, -1, 1, 460,
                              "percentile<br>difference"), width="stretch")
    fig = px.histogram(diff.dropna(), nbins=50, height=260)
    fig.update_layout(showlegend=False, margin=dict(t=10), xaxis_title="percentile difference")
    r.plotly_chart(fig, width="stretch")
    top = df.assign(diff=diff).dropna(subset=["diff"]).reindex(diff.abs().sort_values(ascending=False).index[:15])
    r.dataframe(top[["grid_id", "lon", "lat", C.gaez_col(crop), f"gcs2024_{crop}", "diff", "landcover_label", "elevation_mean_m"]]
                .round(3), hide_index=True, width="stretch", height=300)
    r.caption("Cells where the two products disagree most: good places to check against local knowledge or imagery.")
