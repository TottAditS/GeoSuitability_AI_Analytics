import plotly.express as px
import streamlit as st

import common as C

df = C.load_data()
CAT = C.column_catalogue(df.columns)

C.page_header("Data", "Data Explorer", "Filter the warehouse fact table, inspect it and download exactly what you see.")

with st.container(border=True):
    a, b = st.columns(2)
    recs = a.multiselect("Recommendation", C.recommendation_options(df), default=C.recommendation_options(df))
    lcs = b.multiselect("Land cover", sorted(df["landcover_label"].unique()), default=sorted(df["landcover_label"].unique()))
    c, d = st.columns(2)
    elev = df["elevation_mean_m"]
    lo, hi = c.slider("Elevation (m)", 0.0, float(elev.max()), (0.0, float(elev.max())), step=10.0)
    crop = d.selectbox("Minimum GAEZ index for crop", ["(none)"] + C.CROPS, format_func=lambda k: C.CROP_LABELS.get(k, k))
    min_idx = d.slider("Minimum index", 0, 10000, 0, 250, disabled=crop == "(none)")

mask = df["recommended_crop"].isin(recs) & df["landcover_label"].isin(lcs)
mask &= elev.between(lo, hi) | (elev.isna() & (lo == 0.0))
if crop != "(none)":
    mask &= df[C.gaez_col(crop)] >= min_idx
shown = df[mask]

groups = st.multiselect("Column groups to show", list(dict.fromkeys(CAT["group"])),
                        default=["Key", "Location", "Derived (dashboard)", "Climate", "Soil", "Terrain", "Land cover", "GAEZ index (1 km)"])
cols = CAT.loc[CAT["group"].isin(groups), "column"].tolist()

m = st.columns(3)
m[0].metric("Cells shown", f"{len(shown):,}", help=f"{len(shown) / len(df):.0%} of all cells", border=True)
m[1].metric("Columns shown", f"{len(cols)}", border=True)
m[2].metric("Area", f"{len(shown):,} km²", border=True)

l, r = st.columns([1.2, 1])
with l:
    st.dataframe(shown[cols], width="stretch", height=430, hide_index=True)
    st.download_button("Download the filtered table (CSV)", shown[cols].to_csv(index=False).encode("utf-8"),
                       file_name="tabular_features_filtered.csv", mime="text/csv")
with r:
    if shown.empty:
        st.warning("No grid cells match the current filter.")
    else:
        st.plotly_chart(C.category_map(df.loc[shown.index], shown["recommended_crop"], C.LABEL_COLORS,
                                       "Filtered cells", 330), width="stretch")
        num = [c for c in cols if c in shown.select_dtypes("number").columns and c not in ("row", "col")]
        if num:
            var = st.selectbox("Histogram of", num, index=0)
            fig = px.histogram(shown, x=var, nbins=40, height=260, color_discrete_sequence=[C.PINE])
            fig.update_layout(margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
