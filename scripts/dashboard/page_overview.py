import plotly.express as px
import streamlit as st

import common as C

df = C.load_data()
options = C.recommendation_options(df)
group, ranking = C.rule()
lake = C.lake_inventory()
results = sorted([*C.RESULTS_DIR.glob("m*_*.csv"), *C.RESULTS_DIR.glob("g*_*.csv")]) if C.RESULTS_DIR.exists() else []

C.page_header(
    "Banten Province · 1 km analysis grid",
    "GeoSuitability AI",
    "Which crop fits which piece of land? Ten open geodata sources flow through a local **data lake** and "
    "**warehouse** into one table of 9,333 one-kilometre cells, scored for ten crops with GAEZ v5 at 1 km and described by "
    "Sentinel-2 imagery, and sixteen machine-learning / deep-learning experiments plus a graph analysis test how far those suitability maps can "
    "be trusted.",
)

k = st.columns(5)
k[0].metric("Grid cells", f"{len(df):,}", help="1 km x 1 km cells covering the province", border=True)
k[1].metric("Data lake", f"{lake['size_bytes'].sum() / 1e9:.1f} GB", help="Everything under datasets/raw, as ingested", border=True)
n_s2 = sum(c.startswith("s2_") for c in df.columns)
k[2].metric("Table columns", f"{len(C.fact_columns())}", border=True,
            help="tabular_features.csv, one row per cell" + (f" (+ {n_s2} Sentinel-2 columns in a 1:1 table)" if n_s2 else ""))
k[3].metric("Crops scored", "10 + 6", help="10 crops from GAEZ v5, 6 of them also from GCS 2024", border=True)
k[4].metric("Experiment tables", f"{len(results)}", help="Result tables written by scripts/modeling m01-m16 and g01", border=True)

C.section("Crop recommendation", f"{group} · {ranking.lower()} ranking · GAEZ {C.gaez_res()} · colour depth = how suitable the winning crop is")
left, right = st.columns([1.35, 1])
with left:
    rec = C.recommendation()
    hover = df["recommended_crop"] + "<br>GAEZ index of the winner: " + C.fmt_index(rec["winner_index"])
    st.plotly_chart(C.category_map(df, df["recommended_crop"], C.LABEL_COLORS, height=470,
                                   strength=C.shading(df, C.SHADINGS[0]), hover=hover), width="stretch")
with right:
    dist = df["recommended_crop"].value_counts().reindex(options).dropna().reset_index()
    dist.columns = ["Recommendation", "Cells"]
    dist["Share"] = dist["Cells"] / len(df)
    fig = px.bar(dist.sort_values("Cells"), x="Cells", y="Recommendation", orientation="h", color="Recommendation",
                 color_discrete_map=C.LABEL_COLORS, text=dist.sort_values("Cells")["Share"].map("{:.1%}".format), height=470)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, margin=dict(t=10, r=40), yaxis_title="", xaxis_title="cells")
    st.plotly_chart(fig, width="stretch")
C.note(C.rule_description() + " Colour depth: pale = barely suitable, deep = very suitable (the winning crop's GAEZ "
       "index). Change the rule in the sidebar; every crop has its own map on the Suitability Map page.")

summ, lead, agr = C.load_result("m01_summary_pixel"), C.load_result("m01_leaderboard"), C.load_result("m08_agreement")
if summ is not None and lead is not None:
    C.section("Key findings", "computed from the result tables of the modeling notebooks")
    best = summ.iloc[0]
    gap = lead.set_index("model").loc[best["model"], "leakage_gap"]
    names = {"et": "ExtraTrees", "hgb": "HistGB", "xgb": "XGBoost", "knn": "kNN", "ridge": "Ridge"}
    cards = [
        (f"R² {best['r2']:.2f}", f"{names.get(best['model'], best['model'])} reproduces the 1 km GAEZ on ~9 km blocks it never "
                                 "saw, from 14 climate, soil and terrain drivers."),
        (f"+{gap:.2f}", "How much a random train/test split flatters the same model: neighbouring cells are nearly alike."),
    ]
    dl = [C.load_result(n) for n in ("m09_mlp_vs_trees", "m10_grid_networks")]
    dl = [t.rename(columns={"Unnamed: 0": "m"}) for t in dl if t is not None]
    if dl:
        best_dl = max(t[t["m"].str.contains("MLP|U-Net")]["r2"].max() for t in dl)
        verdict = "close, but tree ensembles win on this table" if best_dl < best["r2"] else "it edges out the tree ensembles"
        cards.append((f"R² {best_dl:.2f}", f"Best neural network (GPU MLP or U-Net on the grid): {verdict}."))
    s2 = C.load_result("m14_summary")
    if s2 is not None:
        top = s2.sort_values("r2").iloc[-1]
        cards.append((f"R² {top['r2']:.2f}", f"With Sentinel-2 imagery ({top['setting']}, m14): the satellite data adds "
                                             "land-use information the physical drivers lack."))
    lb13 = C.load_result("m13_leaderboard")
    if lb13 is not None:
        b9, b1 = (lb13[lb13["resolution"] == r].sort_values("pixel").iloc[-1] for r in ("9km", "1km"))
        cards.append((f"R² {b9['pixel']:.2f}", f"The same models on the ~9 km product of the first round (m13); random K-fold "
                                               f"flattered them by +{b9['leakage_gap']:.2f} there, +{b1['leakage_gap']:.2f} at 1 km."))
    if agr is not None:
        a = agr.set_index("crop")["spearman_cell"].sort_values()
        strong = " and ".join(C.CROP_LABELS[c].lower() for c in a.index[-2:][::-1])
        low = f"{'negative' if a.iloc[0] < 0 else 'weak'} for {C.CROP_LABELS[a.index[0]].lower()}"
        cards.append((f"ρ {a.min():.2f} to {a.max():.2f}",
                      f"Agreement between GAEZ and the independent GCS 2024 product: strongest for {strong}, {low}."))
    ml = C.load_result("m16_summary")
    if ml is not None:
        rf = ml[ml["model"] == "Random forest"].set_index("framework")
        if {"scikit-learn", "Spark MLlib"} <= set(rf.index):
            secs = "wall seconds (10 crops x 5 folds)"
            cards.append((f"R² {rf.loc['Spark MLlib', 'r2']:.2f} = {rf.loc['scikit-learn', 'r2']:.2f}",
                          f"The same random forest in Spark MLlib and in scikit-learn (m16); on one machine Spark takes "
                          f"{rf.loc['Spark MLlib', secs] / rf.loc['scikit-learn', secs]:.0f} times longer, the price of a "
                          "framework built to scale out."))
    nodes = C.load_result("g01_crop_nodes")
    if nodes is not None and "crops nested inside it" in nodes:
        with_land = nodes[nodes["suitable km2"] > 0]
        widest = with_land.sort_values(["crops nested inside it", "suitable km2"]).iloc[-1]
        k, n = int(widest["crops nested inside it"]), len(with_land) - 1
        cards.append((f"{k} of {n}", f"crops have their suitable land (95 % or more) inside that of {widest['crop'].lower()} "
                                     "(g01)" + (": the crops compete for one core of good land rather than dividing the "
                                                "province." if k >= n / 2 else ".")))
    for start in range(0, len(cards), 3):   # rows of three cards
        for col, (big, text) in zip(st.columns(3), cards[start:start + 3]):
            with col.container(border=True):
                st.markdown(f"<div style='font-size:1.6rem;font-weight:700;color:{C.PINE};letter-spacing:-.02em'>{big}</div>",
                            unsafe_allow_html=True)
                st.caption(text)

C.section("Explore")
guide = [
    ("page_map.py", "Suitability Map", ":material/map:", "The recommendation, each crop's GAEZ and GCS score, and every raw layer on a basemap."),
    ("page_gaez_gcs.py", "GAEZ x GCS 2024", ":material/compare_arrows:", "Two independent products combined: agreement, a consensus score, a combined recommendation."),
    ("page_lake.py", "Data Lake & Warehouse", ":material/database:", "Architecture, lake inventory, warehouse schema, table relations, lineage, Spark output, data quality."),
    ("page_eda.py", "Exploratory Analysis", ":material/query_stats:", "Every column profiled: distributions, correlations, relationships, spatial patterns, PCA."),
    ("page_explorer.py", "Data Explorer", ":material/table_view:", "Filter the warehouse table and download exactly what you see."),
    ("page_experiments.py", "ML/DL Experiments", ":material/science:", "Sixteen modeling notebooks, scored with honest spatial validation, with and without Spark MLlib."),
    ("page_ai_lab.py", "AI Lab", ":material/smart_toy:", "Predict any cell, what-if sliders, a live model arena and a deep-learning playground."),
    ("page_graph.py", "Graph Analytics", ":material/hub:", "The grid as a graph: contiguous suitable land, bottlenecks, communities, crop competition, GraphFrames on Spark."),
]
cols = st.columns(2)
for i, (page, label, icon, text) in enumerate(guide):
    with cols[i % 2].container(border=True):
        st.page_link(page, label=label, icon=icon)
        st.caption(text)
