import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ai_models as A
import common as C

df = C.load_data()
CROP_NAMES = [C.CROP_LABELS[c] for c in C.CROPS]

C.page_header("AI · interactive", "AI Lab", "Try the models yourself. Every model here predicts the ten GAEZ suitability indices of a 1 km cell from its "
           "14 physical drivers (climate, soil, terrain), optionally with the Sentinel-2 statistics of the cell, at the GAEZ "
           "resolution chosen in the sidebar; the deep-learning playground trains a neural network live, including the "
           "image-patch fusion network of m14.")

SECTIONS = ["Predict a location", "What-if simulator", "Model arena", "Deep learning playground"]
section = st.segmented_control("Section", SECTIONS, default=SECTIONS[0], key="ai_section") or SECTIONS[0]

# ------------------------------------------------------------------------------------------- cell selection
if "ai_grid_id" not in st.session_state:
    st.session_state.ai_grid_id = df.loc[df[C.gaez_col("rice")].idxmax(), "grid_id"]


def _random_cell():
    st.session_state.ai_grid_id = df["grid_id"].sample(1).iloc[0]


def _nearest_cell():
    d = (df["lat"] - st.session_state.ai_lat) ** 2 + (df["lon"] - st.session_state.ai_lon) ** 2
    st.session_state.ai_grid_id = df.at[d.idxmin(), "grid_id"]


def cell_picker():
    with st.container(border=True):
        a, b, c, d, e = st.columns([1.4, 1, 1, 0.9, 0.9], vertical_alignment="bottom")
        a.selectbox("Grid cell", df["grid_id"].tolist(), key="ai_grid_id")
        b.number_input("Latitude", -7.1, -5.7, -6.40, 0.01, key="ai_lat", format="%.3f")
        c.number_input("Longitude", 105.0, 106.9, 106.00, 0.01, key="ai_lon", format="%.3f")
        d.button("Nearest cell", on_click=_nearest_cell, width="stretch")
        e.button("Random cell", on_click=_random_cell, width="stretch")
    row = df[df["grid_id"] == st.session_state.ai_grid_id].iloc[0]
    return row


def locator(row, height=330):
    fig = C.category_map(df, df["recommended_crop"], C.LABEL_COLORS, None, height)
    fig.add_trace(go.Scatter(x=[row["col"]], y=[row["row"]], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(symbol="star", size=18, color=C.FOREST, line=dict(color="white", width=1.5))))
    fig.update_traces(showscale=False, selector=dict(type="heatmap"))
    return fig


def input_picker(key):
    options = A.available_inputs()
    return st.segmented_control("Model inputs", options, default=options[-1], key=key,
                                help="Sentinel-2 adds the observed land surface (vegetation, moisture, built-up land) of "
                                     "2020-2024 to the 14 physical drivers; m14 shows it raises the honest R² from 0.82 "
                                     "to 0.87.") or options[-1]


def rule_label(pred_row) -> str:
    """The sidebar recommendation rule applied to one row of predicted indices (percentiles against all of Banten)."""
    group, ranking = C.rule()
    one = pd.DataFrame([pred_row], columns=[C.gaez_col(c) for c in C.CROPS])
    return C.recommend(one, C.CROP_GROUPS[group], ranking, reference=df, res=C.gaez_res())["label"].iloc[0]


# ------------------------------------------------------------------------------------------- predict
if section == "Predict a location":
    row = cell_picker()
    inputs = input_picker("ai_inputs")
    bundle = A.get_models(C.gaez_res(), inputs)
    x = row[bundle["features"]].to_numpy(float)
    preds = A.predict_all(bundle, x)

    l, r = st.columns([1, 1.3])
    with l:
        st.plotly_chart(locator(row), width="stretch")
        st.markdown(f"**{row['grid_id']}** · lon {row['lon']:.3f}, lat {row['lat']:.3f} · {row['landcover_label']} · "
                    f"elevation {row['elevation_mean_m']:.0f} m · {row['chirps_annual_mean_mm']:.0f} mm/yr · "
                    f"{row['bio1_temp_mean_c']:.1f} °C")
        st.markdown(f"Recommendation from GAEZ: **{row['recommended_crop']}**")
    with r:
        honest = st.toggle("Honest mode: also retrain without this cell's GAEZ pixel and its 8 neighbours", value=False,
                           help="The models above were trained on every cell, including this one, so they look better than "
                                "they would on a new place. Honest mode retrains ExtraTrees with a ~27 x 27 km hole around the cell.")
        rows = [{"crop": n, "source": "GAEZ (truth)", "index": row[C.gaez_col(c)]} for c, n in zip(C.CROPS, CROP_NAMES)]
        for m, p in preds.items():
            rows += [{"crop": n, "source": m, "index": v} for n, v in zip(CROP_NAMES, p[0])]
        if honest:
            t = A.E.load_table(complete=False)
            pr = t.loc[t["grid_id"] == row["grid_id"], ["pix_row", "pix_col"]].iloc[0]
            ho = A.heldout_prediction(int(pr["pix_row"]), int(pr["pix_col"]), tuple(float(v) for v in x), C.gaez_res(),
                                      inputs)
            preds["ExtraTrees (held out)"] = ho[None]
            rows += [{"crop": n, "source": "ExtraTrees (held out)", "index": v} for n, v in zip(CROP_NAMES, ho)]
        for c in C.GCS_CROPS:
            if pd.notna(row[f"gcs2024_{c}"]):
                rows.append({"crop": C.CROP_LABELS[c], "source": "GCS 2024 x 10,000", "index": row[f"gcs2024_{c}"] * 10000})
        long = pd.DataFrame(rows)
        fig = px.bar(long, x="crop", y="index", color="source", barmode="group", height=430)
        fig.add_hline(y=C.NOT_RECOMMENDED_THRESHOLD, line_dash="dash", line_color="grey",
                      annotation_text="'recommended' threshold")
        fig.update_layout(margin=dict(t=10), xaxis_title="", yaxis_title="suitability index (0-10,000)")
        st.plotly_chart(fig, width="stretch")

    tab = pd.DataFrame({m: p[0] for m, p in preds.items()}, index=CROP_NAMES)
    tab.insert(0, "GAEZ (truth)", [row[C.gaez_col(c)] for c in C.CROPS])
    labels = {"GAEZ (truth)": row["recommended_crop"], **{m: rule_label(p[0]) for m, p in preds.items()}}
    st.dataframe(tab.round(0), width="stretch")
    st.dataframe(pd.Series(labels, name="recommendation (the rule applied to each column)").to_frame(), width="stretch")

    C.section("How sure is the model?", "90 % interval and label probabilities from held-out errors (notebook m15)")
    unc = C.load_uncertainty()
    if unc is None:
        C.no_results_note("m15_uncertainty.ipynb")
    elif C.gaez_res() != "1 km":
        st.info("The uncertainty estimates (m15) belong to the 1 km target: switch the GAEZ resolution to 1 km in the sidebar.")
    else:
        tag = C.UNCERTAINTY_INPUTS.get(inputs, "drivers")
        u = unc.loc[row.name]
        if pd.isna(u[f"{tag}__pred__{C.CROPS[0]}"]):
            st.info("No uncertainty estimate for this cell: GAEZ leaves it without a 1 km value for some crops.")
        else:
            est = np.array([u[f"{tag}__pred__{c}"] for c in C.CROPS])
            lo = np.array([u[f"{tag}__lo__{c}"] for c in C.CROPS])
            hi = np.array([u[f"{tag}__hi__{c}"] for c in C.CROPS])
            ul, ur = st.columns([1.3, 1])
            fig = go.Figure(go.Bar(x=CROP_NAMES, y=est, name="held-out estimate", marker_color=C.SAGE,
                                   error_y=dict(type="data", symmetric=False, array=hi - est, arrayminus=est - lo,
                                                color=C.FOREST, thickness=1.2)))
            fig.add_trace(go.Scatter(x=CROP_NAMES, y=[row[C.gaez_col(c)] for c in C.CROPS], mode="markers", name="GAEZ (truth)",
                                     marker=dict(symbol="diamond", size=10, color=C.CLAY)))
            fig.add_hline(y=C.NOT_RECOMMENDED_THRESHOLD, line_dash="dash", line_color="grey")
            fig.update_layout(height=380, margin=dict(t=10), yaxis_title="suitability index (0-10,000)",
                              legend=dict(orientation="h", y=1.08))
            ul.plotly_chart(fig, width="stretch")
            probs = pd.Series({C.CROP_LABELS.get(lab, lab): u[f"{tag}__plabel__{lab}"] for lab in C.UNCERTAINTY_LABELS})
            probs = probs[probs >= 0.005].sort_values()
            fig = px.bar(probs.rename("probability").rename_axis("label").reset_index(), x="probability", y="label", orientation="h",
                         color="label", color_discrete_map=C.LABEL_COLORS, height=380, range_x=(0, 1),
                         text=probs.map("{:.0%}".format).to_numpy())
            fig.update_layout(showlegend=False, margin=dict(t=10), yaxis_title="", xaxis_title="probability of the recommendation")
            ur.plotly_chart(fig, width="stretch")
            st.dataframe(pd.DataFrame({"held-out estimate": est.round(0), "90 % interval": [f"{a:,.0f} - {b:,.0f}" for a, b in zip(lo, hi)],
                                       "P(index >= 5,000)": [round(u[f"{tag}__psuit__{c}"], 2) for c in C.CROPS]},
                                      index=CROP_NAMES), width="stretch")
            st.caption("The estimate comes from ExtraTrees that never saw this cell's ~9 km block; the bars span the 90 % "
                       "interval calibrated on held-out errors, the diamonds are the true GAEZ values. The probabilities apply "
                       "the absolute rule (highest index wins, 'Not recommended' below 5,000) to 200 plausible versions of the "
                       "cell's ten indices.")

    C.section("Why does the model say that?")
    crop = st.selectbox("Explain the prediction for", C.CROPS, format_func=C.CROP_LABELS.get)
    contrib, base = A.contributions(bundle, x, crop)
    cd = pd.DataFrame({"feature": contrib.index, "contribution": contrib.values,
                       "value here": x, "province median": df[bundle["features"]].median().values})
    cd = cd.reindex(cd["contribution"].abs().sort_values().index)
    cd["effect"] = np.where(cd["contribution"] >= 0, "raises the index", "lowers the index")
    fig = px.bar(cd, x="contribution", y="feature", orientation="h", color="effect", height=max(460, 20 * len(cd)),
                 color_discrete_map={"raises the index": C.FERN, "lowers the index": C.CLAY},
                 hover_data={"value here": ":.2f", "province median": ":.2f", "effect": False})
    fig.update_yaxes(categoryorder="array", categoryarray=cd["feature"].tolist())   # largest effect on top
    fig.update_layout(margin=dict(t=10), yaxis_title="", legend_title_text="", barmode="relative",
                      xaxis_title=f"push away from the average prediction ({base:,.0f}), index points")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"XGBoost SHAP contributions for {C.CROP_LABELS[crop]}: the average prediction over Banten is {base:,.0f}; "
               f"each bar adds or removes index points, ending at {base + contrib.sum():,.0f}. Hover a bar to compare the "
               "cell's value with the province median.")

# ------------------------------------------------------------------------------------------- what-if
elif section == "What-if simulator":
    row = cell_picker()
    bundle = A.get_models(C.gaez_res(), "14 drivers")   # what-if = physical conditions; imagery is observed, not set
    if st.session_state.get("wi_from") != row["grid_id"]:
        for f in A.FEATURES:
            v = row[f]
            st.session_state[f"wi_{f}"] = float(df[f].median() if pd.isna(v) else v)
        st.session_state.wi_from = row["grid_id"]
    st.caption(f"Sliders start at the values of {row['grid_id']} (pick another cell above to reset them).")
    groups = [("Climate", C.FEATURE_GROUPS["Climate"]), ("Soil", C.FEATURE_GROUPS["Soil"]), ("Terrain", C.FEATURE_GROUPS["Terrain"])]
    cols = st.columns(3)
    for (name, feats), col in zip(groups, cols):
        with col.container(border=True):
            st.markdown(f"**{name}**")
            for f in feats:
                lo, hi = float(df[f].min()), float(df[f].max())
                st.slider(C.nice(f), lo, hi, step=(hi - lo) / 200, key=f"wi_{f}", format="%.2f")
    x = np.array([st.session_state[f"wi_{f}"] for f in A.FEATURES])
    preds = A.predict_all(bundle, x)
    ratio, verdict = A.novelty(bundle, x)
    badge = {"typical of Banten": "green", "unusual": "orange"}.get(verdict, "red")
    st.badge(f"This combination is {verdict} (distance to the nearest real cell = {ratio:.1f} x the typical spacing)", color=badge)
    long = pd.DataFrame([{"crop": n, "model": m, "index": v} for m, p in preds.items() for n, v in zip(CROP_NAMES, p[0])])
    q = C.load_result("m15_quantiles")
    if q is not None and C.gaez_res() == "1 km":   # 90 % half-widths of ExtraTrees from held-out errors (m15)
        hw = q[q["inputs"] == "drivers"].set_index("crop")["half_width"]
        long["90 % half-width"] = [hw[c] if m == "ExtraTrees" else np.nan for m in preds for c in C.CROPS]
    fig = px.bar(long, x="crop", y="index", color="model", barmode="group", height=400,
                 error_y="90 % half-width" if "90 % half-width" in long else None)
    fig.add_hline(y=C.NOT_RECOMMENDED_THRESHOLD, line_dash="dash", line_color="grey")
    fig.update_layout(margin=dict(t=10), xaxis_title="", yaxis_title="predicted GAEZ index")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame({m: [rule_label(p[0])] for m, p in preds.items()}, index=["recommendation"]), width="stretch")
    st.caption("The simulator varies the 14 physical drivers, so it uses the models without Sentinel-2 (the imagery is an "
               "observation of the land, not a condition one can set)."
               + (" Error bar on ExtraTrees: its 90 % interval from held-out errors (m15), valid for conditions typical of "
                  "Banten." if "90 % half-width" in long else ""))
    if verdict != "typical of Banten":
        st.warning("Tree models cannot extrapolate: outside the conditions seen in Banten they repeat the nearest known "
                   "answer, while ridge extrapolates linearly. Disagreement between models is a sign not to trust any of them.")

    C.section("Sweep one variable")
    a, b, c = st.columns(3)
    feat = a.selectbox("Variable", A.FEATURES, format_func=C.nice)
    crop = b.selectbox("Crop", C.CROPS, format_func=C.CROP_LABELS.get, key="wi_crop")
    models = c.multiselect("Models", A.MODEL_NAMES, default=["ExtraTrees", "XGBoost", "Ridge"])
    curves = pd.concat([A.ice(bundle, m, x, crop, feat).assign(model=m) for m in models]) if models else pd.DataFrame()
    if not curves.empty:
        fig = px.line(curves, x=feat, y="prediction", color="model", height=380)
        fig.add_vline(x=st.session_state[f"wi_{feat}"], line_dash="dot", annotation_text="current value")
        fig.update_layout(margin=dict(t=10), yaxis_title=f"predicted {C.CROP_LABELS[crop]} index", xaxis_title=C.nice(feat))
        st.plotly_chart(fig, width="stretch")
        st.caption("Everything else stays at the slider values; the curve shows how the model responds to this one variable "
                   "over the range observed in Banten.")

# ------------------------------------------------------------------------------------------- arena
elif section == "Model arena":
    st.caption("Live cross-validation through the same code as the notebooks (5 folds, one repeat, 4 worker processes). "
               "Compare a random K-fold with spatial schemes to see how much random splits flatter a model.")
    with st.form("arena"):
        a, b, c, d = st.columns([1, 1.2, 1, 1])
        crops = a.multiselect("Crops", C.CROPS, default=["rice", "coffee"], format_func=C.CROP_LABELS.get)
        models = b.multiselect("Models", list(A.ARENA_MODELS), default=list(A.ARENA_MODELS), format_func=A.ARENA_MODELS.get)
        schemes = c.multiselect("Validation", ["random", "block10", "pixel"], default=["random", "pixel"],
                                format_func={"random": "random cells", "block10": "10 km blocks", "pixel": "whole ~9 km GAEZ pixels"}.get)
        arena_inputs = d.selectbox("Inputs", A.available_inputs())
        go_ = st.form_submit_button("Run cross-validation", type="primary")
    if go_ and crops and models and schemes:
        st.session_state.arena_args = (tuple(crops), tuple(models), tuple(schemes), arena_inputs)
    if "arena_args" in st.session_state:
        a_crops, a_models, a_schemes, *a_rest = st.session_state.arena_args
        res = A.arena(a_crops, a_models, a_schemes, C.gaez_res(), a_rest[0] if a_rest else "14 drivers")
        st.success(f"{len(res)} model x crop x scheme runs in {res['seconds_total'].iloc[0]:.0f} s (cached for this session).")
        mean = res.groupby(["model", "scheme"])[["r2", "rmse", "spearman"]].mean().reset_index()
        fig = px.bar(mean, x="model", y="r2", color="scheme", barmode="group", height=400,
                     category_orders={"scheme": ["random", "block10", "pixel"]})
        fig.update_layout(margin=dict(t=10), yaxis_title="R² (mean over chosen crops)", xaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(res[["target", "model", "scheme", "r2", "rmse", "mae", "spearman", "fit_seconds"]].round(3),
                     hide_index=True, width="stretch")
    else:
        st.info("Choose crops, models and validation schemes, then run.")

# ------------------------------------------------------------------------------------------- deep learning
else:
    st.caption(f"Train a multi-output neural network (PyTorch) on the GAEZ {C.gaez_res()} index (switch in the sidebar): "
               "80 % of the ~9 km GAEZ pixels for training, the other 20 % held out whole for testing. ExtraTrees with the "
               "same information on the same split is the yardstick. Runs on the GPU if one is available.")
    nets = [k for k in A.NETWORKS if k == "mlp" or A.patches_available()]
    with st.form("dl"):
        arch = st.selectbox("Network", nets, format_func=A.NETWORKS.get,
                            help="Fusion = the m14 network: a CNN reads the 25 x 25 Sentinel-2 patch of each cell, an MLP the "
                                 "14 drivers, and a joint head predicts the ten indices. Its architecture is fixed; epochs, "
                                 "learning rate, batch size and seed apply.")
        a, b, c, d = st.columns(4)
        depth = a.slider("Hidden layers", 1, 4, 2)
        width = b.select_slider("Units per layer", [16, 32, 64, 128, 256, 512], value=128)
        act = c.selectbox("Activation", ["silu", "relu", "gelu", "tanh"])
        dropout = d.slider("Dropout", 0.0, 0.5, 0.0, 0.05)
        e, f, g, h = st.columns(4)
        epochs = e.slider("Epochs", 10, 200, 60, 10)
        lr = f.select_slider("Learning rate", [3e-4, 1e-3, 3e-3, 1e-2, 3e-2], value=3e-3)
        wd = g.select_slider("Weight decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], value=1e-4)
        batch = h.select_slider("Batch size", [64, 128, 256, 512, 1024], value=512)
        seed = st.number_input("Split / initialisation seed", 0, 99, 0)
        go_ = st.form_submit_button("Train network", type="primary")
    if go_:
        st.session_state.dl_args = (arch, (width,) * depth, act, float(dropout), float(wd), int(epochs), float(lr), int(batch),
                                    int(seed))
    out = None
    if "dl_args" in st.session_state:
        with st.spinner("Training in a separate worker process (first run also loads PyTorch)..."):
            try:
                args = st.session_state.dl_args
                if not isinstance(args[0], str):        # settings from before the network choice existed
                    args = ("mlp", *args)
                out = A.train_network(*args, C.gaez_res())
            except Exception as exc:  # noqa: BLE001
                st.error(f"Training failed: {exc}")
    if out is not None:
        k = st.columns(5)
        k[0].metric("Network mean R² (held-out pixels)", f"{out['scores']['Network'].mean():.3f}", border=True)
        k[1].metric(f"{out['yardstick']}, same split", f"{out['scores'][out['yardstick']].mean():.3f}", border=True)
        k[2].metric("Parameters", f"{out['params']:,}", border=True)
        k[3].metric("Training time", f"{out['seconds']:.1f} s", border=True)
        k[4].metric("Device", out["device"].upper(), border=True)
        l, r = st.columns(2)
        hist = out["history"].melt(id_vars="epoch", var_name="curve", value_name="MSE (z-scored targets)")
        fig = px.line(hist, x="epoch", y="MSE (z-scored targets)", color="curve", height=380, title="Learning curves")
        fig.update_layout(margin=dict(t=40))
        l.plotly_chart(fig, width="stretch")
        sc = out["scores"].melt(id_vars="crop", var_name="model", value_name="R²")
        fig = px.bar(sc, x="crop", y="R²", color="model", barmode="group", height=380, title="Held-out R² per crop")
        fig.update_layout(margin=dict(t=40), xaxis_title="")
        r.plotly_chart(fig, width="stretch")
        crop = st.selectbox("Predicted vs true for", C.CROPS, format_func=C.CROP_LABELS.get, key="dl_crop")
        s = out["sample"]
        l, r = st.columns(2)
        fig = px.scatter(s, x=f"true_{crop}", y=f"net_{crop}", opacity=0.4, height=380,
                         labels={f"true_{crop}": "GAEZ (truth)", f"net_{crop}": "network prediction"})
        fig.add_shape(type="line", x0=0, y0=0, x1=10000, y1=10000, line=dict(dash="dash", color="grey"))
        fig.update_layout(margin=dict(t=10))
        l.plotly_chart(fig, width="stretch")
        held = df["grid_id"].isin(set(out["test_ids"])).map({True: "held out (test)", False: "training"})
        r.plotly_chart(C.category_map(df, held, {"training": "#CFE0C3", "held out (test)": C.CLAY},
                                      "Which pixels were held out", 380), width="stretch")
        gap = out["history"]["validation loss"].iloc[-1] - out["history"]["train loss"].iloc[-1]
        st.caption(f"Train on {out['n_train']:,} cells, test on {out['n_test']:,} cells from pixels the network never saw. "
                   f"Final validation minus training loss = {gap:.3f} (a large gap means overfitting; try dropout or weight decay).")
    else:
        st.info("Pick the architecture and training settings, then press 'Train network'.")
