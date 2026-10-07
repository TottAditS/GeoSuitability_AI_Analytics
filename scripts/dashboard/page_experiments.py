import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import common as C

df = C.load_data()


def R(name):
    """A result table with readable model names (first column or 'model')."""
    t = C.load_result(name)
    if t is not None:
        for col in [c for c in ("model", "Unnamed: 0") if c in t.columns]:
            t[col] = t[col].map(C.model_name)
    return t


C.page_header("AI · evaluation", "ML/DL Experiments", "Results of the fifteen notebooks in `scripts/modeling` (m01-m15). Task: predict each crop's GAEZ v5 "
           "suitability index at 1 km from the 14 physical drivers (climate, soil, terrain). All headline scores use "
           "**pixel-grouped cross-validation**: whole ~9 km blocks (the pixels of GAEZ's 5 arc-minute grid) are held out, "
           "because neighbouring 1 km cells are nearly alike. m13 compares the 1 km target with the ~9 km product of the "
           "first round; m14 adds Sentinel-2 satellite imagery, m15 measures the uncertainty.")

if R("m01_leaderboard") is None:
    C.no_results_note("m01_baselines_cv_leakage.ipynb")
    st.stop()

EXPERIMENTS = {
    "Summary": None,
    "m01 Baselines and CV leakage": ("m01_baselines_cv_leakage.ipynb",
        "How well do five model families recover GAEZ, and how much does a naive random K-fold overstate it?",
        "Six models (dummy, ridge, kNN, ExtraTrees, HistGB, XGBoost) x 10 crops x three CV schemes (random cells, "
        "10 km blocks, whole ~9 km GAEZ pixels), 5 folds x 2 repeats, identical folds for every model."),
    "m02 Block size": ("m02_blocksize_sensitivity.ipynb",
        "How large must the held-out blocks be before the score stops being inflated by neighbours?",
        "Square blocks of 5-40 km vs random cells vs whole ~9 km GAEZ pixels; 'pixel overlap' is the share of test cells "
        "whose pixel also has cells in the training set."),
    "m03 Feature ablation and SHAP": ("m03_feature_ablation_shap.ipynb",
        "Which input groups and variables does the surrogate actually need?",
        "Feature groups alone / combined / left out; leave-one-feature-out; SHAP shares of an XGBoost surrogate."),
    "m04 Spatial context": ("m04_spatial_context_features.ipynb",
        "Do neighbourhood averages of the inputs help, and does training on ~9 km averages lose anything?",
        "Focal means over 3-41 km windows and per-pixel means added to the 14 drivers; one row per ~9 km GAEZ pixel "
        "as the coarse alternative."),
    "m05 Recommended-crop classification": ("m05_classify_recommended_crop.ipynb",
        "How well can the recommendation label itself be predicted, and per crop 'suitable (index >= 5,000) or not'?",
        "Direct classifiers vs regressing the ten indices and then applying the dashboard rule."),
    "m06 Extrapolation": ("m06_extrapolation_stress_tests.ipynb",
        "What happens when a model must predict conditions it never saw (other elevation, rainfall, end of the province)?",
        "Train on one regime, test on another, with a gap in between."),
    "m07 Unsupervised zones": ("m07_unsupervised_zones.ipynb",
        "Which agro-ecological zones emerge from the drivers alone, and do they match the crop map?",
        "K-means, Gaussian mixtures, Ward and spatially constrained Ward; quality, contiguity and agreement with the labels."),
    "m08 GAEZ vs GCS 2024": ("m08_cross_source_gcs2024.ipynb",
        "How do the two independent suitability products relate, and do they agree on local detail?",
        "Agreement, learnability of each product, cross-source transfer, and within-pixel agreement with the 1 km product."),
    "m09 MLP on the GPU": ("m09_torch_mlp_sweeps.ipynb",
        "Does a multi-output neural network do better than trees on this table?",
        "One change at a time against a base MLP (architecture, regularisation, normalisation, loss), plus seed averaging."),
    "m10 U-Net on the grid": ("m10_grid_networks_unet.ipynb",
        "Does a convolutional network that sees the neighbourhood help a deep model?",
        "The province as an image (channels = drivers); pointwise network vs U-Net, loss masked to training cells."),
    "m11 Soil gap-fill": ("m11_soil_gapfill_effect.ipynb",
        "Does the interpolation of missing SoilGrids cells change the results?",
        "Gap-filled soil vs raw soil with NaN vs a flag vs no soil; errors on observed vs interpolated cells."),
    "m12 Learning curves and ensembles": ("m12_learning_curves_ensembles.ipynb",
        "Would more labelled pixels help, and do ensembles beat the best single model?",
        "Training on 10-100 % of the training pixels; equal-weight averages of out-of-fold predictions."),
    "m13 GAEZ ~9 km vs 1 km": ("m13_gaez_1km_downscaling.ipynb",
        "What changes when the target is GAEZ at 1 km instead of ~9 km, and was the sub-pixel detail of the ~9 km models real?",
        "The m01 grid on the same cells and folds for both targets; the ~9 km map as is and blurred and models trained on "
        "each target, all scored against GAEZ v5 at 30 arc-sec."),
    "m14 Sentinel-2 imagery": ("m14_sentinel2_imagery.ipynb",
        "Does satellite imagery add to the physical drivers, as per-cell statistics or as image patches?",
        "Trees on the 14 drivers, 23 Sentinel-2 statistics and both; on the GPU an MLP on the drivers, a CNN on the "
        "25 x 25 image patch of each cell and a fusion network with both branches; identical pixel-grouped folds."),
    "m15 Uncertainty": ("m15_uncertainty.ipynb",
        "How sure is a model estimate, and the recommendation that follows from it?",
        "Out-of-fold ExtraTrees, a second model for the size of the error, conformal 90 % intervals checked on held-out "
        "folds, and label probabilities from 200 plausible versions of every cell."),
}
choice = st.selectbox("Experiment", list(EXPERIMENTS))


def chart(fig, h=None):
    if h:
        fig.update_layout(height=h)
    fig.update_layout(margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")


def figures(*names):
    paths = [p for p in (C.figure_path(n) for n in names) if p]
    if paths:
        with st.expander(f"Figures from the notebook ({len(paths)})"):
            for p in paths:
                st.image(str(p), caption=p.name)


def table(name, label=None):
    t = R(name)
    if t is not None:
        with st.expander(label or f"Table `{name}.csv`"):
            st.dataframe(t, width="stretch", hide_index=True)
            st.download_button("Download CSV", t.to_csv(index=False).encode("utf-8"), file_name=f"{name}.csv",
                               mime="text/csv", key=f"dl_{name}")


def model_order(t, col="pixel"):
    return t.sort_values(col, ascending=False)["model"].tolist()


# ------------------------------------------------------------------------------------------------- summary
if choice == "Summary":
    lead, summ = R("m01_leaderboard"), R("m01_summary_pixel")
    best = summ.iloc[0]
    n_target = int(df[[f"gaez1k_index_{c}" for c in C.CROPS]].notna().all(axis=1).sum())
    gap = lead.set_index("model").loc[best["model"], "leakage_gap"]
    dl = []
    if R("m09_mlp_vs_trees") is not None:
        t = R("m09_mlp_vs_trees").rename(columns={"Unnamed: 0": "setting"})
        dl.append(t[t["setting"].str.startswith("MLP")]["r2"].max())
    if R("m10_grid_networks") is not None:
        t = R("m10_grid_networks").rename(columns={"Unnamed: 0": "setting"})
        dl.append(t[t["setting"].str.contains("U-Net|pointwise")]["r2"].max())
    agr = R("m08_agreement")
    k = st.columns(5)
    k[0].metric("Cells with a target", f"{n_target:,}", border=True,
                help="1 km cells with a GAEZ value for all ten crops; GAEZ marks the rest as no cropland or open water")
    k[1].metric(f"{best['model']} R²", f"{best['r2']:.3f}", border=True, help="Best model under pixel-grouped CV")
    k[2].metric("Leakage (R²)", f"+{gap:.2f}", border=True,
                help="R² under random K-fold minus R² under pixel-grouped CV, for the same model")
    k[3].metric("Best deep R²", f"{max(dl):.3f}" if dl else "-", border=True, help="Best MLP or U-Net (m09, m10)")
    k[4].metric("Crops GAEZ ≈ GCS", f"{int((agr['spearman_cell'] > 0.5).sum())} / {len(agr)}" if agr is not None else "-",
                border=True, help="Crops whose GAEZ and GCS 2024 maps agree with a Spearman rho above 0.5")

    l, r = st.columns(2)
    fam = summ[summ["model"] != C.model_name("dummy")][["model", "r2"]].assign(kind="per-cell tabular")
    if R("m09_mlp_vs_trees") is not None:
        t = R("m09_mlp_vs_trees").rename(columns={"Unnamed: 0": "model"})
        fam = pd.concat([fam, t[t["model"].str.startswith("MLP")][["model", "r2"]].assign(kind="deep (MLP)")])
    if R("m10_grid_networks") is not None:
        t = R("m10_grid_networks").rename(columns={"Unnamed: 0": "model"})
        fam = pd.concat([fam, t[t["model"].str.contains("U-Net|pointwise")][["model", "r2"]].assign(kind="deep (grid)")])
    s2 = R("m14_summary")
    if s2 is not None:
        s2 = s2[s2["setting"].str.contains("Sentinel-2|patch")].rename(columns={"setting": "model"})
        fam = pd.concat([fam, s2[["model", "r2"]].assign(kind="with Sentinel-2 imagery")])
    fig = px.bar(fam.sort_values("r2"), x="r2", y="model", color="kind", orientation="h", title="Every model family, honest R²")
    fig.update_layout(yaxis_title="", xaxis_range=[0.4, 0.9])
    with l:
        chart(fig, 420)
    long = lead.melt(id_vars="model", value_vars=["random", "block10", "pixel"], var_name="CV scheme", value_name="R²")
    fig = px.bar(long, x="model", y="R²", color="CV scheme", barmode="group", title="The same models, three ways to validate",
                 category_orders={"model": model_order(lead)})
    with r:
        chart(fig, 420)
    guide = (f"Reading guide: the models reproduce the 1 km GAEZ well on ~9 km blocks they never saw (best R² "
             f"{lead['pixel'].max():.2f}), random K-fold makes them look better than they are ({lead['random'].max():.2f}), "
             "and skill drops with distance from the training data (m02, m06). ")
    if dl:
        guide += (f"Neural networks come close (best R² {max(dl):.2f}) but the trees win on this table. " if max(dl) < best["r2"]
                  else f"The best neural network (R² {max(dl):.2f}) edges out the trees. ")
    g10 = R("m10_grid_networks")
    if g10 is not None:
        g10 = g10.set_index(g10.columns[0])["r2"]
        unet, point = g10.filter(like="U-Net").max(), g10.filter(like="pointwise").max()
        guide += f"Spatial context {'helps' if unet > point else 'does not help'} the networks (U-Net vs pointwise, m10). "
    if agr is not None:
        a = agr.set_index("crop")["spearman_cell"].sort_values()
        guide += (f"GAEZ and GCS 2024 agree most for {C.CROP_LABELS[a.index[-1]].lower()} and "
                  f"{C.CROP_LABELS[a.index[-2]].lower()}, least for {C.CROP_LABELS[a.index[0]].lower()} (m08). ")
    if s2 is not None:
        top = s2.sort_values("r2").iloc[-1]
        guide += (f"Adding Sentinel-2 imagery lifts the best model to R² {top['r2']:.2f} ({top['model']}, m14): the "
                  "imagery carries land-use information the physical drivers lack. ")
    C.note(guide + "m13 compares the 1 km target with the ~9 km product of the first round.")

    C.section("Where do the models err?", "out-of-fold predictions against the target they were trained on")
    TARGETS = {"1 km GAEZ (m01, m09)": ("gaez1k_index_", ["m01_oof_pixel", "m09_mlp_oof_pixel"]),
               "~9 km GAEZ (m13)": ("gaez_index_", ["m13_oof_9km_pixel"])}
    ca, cb, cc = st.columns([1.3, 1, 1])
    target = ca.segmented_control("Target", list(TARGETS), default=list(TARGETS)[0], key="err_target") or list(TARGETS)[0]
    prefix, files = TARGETS[target]
    oof = None
    for name in files:
        if (C.OOF_DIR / f"{name}.parquet").exists():
            part = pd.read_parquet(C.OOF_DIR / f"{name}.parquet")
            oof = part if oof is None else oof.merge(part, on="grid_id", how="left")
    if oof is None:
        st.info(f"Out-of-fold predictions not found (run {', '.join(files)}).")
    else:
        models = sorted({col.split("__")[0] for col in oof.columns if "__" in col})
        m = cb.selectbox("Model", models, format_func=lambda s: {"et": "ExtraTrees", "hgb": "HistGB", "xgb": "XGBoost",
                                                                "mlp3": "MLP (3-seed, GPU)"}.get(s, s))
        crop = cc.selectbox("Crop", C.CROPS, format_func=C.CROP_LABELS.get)
        d = (df[["grid_id", "row", "col", "lon", "lat", f"{prefix}{crop}"]].merge(oof[["grid_id", f"{m}__{crop}"]], on="grid_id")
             .dropna(subset=[f"{prefix}{crop}", f"{m}__{crop}"]))
        truth, pred = d[f"{prefix}{crop}"], d[f"{m}__{crop}"]
        x1, x2, x3 = st.columns(3)
        x1.plotly_chart(C.grid_map(d, truth, f"{target.split(' (')[0]} (target)", C.SEQ, 0, 10000, 360), width="stretch")
        x2.plotly_chart(C.grid_map(d, pred, "Out-of-fold prediction", C.SEQ, 0, 10000, 360), width="stretch")
        x3.plotly_chart(C.grid_map(d, pred - truth, "Error (prediction - target; clay = over, teal = under)", C.DIV_ERR, -4000, 4000, 360), width="stretch")
        err = pred - truth
        st.caption(f"R² {1 - (err ** 2).sum() / ((truth - truth.mean()) ** 2).sum():.3f} · RMSE {np.sqrt((err ** 2).mean()):,.0f} · "
                   "every prediction comes from a model that never saw that cell's GAEZ pixel.")

# ------------------------------------------------------------------------------------------------- per experiment
else:
    nb, question, how = EXPERIMENTS[choice]
    st.markdown(f"**Question.** {question}  \n**How.** {how}  \n**Notebook.** `scripts/modeling/{nb}`")

    if choice.startswith("m01"):
        lead, raw, ceil = R("m01_leaderboard"), R("m01_baselines_raw"), R("m01_ceiling")
        l, r = st.columns(2)
        with l:
            long = lead.melt(id_vars="model", value_vars=["random", "block10", "pixel"], var_name="scheme", value_name="R²")
            chart(px.bar(long, x="model", y="R²", color="scheme", barmode="group", title="Mean R² over 10 crops",
                         category_orders={"model": model_order(lead)}), 400)
        with r:
            chart(px.bar(lead.sort_values("leakage_gap"), x="leakage_gap", y="model", orientation="h",
                         title="Optimism of random K-fold (R² random - R² pixel)", color_discrete_sequence=[C.CLAY]), 400)
        hm = raw[raw.scheme == "pixel"].groupby(["target", "model"])["r2"].mean().unstack("model")[model_order(lead)]
        chart(px.imshow(hm.round(3), text_auto=True, color_continuous_scale=C.SEQ, zmin=0, zmax=1, aspect="auto",
                        title="R² per crop and model (pixel CV)"), 420)
        chart(px.bar(ceil, x="crop", y="best_pixel_r2", color="best_model", title="Best honest R² per crop"), 320)
        figures("m01_leaderboard", "m01_crop_by_model_pixel")
        table("m01_baselines_raw", "All runs (model x crop x scheme x repeat)")

    elif choice.startswith("m02"):
        t = R("m02_blocksize_summary").rename(columns={"Unnamed: 0": "scheme"})
        models = [c for c in t.columns if c not in ("scheme", "pixel_overlap", "train_cells", "test_cells")]
        km = {"random": 1, "block5": 5, "block10": 10, "block20": 20, "block40": 40}
        curve = t[t["scheme"].isin(km)].assign(km=lambda d: d["scheme"].map(km)).melt(id_vars=["scheme", "km"], value_vars=models,
                                                                                     var_name="model", value_name="R²")
        fig = px.line(curve.sort_values("km"), x="km", y="R²", color="model", markers=True, log_x=True,
                      title="Apparent skill vs size of the held-out blocks (1 = random cells)")
        fig.add_vline(x=9.2, line_dash="dash", line_color="grey", annotation_text="GAEZ pixel ~9 km")
        chart(fig, 420)
        fig = px.scatter(t.melt(id_vars=["scheme", "pixel_overlap"], value_vars=models, var_name="model", value_name="R²"),
                         x="pixel_overlap", y="R²", color="model", text="scheme", title="Apparent skill vs 'pixel overlap'")
        fig.update_traces(textposition="top center")
        chart(fig, 420)
        figures("m02_blocksize")
        table("m02_blocksize_summary")

    elif choice.startswith("m03"):
        fs = R("m03_featuresets_summary")
        chart(px.bar(fs.melt(id_vars=["feature_set", "n_features"], var_name="model", value_name="R²"), x="R²", y="feature_set",
                     color="model", barmode="group", orientation="h", title="Feature sets (pixel CV)"), 560)
        l, r = st.columns(2)
        lofo = R("m03_leave_one_feature_out").set_index("target")
        with l:
            chart(px.imshow(lofo.round(3), color_continuous_scale=C.SEQ_CLAY, aspect="auto", title="R² lost when one variable is removed"), 420)
        shap = R("m03_shap_share").rename(columns={"Unnamed: 0": "crop"}).set_index("crop")
        with r:
            chart(px.imshow(shap.round(2), text_auto=True, color_continuous_scale=C.SEQ, aspect="auto",
                            title="Share of mean |SHAP| per crop (XGBoost)"), 420)
        figures("m03_featuresets", "m03_leave_one_feature_out", "m03_shap_share", "m03_shap_beeswarm_rice",
                "m03_shap_beeswarm_coffee", "m03_partial_dependence")
        table("m03_featuresets_raw")

    elif choice.startswith("m04"):
        raw = R("m04_context_raw")
        t = raw.groupby(["feature_set", "model"])["r2"].mean().reset_index()
        chart(px.bar(t, x="r2", y="feature_set", color="model", barmode="group", orientation="h",
                     title="Neighbourhood features (pixel CV)"), 440)
        l, r = st.columns(2)
        fsc = R("m04_focal_scale")
        with l:
            fig = px.line(fsc, x="window (km)", y="hgb", markers=True, log_x=True, title="One neighbourhood scale only (HistGB)")
            chart(fig, 360)
        pl = R("m04_pixel_level_raw")
        with r:
            chart(px.bar(pl.groupby(["model", "scheme"])["r2"].mean().reset_index(), x="model", y="r2", color="scheme",
                         barmode="group", title="Training on ~9 km pixel averages (one row per pixel)"), 360)
        figures("m04_context", "m04_focal_scale")
        table("m04_context_raw")

    elif choice.startswith("m05"):
        lb = R("m05_leaderboard")
        target = st.segmented_control("Label", ["thresh", "top"], default="thresh",
                                      format_func=lambda s: {"thresh": "with 'Not recommended'", "top": "plain argmax"}.get(s, s), key="m05_label") or "thresh"
        sub = lb[lb["target"] == target].melt(id_vars=["target", "scheme", "model"], value_vars=["accuracy", "balanced_acc", "macro_f1"],
                                              var_name="metric", value_name="score")
        chart(px.bar(sub, x="model", y="score", color="metric", barmode="group", facet_col="scheme",
                     title="Direct classifiers vs regress-then-rule ('+ rule')"), 420)
        bc = R("m05_binary_by_crop_pixel").set_index("target")
        chart(px.imshow(bc.round(3), text_auto=True, color_continuous_scale=C.SEQ, aspect="auto",
                        title="Per crop: suitable (index >= 5,000) or not - balanced accuracy, pixel CV"), 380)
        figures("m05_macro_f1", "m05_confusion", "m05_error_map")
        table("m05_leaderboard")

    elif choice.startswith("m06"):
        for name, title, scale, rng in [("m06_extrapolation_r2", "R² on the unseen regime (clipped at -2)", C.DIV, (-2, 1)),
                                        ("m06_extrapolation_spearman", "Rank agreement on the unseen regime", C.DIV, (-0.2, 1))]:
            t = R(name).set_index("model")
            chart(px.imshow(t.clip(lower=rng[0]).round(2), text_auto=True, color_continuous_scale=scale, zmin=rng[0], zmax=rng[1],
                            aspect="auto", title=title), 380)
        figures("m06_extrapolation", "m06_elevation_by_crop")
        table("m06_extrapolation_raw")

    elif choice.startswith("m07"):
        ks = R("m07_k_sweep")
        l, r = st.columns(2)
        with l:
            chart(px.line(ks, x="k", y=["silhouette", "davies_bouldin"], markers=True, title="Choosing k"), 360)
        with r:
            chart(px.line(ks, x="k", y="gmm_bic", markers=True, title="Gaussian mixture BIC (lower is better)"), 360)
        alg = R("m07_algorithms")
        st.dataframe(alg.round(3), hide_index=True, width="stretch")
        comp = R("m07_zone_label_composition").rename(columns={"row_0": "zone"})
        chart(px.bar(comp.melt(id_vars="zone", var_name="label", value_name="share"), x="zone", y="share", color="label",
                     title="Crop recommendation inside each zone (zone 0 = lowest)"), 360)
        pg = R("m07_physical_vs_gaez_zones")
        st.caption(f"Agreement between zones from the drivers and zones from the GAEZ indices: NMI "
                   f"{pg['NMI'].min():.2f}-{pg['NMI'].max():.2f} over k = {pg['k'].min()}-{pg['k'].max()}.")
        figures("m07_k_sweep", "m07_zones")
        table("m07_algorithms")

    elif choice.startswith("m08"):
        ag = R("m08_agreement")
        l, r = st.columns(2)
        with l:
            chart(px.bar(ag, x="crop", y=["spearman_cell", "spearman_pixel_mean"], barmode="group",
                         title="Agreement between GAEZ and GCS (Spearman)"), 380)
        ln = R("m08_learnability")
        with r:
            chart(px.bar(ln.melt(id_vars=["source", "model"], value_vars=["random", "block10", "pixel"], var_name="scheme",
                                 value_name="R²"), x="model", y="R²", color="scheme", barmode="group", facet_col="source",
                         title="Which product is easier to learn?"), 380)
        tr = R("m08_transfer")
        chart(px.imshow(tr[tr["model"] == "HistGB"].drop(columns="model").set_index("crop").round(2), text_auto=True,
                        color_continuous_scale=C.DIV, zmin=-0.3, zmax=1, aspect="auto",
                        title="Cross-source transfer (Spearman of out-of-fold predictions, HistGB)"), 380)
        wp = R("m08_within_pixel").set_index("crop")
        num = wp.select_dtypes("number")
        chart(px.bar(num.reset_index().melt(id_vars="crop", var_name="series", value_name="rho"), x="crop", y="rho",
                     color="series", barmode="group", title="Within-pixel agreement with the 1 km product"), 420)
        gaez_rho = wp["within-pixel rho(raw GAEZ, GCS)"]
        sur = wp[[c for c in wp.columns if c.endswith("GAEZ-surrogate")]]
        st.caption("Each map minus its ~9 km pixel mean, compared with GCS: how far the products agree on local detail. "
                   f"GAEZ itself (1 km) agrees with GCS at a within-pixel Spearman of {gaez_rho.min():.2f} to "
                   f"{gaez_rho.max():.2f} across the six crops and its surrogates land close to that "
                   f"({sur.min().min():.2f} to {sur.max().max():.2f}); the GCS surrogates show how much of GCS's own local "
                   "pattern the 14 drivers carry.")
        figures("m08_learnability", "m08_within_pixel", "m08_disagreement_maps")
        table("m08_transfer")

    elif choice.startswith("m09"):
        sw = R("m09_mlp_sweeps")
        chart(px.bar(sw.sort_values("r2"), x="r2", y="setting", color="section", orientation="h",
                     title="One change at a time vs the base MLP"), 460)
        vs = R("m09_mlp_vs_trees").rename(columns={"Unnamed: 0": "model"})
        chart(px.bar(vs.sort_values("r2"), x="r2", y="model", orientation="h", title="MLP vs trees (pixel CV)"), 320)
        figures("m09_mlp_sweeps")
        table("m09_mlp_sweeps")

    elif choice.startswith("m10"):
        g = R("m10_grid_networks").rename(columns={"Unnamed: 0": "model"})
        chart(px.bar(g.sort_values("r2"), x="r2", y="model", orientation="h", title="Spatial context vs per-cell models"), 320)
        sp = R("m10_subpixel_vs_gcs")
        chart(px.bar(sp.melt(id_vars="crop", var_name="model", value_name="rho"), x="crop", y="rho", color="model",
                     barmode="group", title="Local (within-pixel) agreement with GCS 2024"), 360)
        figures("m10_context", "m10_maps")
        table("m10_grid_networks")

    elif choice.startswith("m11"):
        sv = R("m11_soil_variants")
        chart(px.bar(sv.melt(id_vars="variant", var_name="model", value_name="R²"), x="variant", y="R²", color="model",
                     barmode="group", title="Soil input variants (pixel CV)"), 380)
        es = R("m11_error_split")
        chart(px.bar(es, x="model", y="r2", color="cells", barmode="group", title="Error on observed vs interpolated soil cells"), 340)
        figures("m11_interpolated_cells")
        table("m11_soil_variants_raw")

    elif choice.startswith("m12"):
        lc = R("m12_learning_curves")
        xcol = lc.columns[0]
        chart(px.line(lc.melt(id_vars=xcol, var_name="model", value_name="R²"), x=xcol, y="R²", color="model", markers=True,
                      log_x=True, title="Learning curves (still rising = more labelled pixels would help)"), 400)
        en = R("m12_ensembles")
        l, r = st.columns(2)
        with l:
            chart(px.bar(en.sort_values("r2"), x="r2", y="ensemble", orientation="h", title="Ensembles of out-of-fold predictions"), 380)
        ec = R("m12_error_correlation").rename(columns={"Unnamed: 0": "model"}).set_index("model")
        with r:
            chart(px.imshow(ec.round(2), text_auto=True, color_continuous_scale=C.SEQ, zmin=0.5, zmax=1,
                            title="Correlation of errors between models"), 380)
        figures("m12_learning_curves", "m12_error_correlation")
        table("m12_ensembles")

    elif choice.startswith("m13"):
        res13, down, lb = R("m13_resolution"), R("m13_downscaling"), R("m13_leaderboard")
        if res13 is None or down is None or lb is None:
            C.no_results_note(nb)
            st.stop()
        order = ["~9 km GAEZ as is", "~9 km GAEZ smoothed", "ExtraTrees, ~9 km target", "HistGB, ~9 km target",
                 "XGBoost, ~9 km target", "ExtraTrees, 1 km target"]
        colors = dict(zip(order, [C.STONE, C.SAGE, C.OCHRE, C.CLAY, C.BARK, C.PINE]))
        l, r = st.columns(2)
        with l:
            chart(px.bar(down, x="crop", y="r2_vs_1km", color="candidate", barmode="group", color_discrete_map=colors,
                         category_orders={"candidate": order}, range_y=(0.5, 0.9),
                         title="Whole map: R² against the real 1 km GAEZ"), 420)
        with r:
            chart(px.bar(down, x="crop", y="within_pixel_rho", color="candidate", barmode="group", color_discrete_map=colors,
                         category_orders={"candidate": order}, title="Inside each ~9 km pixel: Spearman with the 1 km GAEZ"), 420)
        g = down.groupby("candidate")["within_pixel_rho"].mean()
        st.caption(f"Inside a GAEZ pixel the models trained on the ~9 km target follow the 1 km GAEZ with a Spearman of "
                   f"{g[[c for c in g.index if c.endswith('~9 km target')]].mean():.2f}: better than blurring the ~9 km map "
                   f"({g['~9 km GAEZ smoothed']:.2f}), well below a model trained on the 1 km target "
                   f"({g['ExtraTrees, 1 km target']:.2f}). Their sub-pixel detail was partly real; the 1 km product now "
                   "makes it unnecessary.")
        long = lb.melt(id_vars=["resolution", "model"], value_vars=["random", "block10", "pixel"], var_name="scheme",
                       value_name="R²")
        long["resolution"] = long["resolution"].map({"9km": "~9 km target", "1km": "1 km target"})
        chart(px.bar(long, x="model", y="R²", color="scheme", barmode="group", facet_col="resolution",
                     category_orders={"resolution": ["~9 km target", "1 km target"], "scheme": ["random", "block10", "pixel"]},
                     title="The m01 grid on both targets (same cells, same folds)"), 400)
        l, r = st.columns(2)
        share = res13.melt(id_vars="crop", value_vars=["within_pixel_share_9km", "within_pixel_share_1km"],
                           var_name="target", value_name="share")
        share["target"] = share["target"].map({"within_pixel_share_9km": "GAEZ ~9 km", "within_pixel_share_1km": "GAEZ 1 km"})
        with l:
            chart(px.bar(share, x="crop", y="share", color="target", barmode="group",
                         color_discrete_map={"GAEZ ~9 km": C.SAGE, "GAEZ 1 km": C.PINE},
                         title="Share of the variance inside the ~9 km pixels"), 360)
        suit = res13.melt(id_vars="crop", value_vars=["suitable_cells_9km", "suitable_cells_1km"], var_name="target",
                          value_name="cells")
        suit["target"] = suit["target"].map({"suitable_cells_9km": "GAEZ ~9 km", "suitable_cells_1km": "GAEZ 1 km"})
        with r:
            chart(px.bar(suit, x="crop", y="cells", color="target", barmode="group",
                         color_discrete_map={"GAEZ ~9 km": C.SAGE, "GAEZ 1 km": C.PINE},
                         title="Cells with an index of 5,000 or more"), 360)
        figures("m13_downscaling", "m13_leaderboard", "m13_maps_rice")
        table("m13_cv_raw", "All runs (model x crop x target resolution x scheme)")

    elif choice.startswith("m14"):
        sm, cg = R("m14_summary"), R("m14_gain_by_crop")
        if sm is None or cg is None:
            C.no_results_note(nb)
            st.stop()
        chart(px.bar(sm.sort_values("r2"), x="r2", y="setting", color="family", orientation="h", range_x=(0.5, 0.9),
                     color_discrete_map={"trees, per-cell statistics": C.PINE, "neural networks, image patches": C.TEAL},
                     title="The 14 drivers, Sentinel-2 and both (pixel CV, identical folds)"), 400)
        cg = cg.rename(columns={"Unnamed: 0": "crop", "target": "crop"})
        cg["crop"] = cg["crop"].map(C.CROP_LABELS)
        chart(px.bar(cg.melt(id_vars="crop", var_name="comparison", value_name="R² gain"), x="crop", y="R² gain",
                     color="comparison", barmode="group", color_discrete_sequence=[C.PINE, C.TEAL],
                     title="What Sentinel-2 adds, per crop"), 360)
        best = sm.sort_values("r2").iloc[-1]
        trees = sm.set_index("setting")["r2"]
        st.caption(f"Best model: {best['setting']} (R² {best['r2']:.3f}). For ExtraTrees the Sentinel-2 statistics add "
                   f"{trees['ExtraTrees, drivers + Sentinel-2'] - trees['ExtraTrees, 14 drivers']:+.3f}; the imagery alone "
                   f"reaches {trees['ExtraTrees, Sentinel-2 only']:.3f}. Sentinel-2 shows current land cover and use, which "
                   "the 1 km GAEZ partly reflects, so the gain is information the drivers lack rather than suitability seen "
                   "from space.")
        figures("m14_summary", "m14_error_map")
        table("m14_tabular_raw", "All tree runs (model x crop x feature set)")

    elif choice.startswith("m15"):
        cov, rel, sc = R("m15_coverage"), R("m15_reliability"), R("m15_probability_scores")
        if cov is None or rel is None:
            C.no_results_note(nb)
            st.stop()
        l, r = st.columns(2)
        with l:
            fig = px.bar(cov, x="crop", y="coverage", color="interval", barmode="group", facet_row="inputs", range_y=(0.6, 1),
                         color_discrete_map={"adaptive": C.PINE, "constant width": C.SAGE}, title="Coverage of the 90 % intervals")
            fig.add_hline(y=0.9, line_dash="dash", line_color="grey")
            chart(fig, 480)
        with r:
            fig = px.line(rel, x="predicted", y="observed", color="inputs", facet_row="event", markers=True,
                          title="Reliability of the probabilities", range_x=(0, 1), range_y=(0, 1))
            chart(fig, 480)
        if sc is not None:
            st.dataframe(sc.round(3), hide_index=True, width="stretch")
        ad = cov[cov["interval"] == "adaptive"].groupby("inputs")[["coverage", "mean width"]].mean()
        st.caption("Adaptive intervals: " + "; ".join(f"{k}: coverage {v['coverage']:.0%}, mean width {v['mean width']:,.0f}"
                                                   for k, v in ad.iterrows())
                   + ". The maps are on the Suitability Map page, view 'Uncertainty'.")
        figures("m15_coverage", "m15_reliability", "m15_maps_rice")
        table("m15_coverage_by_landcover", "Coverage by land cover")
