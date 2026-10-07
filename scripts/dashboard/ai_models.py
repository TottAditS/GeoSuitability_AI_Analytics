"""Model training and inference behind the AI Lab page (scikit-learn, XGBoost and PyTorch).

Models are trained on demand and cached for the lifetime of the dashboard process. Thread pools are capped
at N_THREADS so a click never takes the whole CPU. PyTorch never enters the dashboard process: the deep-learning
playground trains in a separate worker (dl_worker.py), see that file for why.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from threadpoolctl import threadpool_limits

import common as C

if str(C.MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(C.MODELING_DIR))
import exp_lib as E  # noqa: E402  (shared experiment library: feature groups, CV schemes, model zoo)

N_THREADS = 4
FEATURES = C.BASE_FEATURES

# threadpoolctl warns on every call that two OpenMP runtimes are loaded (MKL's libiomp next to the LLVM libomp of
# other packages). The crash it describes is a Linux issue; the real clash on this machine is with PyTorch, which
# never loads in this process (see dl_worker.py).
warnings.filterwarnings("ignore", message=r"\s*Found Intel OpenMP", category=RuntimeWarning)
MODEL_NAMES = ["Ridge", "kNN", "ExtraTrees", "HistGB", "XGBoost"]
ARENA_MODELS = {"ridge": "Ridge", "knn": "kNN", "et": "ExtraTrees", "hgb": "HistGB", "xgb": "XGBoost"}


def targets_for(res: str) -> list[str]:
    """The ten GAEZ index columns the models learn: 1 km (30 arc-sec) or ~9 km (5 arc-min)."""
    return [C.gaez_col(c, "index", res) for c in C.CROPS]


class ScaledTargets:
    """Fits a multi-output regressor on z-scored targets (so every crop weighs the same) and undoes it on predict."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, Y):
        self.mu, self.sd = Y.mean(axis=0), Y.std(axis=0) + 1e-9
        self.model.fit(X, (Y - self.mu) / self.sd)
        return self

    def predict(self, X):
        return self.model.predict(X) * self.sd + self.mu


def _make(name, n_trees=150):
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    imp = SimpleImputer(strategy="median")
    if name == "Ridge":
        return make_pipeline(imp, StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if name == "kNN":
        return make_pipeline(imp, StandardScaler(), KNeighborsRegressor(n_neighbors=20, weights="distance"))
    if name == "ExtraTrees":
        return ScaledTargets(make_pipeline(imp, ExtraTreesRegressor(n_trees, min_samples_leaf=3, max_features=0.7,
                                                                    n_jobs=N_THREADS, random_state=0)))
    if name == "HistGB":
        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0,
                                             random_state=0)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=3, reg_lambda=2.0, tree_method="hist", n_jobs=N_THREADS, random_state=0)
    raise ValueError(name)


def _fit(name, X, Y, n_trees=150):
    """Multi-output models get the whole target matrix; HistGB / XGBoost get one model per crop."""
    if name in ("HistGB", "XGBoost"):
        return [_make(name).fit(X, Y[:, j]) for j in range(Y.shape[1])]
    return _make(name, n_trees).fit(X, Y)


def _predict(model, X):
    if isinstance(model, list):
        return np.column_stack([m.predict(X) for m in model])
    return model.predict(X)


@st.cache_resource(show_spinner="Training the AI Lab models on every cell with a GAEZ value (first time only)...")
def get_models(res: str = "1 km"):
    from sklearn.neighbors import NearestNeighbors

    df = C.load_data()
    tcols = targets_for(res)
    ok = df[tcols].notna().all(axis=1).to_numpy()      # 1 km: no target where GAEZ marks no cropland
    X, Y = df[FEATURES].to_numpy(float), df[tcols].to_numpy(float)
    t0 = time.perf_counter()
    with threadpool_limits(N_THREADS):
        models = {name: _fit(name, X[ok], Y[ok]) for name in MODEL_NAMES}
    secs = time.perf_counter() - t0
    med = np.nanmedian(X, axis=0)
    Xf = np.where(np.isnan(X), med, X)
    mu, sd = Xf.mean(axis=0), Xf.std(axis=0) + 1e-9
    nn = NearestNeighbors(n_neighbors=2).fit((Xf - mu) / sd)
    typical = float(np.median(nn.kneighbors((Xf - mu) / sd)[0][:, 1]))
    return {"models": models, "seconds": secs, "median": med, "mu": mu, "sd": sd, "nn": nn, "typical_nn": typical,
            "x_min": np.nanmin(X, axis=0), "x_max": np.nanmax(X, axis=0)}


def predict_all(bundle, X) -> dict[str, np.ndarray]:
    """{model name: (n, 10) predicted GAEZ indices, clipped to 0-10,000}."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    with threadpool_limits(N_THREADS):
        return {n: np.clip(_predict(m, X), 0, 10000) for n, m in bundle["models"].items()}


def novelty(bundle, x) -> tuple[float, str]:
    """Distance to the closest real cell in standardised feature space, relative to the typical cell-to-cell distance."""
    x = np.where(np.isnan(x), bundle["median"], x)
    d = float(bundle["nn"].kneighbors(((x - bundle["mu"]) / bundle["sd"])[None], n_neighbors=1)[0][0, 0])
    ratio = d / bundle["typical_nn"]
    verdict = "typical of Banten" if ratio <= 2 else "unusual" if ratio <= 6 else "far outside the data"
    return ratio, verdict


def contributions(bundle, x, crop) -> tuple[pd.Series, float]:
    """Per-feature contributions (SHAP values computed natively by XGBoost) to one crop's prediction."""
    import xgboost as xgb

    model = bundle["models"]["XGBoost"][C.CROPS.index(crop)]
    contrib = model.get_booster().predict(xgb.DMatrix(np.atleast_2d(x), feature_names=None), pred_contribs=True)[0]
    return pd.Series(contrib[:-1], index=FEATURES), float(contrib[-1])


def ice(bundle, model_name, x, crop, feature, n=40):
    """Prediction for one crop while one feature is swept over its observed range (all else fixed)."""
    df = C.load_data()
    grid = np.linspace(np.nanquantile(df[feature], 0.005), np.nanquantile(df[feature], 0.995), n)
    Xs = np.repeat(np.atleast_2d(x), n, axis=0)
    Xs[:, FEATURES.index(feature)] = grid
    with threadpool_limits(N_THREADS):
        pred = np.clip(_predict(bundle["models"][model_name], Xs), 0, 10000)[:, C.CROPS.index(crop)]
    return pd.DataFrame({feature: grid, "prediction": pred})


@st.cache_data(show_spinner="Retraining ExtraTrees without this GAEZ pixel and its neighbours...", max_entries=64)
def heldout_prediction(pix_row: int, pix_col: int, x: tuple, res: str = "1 km") -> np.ndarray:
    """ExtraTrees trained on every cell outside the 3 x 3 block of GAEZ pixels around the query: an honest
    prediction for a place the model has never seen (the in-sample prediction is optimistic)."""
    t = E.load_table(complete=False)   # every cell; the targets of the chosen resolution are filtered below
    ring = (t["pix_row"].sub(pix_row).abs() <= 1) & (t["pix_col"].sub(pix_col).abs() <= 1)
    keep = ~ring & t[targets_for(res)].notna().all(axis=1)
    X, Y = t.loc[keep, FEATURES].to_numpy(float), t.loc[keep, targets_for(res)].to_numpy(float)
    with threadpool_limits(N_THREADS):
        model = _fit("ExtraTrees", X, Y, n_trees=80)
        return np.clip(model.predict(np.atleast_2d(np.array(x, dtype=float))), 0, 10000)[0]


@st.cache_data(show_spinner="Running cross-validation...", max_entries=32)
def arena(crops: tuple, models: tuple, schemes: tuple, res: str = "1 km") -> pd.DataFrame:
    """Quick 5-fold CV (one repeat) of the chosen models, through the same code path as the notebooks."""
    t = E.load_table(complete=False)   # every cell; the targets of the chosen resolution are filtered below
    cols = {c: C.gaez_col(c, "index", res) for c in crops}
    t = t[t[list(cols.values())].notna().all(axis=1)].reset_index(drop=True)
    targets = {c: t[col].to_numpy() for c, col in cols.items()}
    t0 = time.perf_counter()
    out, _ = E.run_grid(t, targets, list(models), list(schemes), n_repeats=1, n_jobs=N_THREADS, verbose=False)
    out["seconds_total"] = time.perf_counter() - t0
    out["model"] = out["model"].map(ARENA_MODELS)
    return out


WORKER = C.ROOT / "scripts" / "dashboard" / "dl_worker.py"


@st.cache_data(show_spinner=False, max_entries=16)
def train_mlp(hidden: tuple, act: str, dropout: float, weight_decay: float, epochs: int, lr: float, batch: int,
              seed: int, res: str = "1 km") -> dict:
    """Train a multi-output MLP on 80 % of the GAEZ pixels and test it on the other 20 % (one pixel-grouped fold), in a
    separate process. Returns the loss curves, per-crop R² of the MLP and of an ExtraTrees baseline on the same split,
    and a sample of test predictions. Uses the GPU when PyTorch sees one."""
    args = dict(hidden=list(hidden), act=act, dropout=dropout, weight_decay=weight_decay, epochs=epochs, lr=lr,
                batch=batch, seed=seed, n_threads=N_THREADS,
                target_prefix="gaez1k_index_" if res == "1 km" else "gaez_index_")
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    proc = subprocess.run([sys.executable, str(WORKER), json.dumps(args)], capture_output=True, text=True,
                          timeout=1800, env=env, cwd=str(WORKER.parent))
    if proc.returncode != 0:
        raise RuntimeError(f"training worker failed (exit {proc.returncode}): {proc.stderr[-1500:]}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    out["history"] = pd.DataFrame(out["history"])
    out["scores"] = pd.DataFrame(out["scores"]).assign(crop=lambda d: d["crop"].map(C.CROP_LABELS))
    out["sample"] = pd.DataFrame(out["sample"])
    return out
