"""Shared building blocks for the ML experiments: data access, feature groups, CV schemes,
model zoo, metrics and a parallel grid runner.

Deliberately torch-free so joblib worker processes start fast; the PyTorch models live in
exp_torch.py. If a notebook uses both SHAP and torch, `import torch` must come first (on Windows,
importing shap first loads a conflicting runtime DLL and torch then fails with WinError 127).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJECT_ROOT / "datasets" / "processed" / "features" / "tabular_features.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"        # small summary tables (tracked)
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
OOF_DIR = PROJECT_ROOT / "datasets" / "processed" / "experiments"  # bulky per-cell predictions (ignored)

CROPS = ["rice", "maize", "cassava", "soybean", "oilpalm", "rubber", "sugarcane", "coffee", "cacao", "coconut"]
GCS_CROPS = ["rice", "maize", "cassava", "soybean", "oilpalm", "sugarcane"]  # crops covered by Global Crop Suitability 2024

NOT_RECOMMENDED_THRESHOLD = 5000  # same rule the dashboard applies to the GAEZ index (0-10,000)

GAEZ_CELL_DEG = 1 / 12  # GAEZ v5 native grid: 5 arc-minutes (~9 km)

N_JOBS = 6  # parallel workers for the grid runners; deliberately not all cores so the machine stays usable

LC_CODES = [10, 30, 40, 50, 60, 80, 90, 95]  # ESA WorldCover classes present in Banten

FEATURE_GROUPS = {
    "climate": ["bio1_temp_mean_c", "bio12_precip_annual_mm", "bio15_precip_seasonality",
                "chirps_annual_mean_mm", "chirps_annual_std_mm"],
    "soil": ["soil_ph", "soil_soc", "soil_nitrogen", "soil_clay_pct", "soil_sand_pct", "soil_cec",
             "soil_bulk_density"],
    "terrain": ["elevation_mean_m", "slope_mean_deg"],
    "landcover": ["cropland_pct"] + [f"lc_{c}" for c in LC_CODES],
    "coords": ["lon", "lat"],
}
BASE_GROUPS = ("climate", "soil", "terrain")  # the physical drivers GAEZ itself is built from


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def load_table() -> pd.DataFrame:
    """Warehouse fact table plus helper columns (GAEZ pixel ids, land-cover one-hot)."""
    df = pd.read_csv(FEATURES_CSV)
    # Which native GAEZ pixel each 1 km cell centroid falls in (the 5' grid is anchored at -180/90).
    df["pix_row"] = np.floor((90 - df["lat"].to_numpy()) / GAEZ_CELL_DEG).astype(int)
    df["pix_col"] = np.floor((df["lon"].to_numpy() + 180) / GAEZ_CELL_DEG).astype(int)
    df["pix"] = df["pix_row"] * 100_000 + df["pix_col"]
    for code in LC_CODES:
        df[f"lc_{code}"] = (df["landcover_majority"] == code).astype(float)
    return df


def add_focal_features(df: pd.DataFrame, columns: list[str], sizes=(3, 5, 9, 15, 25, 41)) -> pd.DataFrame:
    """Neighbourhood means (size x size cells on the 1 km grid) of each column, named `<col>__f<size>`.

    Cells outside the province are not rows of the table, so each mean runs over valid cells only
    (normalised convolution). Only inputs are smoothed, never targets, so this is leak-free.
    """
    from scipy.ndimage import uniform_filter

    r, c = df["row"].to_numpy(), df["col"].to_numpy()
    shape = (r.max() + 1, c.max() + 1)
    out = {}
    for col in columns:
        vals = df[col].to_numpy(dtype=float)
        ok = ~np.isnan(vals)
        grid, w = np.zeros(shape), np.zeros(shape)
        grid[r[ok], c[ok]] = vals[ok]
        w[r[ok], c[ok]] = 1.0
        for s in sizes:
            num = uniform_filter(grid, size=s, mode="constant", cval=0.0)
            den = uniform_filter(w, size=s, mode="constant", cval=0.0)
            out[f"{col}__f{s}"] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)[r, c]
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def add_pixel_mean_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Mean of each column over the cells that share a native GAEZ pixel, named `<col>__pix`."""
    means = df.groupby("pix")[columns].transform("mean")
    means.columns = [f"{c}__pix" for c in columns]
    return pd.concat([df, means], axis=1)


def feature_columns(groups=BASE_GROUPS) -> list[str]:
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def gaez_cols(crops=CROPS) -> list[str]:
    return [f"gaez_index_{c}" for c in crops]


def gcs_cols(crops=GCS_CROPS) -> list[str]:
    return [f"gcs2024_{c}" for c in crops]


def crop_labels(df: pd.DataFrame, mode: str = "top") -> pd.Series:
    """Classification targets derived from the 10 GAEZ indices.

    mode="top"      : argmax crop, no threshold (8 of the 10 crops ever win).
    mode="thresh"   : argmax crop, or "Not recommended" if the best index is below the 5000 cut-off.
    """
    idx = df[gaez_cols()].to_numpy()
    best = idx.argmax(axis=1)
    labels = np.array(CROPS, dtype=object)[best]
    if mode == "thresh":
        labels = np.where(idx.max(axis=1) < NOT_RECOMMENDED_THRESHOLD, "Not recommended", labels)
    return pd.Series(labels, index=df.index, name="label")


# --------------------------------------------------------------------------------------------
# Cross-validation schemes
# --------------------------------------------------------------------------------------------
def make_splits(df: pd.DataFrame, scheme: str, n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Train/test index pairs for one CV scheme.

    random      plain shuffled K-fold over 1 km cells (what a naive tabular workflow does)
    block<k>    square blocks of k x k cells (k km) kept together, e.g. block10
    pixel       whole native GAEZ pixels kept together (the source's own resolution)
    <scheme>_buf  any grouped scheme, but training cells in the 8 GAEZ pixels around every test pixel are
                  dropped as well (only sensible with large blocks, e.g. block40_buf; with scattered
                  pixel folds the ring swallows most of the training set)
    """
    n = len(df)
    if scheme == "random":
        return [(tr, te) for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(n))]
    buffered = scheme.endswith("_buf")
    base = scheme.removesuffix("_buf")
    if base.startswith("block"):
        k = int(base.removeprefix("block"))
        groups = (df["row"].to_numpy() // k) * 10_000 + (df["col"].to_numpy() // k)
    elif base == "pixel":
        groups = df["pix"].to_numpy()
    else:
        raise ValueError(f"unknown CV scheme: {scheme}")

    pix_r, pix_c = df["pix_row"].to_numpy(), df["pix_col"].to_numpy()
    splits = []
    for tr, te in GroupKFold(n_splits, shuffle=True, random_state=seed).split(np.arange(n), groups=groups):
        if buffered:
            test_pix = set(zip(pix_r[te], pix_c[te]))
            ring = {(r + dr, c + dc) for r, c in test_pix for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            tr = tr[np.fromiter(((r, c) not in ring for r, c in zip(pix_r[tr], pix_c[tr])), dtype=bool, count=len(tr))]
        splits.append((tr, te))
    return splits


# --------------------------------------------------------------------------------------------
# Model zoo: deliberately small and light, every model fits in a few seconds on ~7,500 cells
#   regression      dummy, ridge, knn, et (ExtraTrees), hgb (HistGradientBoosting), xgb (XGBoost)
#   classification  dummy, logreg, knn, et, hgb, xgb
# --------------------------------------------------------------------------------------------
def make_regressor(name: str, seed: int = 0, n_jobs: int = 1):
    imp = SimpleImputer(strategy="median")
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return make_pipeline(imp, StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if name == "knn":
        return make_pipeline(imp, StandardScaler(), KNeighborsRegressor(n_neighbors=20, weights="distance", n_jobs=n_jobs))
    if name == "et":
        return make_pipeline(imp, ExtraTreesRegressor(100, min_samples_leaf=3, max_features=0.7, n_jobs=n_jobs, random_state=seed))
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0,
                                             random_state=seed)
    if name == "xgb":
        return XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=3, reg_lambda=2.0, tree_method="hist", n_jobs=n_jobs, random_state=seed)
    raise ValueError(f"unknown regressor: {name}")


def make_classifier(name: str, seed: int = 0, n_jobs: int = 1):
    imp = SimpleImputer(strategy="median")
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return make_pipeline(imp, StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"))
    if name == "knn":
        return make_pipeline(imp, StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=n_jobs))
    if name == "et":
        return make_pipeline(imp, ExtraTreesClassifier(100, min_samples_leaf=2, max_features="sqrt", n_jobs=n_jobs,
                                                       class_weight="balanced", random_state=seed))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0,
                                              class_weight="balanced", random_state=seed)
    if name == "xgb":
        return XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=2, reg_lambda=2.0, tree_method="hist", n_jobs=n_jobs, random_state=seed)
    raise ValueError(f"unknown classifier: {name}")


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# --------------------------------------------------------------------------------------------
# Grid runners (one joblib task = one model x target x CV scheme x repeat, all folds inside)
# --------------------------------------------------------------------------------------------
def _fit_predict_oof(model_name: str, kind: str, X: np.ndarray, y: np.ndarray, splits, seed: int, clip=None):
    """Out-of-fold predictions for one target. Cells that are never tested stay NaN (nothing here)."""
    oof = np.full(len(y), np.nan if kind == "reg" else -1, dtype=float if kind == "reg" else int)
    fit_s = 0.0
    for tr, te in splits:
        model = make_regressor(model_name, seed) if kind == "reg" else make_classifier(model_name, seed)
        # A class can be missing from a training fold (e.g. a rare crop that wins in a single pixel);
        # re-encode to 0..k-1 so every classifier, XGBoost included, accepts the fold.
        classes, y_tr = np.unique(y[tr], return_inverse=True) if kind == "clf" else (None, y[tr])
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[tr], y_tr)
        fit_s += time.perf_counter() - t0
        pred = model.predict(X[te])
        if kind == "reg" and clip is not None:
            pred = np.clip(pred, *clip)
        oof[te] = classes[pred.astype(int)] if kind == "clf" else pred
    return oof, fit_s


def _task(kind, model_name, target, scheme, repeat, X, y, splits, seed, clip, keep_oof):
    oof, fit_s = _fit_predict_oof(model_name, kind, X, y, splits, seed, clip)
    seen = oof >= 0 if kind == "clf" else ~np.isnan(oof)
    metrics = regression_metrics(y[seen], oof[seen]) if kind == "reg" else classification_metrics(y[seen], oof[seen])
    row = {"target": target, "model": model_name, "scheme": scheme, "repeat": repeat, "fit_seconds": fit_s,
           "n_train_mean": float(np.mean([len(tr) for tr, _ in splits])), **metrics}
    return row, (oof if keep_oof else None)


def run_grid(df: pd.DataFrame, targets: dict[str, np.ndarray], models: list[str], schemes: list[str],
             kind: str = "reg", groups=BASE_GROUPS, columns: list[str] | None = None, n_repeats: int = 2,
             n_splits: int = 5, clip=(0.0, 10_000.0), n_jobs: int = N_JOBS, keep_oof: bool = False, verbose: bool = True,
             custom_splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] | None = None):
    """Evaluate every (model, target, scheme, repeat) with identical splits across models.

    targets        {name: y vector aligned with df}; for kind="clf" the vectors must be integer-encoded.
    custom_splits  {scheme name: [(train_idx, test_idx), ...]} replaces the built-in schemes (one repeat);
                   metrics are then computed on the cells that appear in some test set.
    Returns (results DataFrame with one row per task, {key: oof vector} if keep_oof else {}).
    """
    cols = columns if columns is not None else feature_columns(groups)
    X = df[cols].to_numpy(dtype=float)
    if custom_splits is not None:
        schemes, n_repeats = list(custom_splits), 1
        split_cache = {(s, 0): custom_splits[s] for s in schemes}
    else:
        split_cache = {(s, r): make_splits(df, s, n_splits, seed=r) for s in schemes for r in range(n_repeats)}
    jobs = [(m, t, s, r) for t in targets for m in models for s in schemes for r in range(n_repeats)]
    if verbose:
        print(f"{len(jobs)} tasks ({len(models)} models x {len(targets)} targets x {len(schemes)} schemes x {n_repeats} repeats), "
              f"{len(cols)} features", flush=True)
    t0 = time.perf_counter()
    out = Parallel(n_jobs=n_jobs, batch_size=1, max_nbytes=None, inner_max_num_threads=1)(
        delayed(_task)(kind, m, t, s, r, X, targets[t], split_cache[(s, r)], r, clip if kind == "reg" else None, keep_oof)
        for m, t, s, r in jobs)
    if verbose:
        print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)
    res = pd.DataFrame([o[0] for o in out])
    oof = {(m, t, s, r): o[1] for (m, t, s, r), o in zip(jobs, out) if o[1] is not None}
    return res, oof


def summarise(res: pd.DataFrame, by=("model", "scheme"), metric: str = "r2") -> pd.DataFrame:
    """Mean and std over repeats and targets for a quick leaderboard."""
    g = res.groupby(list(by))[metric]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(), "median": g.median()}).sort_values("mean", ascending=False)


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=index, float_format="%.5g")
    return path


def save_fig(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def save_oof(oof: pd.DataFrame, name: str) -> Path:
    """Per-cell out-of-fold predictions (bulky, git-ignored) so later experiments can reuse them."""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    path = OOF_DIR / f"{name}.parquet"
    oof.to_parquet(path, index=False)
    return path
