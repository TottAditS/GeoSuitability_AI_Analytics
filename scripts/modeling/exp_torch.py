"""PyTorch models for the surrogate experiments (tabular MLP, grid CNN, Sentinel-2 patch CNN and fusion network).
Runs on the GPU when there is one.

`import torch` stays first: on Windows, importing shap before torch breaks torch's DLL loading.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

import numpy as np

import exp_lib as E

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(E.N_JOBS)   # only matters when there is no GPU
_ACTS = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}

DEFAULT_CFG = dict(hidden=(128, 64), act="silu", norm=None, dropout=0.0, lr=3e-3, weight_decay=1e-4, epochs=100,
                   batch_size=256, loss="mse", input_noise=0.0, huber_delta=1.0)


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden=(128, 64), act="silu", norm=None, dropout=0.0):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(h))
            elif norm == "layer":
                layers.append(nn.LayerNorm(h))
            layers.append(_ACTS[act]())
            if dropout:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _loss_fn(name: str, delta: float):
    return {"mse": nn.MSELoss(), "l1": nn.L1Loss(), "huber": nn.HuberLoss(delta=delta)}[name]


def fit_predict(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, cfg: dict | None = None, seed: int = 0) -> np.ndarray:
    """Train on (Xtr, Ytr) and predict Xte. Inputs are median-imputed and z-scored with train statistics; each
    target column is z-scored too, so a multi-output net weighs every crop equally. Fixed epochs with a cosine
    learning-rate decay (no early stopping, so no validation split is needed)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    Ytr = Ytr.reshape(len(Ytr), -1)
    med = np.nanmedian(Xtr, axis=0)
    Xtr, Xte = np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-9

    torch.manual_seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    xt = torch.as_tensor((Xtr - mu) / sd, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor((Ytr - ymu) / ysd, dtype=torch.float32, device=DEVICE)
    model = MLP(xt.shape[1], yt.shape[1], cfg["hidden"], cfg["act"], cfg["norm"], cfg["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, bs = len(xt), cfg["batch_size"]
    steps = cfg["epochs"] * int(np.ceil(n / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    loss_fn = _loss_fn(cfg["loss"], cfg["huber_delta"])

    model.train()
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(n, device=DEVICE, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = xt[idx]
            if cfg["input_noise"]:
                xb = xb + cfg["input_noise"] * torch.randn(xb.shape, device=DEVICE, generator=gen)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), yt[idx]).backward()
            opt.step()
            sched.step()

    model.eval()
    with torch.no_grad():
        xe = torch.as_tensor((Xte - mu) / sd, dtype=torch.float32, device=DEVICE)
        pred = model(xe).cpu().numpy()
    return pred * ysd + ymu


def cv_multioutput(df, targets: dict[str, np.ndarray], cfg: dict | None = None, scheme: str = "pixel", n_repeats: int = 2,
                   columns: list[str] | None = None, n_splits: int = 5, clip=(0.0, 10_000.0), single_output: bool = False,
                   n_seeds: int = 1):
    """Out-of-fold evaluation of the MLP; returns a tidy DataFrame (one row per crop and repeat) and the OOF cube.

    single_output=True trains one network per crop instead of one network for all crops.
    n_seeds > 1 averages that many networks (different initialisations) before scoring."""
    cols = columns if columns is not None else E.feature_columns()
    X = df[cols].to_numpy(float)
    names = list(targets)
    Y = np.column_stack([targets[k] for k in names]).astype(float)
    rows, oofs = [], []
    t0 = time.perf_counter()
    for rep in range(n_repeats):
        oof = np.full(Y.shape, np.nan)
        for tr, te in E.make_splits(df, scheme, n_splits, seed=rep):
            preds = []
            for s in range(n_seeds):
                seed = rep * 100 + s
                if single_output:
                    p = np.column_stack([fit_predict(X[tr], Y[tr, j], X[te], cfg, seed)[:, 0] for j in range(Y.shape[1])])
                else:
                    p = fit_predict(X[tr], Y[tr], X[te], cfg, seed)
                preds.append(p)
            oof[te] = np.clip(np.mean(preds, axis=0), *clip)
        for j, k in enumerate(names):
            rows.append({"target": k, "repeat": rep, **E.regression_metrics(Y[:, j], oof[:, j])})
        oofs.append(oof)
    import pandas as pd
    out = pd.DataFrame(rows)
    out["fit_seconds"] = (time.perf_counter() - t0) / max(n_repeats, 1)
    return out, np.stack(oofs)


# --------------------------------------------------------------------------------------------
# Grid models: the province as a 136 x 186 image (channels = physical drivers), per-cell targets masked in the loss
# --------------------------------------------------------------------------------------------
DEFAULT_GRID_CFG = dict(arch="unet", base=32, depth=3, act="silu", lr=2e-3, weight_decay=1e-4, steps=1500, batch=16,
                        crop=64, augment=True, hidden=(64, 64))


class _Block(nn.Module):
    def __init__(self, c_in: int, c_out: int, act: str):
        super().__init__()
        groups = 8 if c_out % 8 == 0 else 1
        self.f = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(groups, c_out), _ACTS[act](),
                               nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(groups, c_out), _ACTS[act]())

    def forward(self, x):
        return self.f(x)


class UNet(nn.Module):
    def __init__(self, c_in: int, c_out: int, base: int = 32, depth: int = 3, act: str = "silu"):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth)]
        self.enc, c = nn.ModuleList(), c_in
        for ch in chans:
            self.enc.append(_Block(c, ch, act))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for ch in reversed(chans[:-1]):
            self.up.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.dec.append(_Block(2 * ch, ch, act))
            c = ch
        self.head = nn.Conv2d(c, c_out, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


class PointwiseNet(nn.Module):
    """1 x 1 convolutions only: the same network as a per-cell MLP, so the U-Net comparison isolates spatial context."""

    def __init__(self, c_in: int, c_out: int, hidden=(64, 64), act: str = "silu"):
        super().__init__()
        layers, c = [], c_in
        for h in hidden:
            layers += [nn.Conv2d(c, h, 1), _ACTS[act]()]
            c = h
        layers.append(nn.Conv2d(c, c_out, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def grid_data(df, columns: list[str], targets: dict[str, np.ndarray], multiple: int = 8):
    """Image tensors for the province: inputs (channels = z-scored features + validity mask), targets (/10,000)."""
    r, c = df["row"].to_numpy(), df["col"].to_numpy()
    H, W = r.max() + 1, c.max() + 1
    Hp, Wp = int(np.ceil(H / multiple) * multiple), int(np.ceil(W / multiple) * multiple)
    vals = df[columns].to_numpy(float)
    vals = np.where(np.isnan(vals), np.nanmedian(vals, axis=0), vals)
    vals = (vals - vals.mean(0)) / (vals.std(0) + 1e-9)
    X = np.zeros((len(columns) + 1, Hp, Wp), np.float32)
    X[:-1, r, c] = vals.T
    X[-1, r, c] = 1.0
    Y = np.zeros((len(targets), Hp, Wp), np.float32)
    for j, k in enumerate(targets):
        Y[j, r, c] = np.asarray(targets[k], float) / 10_000
    return X, Y, r, c


def fit_predict_grid(X, Y, r, c, train_idx, cfg: dict | None = None, seed: int = 0) -> np.ndarray:
    """Train a grid network on the cells `train_idx` (loss masked to those cells) and return per-cell predictions
    (n_cells x n_targets, 0-10,000 scale) for every cell. Inputs are visible everywhere, labels only for train cells."""
    cfg = {**DEFAULT_GRID_CFG, **(cfg or {})}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_t = Y.shape[0]
    ymu = Y[:, r[train_idx], c[train_idx]].mean(axis=1)
    ysd = Y[:, r[train_idx], c[train_idx]].std(axis=1) + 1e-9
    Yz = ((Y - ymu[:, None, None]) / ysd[:, None, None]).astype(np.float32)
    M = np.zeros(Y.shape[1:], np.float32)
    M[r[train_idx], c[train_idx]] = 1.0

    xt = torch.as_tensor(X, device=DEVICE)
    yt = torch.as_tensor(Yz, device=DEVICE)
    mt = torch.as_tensor(M, device=DEVICE)
    if cfg["arch"] == "unet":
        model = UNet(X.shape[0], n_t, cfg["base"], cfg["depth"], cfg["act"]).to(DEVICE)
    else:
        model = PointwiseNet(X.shape[0], n_t, cfg["hidden"], cfg["act"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"], pct_start=0.1)
    Hp, Wp, cs, bs = X.shape[1], X.shape[2], cfg["crop"], cfg["batch"]

    model.train()
    for _ in range(cfg["steps"]):
        i = rng.integers(0, Hp - cs + 1, bs)
        j = rng.integers(0, Wp - cs + 1, bs)
        xb = torch.stack([xt[:, a:a + cs, b:b + cs] for a, b in zip(i, j)])
        yb = torch.stack([yt[:, a:a + cs, b:b + cs] for a, b in zip(i, j)])
        mb = torch.stack([mt[a:a + cs, b:b + cs] for a, b in zip(i, j)])[:, None]
        if cfg["augment"]:                      # same flip / rotation for the whole batch
            k = int(rng.integers(0, 4))
            xb, yb, mb = (torch.rot90(t, k, dims=(2, 3)) for t in (xb, yb, mb))
            if rng.random() < 0.5:
                xb, yb, mb = (torch.flip(t, dims=(3,)) for t in (xb, yb, mb))
        opt.zero_grad(set_to_none=True)
        loss = (((model(xb) - yb) ** 2) * mb).sum() / (mb.sum() * n_t).clamp(min=1.0)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred = model(xt[None])[0].cpu().numpy()          # (n_targets, Hp, Wp), z-scored
    pred = pred * ysd[:, None, None] + ymu[:, None, None]
    return pred[:, r, c].T * 10_000


def cv_grid(df, targets: dict[str, np.ndarray], cfg: dict | None = None, scheme: str = "pixel", n_repeats: int = 2,
            columns: list[str] | None = None, n_splits: int = 5, clip=(0.0, 10_000.0)):
    """Out-of-fold evaluation of a grid network; same output layout as cv_multioutput."""
    import pandas as pd

    cols = columns if columns is not None else E.feature_columns()
    cfg_full = {**DEFAULT_GRID_CFG, **(cfg or {})}
    multiple = 8 if cfg_full["arch"] == "unet" else 1
    X, Y, r, c = grid_data(df, cols, targets, multiple=multiple)
    names = list(targets)
    Ytab = np.column_stack([targets[k] for k in names]).astype(float)
    rows, oofs = [], []
    t0 = time.perf_counter()
    for rep in range(n_repeats):
        oof = np.full(Ytab.shape, np.nan)
        for tr, te in E.make_splits(df, scheme, n_splits, seed=rep):
            oof[te] = np.clip(fit_predict_grid(X, Y, r, c, tr, cfg_full, seed=rep)[te], *clip)
        for j, k in enumerate(names):
            rows.append({"target": k, "repeat": rep, **E.regression_metrics(Ytab[:, j], oof[:, j])})
        oofs.append(oof)
    out = pd.DataFrame(rows)
    out["fit_seconds"] = (time.perf_counter() - t0) / max(n_repeats, 1)
    return out, np.stack(oofs)


# ----------------------------------------------------------------------------------------------------------------
# Sentinel-2 image patches: CNN branch and fusion with the drivers (m14, the AI Lab deep-learning playground)
# ----------------------------------------------------------------------------------------------------------------
def load_patch_composite(grid_ids, years=range(2020, 2025)) -> np.ndarray:
    """Five-year Sentinel-2 composite patch per cell, (n, 10, 25, 25): the 9 band means of every 4 x 4 block weighted by
    the share of cloud-free pixels behind it, plus that share averaged over the years. Rows follow `grid_ids`; notebook 11
    wrote the yearly arrays in the row order of tabular_features.csv."""
    import pandas as pd

    order = pd.read_csv(E.FEATURES_CSV, usecols=["grid_id"])["grid_id"]
    pos = pd.Series(np.arange(len(order)), index=order).loc[list(grid_ids)].to_numpy()
    num = den = None
    for y in years:
        a = np.load(E.S2_PATCH_DIR / f"patches_{y}.npy", mmap_mode="r")[pos].astype(np.float32)
        w = a[:, 9:10]
        num = a[:, :9] * w if num is None else num + a[:, :9] * w
        den = w if den is None else den + w
    return np.concatenate([num / np.maximum(den, 1e-6), den / len(years)], axis=1)


class PatchCNN(nn.Module):
    """Three 3 x 3 convolution blocks (25 -> 13 -> 7 px) and global average pooling: one 64-d vector per patch."""

    def __init__(self, c_in=10, width=32, d_out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, width, 3, padding=1), nn.BatchNorm2d(width), nn.SiLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.BatchNorm2d(2 * width), nn.SiLU(),
            nn.Conv2d(2 * width, d_out, 3, stride=2, padding=1), nn.BatchNorm2d(d_out), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, p):
        return self.net(p)


class FusionNet(nn.Module):
    """Patch branch, driver branch or both, joined before a small multi-output head (the ten GAEZ indices)."""

    def __init__(self, use_patch=True, use_drivers=True, n_drivers=14, n_out=10):
        super().__init__()
        self.cnn = PatchCNN() if use_patch else None
        self.mlp = nn.Sequential(nn.Linear(n_drivers, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU()) if use_drivers else None
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(64 * (use_patch + use_drivers), 64), nn.SiLU(), nn.Linear(64, n_out))

    def forward(self, p, x):
        parts = ([self.cnn(p)] if self.cnn is not None else []) + ([self.mlp(x)] if self.mlp is not None else [])
        return self.head(torch.cat(parts, dim=1))


def fit_predict_fusion(P, X, Y, tr, te, use_patch=True, use_drivers=True, seed=0, epochs=40, batch=256, lr=2e-3,
                       track=False):
    """Train a FusionNet on the rows `tr` and predict the rows `te` (0-10,000 scale). P: patches (n, 10, 25, 25), X: drivers
    (n, d), Y: targets (n, k). Inputs, patches and targets are standardised with training statistics only; patches are
    flipped at random during training (orientation does not matter for suitability). track=True also returns the
    per-epoch training and validation loss (MSE on z-scored targets)."""
    import torch.nn.functional as F

    torch.manual_seed(seed)
    med = np.nanmedian(X[tr], axis=0)
    Xf = np.where(np.isnan(X), med, X)
    Xs = (Xf - Xf[tr].mean(0)) / (Xf[tr].std(0) + 1e-9)
    Ps = (P - P[tr].mean(axis=(0, 2, 3), keepdims=True)) / (P[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6)
    ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-9
    to = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=DEVICE)  # noqa: E731
    xt, pt, yt = to(Xs[tr]), to(Ps[tr]), to((Y[tr] - ym) / ys)
    net = FusionNet(use_patch, use_drivers, n_drivers=X.shape[1], n_out=Y.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def predict(rows):
        net.eval()
        with torch.no_grad():
            return torch.cat([net(to(Ps[rows[i:i + 2048]]), to(Xs[rows[i:i + 2048]])) for i in range(0, len(rows), 2048)])

    hist = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr), device=DEVICE, generator=gen)
        total = 0.0
        for i in range(0, len(tr), batch):
            b = perm[i:i + batch]
            pb = pt[b]
            if use_patch and torch.rand(1, device=DEVICE, generator=gen).item() < 0.5:
                pb = pb.flip(-1)
            if use_patch and torch.rand(1, device=DEVICE, generator=gen).item() < 0.5:
                pb = pb.flip(-2)
            loss = F.mse_loss(net(pb, xt[b]), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        if track:
            val = F.mse_loss(predict(te), to((Y[te] - ym) / ys)).item()
            hist.append({"epoch": ep + 1, "train loss": total / len(tr), "validation loss": val})
    pred = np.clip(predict(te).cpu().numpy() * ys + ym, 0, 10000)
    return (pred, hist) if track else pred
