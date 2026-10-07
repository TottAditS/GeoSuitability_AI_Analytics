"""Deep-learning playground worker: trains one network in its own process and prints the result as JSON.

Two networks: an MLP on the 14 physical drivers (architecture from the playground sliders) or the m14 fusion network
(exp_torch.FusionNet: a CNN on the Sentinel-2 image patch of each cell joined with the drivers).

Why a separate process: PyTorch ships its own OpenMP runtime (libiomp5md.dll) while numpy's multithreaded BLAS in this
environment uses another one (libomp.dll). In a single long-lived process (the dashboard runs correlations, PCA and
regressions) whichever loads second fails - "WinError 127 ... shm.dll" or "OMP: Error #15". Here torch is the first
import and nothing in the worker uses multithreaded BLAS, so the two never meet.

Usage (called by ai_models.train_network):  python dl_worker.py '<json arguments>'
"""
import torch  # noqa: I001  - must stay the first import (see above)

import json
import sys
import time
from pathlib import Path

import numpy as np

MODELING_DIR = Path(__file__).resolve().parents[1] / "modeling"
sys.path.insert(0, str(MODELING_DIR))
import exp_lib as E  # noqa: E402
import exp_torch as T  # noqa: E402


def main(args: dict) -> dict:
    torch.set_num_threads(args.get("n_threads", 4))
    hidden, act, dropout = tuple(args["hidden"]), args["act"], args["dropout"]
    epochs, lr, batch, seed, wd = args["epochs"], args["lr"], args["batch"], args["seed"], args["weight_decay"]
    features, crops = E.feature_columns(), E.CROPS
    targets = [f"{args.get('target_prefix', 'gaez_index_')}{c}" for c in crops]

    t = E.load_table(complete=False)   # every cell; the targets of the chosen resolution are filtered below
    t = t[t[targets].notna().all(axis=1)].reset_index(drop=True)   # 1 km: GAEZ leaves no-cropland cells empty
    tr, te = E.make_splits(t, "pixel", 5, seed=seed)[0]
    X, Y = t[features].to_numpy(float), t[targets].to_numpy(float)
    if args.get("arch") == "fusion":
        return fusion(args, t, X, Y, tr, te, crops, seed)
    med = np.nanmedian(X[tr], axis=0)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    ymu, ysd = Y[tr].mean(0), Y[tr].std(0) + 1e-9
    dev = T.DEVICE
    xt = torch.as_tensor((X[tr] - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.as_tensor((Y[tr] - ymu) / ysd, dtype=torch.float32, device=dev)
    xv = torch.as_tensor((X[te] - mu) / sd, dtype=torch.float32, device=dev)
    yv = torch.as_tensor((Y[te] - ymu) / ysd, dtype=torch.float32, device=dev)

    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    net = T.MLP(len(features), len(targets), hidden, act, None, dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(xt) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    loss_fn = torch.nn.MSELoss()
    hist = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=dev, generator=gen)
        total = 0.0
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(xv), yv).item()
        hist.append({"epoch": ep + 1, "train loss": total / len(xt), "validation loss": val})
    secs = time.perf_counter() - t0
    with torch.no_grad():
        pred = np.clip(net(xv).cpu().numpy() * ysd + ymu, 0, 10000)

    # Yardstick on exactly the same split (tree models are safe next to torch; only multithreaded BLAS is not).
    from sklearn.ensemble import ExtraTreesRegressor
    et = ExtraTreesRegressor(100, min_samples_leaf=3, max_features=0.7, n_jobs=args.get("n_threads", 4), random_state=0)
    et.fit(X[tr], (Y[tr] - ymu) / ysd)
    et_pred = np.clip(et.predict(X[te]) * ysd + ymu, 0, 10000)

    return package(hist, pred, et_pred, Y, t, te, crops, secs, int(sum(p.numel() for p in net.parameters())), len(tr),
                   "ExtraTrees, 14 drivers")


def fusion(args, t, X, Y, tr, te, crops, seed):
    """The m14 fusion network on one pixel-grouped fold; the yardstick is ExtraTrees on the drivers plus the Sentinel-2
    cell statistics, i.e. the same information in tabular form."""
    P = T.load_patch_composite(t["grid_id"])
    t0 = time.perf_counter()
    pred, hist = T.fit_predict_fusion(P, X, Y, tr, te, seed=seed, epochs=args["epochs"], batch=args["batch"], lr=args["lr"],
                                      track=True)
    secs = time.perf_counter() - t0
    params = int(sum(p.numel() for p in T.FusionNet(n_drivers=X.shape[1], n_out=Y.shape[1]).parameters()))

    from sklearn.ensemble import ExtraTreesRegressor
    Xs2 = t[E.feature_columns() + E.FEATURE_GROUPS["sentinel2"]].to_numpy(float)
    med = np.nanmedian(Xs2[tr], axis=0)
    Xs2 = np.where(np.isnan(Xs2), med, Xs2)
    ymu, ysd = Y[tr].mean(0), Y[tr].std(0) + 1e-9
    et = ExtraTreesRegressor(100, min_samples_leaf=3, max_features=0.7, n_jobs=args.get("n_threads", 4), random_state=0)
    et.fit(Xs2[tr], (Y[tr] - ymu) / ysd)
    et_pred = np.clip(et.predict(Xs2[te]) * ysd + ymu, 0, 10000)
    return package(hist, pred, et_pred, Y, t, te, crops, secs, params, len(tr), "ExtraTrees, drivers + Sentinel-2")


def r2(a, b):
    return (1 - ((a - b) ** 2).sum(0) / ((a - a.mean(0)) ** 2).sum(0)).tolist()


def package(hist, pred, et_pred, Y, t, te, crops, secs, params, n_train, yardstick):
    rng = np.random.default_rng(0)
    pick = rng.choice(len(te), size=min(1500, len(te)), replace=False)
    sample = {"grid_id": t["grid_id"].to_numpy()[te][pick].tolist()}
    for j, c in enumerate(crops):
        sample[f"true_{c}"] = Y[te][pick, j].tolist()
        sample[f"net_{c}"] = pred[pick, j].tolist()
    return {"history": hist, "scores": {"crop": crops, "Network": r2(Y[te], pred), yardstick: r2(Y[te], et_pred)},
            "yardstick": yardstick, "seconds": secs, "device": str(T.DEVICE), "params": params, "n_train": int(n_train),
            "n_test": int(len(te)), "test_ids": t["grid_id"].to_numpy()[te].tolist(), "sample": sample}


if __name__ == "__main__":
    print(json.dumps(main(json.loads(sys.argv[1]))))
