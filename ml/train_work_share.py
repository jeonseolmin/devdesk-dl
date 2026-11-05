"""Focus-drop forecast as "how many of the next H minutes go to work?" (ml/work_share.py): baselines vs LightGBM
vs GRU, with what DevDesk records ("base") and with BEHACOM-only extras ("+extras": network, typing, scroll, CPU).

Usage: uv run --group ml --group wake python -m ml.train_work_share [--data <behacom_slim.csv>] [--devdesk <csv>]
Metric: MAE in minutes on each user's last 15 % (time order); paired block bootstrap by session vs the user-mean
baseline. Writes only ml/models/work_share_model_<tag>_metrics.json (no model file: nothing loads it yet).
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml import devdesk_data, work_share
from ml.train_next_app import block_ci

ROOT = Path(__file__).resolve().parent
SEED = 42
META = ("device_id", "m", "session", "y", "seq")


def table(ex):
    x = ex.drop(columns=list(META))
    return x.assign(user=ex["device_id"].astype("category"))


def fit_lgb(parts, seed=SEED):
    model = lgb.LGBMRegressor(objective="l1", n_estimators=600, learning_rate=0.03, num_leaves=15,
                              min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=seed, verbose=-1)
    model.fit(table(parts["train"]), parts["train"]["y"], eval_X=(table(parts["val"]),), eval_y=(parts["val"]["y"],),
              eval_metric="l1", callbacks=[lgb.early_stopping(50, verbose=False)])
    return lambda ex: np.clip(model.predict(table(ex)), 0, 1)


def fit_gru(parts, n_users, dev, seed, epochs=60):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    feat = parts["train"]["seq"].iloc[0].shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(feat, 32, batch_first=True)
            self.user = nn.Embedding(n_users, 4)
            self.out = nn.Sequential(nn.Linear(36, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, seq, user):
            h = self.gru(seq)[1][-1]
            return torch.sigmoid(self.out(torch.cat([h, self.user(user)], 1))).squeeze(1)

    def tensors(ex):
        return (torch.tensor(np.stack(ex["seq"].to_numpy()), device=dev),
                torch.tensor(ex["device_id"].to_numpy(), device=dev),
                torch.tensor(ex["y"].to_numpy(), dtype=torch.float32, device=dev))

    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    tr, va = tensors(parts["train"]), tensors(parts["val"])
    gen = torch.Generator().manual_seed(seed)
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(tr[2]), generator=gen).split(128):
            b = b.to(dev)
            loss = (model(tr[0][b], tr[1][b]) - tr[2][b]).abs().mean()  # L1 = the reported MAE
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = (model(va[0], va[1]) - va[2]).abs().mean().item()
        if val < best - 1e-5:
            best, state, bad = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    if state is None:
        raise SystemExit("work_share: the GRU's val loss never became a number")
    model.load_state_dict(state)
    model.eval()

    def predict(ex):
        with torch.no_grad():
            t = tensors(ex)
            return model(t[0], t[1]).cpu().numpy()

    predict.model = model  # for ml/scripts/export_demos.py
    return predict


def evaluate(ex, horizon, n_seeds, dev, extras):
    ex = ex.assign(part=work_share.split(ex, horizon))
    parts = {p: ex[ex["part"] == p].drop(columns="part").reset_index(drop=True) for p in ("train", "val", "test")}
    if any(len(g) < 50 for g in parts.values()):
        raise SystemExit(f"work_share: too few decision points {({p: len(g) for p, g in parts.items()})}")
    test, y = parts["test"], parts["test"]["y"].to_numpy()
    user_mean = test["device_id"].map(parts["train"].groupby("device_id")["y"].mean()).fillna(parts["train"]["y"].mean())
    preds = {"user mean": user_mean.to_numpy(), "like the last H min": test["past"].fillna(user_mean).to_numpy()}
    sets = {"base": [c for c in parts["train"].columns if not c.startswith("x_")]}
    if extras:
        sets["+extras"] = list(parts["train"].columns)
    gru_runs = {}
    keep = base_seq_columns(extras)
    for name, cols in sets.items():
        sub = {p: g[cols] if name == "+extras" else g[cols].assign(seq=g["seq"].map(lambda s: s[:, keep]))
               for p, g in parts.items()}
        preds[f"lightgbm ({name})"] = fit_lgb(sub)(sub["test"])
        runs = [fit_gru(sub, int(ex["device_id"].max()) + 1, dev, SEED + s)(sub["test"]) for s in range(n_seeds)]
        preds[f"gru ({name})"] = runs[0]
        gru_runs[name] = runs
    err = {k: np.abs(p - y) * horizon for k, p in preds.items()}
    res = {"points": {p: len(g) for p, g in parts.items()}, "users": int(ex["device_id"].nunique()),
           "y_std_minutes": float(np.std(y) * horizon), "mae_minutes": {}, "vs_user_mean_ci95": {}}
    for k, e in err.items():
        res["mae_minutes"][k] = float(e.mean())
        if k != "user mean":  # negative = this model errs less than the user's mean
            ci = block_ci(e - err["user mean"], test["session"])
            res["vs_user_mean_ci95"][k] = ci
    res["gru_seed_spread"] = {name: [float(np.abs(r - y).mean() * horizon) for r in runs]
                              for name, runs in gru_runs.items()}
    return res


def base_seq_columns(extras):
    """Indices of a sequence row's DevDesk-recordable columns (every seq column but the x_* extras)."""
    return [i for i, n in enumerate(work_share.seq_names(extras)) if not n.startswith("x_")]


def main():
    ap = argparse.ArgumentParser()
    devdesk_data.add_source_args(ap, ROOT / "data" / "raw" / "behacom" / "behacom_slim.csv")
    ap.add_argument("--seeds", type=int, default=3, help="GRU trainings per setting (seed 42 first)")
    a = ap.parse_args()
    if a.seeds < 1:
        ap.error("--seeds must be at least 1")
    df, tz, tag = devdesk_data.load_source(ap, a, behacom_extras=work_share.EXTRAS)
    extras = tag == "behacom"  # the extras exist only in BEHACOM
    df["device_id"] = pd.factorize(df["device_id"], sort=True)[0]
    d = work_share.minutes(df)

    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for h in work_share.HORIZONS:
        ex = work_share.examples(d, h, tz, extras)
        results[str(h)] = evaluate(ex, h, a.seeds, dev, extras)
        r = results[str(h)]
        print(f"\nH = {h} min: points {r['points']}, y std {r['y_std_minutes']:.2f} min", flush=True)
        for k, v in r["mae_minutes"].items():
            ci = r["vs_user_mean_ci95"].get(k)
            print(f"  {k:28s} MAE {v:.3f} min" + (f"   vs user mean {ci[0]:+.3f} .. {ci[1]:+.3f}" if ci else ""))
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    (out / f"work_share_model_{tag}_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")


if __name__ == "__main__":
    main()
