"""Daily-pattern anomaly detection on BEHACOM: user-mean distance vs PCA vs conditional autoencoder.

Usage: uv run --group ml --group wake python -m ml.train_day_anomaly [--data ml/data/raw/behacom/behacom_slim.csv]
With --devdesk <your export CSV> (ml/scripts/export_devdesk.py) it runs on your own DevDesk rows instead.
Writes ml/models/day_anomaly_<behacom|devdesk>_metrics.json.

No public day has an "anomaly" label, so three proxies, each scored as AUC of normal test days vs:
  night_game  the same test day with 3 straight hours of gaming between 0 and 5 h
  shift       the same test day moved 6 hours later
  swapped     another user's test day, scored as if it were this user's (does it notice "not you"?)
AUC 0.5 = chance, 1.0 = every anomaly scores above every normal day.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from ml import day_anomaly, devdesk_data

ROOT = Path(__file__).resolve().parent
SEED = 42
KINDS = ("night_game", "shift")


def eval_sets(test, rng, swap=True):
    """{'normal'|kind|'swapped': (X, claimed user)}. swap=False: no 'swapped' (DevDesk: every PC is one person)."""
    X, U = np.stack(test["x"]), test["device_id"].to_numpy()
    sets = {"normal": (X, U)}
    for k in KINDS:
        sets[k] = (np.stack([day_anomaly.inject(x, k, rng) for x in X]), U)
    users = np.unique(U)
    if swap and len(users) > 1:  # one user (DevDesk on one PC): there is no "someone else's day" to swap in
        sets["swapped"] = (X, np.array([rng.choice(users[users != u]) for u in U]))
    return sets


def pca_dims(n_train):
    """At most 8 components, fewer than train days: a first DevDesk week has 5-7 train days and PCA(8) would raise."""
    return max(1, min(8, n_train - 1))


def aucs(score, sets):
    s0 = score(*sets["normal"])
    return {k: float(roc_auc_score(np.r_[np.zeros(len(s0)), np.ones(len(s))], np.r_[s0, s]))
            for k, s in ((k, score(*v)) for k, v in sets.items() if k != "normal")}


def train_ae(train, n_users, seed, epochs, dev):
    import torch

    torch.manual_seed(seed)
    X = torch.tensor(np.stack(train["x"]), dtype=torch.float32, device=dev)
    U = torch.tensor(train["device_id"].to_numpy(), device=dev)
    # early stopping on each user's last 15 % of train days (date order, like every other split here)
    by_user = train.groupby("device_id")
    val = (by_user.cumcount() / by_user["day"].transform("size") >= 0.85).to_numpy()
    if val.all() or not val.any():
        raise SystemExit(f"day_anomaly: {len(X)} train days cannot be split into fit / early-stopping days")
    val = torch.tensor(val, device=dev)  # a copy: pandas hands out a read-only array
    model = day_anomaly.autoencoder(n_users).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    best, best_state, bad = float("inf"), None, 0
    fit_idx = torch.nonzero(~val).squeeze(1)
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(fit_idx), device=dev).split(32):
            idx = fit_idx[b]
            loss = model.error(X[idx], U[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = model.error(X[val], U[val]).mean().item()
        if v < best - 1e-6:
            best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 20:
                break
    if best_state is None:  # the early-stopping error was NaN every epoch
        raise SystemExit("day_anomaly: the autoencoder's early-stopping error never became a number")
    model.load_state_dict(best_state)
    model.eval()

    def score(x, u):
        with torch.no_grad():
            return model.error(torch.tensor(x, dtype=torch.float32, device=dev),
                               torch.tensor(u, device=dev)).cpu().numpy()

    score.model = model  # for ml/scripts/export_demos.py
    return score


def main():
    ap = argparse.ArgumentParser()
    devdesk_data.add_source_args(ap, ROOT / "data" / "raw" / "behacom" / "behacom_slim.csv")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    if a.seeds < 1:
        ap.error("--seeds must be at least 1")

    df, tz, tag = devdesk_data.load_source(ap, a)
    df["device_id"] = pd.factorize(df["device_id"], sort=True)[0]  # 0..n-1 for embeddings; DevDesk's are uuids
    p = day_anomaly.profiles(df, tz)
    if p.empty:
        raise SystemExit(f"day_anomaly: no user has {day_anomaly.MIN_USER_DAYS}+ days of "
                         f"{day_anomaly.MIN_DAY_MINUTES}+ minutes yet")
    train, test = p[p["part"] == "train"], p[p["part"] == "test"]
    print(f"{len(p)} user-days ({len(train)} train / {len(test)} test), {p['device_id'].nunique()} users", flush=True)
    # devdesk: every row is yours (RLS); two PCs are one person, so "someone else's day" is not tested
    sets = eval_sets(test, np.random.default_rng(SEED), swap=tag != "devdesk")
    Xtr = np.stack(train["x"])
    results = {}

    means = {u: np.stack(g["x"]).mean(0) for u, g in train.groupby("device_id")}
    results["user mean distance"] = aucs(lambda x, u: np.array([((r - means[k]) ** 2).mean()
                                                                for r, k in zip(x, u, strict=True)]), sets)
    pca = PCA(pca_dims(len(Xtr)), random_state=SEED).fit(Xtr)
    results[f"pca ({pca.n_components_}, all users)"] = aucs(lambda x, u: ((pca.inverse_transform(pca.transform(x)) - x) ** 2).mean(1),
                                         sets)

    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_users = int(p["device_id"].max()) + 1
    runs = [aucs(train_ae(train, n_users, SEED + s, a.epochs, dev), sets) for s in range(a.seeds)]
    results["autoencoder (per user)"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    results["autoencoder (per user)"]["seed_spread"] = {
        k: [float(min(r[k] for r in runs)), float(max(r[k] for r in runs))] for k in runs[0]}

    metrics = {"user_days": {"train": len(train), "test": len(test)}, "users": int(p["device_id"].nunique()),
               "auc": results}
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    # tag (behacom / devdesk) in the name: one source never overwrites the other's results
    (out / f"day_anomaly_{tag}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, r in results.items():
        print(f"{name:24s} " + "  ".join(f"{k} {r[k]:.3f}" for k in sets if k in r))


if __name__ == "__main__":
    main()
