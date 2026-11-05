"""Next-app prediction on BEHACOM: baselines vs LightGBM vs GRU, same per-user time split.

Usage: uv run --group ml --group wake python -m ml.train_next_app [--data ml/data/raw/behacom/behacom_slim.csv]
With --devdesk <your export CSV> (ml/scripts/export_devdesk.py) it runs on your own DevDesk rows instead.
Writes ml/models/next_app_gru_<behacom|devdesk>.pt and next_app_<behacom|devdesk>_metrics.json.
See ml/next_app.py for the task and the split.
Every model gets the same rule: the current app is never offered as the next one (except OTHER).
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml import devdesk_data, next_app

ROOT = Path(__file__).resolve().parent
SEED = 42
MIN_CI_GROUPS = 10  # fewer test sessions than this: no bootstrap interval (measured 2026-11-02: 1 session, width 0)


def scores(ranked, ex):
    t = ex["target"].to_numpy()
    per_user = [next_app.topk_hits([ranked[i] for i in g], t[g], 1)
                for g in ex.groupby("device_id").indices.values()]
    return {"top1": next_app.topk_hits(ranked, t, 1), "top3": next_app.topk_hits(ranked, t, 3),
            "top1_user_macro": float(np.mean(per_user)), "n": len(t)}


def paired_ci(ranked_a, ranked_b, target, groups, k=1, n=2000):
    """95 % interval of top-k(a) - top-k(b) on the same test rows, resampling whole `groups` (sessions).

    Paired (one resample for both models) and by block: examples of one session follow each other and are not
    independent, so resampling single rows gave too narrow an interval (code-review 2026-10-31).
    """
    ha = np.array([t in r[:k] for r, t in zip(ranked_a, target, strict=True)], float)
    hb = np.array([t in r[:k] for r, t in zip(ranked_b, target, strict=True)], float)
    return block_ci(ha - hb, groups, n)


def block_ci(diff, groups, n=2000):
    """95 % interval of mean(diff) resampling whole `groups`; None below MIN_CI_GROUPS groups."""
    codes = pd.factorize(np.asarray(groups))[0]
    sums, counts = np.bincount(codes, weights=diff), np.bincount(codes).astype(float)
    rng, g = np.random.default_rng(SEED), len(sums)
    if g < MIN_CI_GROUPS:  # one test session resamples to itself: a zero-width "interval" that means nothing
        return None
    chunk = max(1, 2_000_000 // max(1, g))  # resamples per batch: ~16 MB of indices at any test size
    means = np.concatenate([
        sums[i].sum(1) / counts[i].sum(1) for i in (rng.integers(0, g, (min(chunk, n - s), g)) for s in range(0, n, chunk))
    ])
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def lgb_frame(ex):
    return pd.DataFrame({
        "user": ex["device_id"].astype("category"), "cur": ex["cur"].astype("category"),
        "prev": ex["prev"].astype("category"),
        "prev2": ex["hist"].map(lambda h: h[-3] if len(h) >= 3 else -1).astype("category"),
        "log_minutes": np.log1p(ex["minutes"]), "hour": ex["hour"],
    })


def gru_tensors(ex, n_apps):
    import torch

    n = len(ex)
    apps = torch.full((n, next_app.HISTORY), n_apps, dtype=torch.long)  # n_apps = padding id
    num = torch.zeros((n, next_app.HISTORY, 3))
    lengths = torch.tensor(ex["hist"].map(len).to_numpy())
    for i, (h, m, hr) in enumerate(zip(ex["hist"], ex["hist_minutes"], ex["hist_hour"], strict=True)):
        k = len(h)
        apps[i, :k] = torch.tensor(h)
        num[i, :k] = torch.from_numpy(next_app.visit_features(m, hr))
    users = torch.tensor(ex["device_id"].to_numpy(), dtype=torch.long)
    target = torch.tensor(ex["target"].to_numpy(), dtype=torch.long)
    return apps, num, lengths, users, target


def train_gru(parts, n_apps, n_users, epochs, dev, seed=SEED):
    import torch

    torch.manual_seed(seed)
    model = next_app.gru_model(n_apps, n_users).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    if not len(parts["val"]) or not len(parts["train"]):
        raise SystemExit(f"next_app: need train and val examples, got {len(parts['train'])} / {len(parts['val'])}")
    tr = [t.to(dev) for t in gru_tensors(parts["train"], n_apps)]
    va = [t.to(dev) for t in gru_tensors(parts["val"], n_apps)]
    best, best_state, bad = float("inf"), None, 0
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        for b in torch.randperm(len(tr[4]), generator=gen).split(256):
            b = b.to(dev)
            loss = loss_fn(model(tr[0][b], tr[1][b], tr[2][b], tr[3][b]), tr[4][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(*va[:4]), va[4]).item()
        print(f"gru epoch {ep + 1:2d}: val loss {val:.4f}", flush=True)
        if val < best - 1e-4:  # val loss is not saturated here, so it picks the epoch (early stopping)
            best, best_state, bad = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:
                break
    if best_state is None:  # val loss was NaN every epoch
        raise SystemExit("next_app: the GRU's val loss never became a number; check the val examples")
    model.load_state_dict(best_state)
    model.eval()

    def prob(ex):
        with torch.no_grad():
            t = [x.to(dev) for x in gru_tensors(ex, n_apps)]
            return torch.softmax(model(*t[:4]), dim=1).cpu().numpy()

    return model, prob, best


def main():
    ap = argparse.ArgumentParser()
    devdesk_data.add_source_args(ap, ROOT / "data" / "raw" / "behacom" / "behacom_slim.csv")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5, help="GRU trainings in total (seed 42 first) for the seed spread")
    a = ap.parse_args()
    if a.seeds < 1:
        ap.error("--seeds must be at least 1")

    df, tz, tag = devdesk_data.load_source(ap, a)
    # 0..n-1 for embeddings; DevDesk's are uuids. With --devdesk a "user" is a PC (one person may have two)
    df["device_id"] = pd.factorize(df["device_id"], sort=True)[0]
    v = next_app.visits(df, tz)
    voc = next_app.vocab(v)
    ex = next_app.examples(v, voc)
    parts = {p: ex[ex["part"] == p].reset_index(drop=True) for p in ("train", "val", "test")}
    if any(g.empty for g in parts.values()):
        raise SystemExit(f"next_app: too few app switches yet ({ {p: len(g) for p, g in parts.items()} })")
    n_apps, n_users = len(voc) + 1, int(ex["device_id"].max()) + 1
    other_share = {p: float((g["target"] == next_app.OTHER).mean()) for p, g in parts.items()}
    print({p: len(g) for p, g in parts.items()}, "examples; OTHER target share", other_share, flush=True)

    test = parts["test"]
    markov = next_app.Markov().fit(parts["train"])
    fallback = [a for a, _ in markov.all.most_common()]
    ranked = {"back (A→B→A)": next_app.rank_back(test, fallback), "markov (per user)": markov.rank(test)}

    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=10,
                             random_state=SEED, verbose=-1)
    clf.fit(lgb_frame(parts["train"]), parts["train"]["target"],
            eval_X=(lgb_frame(parts["val"]),), eval_y=(parts["val"]["target"],),
            callbacks=[lgb.early_stopping(30, verbose=False)])
    prob = np.zeros((len(test), n_apps))
    prob[:, clf.classes_] = clf.predict_proba(lgb_frame(test))
    ranked["lightgbm"] = next_app.mask_current(prob, test["cur"].to_numpy())

    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, gru_prob, val_loss = train_gru(parts, n_apps, n_users, a.epochs, dev)
    ranked["gru"] = next_app.mask_current(gru_prob(test), test["cur"].to_numpy())
    results = {k: scores(r, test) for k, r in ranked.items()}
    t = test["target"].to_numpy()
    results["gru"]["vs_ci95"] = {  # GRU minus each other model, same test rows
        name: {f"top{k}": paired_ci(ranked["gru"], r, t, test["session"], k) for k in (1, 3)}
        for name, r in ranked.items() if name != "gru"
    }
    # the CI above is for one GRU (seed SEED); how much does another seed move it? (the table reports the first)
    spread = [scores(next_app.mask_current(train_gru(parts, n_apps, n_users, a.epochs, dev, SEED + s)[1](test),
                                  test["cur"].to_numpy()), test) for s in range(1, a.seeds)]
    results["gru"]["seed_spread"] = {
        k: [min(r[k] for r in spread + [results["gru"]]), max(r[k] for r in spread + [results["gru"]])]
        for k in ("top1", "top3")}

    metrics = {
        "examples": {p: len(g) for p, g in parts.items()}, "other_target_share": other_share,
        "vocab": list(voc), "users": n_users, "gru_val_loss": val_loss,
        "gru_params": sum(p.numel() for p in model.parameters()), "test": results,
    }
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    # tag (behacom / devdesk) in the names: one source never overwrites the other's results
    torch.save(model.state_dict(), out / f"next_app_gru_{tag}.pt")
    (out / f"next_app_{tag}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, r in results.items():
        print(f"{k:20s} top1 {r['top1']:.3f}  top3 {r['top3']:.3f}  user-macro top1 {r['top1_user_macro']:.3f}")
    print(f"gru over {a.seeds} seeds: top1 {results['gru']['seed_spread']['top1']}, "
          f"top3 {results['gru']['seed_spread']['top3']}")
    def show(iv):
        return "n/a" if iv is None else f"[{iv[0]:+.3f}, {iv[1]:+.3f}]"

    sessions = test["session"].nunique()
    if sessions < MIN_CI_GROUPS:
        print(f"no 95% intervals: only {sessions} test sessions (need {MIN_CI_GROUPS}); the numbers above are anecdotes")
    for name, ci in results["gru"]["vs_ci95"].items():
        print(f"gru - {name:18s} top1 95% {show(ci['top1'])}  top3 {show(ci['top3'])}")


if __name__ == "__main__":
    main()
