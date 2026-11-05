"""Retrain the three activity models on BEHACOM (seed 42) and export them for ml/streamlit_app.py's checks.

Usage: uv run --group ml --group wake python -m ml.scripts.export_demos --data <behacom_slim.csv>
Writes ml/models/demo_next_app.onnx, demo_day.onnx, demo_focus.onnx + demos.json: each model's sha256, its test
numbers from this run, and what its page needs (app names, one held-out day per user, each user's mean). No raw rows
beyond those days. BEHACOM only: the files are committed, and a DevDesk export is the user's own data. Each graph is
checked against the torch model and refused if the outputs differ; a run that fails leaves the old files as they are.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml import behacom, day_anomaly, next_app, train_day_anomaly, train_next_app, train_work_share, work_share

SEED = 42
HORIZON = 10  # minutes; the report's headline forecast
TOLERANCE = 1e-4
OUT = train_next_app.ROOT / "models"


class NextGraph(torch.nn.Module):
    """NextAppGRU for one unpadded history: no packing, the same weights."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, apps, num, users):
        _, h = self.model.gru(torch.cat([self.model.app(apps), num], dim=2))
        return torch.softmax(self.model.out(torch.cat([h[-1], self.model.user(users)], dim=1)), dim=1)


def export(graph, examples, name, inputs, axes, want):
    """Write <name>.onnx.tmp if onnxruntime matches want(example) on every example; its sha256. main moves it."""
    import onnxruntime as ort

    graph = graph.cpu().eval()
    tmp = OUT / f"{name}.onnx.tmp"
    torch.onnx.export(graph, examples[0], str(tmp), input_names=inputs, output_names=["out"], opset_version=17,
                      dynamic_axes=axes, dynamo=False)
    session = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    for ex in examples:
        with torch.no_grad():
            expected = want(*ex).numpy()
        got = session.run(None, {k: v.numpy() for k, v in zip(inputs, ex, strict=True)})[0]
        if not np.abs(expected - got).max() <= TOLERANCE:  # `not <=`: NaN is refused too
            tmp.unlink()
            raise SystemExit(f"{name}: ONNX differs from torch by {np.abs(expected - got).max():.1e}; not written")
    print(f"checked {name}.onnx ({len(examples)} checks)", flush=True)
    return hashlib.sha256(tmp.read_bytes()).hexdigest()


def next_app_demo(df, tz, dev):
    v = next_app.visits(df, tz)
    voc = next_app.vocab(v)
    ex = next_app.examples(v, voc)
    parts = {p: ex[ex["part"] == p].reset_index(drop=True) for p in ("train", "val", "test")}
    n_apps, n_users = len(voc) + 1, int(ex["device_id"].max()) + 1
    model, prob, _ = train_next_app.train_gru(parts, n_apps, n_users, 60, dev)
    test, markov = parts["test"], next_app.Markov().fit(parts["train"])
    fallback = [a for a, _ in markov.all.most_common()]
    ranked = {"gru": next_app.mask_current(prob(test), test["cur"].to_numpy()),
              "back": next_app.rank_back(test, fallback)}
    scores = {k: train_next_app.scores(r, test) for k, r in ranked.items()}

    model = model.cpu().eval()
    # every user the page offers, and one-visit chains, which the page sends when only "지금 앱" is picked
    rows = pd.concat([test.groupby("device_id").head(3), test[test["hist"].map(len) == 1].groupby("device_id").head(1)])
    examples = []
    for hist, m, hr, u in zip(rows["hist"], rows["hist_minutes"], rows["hist_hour"], rows["device_id"], strict=True):
        num = next_app.visit_features(m, hr)
        examples.append((torch.tensor([hist]), torch.tensor(num[None]), torch.tensor([int(u)])))

    def want(apps, num, users):
        return torch.softmax(model(apps, num, torch.tensor([apps.shape[1]]), users), dim=1)

    digest = export(NextGraph(model), examples, "demo_next_app", ["apps", "num", "users"],
                    {"apps": {1: "visits"}, "num": {1: "visits"}}, want)
    return {"sha256": digest, "apps": ["(기타)", *voc], "fallback": [int(a) for a in fallback], "users": sorted(int(u) for u in parts["train"]["device_id"].unique()),
            "gru": {k: scores["gru"][k] for k in ("top1", "top3")}, "back": {k: scores["back"][k] for k in ("top1", "top3")}}


def day_demo(df, tz, dev):
    p = day_anomaly.profiles(df, tz)
    train, test = p[p["part"] == "train"], p[p["part"] == "test"]
    n_users = int(p["device_id"].max()) + 1
    score = train_day_anomaly.train_ae(train, n_users, SEED, 500, dev)
    sets = train_day_anomaly.eval_sets(test, np.random.default_rng(SEED))
    auc = train_day_anomaly.aucs(score, sets)
    X, U = sets["normal"]
    errors = score(X, U)
    days = {}  # one held-out day per user: that user's median error, neither best nor worst
    for u in np.unique(U):
        idx = np.flatnonzero(U == u)
        days[int(u)] = np.round(X[idx[np.argsort(errors[idx])[(len(idx) - 1) // 2]]], 4).tolist()

    # the verdict's limit per user ("unlike THIS user"), from days the test never sees: the same last train days that
    # stopped the training (at least the last one); test only measures
    by_user = train.groupby("device_id")
    late = train[(by_user.cumcount() / by_user["day"].transform("size") >= 0.85) | (by_user.cumcount(ascending=False) == 0)]
    limits = {int(u): float(np.percentile(score(np.stack(g["x"]), g["device_id"].to_numpy()), 95))
              for u, g in late.groupby("device_id")}

    model = score.model.cpu().eval()
    examples = [(torch.tensor(X[i:i + 1], dtype=torch.float32), torch.tensor(U[i:i + 1])) for i in range(len(X))]
    digest = export(model, examples, "demo_day", ["x", "user"], None, lambda x, u: model(x, u))
    return {"sha256": digest, "days": days, "auc": auc, "normal_p95": limits}


def focus_demo(df, tz, dev):
    ex = work_share.examples(work_share.minutes(df), HORIZON, tz, extras=False)
    ex = ex.assign(part=work_share.split(ex, HORIZON))
    parts = {p: ex[ex["part"] == p].drop(columns="part").reset_index(drop=True) for p in ("train", "val", "test")}
    n_users = int(ex["device_id"].max()) + 1
    predict = train_work_share.fit_gru(parts, n_users, dev, SEED)
    test, y = parts["test"], parts["test"]["y"].to_numpy()
    means = parts["train"].groupby("device_id")["y"].mean()
    baseline = test["device_id"].map(means).fillna(parts["train"]["y"].mean()).to_numpy()
    mae_gru = float(np.abs(predict(test) - y).mean() * HORIZON)  # before the model moves to the CPU for export

    model = predict.model.cpu().eval()
    sample = test.groupby("device_id").head(3)
    examples = [(torch.tensor(s[None]), torch.tensor([int(u)])) for s, u in
                zip(sample["seq"], sample["device_id"], strict=True)]
    digest = export(model, examples, "demo_focus", ["seq", "user"], None, lambda s, u: model(s, u))
    return {"sha256": digest, "horizon": HORIZON, "user_mean": {int(u): float(v) for u, v in means.items()},
            "mae_gru": mae_gru,
            "mae_user_mean": float(np.abs(baseline - y).mean() * HORIZON)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=train_next_app.ROOT / "data" / "raw" / "behacom" / "behacom_slim.csv")
    a = ap.parse_args()
    df, tz = behacom.load(a.data), behacom.TZ
    df["device_id"] = pd.factorize(df["device_id"], sort=True)[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)
    meta = {"seed": SEED, "next_app": next_app_demo(df, tz, dev), "day": day_demo(df, tz, dev),
            "focus": focus_demo(df, tz, dev)}
    tmp = OUT / "demos.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
    for key in ("next_app", "day", "focus"):  # only now: a failed run must not split the models from their hashes
        (OUT / f"demo_{key}.onnx.tmp").replace(OUT / f"demo_{key}.onnx")
    tmp.replace(OUT / "demos.json")
    print(json.dumps({"next_app": {k: meta["next_app"][k] for k in ("gru", "back")}, "day": meta["day"]["auc"],
                      "focus": {k: meta["focus"][k] for k in ("mae_gru", "mae_user_mean")}}, indent=2))


if __name__ == "__main__":
    main()
