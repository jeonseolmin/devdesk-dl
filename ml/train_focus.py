"""ML1: train and compare focus-state classifiers on rule-labelled activity samples.

Usage: uv run --group ml python -m ml.train_focus [--data ml/data/raw/behacom/behacom_slim.csv]
Writes ml/models/focus_lgbm.txt, focus_lgbm.json (features/classes) and focus_lgbm_metrics.json.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml import behacom
from ml.focus_features import CATEGORIES, FEATURE_COLUMNS, LABELS, build_features, rule_labels

ROOT = Path(__file__).resolve().parent
SEED = 42


def make_dataset(samples: pd.DataFrame, tz: str):
    feats = build_features(samples, tz)
    idx = feats["sample_index"].to_numpy(dtype=int)  # positional: index may repeat; dtype for the empty frame
    feats["label"] = rule_labels(samples).to_numpy()[idx]
    feats = feats[feats["label"].notna()].reset_index(drop=True)
    return feats[FEATURE_COLUMNS].astype(float), feats["label"].astype(str), feats["device_id"]


class Persistence:
    """Baseline: the state is whatever the rule said one minute ago (majority class if unknown)."""

    def fit(self, X, y):
        self.fallback_ = y.value_counts().idxmax()
        return self

    def predict(self, X):
        prev = X[[f"prev_{c}" for c in CATEGORIES]]
        category = prev.fillna(0).idxmax(axis=1).str.removeprefix("prev_").where(prev.notna().all(axis=1))
        rule = rule_labels(pd.DataFrame({"app_category": category, "idle_sec": X["prev_idle_sec"]}))
        return rule.fillna(self.fallback_).to_numpy()


def models():
    """Each learner with and without class weighting; "lightgbm" is the shipped model."""
    def logreg(cw):
        return lambda: make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight=cw))

    def forest(cw):
        return lambda: make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight=cw,
                                   n_jobs=-1, random_state=SEED))

    def boost(cw):
        # Small leaves + large min_child_samples: fewer user-specific splits, generalises to unseen users.
        return lambda: lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=100, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, class_weight=cw,
            random_state=SEED, verbose=-1)

    return {
        "majority": lambda: DummyClassifier(strategy="most_frequent"),
        "persistence": Persistence,
        "logreg": logreg(None), "logreg_balanced": logreg("balanced"),
        "random_forest": forest(None), "random_forest_balanced": forest("balanced"),
        "lightgbm": boost(None), "lightgbm_balanced": boost("balanced"),
    }


def evaluate(make, X, y, splits):
    pred = np.empty(len(y), dtype=object)
    fit_sec = 0.0
    for tr, te in splits:
        m = make()
        t0 = time.perf_counter()
        m.fit(X.iloc[tr], y.iloc[tr])
        fit_sec += time.perf_counter() - t0
        pred[te] = m.predict(X.iloc[te])
    per_class = f1_score(y, pred, labels=list(LABELS), average=None, zero_division=0)
    return {
        "macro_f1": round(float(f1_score(y, pred, labels=list(LABELS), average="macro", zero_division=0)), 4),
        # break is nearly given by prev_idle_sec; this is the part that needs a model
        "macro_f1_focus_distracted": round(float(f1_score(
            y, pred, labels=["focus", "distracted"], average="macro", zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "f1": {c: round(float(v), 4) for c, v in zip(LABELS, per_class, strict=True)},
        "fit_sec_total": round(fit_sec, 2),
    }


def deploy_cost(model, X, repeats=50):
    """Size and predict latency of a fitted model, measured the same way for every candidate.

    Size is the pickled model so learners compare on one format. Timing is single-threaded (the app predicts
    in-process) and after untimed warm-up calls; the model's n_jobs is restored afterwards. The app predicts
    one row per minute, so single_row_ms is the number that matters (it includes per-call overhead such as
    pandas checks); batch_ms_per_row is for reference. Both are medians.
    """
    jobs = {k: v for k, v in model.get_params().items() if k.endswith("n_jobs")}
    prev = X[[f"prev_{c}" for c in CATEGORIES]].notna().all(axis=1).to_numpy()
    batch, one = X.iloc[:1000], X.iloc[[int(prev.argmax())]]  # a row with previous-minute values, if any

    def median_ms(rows, n):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            model.predict(rows)
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))

    model.set_params(**{k: 1 for k in jobs})
    try:
        model.predict(batch)
        model.predict(one)
        batch_ms = median_ms(batch, max(repeats // 10, 1)) / len(batch)
        single_ms = median_ms(one, repeats)
    finally:
        model.set_params(**jobs)
    return {
        "pickle_kb": round(len(pickle.dumps(model)) / 1024, 1),
        "batch_ms_per_row": round(batch_ms, 4),
        "single_row_ms": round(single_ms, 3),
        "single_row_has_prev": bool(prev.any()),  # False: timed on a row without previous-minute values
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/behacom/behacom_slim.csv"))
    args = ap.parse_args()

    X, y, groups = make_dataset(behacom.load(args.data), behacom.TZ)
    print(f"labelled rows {len(y)}, devices {groups.nunique()}, labels {y.value_counts().to_dict()}")
    if groups.nunique() < 2:
        sys.exit("need labelled rows from at least 2 devices for leave-one-user-out")

    louo = list(GroupKFold(n_splits=groups.nunique()).split(X, y, groups))
    results = {}
    for name, make in models().items():
        results[name] = evaluate(make, X, y, louo)
        print(name, results[name])
    # Same model with a random split: rows of one user land in both train and test (leakage check).
    shuffled = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    results["lightgbm_random_split"] = evaluate(models()["lightgbm"], X, y, shuffled)
    print("lightgbm_random_split", results["lightgbm_random_split"])

    final = models()["lightgbm"]().fit(X, y)
    importance = pd.Series(final.booster_.feature_importance("gain"), index=FEATURE_COLUMNS)
    top = (importance / importance.sum()).sort_values(ascending=False).head(10).round(4).to_dict()

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    final.booster_.save_model(str(models_dir / "focus_lgbm.txt"))
    meta = {
        "features": FEATURE_COLUMNS, "classes": [str(c) for c in final.classes_],
        "trained_on": "BEHACOM rule labels", "rows": int(len(y)), "devices": int(groups.nunique()),
    }
    (models_dir / "focus_lgbm.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Deployment cost of the shipped model and the best other learner family, for the model choice in the
    # report. Measured after the model files are written, and a failure only marks the cost as missing, so
    # the shipped model and the cross-validation metrics are still written.
    others = [k for k in models() if k not in ("majority", "persistence") and not k.startswith("lightgbm")]
    rival = max(others, key=lambda k: results[k]["macro_f1"])
    try:
        cost = {"lightgbm": deploy_cost(final, X), rival: deploy_cost(models()[rival]().fit(X, y), X)}
    except Exception as e:  # reference-only measurement, e.g. MemoryError on the full-data forest
        print(f"deploy_cost failed: {e!r}")
        cost = {"error": repr(e)}

    report = {
        "dataset": {"rows": int(len(y)), "devices": int(groups.nunique()),
                    "labels": {k: int(v) for k, v in y.value_counts().items()}},
        "validation": "leave-one-user-out (GroupKFold by device_id), pooled out-of-fold predictions",
        "results": results,
        "lightgbm_top_features_gain": top,
        "model_file_kb": round((models_dir / "focus_lgbm.txt").stat().st_size / 1024, 1),
        "deploy_cost": cost,
    }
    (models_dir / "focus_lgbm_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("lightgbm_top_features_gain", "model_file_kb", "deploy_cost")}, indent=2))


if __name__ == "__main__":
    main()
