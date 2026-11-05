"""Next-app prediction: from the apps a user just visited, which app comes next (inside one session).

activity_samples-shaped minutes (ml/behacom.py today, real DevDesk rows later) → visits (a run of minutes on one
exe) → one example per visit whose successor is in the same session. Splits are per user and in time order
(first 70 % of a user's visits train, next 15 % val, last 15 % test): the app would learn from your own past.
Pure pandas/numpy here; the GRU lives below the baselines and needs torch (the `wake` dependency group).
"""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from devdesk.categories import exe_stem  # the same app name the category map uses
from ml.focus_features import SESSION_GAP_SEC, one_row_per_minute

HISTORY = 20  # visits the sequence model sees
TOP_APPS = 30  # vocabulary = the most visited train apps; the rest is OTHER
OTHER = 0
SPLITS = (0.70, 0.85)  # per-user cut points in time order: train | val | test
_NONE = Counter()  # read-only stand-in for an unseen key


def visits(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Minutes (device_id, ts, exe_name) → one row per visit, in time order per user. `tz` = the PC's zone.

    Columns: device_id, session, app, minutes (first to last minute of the visit, inclusive: elapsed time, so a
    ~70 s logger and DevDesk's one row per minute read the same), hour (local, at the first minute), part.
    Sessions split on the raw ts gaps; then rows in one device-minute collapse to the last (BEHACOM logs duplicates;
    without this, code / chrome / code at one ts would be three visits and a fake A → B → A).
    """
    # no exe (DevDesk writes None for the desktop or a protected process): not an app, so code / desktop / code
    # stays one code visit instead of a fake A → B → A; the minute still counts toward the session gap below
    name = df["exe_name"]
    keep = name.notna() & (name.astype(str).str.strip() != "")
    df = df.sort_values(["device_id", "ts"], kind="stable")
    gap = df.groupby("device_id")["ts"].diff().dt.total_seconds()
    df = df.assign(_keep=keep.reindex(df.index).to_numpy())
    df = one_row_per_minute(df.assign(session=np.cumsum((gap.isna() | (gap >= SESSION_GAP_SEC)).to_numpy())))
    df = df[df["_keep"]]
    if df.empty:  # DevDesk's first minutes: nothing to learn from yet, not an error
        return pd.DataFrame(columns=["device_id", "session", "app", "hour", "minutes", "part"])
    app = df["exe_name"].map(exe_stem).to_numpy()
    session = df["session"].to_numpy()
    new_visit = np.r_[True, (session[1:] != session[:-1]) | (app[1:] != app[:-1])]
    v = pd.DataFrame({
        "device_id": df["device_id"].to_numpy(),
        "session": session,
        "visit": np.cumsum(new_visit),
        "app": app,
        "minute": df["minute"].to_numpy(),
        "hour": df["minute"].dt.tz_convert(tz).dt.hour.to_numpy(),
    })
    out = v.groupby("visit", sort=True).agg(
        device_id=("device_id", "first"), session=("session", "first"), app=("app", "first"),
        hour=("hour", "first"), first=("minute", "first"), last=("minute", "last"),
    ).reset_index(drop=True)
    out["minutes"] = ((out["last"] - out["first"]).dt.total_seconds() // 60 + 1).astype(int)
    out = out.drop(columns=["first", "last"])
    pos = out.groupby("device_id").cumcount() / out.groupby("device_id")["app"].transform("size")
    out["part"] = np.select([pos < SPLITS[0], pos < SPLITS[1]], ["train", "val"], "test")
    return out


def vocab(v: pd.DataFrame, k=TOP_APPS) -> dict:
    """app → id 1..k from train visits (0 = OTHER)."""
    top = v.loc[v["part"] == "train", "app"].value_counts().head(k).index
    return {a: i + 1 for i, a in enumerate(top)}


def examples(v: pd.DataFrame, voc: dict) -> pd.DataFrame:
    """One row per visit with a next visit in the same session and the same part: ids, history, target, part.

    Same part: a train example whose answer is a val visit would teach the models val data (the history may
    still reach back across the cut; that is past, not answer).

    hist / hist_minutes / hist_hour = the last HISTORY visits of this session up to and including this one
    (oldest first, unpadded).
    """
    v = v.reset_index(drop=True)  # positions below; a filtered / sampled frame keeps its old labels
    ids = v["app"].map(voc).fillna(OTHER).astype(int).to_numpy()
    user, minutes, hour, part = (v[c].to_numpy() for c in ("device_id", "minutes", "hour", "part"))
    rows = []
    for sess, idx in v.groupby("session", sort=False).indices.items():
        for j in range(len(idx) - 1):
            i, h = idx[j], idx[max(0, j - HISTORY + 1) : j + 1]
            if part[idx[j + 1]] != part[i]:
                continue
            rows.append({
                "row": i, "session": sess, "device_id": user[i], "part": part[i],
                "cur": ids[i], "prev": ids[idx[j - 1]] if j else -1, "target": ids[idx[j + 1]],
                "hist": ids[h].tolist(), "hist_minutes": minutes[h].tolist(), "hist_hour": hour[h].tolist(),
                "minutes": minutes[i], "hour": hour[i],
            })
    cols = ["row", "session", "device_id", "part", "cur", "prev", "target", "hist", "hist_minutes", "hist_hour", "minutes",
            "hour"]
    return pd.DataFrame(rows, columns=cols)  # no transitions: an empty frame with the columns, not a KeyError


def topk_hits(ranked, target, k):
    """Share of rows whose target is among the first k ranked ids."""
    return float(np.mean([t in r[:k] for r, t in zip(ranked, target, strict=True)])) if len(target) else float("nan")


def _not_current(c):
    """Ids a ranking may not offer after current id c: c itself (a visit ends on a switch), except OTHER
    (two different rare apps both map to OTHER, so OTHER → OTHER happens)."""
    return set() if c == OTHER else {c}


def mask_current(prob, cur):
    """prob (n, classes) → rankings without each row's current app (see _not_current), like the baselines."""
    order = np.argsort(-np.asarray(prob, dtype=np.float64), axis=1, kind="stable")
    return [[a for a in r if a not in _not_current(c)] for r, c in zip(order.tolist(), cur, strict=True)]


def visit_features(minutes, hours):
    """One history's numbers per visit, as the GRU trains on them: log minutes, then the hour as sin and cos."""
    angle = np.asarray(hours, np.float32) * (2 * np.pi / 24)
    return np.stack([np.log1p(np.asarray(minutes, np.float32)), np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)


def rank_back(ex: pd.DataFrame, fallback):
    """Baseline 'go back': the app before this one first (A → B → A), then the fallback ranking."""
    out = []
    for p, c in zip(ex["prev"], ex["cur"], strict=True):
        skip = _not_current(c)
        first = [p] if p >= 0 and p not in skip else []
        out.append(first + [a for a in fallback if a not in skip and a not in first])
    return out


class Markov:
    """Baseline: per user, the most frequent next app after the current one in train; backs off to all users,
    then to the user's overall app counts. Never offers the current app (see _not_current)."""

    def fit(self, train: pd.DataFrame):
        self.user, self.glob = defaultdict(Counter), defaultdict(Counter)
        self.freq = defaultdict(Counter)
        for u, c, t in zip(train["device_id"], train["cur"], train["target"], strict=True):
            self.user[(u, c)][t] += 1
            self.glob[c][t] += 1
            self.freq[u][t] += 1
        self.all = Counter(train["target"])
        return self

    def rank(self, ex: pd.DataFrame):
        out = []
        for u, c in zip(ex["device_id"], ex["cur"], strict=True):
            seen, r = _not_current(c), []
            # .get: ranking must not add empty counters to the fitted model
            for counts in (self.user.get((u, c), _NONE), self.glob.get(c, _NONE), self.freq.get(u, _NONE), self.all):
                for a, _ in counts.most_common():
                    if a not in seen:
                        seen.add(a)
                        r.append(a)
            out.append(r)
        return out


def gru_model(n_apps, n_users):
    """GRU over the session's last HISTORY visits (app embedding + log minutes + hour) + a user embedding."""
    import torch
    from torch import nn

    class NextAppGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.app = nn.Embedding(n_apps + 1, 16, padding_idx=n_apps)  # last id = padding
            self.user = nn.Embedding(n_users, 4)
            self.gru = nn.GRU(16 + 3, 64, batch_first=True)
            self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 + 4, n_apps))

        def forward(self, apps, num, lengths, users):
            x = torch.cat([self.app(apps), num], dim=2)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            return self.out(torch.cat([h[-1], self.user(users)], dim=1))

    return NextAppGRU()
