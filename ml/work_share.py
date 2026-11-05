"""Can "focus drop" be predicted? Measured with a widened, threshold-free target (2026-11-02).

The 10-31 attempt (IDE/terminal + input = focus, "any non-focus minute in the next 10" = drop) gave 94 % drops on
BEHACOM: a 10-second lookup counted as a drop. Here:
  work minute   = an IDE/terminal minute or a document app (DOCS, kept here, not in devdesk.categories) with input
                  in the last minute (ACTIVE_SEC). Browsers are never work: without URLs (never collected) a
                  browser minute can be study or play.
  target        = at a work minute, the share of the next H minutes' observed rows that are work (no "drop" cut-off
                  to pick); needs >= MIN_SEEN observed rows (BEHACOM logs every ~70 s, so a missing minute is no data)
  models        = scored in ml/train_work_share.py (baselines: the user's mean share, "like the last H minutes")
  away runs     = leave work → next work minute in the same session, wall minutes; a two-group split ("a quick
                  lookup" vs "a break") would show as a valley here
Usage: uv run --group ml python -m ml.work_share [--devdesk <csv> --tz Asia/Seoul]; writes only ml/models/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from devdesk.categories import exe_stem
from ml import devdesk_data
from ml.focus_features import ACTIVE_SEC, SESSION_GAP_SEC, one_row_per_minute

ROOT = Path(__file__).resolve().parent
# Document, PDF, office and design apps seen in BEHACOM's "other" (2026-11-02 listing); file managers, players,
# calculators and system tools stay out (work or not cannot be told).
DOCS = frozenset({"soffice.bin", "winword", "excel", "powerpnt", "wps", "wpspdf", "acrobat", "acrord32", "evince",
                  "okular", "mendeleydesktop", "mendeleydesktop.x86_64", "acad", "inkscape", "photoshop", "kdenlive"})
HORIZONS = (5, 10, 15)  # minutes; 10 is the plan's, 5 and 15 show whether the conclusion depends on it
CATS = ("ide", "terminal", "browser", "chat", "game", "other")
WINDOWS = (5, 15)  # minutes of history the window features summarise
SEQ = 30  # rows the sequence model sees
# BEHACOM-only columns (DevDesk records none): network volume (video vs reading in a browser, without URLs),
# typing / scrolling, the app's CPU. Experiment only: DevDesk never adds keyboard hooks.
EXTRAS = ("received_bytes", "sent_bytes", "word_counter", "erase_keys_counter", "mouse_action_counter_3",
          "current_app_foreground_time", "current_app_average_cpu")
MIN_SEEN = 0.5  # share of the horizon's minutes that must have a row
AWAY_EDGES = (1, 2, 3, 5, 10, 20, 60)  # histogram bins (minutes); the last bin is open


def minutes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per device-minute in time order with `work`, `work_narrow` (IDE/terminal only), `session` and
    `m` (integer minute)."""
    df = one_row_per_minute(df).sort_values(["device_id", "minute"], kind="stable").reset_index(drop=True)
    stem = df["exe_name"].map(lambda e: exe_stem(e) if isinstance(e, str) else "")
    active = df["idle_sec"] < ACTIVE_SEC
    code = df["app_category"].isin(["ide", "terminal"])
    gap = df.groupby("device_id")["minute"].diff().dt.total_seconds()
    return df.assign(
        work=(code | stem.isin(DOCS)) & active,
        work_narrow=code & active,
        session=(gap.isna() | (gap >= SESSION_GAP_SEC)).cumsum(),
        # by Timedelta, not astype("int64"): pandas 3 may store datetimes in µs, not ns
        m=(df["minute"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1),
    )


def away_runs(df: pd.DataFrame):
    """(lengths in wall minutes of away runs that came back to work, runs that ended the session away)."""
    lengths, no_return = [], 0
    for _, s in df.groupby("session"):
        w, m = s["work"].to_numpy(), s["m"].to_numpy()
        for i in np.flatnonzero(w[:-1] & ~w[1:]) + 1:
            back = np.flatnonzero(w[i:])
            if len(back):
                lengths.append(int(m[i + back[0]] - m[i - 1] - 1))
            else:
                no_return += 1
    return np.array(lengths, dtype=int), no_return


def row_inputs(d: pd.DataFrame, extras: bool) -> np.ndarray:
    """Per-row numbers the features summarise: work, active, log switches, one-hot category [, extras]."""
    cols = [d["work"].to_numpy(float), (d["idle_sec"] < ACTIVE_SEC).to_numpy(float),
            np.log1p(d["switch_count"].to_numpy(float)), *(d["app_category"].eq(c).to_numpy(float) for c in CATS)]
    if extras:
        e = {c: np.nan_to_num(d[c].to_numpy(float)) for c in EXTRAS}
        cols += [np.log1p(np.clip(e[c], 0, None)) for c in EXTRAS[:5]]
        # foreground time: the readme says <= 60 s per window; 0.03 % of rows hold up to 857,627 (bad records,
        # measured 2026-11-02): clipped. CPU: 0.6 % of rows exceed 100 (several cores, up to 334 %): kept
        cols += [np.clip(e["current_app_foreground_time"], 0, 60) / 60, np.clip(e["current_app_average_cpu"], 0, None) / 100]
    return np.column_stack(cols)


def seq_names(extras: bool):
    """Columns of one `seq` row in examples(): the row inputs, then present (1 = a real row) and gap."""
    return row_names(extras) + ["present", "gap"]


def row_names(extras: bool):
    base = ["work", "active", "switches", *(f"cat_{c}" for c in CATS)]
    return base + ([f"x_{c}" for c in EXTRAS] if extras else [])


def examples(d: pd.DataFrame, horizon: int, tz: str, extras: bool) -> pd.DataFrame:
    """Decision points (work minutes with enough rows ahead) with the target y, window features, the history
    sequence (`seq`, SEQ rows ending at the point, oldest first, zero-padded) and its session."""
    names = row_names(extras)
    out = []
    for dev, g in d.groupby("device_id", sort=True):
        m, w, sess = g["m"].to_numpy(), g["work"].to_numpy(), g["session"].to_numpy()
        x = row_inputs(g, extras)
        cs = np.vstack([np.zeros(x.shape[1]), np.cumsum(x, axis=0)])
        ahead = np.searchsorted(m, m + horizon, side="right")
        first = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1]  # session start rows
        sess_start = first[np.searchsorted(first, np.arange(len(m)), side="right") - 1]
        last_off = np.maximum.accumulate(np.where(~w, np.arange(len(m)), -1))
        local = g["minute"].dt.tz_convert(tz)
        angle = (2 * np.pi * (local.dt.hour + local.dt.minute / 60) / 24).to_numpy()
        gap = np.r_[0, np.diff(m)].astype(float)
        for i in np.flatnonzero(w):
            seen = w[i + 1 : ahead[i]]
            if len(seen) < MIN_SEEN * horizon:
                continue
            row = {"device_id": dev, "m": m[i], "session": f"{dev}:{sess[i]}", "y": seen.mean()}
            lo = np.searchsorted(m, m[i] - horizon, side="left")
            row["past"] = w[lo:i].mean() if i > lo else np.nan  # "like the last `horizon` minutes"
            for win in WINDOWS:
                lo = np.searchsorted(m, m[i] - win + 1, side="left")
                mean = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
                row.update({f"{n}_{win}m": v for n, v in zip(names, mean, strict=True)})
            start = max(last_off[i] + 1, sess_start[i])
            row["streak_min"] = float(m[i] - m[start] + 1)
            row["session_min"] = float(m[i] - m[sess_start[i]])
            row["hour_sin"], row["hour_cos"] = np.sin(angle[i]), np.cos(angle[i])
            lo = max(0, i + 1 - SEQ)
            seq = np.zeros((SEQ, len(seq_names(extras))), np.float32)  # row inputs + present flag + gap
            seq[SEQ - (i + 1 - lo):, :-2] = x[lo : i + 1]
            seq[SEQ - (i + 1 - lo):, -2] = 1
            seq[SEQ - (i + 1 - lo):, -1] = np.clip(gap[lo : i + 1], 0, 10) / 10
            row["seq"] = seq
            out.append(row)
    # no decision point (a short --devdesk export): still the columns callers read
    return pd.DataFrame(out) if out else pd.DataFrame(columns=["device_id", "m", "session", "y", "past", "seq"])


def split(ex: pd.DataFrame, horizon: int, cuts=(0.70, 0.85)) -> pd.Series:
    """Per user in time order: train | val | test. A point whose target window reaches into a later part is
    dropped (NaN): its label would be read from a later part. Returned in the caller's row order.
    Conservative (>=): a window ending exactly on a later part's first point (that point's own, feature, minute)
    is dropped too, one point per boundary."""
    order = ex.index
    ex = ex.sort_values(["device_id", "m"], kind="stable")
    rank = ex.groupby("device_id").cumcount() / ex.groupby("device_id")["m"].transform("size")
    part = pd.Series(np.where(rank < cuts[0], "train", np.where(rank < cuts[1], "val", "test")), index=ex.index)
    later = {"train": ("val", "test"), "val": ("test",)}  # every later part: a user may have no val points
    for prev, nxts in later.items():
        start = ex["m"].where(part.isin(nxts)).groupby(ex["device_id"]).transform("min")
        part[(part == prev) & (ex["m"] + horizon >= start)] = None
    return part.reindex(order)


def measure(df: pd.DataFrame, tz: str) -> dict:
    """The data side only: work minutes, away runs, and the target's spread per H (examples(), the same points
    ml/train_work_share.py trains on; its baselines and models are scored there, on the purged split)."""
    d = minutes(df)
    lengths, no_return = away_runs(d)
    counts = np.histogram(lengths, bins=[*AWAY_EDGES, 10**9])[0]
    share = d["app_category"].value_counts(normalize=True)
    out = {
        "minutes": len(d),
        "users": int(d["device_id"].nunique()),
        "category_share": {k: round(float(v), 4) for k, v in share.items()},
        "work_minutes": {"narrow": int(d["work_narrow"].sum()), "widened": int(d["work"].sum())},
        "away": {"returned": len(lengths), "no_return": no_return,
                 "bins_min": [f"{lo}-{hi}" for lo, hi in zip(AWAY_EDGES, [*AWAY_EDGES[1:], "inf"], strict=True)],
                 "counts": [int(c) for c in counts]},
        "horizons": {},
    }
    for h in HORIZONS:
        t = examples(d, h, tz, extras=False)
        y = h * t["y"]
        stat = (lambda v: float(v)) if len(t) > 1 else (lambda v: None)  # null, not NaN: strict JSON readers
        out["horizons"][str(h)] = {
            "points": len(t), "users": int(t["device_id"].nunique()),
            "y_mean_minutes": stat(y.mean()), "y_std_minutes": stat(y.std()),
            "y_share_at_least_80pct": stat((t["y"] >= 0.8).mean()),
            "hist_minutes": [int(c) for c in np.histogram(y, bins=np.arange(h + 2) - 0.5)[0]],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    devdesk_data.add_source_args(ap, ROOT / "data" / "raw" / "behacom" / "behacom_slim.csv")
    a = ap.parse_args()
    df, tz, tag = devdesk_data.load_source(ap, a)
    result = measure(df, tz)
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    (out / f"work_share_{tag}_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
