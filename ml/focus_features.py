"""ML1 focus-state features and rule labels over `activity_samples`-shaped rows.

Input columns (one row per device-minute, same as the Supabase table):
    device_id, ts (tz-aware UTC), exe_name, app_category, idle_sec, switch_count
The same functions run on public data (via an adapter) and later on DevDesk's own samples,
so a model trained on one can be applied to the other.
"""
import numpy as np
import pandas as pd

# One category map for training and the app: the sampler fills app_category with the same function.
from devdesk.categories import _EXE_CATEGORY, CATEGORIES, categorize  # noqa: F401 (re-exported)

LABELS = ("focus", "distracted", "break")
ACTIVE_SEC = 60  # input within the last minute
BREAK_SEC = 600  # idle ≥10 min
# shortcut: idle > 1 h is "not at the PC" (unknown, not break) so break does not swamp the labels (BEHACOM: 96%);
# revisit when DevDesk's own samples show real break lengths.
UNATTENDED_SEC = 3600
# shortcut: DevDesk's idle_sec is the idle time at the end of the minute, so an active minute reads 0..59 while
# BEHACOM's active minutes are mostly 0; compare the distributions before applying a BEHACOM-trained model.


SESSION_GAP_SEC = 600  # >= 10 min between rows = a session boundary (PC off), not idle time; shared by all ML tasks


def one_row_per_minute(df: pd.DataFrame) -> pd.DataFrame:
    """Rows in the same device-minute (UTC) collapse to the last one in time order (BEHACOM logs duplicates and
    runs at ~70 s; DevDesk writes one row per minute). Adds `minute` = ts floored to the minute; keeps the order."""
    out = df.assign(minute=df["ts"].dt.floor("min"))
    out = out.sort_values(["device_id", "ts"], kind="stable")
    return out.drop_duplicates(["device_id", "minute"], keep="last")


def rule_labels(df: pd.DataFrame) -> pd.Series:
    """Plan's rule labels: idle ≥10 min = break, game = distracted, ide/terminal + input in the last minute = focus.

    Everything else, and idle > UNATTENDED_SEC, stays NaN (unknown), so it is never used as a training target.
    The rows themselves are kept: the features of the minutes after a long absence need that context.
    """
    label = pd.Series(np.nan, index=df.index, dtype=object)
    label[df["app_category"].isin(["ide", "terminal"]) & (df["idle_sec"] < ACTIVE_SEC)] = "focus"
    label[df["app_category"] == "game"] = "distracted"
    label[df["idle_sec"] >= BREAK_SEC] = "break"
    label[df["idle_sec"] > UNATTENDED_SEC] = np.nan
    return label


def build_features(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Per-row features from the previous 5 and 15 minutes only (the current minute is excluded).

    Excluding minute t means the rule label of minute t is never computed from its own inputs; the model
    predicts the state from recent context. `break` stays nearly predictable anyway (idle_sec is a running
    count, so prev_idle_sec ≥ 540 almost implies break), which is why reports also give focus/distracted F1.
    `sample_index` is the row's position in `df`, for joining labels. `tz` is the PC's local time zone.

    Window features are means, not sums, so a logger with a different cadence (BEHACOM: ~70 s, duplicates)
    yields the same scale as DevDesk's one row per minute. Rows in the same device-minute collapse to the last.
    """
    if df.empty:  # first minutes in the app: no history yet
        return pd.DataFrame(columns=["ts", *FEATURE_COLUMNS, "device_id", "sample_index"])
    df = df.reset_index(drop=True).rename_axis("sample_index").reset_index()
    df = one_row_per_minute(df)
    df["ts"] = df["minute"]
    out = []
    for _, g in df.groupby("device_id", sort=False):
        g = g.set_index("ts")
        cols = pd.DataFrame(index=g.index)
        for c in CATEGORIES:
            cols[f"is_{c}"] = (g["app_category"] == c).astype(float)
        cols["switch_count"] = g["switch_count"].astype(float)
        cols["idle_sec"] = g["idle_sec"].astype(float)
        cols["active"] = (g["idle_sec"] < ACTIVE_SEC).astype(float)

        f = pd.DataFrame(index=g.index)
        for w in (5, 15):
            mean = cols.rolling(f"{w}min", closed="left").mean()
            for c in CATEGORIES:
                f[f"ratio_{c}_{w}m"] = mean[f"is_{c}"]
            f[f"switch_mean_{w}m"] = mean["switch_count"]
            f[f"idle_mean_{w}m"] = mean["idle_sec"]
            f[f"active_ratio_{w}m"] = mean["active"]
        prev = cols.shift(1)
        recent = (g.index.to_series().diff() <= pd.Timedelta("2min")).to_numpy()
        f["prev_idle_sec"] = prev["idle_sec"].where(recent)
        for c in CATEGORIES:  # one-hot so linear models can use it too
            f[f"prev_{c}"] = prev[f"is_{c}"].where(recent)

        local = g.index.tz_convert(tz)
        hour = local.hour + local.minute / 60
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        f["weekday"] = local.day_of_week
        f["device_id"] = g["device_id"].to_numpy()
        f["sample_index"] = g["sample_index"].to_numpy()
        out.append(f.reset_index())
    return pd.concat(out, ignore_index=True)


FEATURE_COLUMNS = [
    *(f"ratio_{c}_{w}m" for w in (5, 15) for c in CATEGORIES),
    *(f"{k}_{w}m" for w in (5, 15) for k in ("switch_mean", "idle_mean", "active_ratio")),
    "prev_idle_sec", *(f"prev_{c}" for c in CATEGORIES), "hour_sin", "hour_cos", "weekday",
]
