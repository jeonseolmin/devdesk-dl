"""BEHACOM (behacom_slim.csv from ml/scripts/slim_behacom.py) → activity_samples-shaped rows."""
import numpy as np
import pandas as pd

from ml.focus_features import SESSION_GAP_SEC, categorize

TZ = "Europe/Madrid"  # University of Murcia, Spain
CONTIGUOUS_SEC = SESSION_GAP_SEC  # ≥10 min between rows = session boundary (PC off); see reports/datasets.md


def load(path, extras=()) -> pd.DataFrame:
    """extras: more BEHACOM columns kept as they are (e.g. received_bytes); DevDesk records none of them."""
    mouse = [f"mouse_action_counter_{i}" for i in range(7)]
    raw = pd.read_csv(  # same encoding as the writer, ml/scripts/slim_behacom.py
        path, encoding="utf-8", encoding_errors="surrogateescape", dtype={"current_app": object},
        usecols=["timestamp", "USER", "current_app", "changes_between_apps", "keystroke_counter", *mouse,
                 *(c for c in extras if c not in mouse)],
    )
    bad = raw["USER"].isna() | raw["timestamp"].isna()
    if bad.any():  # ml/data is untrusted input: drop unusable rows loudly instead of crashing on astype
        print(f"behacom: dropping {int(bad.sum())} rows without USER/timestamp")
        raw = raw[~bad]
    df = pd.DataFrame({
        "device_id": raw["USER"].astype(int),
        "ts": pd.to_datetime(raw["timestamp"], unit="ms", utc=True),
        # surrogates (app names that are not UTF-8, kept by the writer) become U+FFFD: pyarrow, when installed, refuses them
        "exe_name": raw["current_app"].map(lambda s: s.encode("utf-8", "surrogateescape").decode("utf-8", "replace"),
                                           na_action="ignore"),
        "switch_count": raw["changes_between_apps"].fillna(0).astype(int),
        **{c: pd.to_numeric(raw[c], errors="coerce").to_numpy() for c in extras},  # untrusted: junk → NaN
    })
    df["app_category"] = df["exe_name"].map(categorize)
    idle_minute = (raw["keystroke_counter"].fillna(0) == 0) & (raw[mouse].fillna(0).sum(axis=1) == 0)
    df = (df.assign(_idle=idle_minute.to_numpy())
          .sort_values(["device_id", "ts"], kind="stable").reset_index(drop=True))  # duplicate ts keep CSV order
    df["idle_sec"] = _idle_seconds(df)
    return df.drop(columns="_idle")


def _idle_seconds(df: pd.DataFrame) -> np.ndarray:
    """BEHACOM has no idle time; approximate DevDesk's idle_sec (seconds since last input at minute end).

    Active minute → 0, idle minute → previous idle_sec + elapsed time. After a logging gap the idle run
    restarts at 60 s: missing rows (PC asleep, logger paused) are unknown, not observed idleness.
    """
    idle = np.zeros(len(df))
    gap = df.groupby("device_id")["ts"].diff().dt.total_seconds().to_numpy()
    flags = df["_idle"].to_numpy()
    for i in range(len(df)):
        if not flags[i]:
            continue
        cont = i > 0 and not np.isnan(gap[i]) and gap[i] < CONTIGUOUS_SEC
        idle[i] = max(idle[i - 1] + gap[i], 60.0) if cont else 60.0  # a no-input minute is idle ≥ 60 s
    return idle
