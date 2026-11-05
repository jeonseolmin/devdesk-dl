"""Daily-pattern anomaly detection: does this day look like this user's usual days?

activity_samples-shaped minutes → one profile per user-day: 24 hours × (6 category shares + active share) = 168
values in [0, 1] (minutes / 60; an hour without rows is all zeros = PC off). Split per user in date order.
There are no anomaly labels in public data, so evaluation uses proxies (see `inject` and train_day_anomaly.py).
Pure pandas/numpy except `autoencoder` (torch, imported inside).
"""
import numpy as np
import pandas as pd

from ml.focus_features import ACTIVE_SEC, CATEGORIES, one_row_per_minute

CHANNELS = (*CATEGORIES, "active")
DIM = 24 * len(CHANNELS)
MIN_DAY_MINUTES = 60  # a day with less is not a usage pattern
MIN_USER_DAYS = 7  # a user with fewer days has no "usual"
TRAIN_SHARE = 0.7  # per user, first 70 % of days (date order) train, the rest test


def profiles(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Minutes (device_id, ts, app_category, idle_sec) → rows (device_id, day, part, x: 168 floats).

    No day or user left (DevDesk's first days): an empty frame with these columns."""
    df = one_row_per_minute(df)  # UTC minutes: flooring local time raises on the autumn DST hour (ambiguous)
    local = df["ts"].dt.tz_convert(tz)
    d = pd.DataFrame({"device_id": df["device_id"].to_numpy(), "day": local.dt.date.to_numpy(),
                      "hour": local.dt.hour.to_numpy()})
    for c in CATEGORIES:
        d[c] = (df["app_category"] == c).to_numpy()
    d["active"] = (df["idle_sec"] < ACTIVE_SEC).to_numpy()
    # clip: on the autumn DST day the local hour 2 happens twice (up to 120 minutes); a share stays a share
    hourly = (d.groupby(["device_id", "day", "hour"])[list(CHANNELS)].sum() / 60).clip(upper=1.0)
    minutes = d.groupby(["device_id", "day"]).size()
    rows = []
    for (u, day), g in hourly.groupby(level=[0, 1]):
        if minutes[(u, day)] < MIN_DAY_MINUTES:
            continue
        x = np.zeros((24, len(CHANNELS)))
        x[g.index.get_level_values(2)] = g.to_numpy()
        rows.append({"device_id": u, "day": day, "x": x.reshape(-1)})
    out = pd.DataFrame(rows, columns=["device_id", "day", "x"])
    n = out.groupby("device_id")["day"].transform("size")
    out = out[n >= MIN_USER_DAYS].sort_values(["device_id", "day"]).reset_index(drop=True)
    pos = out.groupby("device_id").cumcount() / out.groupby("device_id")["day"].transform("size")
    out["part"] = np.where(pos < TRAIN_SHARE, "train", "test")
    return out


def inject(x, kind, rng):
    """A synthetic anomaly from a normal day profile x (168,):
    'night_game'  3 straight hours of active gaming between 0 and 5 h (overwrites those hours) at the level of the
                  day's own busiest hour, not 1.0: a ~70 s logger fills at most ~0.86 of an hour, and a 1.0 would
                  be caught for its out-of-range size, not for being night gaming (code-review, 2026-10-31),
    'shift'       the whole day moved 6 hours later (wrapping: what happened at 20 h now shows at 2 h)."""
    x = x.reshape(24, len(CHANNELS)).copy()
    if kind == "night_game":
        level = x[:, :len(CATEGORIES)].sum(1).max() or 1.0  # minutes logged in the fullest hour, as a share
        start = rng.integers(0, 3)
        x[start : start + 3] = 0
        x[start : start + 3, CHANNELS.index("game")] = level
        x[start : start + 3, CHANNELS.index("active")] = level
    elif kind == "shift":
        x = np.roll(x, 6, axis=0)
    else:
        raise ValueError(kind)
    return x.reshape(-1)


def autoencoder(n_users, latent=8):
    """Conditional autoencoder: the user embedding goes into encoder and decoder, so the error measures
    "unlike THIS user's days", not "unlike anyone's"."""
    import torch
    from torch import nn

    class DayAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.user = nn.Embedding(n_users, 4)
            self.enc = nn.Sequential(nn.Linear(DIM + 4, 64), nn.ReLU(), nn.Linear(64, latent))
            self.dec = nn.Sequential(nn.Linear(latent + 4, 64), nn.ReLU(), nn.Linear(64, DIM), nn.Sigmoid())

        def forward(self, x, u):
            e = self.user(u)
            return self.dec(torch.cat([self.enc(torch.cat([x, e], 1)), e], 1))

        def error(self, x, u):
            return ((self(x, u) - x) ** 2).mean(1)

    return DayAE()
