"""DevDesk's own activity_samples → the minute frame the ML tasks take (same columns as ml/behacom.load).

Rows come from ml/scripts/export_devdesk.py (a CSV under ml/data/devdesk/). The sampler writes exe names and
categories only, never window titles. Columns out: device_id, ts (UTC), exe_name, app_category, idle_sec,
switch_count; sorted by device and time.
"""
import argparse
from pathlib import Path

import pandas as pd

from devdesk.categories import exe_stem

COLUMNS = ("device_id", "ts", "exe_name", "app_category", "idle_sec", "switch_count")
# Remote-desktop clients: a minute with one in front is spent on ANOTHER PC, which records it itself (the user
# streams the desktop PC to the laptop with Sunshine/Moonlight; Sunshine runs as a service, never in front).
# Kept, those minutes would be counted twice and look like hours of one "app". exe stems (devdesk.categories.exe_stem)
REMOTE_CLIENTS = frozenset({"moonlight", "mstsc", "msrdc", "rustdesk", "anydesk", "teamviewer", "parsecd"})
PAGE = 1000  # PostgREST's default max rows per request
TZ = "Asia/Seoul"  # the default PC time zone


def zone(name):
    """argparse type for --tz: a typo fails at the command line, not after a long export or training run."""
    from zoneinfo import ZoneInfo

    try:
        ZoneInfo(name)
    except (ValueError, KeyError) as e:  # ZoneInfoNotFoundError is a KeyError; "../x" is a ValueError
        raise argparse.ArgumentTypeError(f"unknown time zone {name!r} (e.g. {TZ}, Europe/Madrid)") from e
    return name


def to_frame(rows) -> pd.DataFrame:
    """API rows (dicts) or a CSV's frame → the minute frame. Rows missing a device or a time are dropped, loudly."""
    df = pd.DataFrame(rows if isinstance(rows, pd.DataFrame) else list(rows), columns=list(COLUMNS))
    bad = df["device_id"].isna() | df["ts"].isna()
    if bad.any():  # ml/data is untrusted input
        print(f"devdesk_data: dropping {int(bad.sum())} rows without device_id/ts")
        df = df[~bad]
    df = df.assign(
        ts=pd.to_datetime(df["ts"], utc=True, format="ISO8601"),  # a missing exe stays missing (next_app skips it)
        idle_sec=df["idle_sec"].fillna(0).astype(int),
        switch_count=df["switch_count"].fillna(0).astype(int),
    )
    return df.sort_values(["device_id", "ts"], kind="stable").reset_index(drop=True)


# = the table's unique index (device_id, ts), so the order needs no sort; the `or` keyset filter is not an index
# range condition though, so later pages filter more rows. shortcut: fine for weeks of minutes; page per device
# (device_id=eq.X & ts=gt.T) if exports grow to years
ORDER = ("device_id", "ts")


def after_filter(device_id, ts):
    """PostgREST `or` filter for rows strictly after (device_id, ts) in ORDER: the rest of this PC's minutes, then
    the next PCs. Values are double-quoted: timestamps carry ':' and '+', which PostgREST would read as syntax."""
    return f'device_id.gt."{device_id}",and(device_id.eq."{device_id}",ts.gt."{ts}")'


def fetch_all(client):
    """Every activity_samples row RLS lets this session see, in PAGE-row requests (keyset paging on ORDER).

    Stops on an empty page, not a short one: a project whose max_rows is below PAGE answers every page short.
    Rows come back by PC, then time; to_frame sorts them the same way."""
    out, last = [], None
    while True:
        q = client.table("activity_samples").select(",".join(COLUMNS))
        for col in ORDER:
            q = q.order(col)
        q = q.limit(PAGE)
        if last is not None:
            q = q.or_(after_filter(last["device_id"], last["ts"]))
        page = q.execute().data
        if not page:
            return out
        out += page
        last = page[-1]


def add_source_args(ap, behacom_csv):
    """--data (BEHACOM, the default) or --devdesk (your export) + --tz; read back by load_source."""
    ap.add_argument("--data", type=Path, default=None, help=f"BEHACOM slim CSV (default {behacom_csv})")
    ap.add_argument("--devdesk", type=Path, help="your own export (ml/scripts/export_devdesk.py) instead of BEHACOM")
    ap.add_argument("--tz", type=zone, default=None, help=f"with --devdesk: the PC's time zone (default {TZ})")
    ap.set_defaults(_behacom_csv=behacom_csv)


def load_source(ap, a, behacom_extras=()):
    """(minute frame, local zone, tag) for the chosen source; a mixed-up combination is an argparse error,
    not a silent run on the other source. behacom_extras: more BEHACOM columns (DevDesk has none of them)."""
    from ml import behacom

    if a.devdesk:
        if a.data:
            ap.error("--data (BEHACOM) and --devdesk exclude each other")
        return load_csv(a.devdesk), a.tz or TZ, "devdesk"
    if a.tz:
        ap.error("--tz only applies to --devdesk (BEHACOM is always Europe/Madrid)")
    return behacom.load(a.data or a._behacom_csv, extras=behacom_extras), behacom.TZ, "behacom"


def without_remote_clients(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose front app is a remote-desktop client (REMOTE_CLIENTS) dropped; the streamed PC has those minutes."""
    stem = df["exe_name"].map(lambda e: exe_stem(e) if isinstance(e, str) else "")
    dropped = stem.isin(REMOTE_CLIENTS)
    if dropped.any():
        print(f"devdesk_data: dropping {int(dropped.sum())} remote-client minutes (the streamed PC records them)")
    return df[~dropped].reset_index(drop=True)


def load_csv(path) -> pd.DataFrame:
    """The export written by ml/scripts/export_devdesk.py, without remote-client minutes."""
    raw = pd.read_csv(path, dtype={"exe_name": "string", "device_id": "string"})
    return without_remote_clients(to_frame(raw))
