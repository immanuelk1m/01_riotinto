"""Leaderboard writes for the E022-E026 engine.

One row per ``(series, horizon)`` plus an ``all`` row, following the convention the
E022-E026 round already uses in the database: ``<EXP>_<direction>`` for the candidate and
``<EXP>_persistence`` for the direction-free baseline.  ``mape`` is the capped mean APE percent that the audit reports and
``max_ape_pct`` is the worst single cell of the same set, in percent, *without* the
1.0 cap: a capped maximum saturates at 100 % and says nothing about tail risk.

Only dev-window cells are written (the sealed block stays out of the leaderboard
exactly as it stays out of the eligibility computation).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

COLUMN = "max_ape_pct"
SCHEMA = (
    "CREATE TABLE IF NOT EXISTS leaderboard(experiment TEXT, commodity TEXT, horizon TEXT, "
    "mape REAL, rmse REAL, n INTEGER, target_series TEXT, created_at TEXT, max_ape_pct REAL, "
    "PRIMARY KEY(experiment, commodity, horizon))"
)
INSERT = (
    "INSERT OR REPLACE INTO leaderboard(experiment, commodity, horizon, mape, rmse, n, "
    "target_series, created_at, max_ape_pct) VALUES (?,?,?,?,?,?,?,?,?)"
)


def ensure_schema(con: sqlite3.Connection) -> None:
    """Create the table if needed and add ``max_ape_pct`` to a pre-migration database."""
    con.execute(SCHEMA)
    if COLUMN not in [row[1] for row in con.execute("PRAGMA table_info(leaderboard)")]:
        con.execute(f"ALTER TABLE leaderboard ADD COLUMN {COLUMN} REAL")


def rows(experiment: str, cells: pd.DataFrame, col_names: dict[str, str], created_at: str | None = None,
         kind: str = "candidate"):
    """Leaderboard rows for one direction, scored over ``cells``."""
    stamp = created_at or datetime.now().isoformat(timespec="seconds")
    mape_col = "mape_cand" if kind == "candidate" else "mape_pers"
    raw_col = "ape_cand_raw" if kind == "candidate" else "ape_pers_raw"
    payload = []
    groups = [(str(h), g) for h, g in cells.groupby("horizon")] + [("all", cells)]
    for horizon, group in groups:
        for series, block in group.groupby("series"):
            payload.append(
                (
                    experiment,
                    series,
                    horizon,
                    float(block[mape_col].mean()),
                    None,
                    int(len(block)),
                    col_names.get(series, series),
                    stamp,
                    float(block[raw_col].max()),
                )
            )
    return payload


def write(db: Path | str, experiment: str, cells: pd.DataFrame, col_names: dict[str, str],
          persistence: str | None = None) -> int:
    """Upsert the rows for ``experiment`` (plus ``persistence`` if given)."""
    if cells.empty:
        return 0
    payload = rows(experiment, cells, col_names, None, "candidate")
    if persistence and {"mape_pers", "ape_pers_raw"} <= set(cells.columns):
        payload += rows(persistence, cells, col_names, None, "persistence")
    con = sqlite3.connect(Path(db), timeout=60)
    try:
        con.execute("PRAGMA busy_timeout=60000")
        ensure_schema(con)
        with con:
            con.executemany(INSERT, payload)
    finally:
        con.close()
    return len(payload)
