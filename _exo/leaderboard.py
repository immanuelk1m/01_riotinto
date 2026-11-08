"""Leaderboard writes — schema v2.

Why v2
------
PROTOCOL.md §6.3 already required every row to record ``n_series``, ``cells``,
``evaluator_version`` and ``protocol_version`` so that comparability can be
checked mechanically instead of remembered.  The v1 table recorded none of the
four, and it accepted rows that cannot be a MAPE at all: ``E021_ridge_validation``
stored a *validation gain* (``mape = −0.002``, ``n = 0``) in the metric column,
``E026_mlp_broad`` stored a diverged fit (``mape = 50.2``), and aggregate rows
(``horizon = 'all'``) sat in the same column as cells that §5.1 says must never
enter a cell mean.

v2 therefore writes and *gates*:

* ``aggregate`` — 1 for ``horizon = 'all'``; ranking queries must filter it out
* ``window``    — ``dev`` or ``sealed``; the sealed block never enters eligibility
* ``params_hash`` / ``params_version`` — which registered constant set produced the row
* ``protocol_version`` / ``evaluator_version`` — §6.3
* ``pipeline``  — ``clean`` (registered constants, parameter-free model) or ``legacy``
* ``status``    — ``ok`` | ``retired-heuristic`` | ``invalid`` | ``diverged``

A row is refused at write time when it is not a number a MAPE can be: negative,
non-finite, or with ``n <= 0``.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from .params import PARAMS_HASH, PARAMS_VERSION, PROTOCOL_VERSION

#: cell-v3 (2026-11-08): persistence APE is |P_t/P_{t+h} - 1| as PROTOCOL §5.1 states; cell-v2 rows
#: divided by P_t instead and are not comparable with cell-v3 rows (§5.3)
EVALUATOR_VERSION = "cell-v3"
COLUMN = "max_ape_pct"

SCHEMA = """
CREATE TABLE IF NOT EXISTS leaderboard(
  experiment TEXT, direction TEXT, commodity TEXT, horizon TEXT,
  aggregate INTEGER DEFAULT 0, window TEXT DEFAULT 'dev',
  mape REAL, rmse REAL, n INTEGER, n_series INTEGER, cells INTEGER,
  max_ape_pct REAL, target_series TEXT,
  params_hash TEXT, params_version TEXT, protocol_version TEXT, evaluator_version TEXT,
  pipeline TEXT DEFAULT 'clean', status TEXT DEFAULT 'ok', removal_reason TEXT,
  created_at TEXT,
  PRIMARY KEY(experiment, commodity, horizon, window)
)
"""

COLUMNS = (
    "experiment", "direction", "commodity", "horizon", "aggregate", "window",
    "mape", "rmse", "n", "n_series", "cells", "max_ape_pct", "target_series",
    "params_hash", "params_version", "protocol_version", "evaluator_version",
    "pipeline", "status", "removal_reason", "created_at",
)
INSERT = f"INSERT OR REPLACE INTO leaderboard({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})"

#: statuses a row may carry; ``ok`` is the only one that may be ranked
STATUSES = ("ok", "retired-heuristic", "invalid", "diverged")


class LeaderboardRowError(ValueError):
    """Raised instead of writing a row that cannot be a result."""


def check(row: dict) -> dict:
    """Reject anything that is not a rankable row; return the row otherwise."""
    mape = row.get("mape")
    if mape is None or not isinstance(mape, (int, float)) or mape != mape:  # NaN check
        raise LeaderboardRowError(f"{row['experiment']}/{row['commodity']}/{row['horizon']}: mape {mape!r}")
    if float(mape) < 0:
        raise LeaderboardRowError(
            f"{row['experiment']}/{row['commodity']}/{row['horizon']}: negative mape {mape!r} — "
            f"a MAPE cannot be negative; store a gain in its own column"
        )
    if int(row.get("n") or 0) <= 0:
        raise LeaderboardRowError(
            f"{row['experiment']}/{row['commodity']}/{row['horizon']}: n={row.get('n')!r}"
        )
    if row.get("status", "ok") not in STATUSES:
        raise LeaderboardRowError(f"unknown status {row.get('status')!r}")
    if row.get("status", "ok") == "ok" and row.get("direction") and not row.get("params_hash"):
        raise LeaderboardRowError(
            f"{row['experiment']}/{row['direction']}: an eligible row must record params_hash"
        )
    return row


def ensure_schema(con: sqlite3.Connection) -> None:
    con.execute(SCHEMA)


def rows(experiment: str, cells: pd.DataFrame, col_names: dict[str, str],
         created_at: str | None = None, kind: str = "candidate",
         direction: str | None = None, window: str = "dev",
         params_hash: str | None = None, status: str = "ok",
         pipeline: str = "clean", note: str | None = None) -> list[tuple]:
    """Leaderboard rows for one direction, scored over ``cells``."""
    stamp = created_at or datetime.now().isoformat(timespec="seconds")
    mape_col = "mape_cand" if kind == "candidate" else "mape_pers"
    raw_col = "ape_cand_raw" if kind == "candidate" else "ape_pers_raw"
    n_series = int(cells["series"].nunique())
    payload = []
    groups = [(str(h), g) for h, g in cells.groupby("horizon")] + [("all", cells)]
    for horizon, group in groups:
        for series, block in group.groupby("series"):
            row = dict(
                experiment=experiment, direction=direction, commodity=series, horizon=horizon,
                aggregate=int(horizon == "all"), window=window,
                mape=float(block[mape_col].mean()), rmse=None, n=int(len(block)),
                n_series=n_series, cells=int(len(cells)),
                max_ape_pct=float(block[raw_col].max()),
                target_series=col_names.get(series, series),
                params_hash=params_hash, params_version=PARAMS_VERSION,
                protocol_version=PROTOCOL_VERSION, evaluator_version=EVALUATOR_VERSION,
                pipeline=pipeline, status=status, removal_reason=note, created_at=stamp,
            )
            check(row)
            payload.append(tuple(row[column] for column in COLUMNS))
    return payload


def write(db: Path | str, experiment: str, cells: pd.DataFrame, col_names: dict[str, str],
          persistence: str | None = None, direction: str | None = None,
          params_hash: str = PARAMS_HASH, status: str = "ok") -> int:
    """Upsert the rows for ``experiment`` (plus ``persistence`` if given)."""
    if cells.empty:
        return 0
    payload = rows(experiment, cells, col_names, None, "candidate", direction=direction,
                   params_hash=params_hash, status=status)
    if persistence and {"mape_pers", "ape_pers_raw"} <= set(cells.columns):
        # a baseline has no direction: that is what marks it as the reference, not a candidate
        payload += rows(persistence, cells, col_names, None, "persistence",
                        direction=None, params_hash=params_hash, status=status)
    con = sqlite3.connect(Path(db), timeout=60)
    try:
        con.execute("PRAGMA busy_timeout=60000")
        ensure_schema(con)
        with con:
            con.executemany(INSERT, payload)
    finally:
        con.close()
    return len(payload)
