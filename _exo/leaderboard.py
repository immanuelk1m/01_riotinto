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
* ``params_hash`` / ``params_version`` — which registered constant set produced the row
* ``protocol_version`` / ``evaluator_version`` — §6.3
* ``pipeline``  — ``clean`` (registered constants, parameter-free model) or ``legacy``
* ``status``    — ``ok`` | ``retired-heuristic`` | ``invalid`` | ``diverged``

A row is refused at write time when it is not a number a MAPE can be: negative,
non-finite, or with ``n <= 0``.
"""
from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from .params import PARAMS_HASH, PARAMS_VERSION, PROTOCOL_VERSION

#: cell-v4 (2026-11-09): the evaluated cells are the monthly-first-week TSCV origins (PROTOCOL §4); the
#: dev/sealed split is gone and rows scored on other origins are not comparable (§5.3).
#: cell-v3 fixed the naive-forecast APE to |P_t/P_{t+h} - 1| as §5.1 states
EVALUATOR_VERSION = "cell-v4"
COLUMN = "max_ape_pct"

#: the ledger lives in W&B: artifact ``ledger`` (one parquet per table) is the only copy
PROJECT = "riotinto"
TABLES = ("leaderboard", "exogenous_round")

SCHEMA = """
CREATE TABLE IF NOT EXISTS exogenous_round(
  experiment text, direction text, model text, n_series integer, origins integer, blocks integer,
  mape_persistence real, mape_candidate real, delta_pp real, mean_d real, ci_low real, ci_high real,
  p_one_sided real, holm_alpha real, verdict text, protocol_version text, evaluator_version text, created_at text,
  primary key(experiment, direction));
CREATE TABLE IF NOT EXISTS leaderboard(
  experiment TEXT, direction TEXT, commodity TEXT, horizon TEXT,
  aggregate INTEGER DEFAULT 0,
  mape REAL, rmse REAL, n INTEGER, n_series INTEGER, cells INTEGER,
  max_ape_pct REAL, target_series TEXT,
  params_hash TEXT, params_version TEXT, protocol_version TEXT, evaluator_version TEXT,
  pipeline TEXT DEFAULT 'clean', status TEXT DEFAULT 'ok', removal_reason TEXT,
  created_at TEXT,
  PRIMARY KEY(experiment, commodity, horizon)
)
"""

COLUMNS = (
    "experiment", "direction", "commodity", "horizon", "aggregate",
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
    con.executescript(SCHEMA)


def load(frames: dict[str, pd.DataFrame]) -> sqlite3.Connection:
    """In-memory database with the ledger schema (primary keys kept) holding ``frames``."""
    con = sqlite3.connect(":memory:")
    ensure_schema(con)
    for table, frame in frames.items():
        frame.to_sql(table, con, if_exists="append", index=False)
    return con


def dump(con: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    return {table: pd.read_sql(f"select * from {table}", con) for table in TABLES}


def connect() -> sqlite3.Connection:
    """The ledger as an in-memory sqlite, loaded from W&B ``ledger:latest``."""
    import wandb

    root = wandb.Api().artifact(f"{PROJECT}/ledger:latest").download(root=tempfile.mkdtemp())
    frames = {table: pd.read_parquet(Path(root) / f"{table}.parquet") for table in TABLES}
    board = frames["leaderboard"]
    if "window" in board:   # pre-TSCV ledger: rows were scored on dev/sealed origins, not comparable
        print("[ledger] pre-TSCV schema: dropping `window` and the rows not scored as "
              f"{EVALUATOR_VERSION}; they are superseded by the rerun", flush=True)
        keep = board.evaluator_version == EVALUATOR_VERSION
        frames["leaderboard"] = board[keep].drop(columns="window")
        old = frames["exogenous_round"]
        frames["exogenous_round"] = old[old.evaluator_version == EVALUATOR_VERSION]
    return load(frames)


def commit(con: sqlite3.Connection, label: str, group: str, results: Path | None = None,
           ledger: bool = True) -> None:
    """One W&B run named ``group`` (``label`` minus its ``_*`` suffix): its rows as a table, and the whole ledger as a new version."""
    import wandb

    frames = dump(con)
    board = frames["leaderboard"]
    mine = board[board.experiment == label]
    # provenance (params_hash, versions, direction, pipeline) lives in the rows table and the ledger
    # artifact; the runs table only carries status, so it stays readable
    # one run per group: a fixed id makes later commits resume it, and it logs every label of the group
    run = wandb.init(project=PROJECT, id=group, name=group, group=group, resume="allow")
    try:
        if not mine.empty:
            run.config.update({"status": mine["status"].iloc[0]}, allow_val_change=True)
            run.log({"rows": wandb.Table(dataframe=board[board.experiment.str.split("_").str[0] == group])})
        if ledger:
            # ponytail: last-writer-wins on ledger:latest; add a version check if two machines ever commit concurrently
            art = wandb.Artifact("ledger", type="ledger")
            with tempfile.TemporaryDirectory() as tmp:
                for table, frame in frames.items():
                    frame.to_parquet(Path(tmp) / f"{table}.parquet", index=False)
                    art.add_file(str(Path(tmp) / f"{table}.parquet"))
                run.log_artifact(art).wait()
        if results is not None and Path(results).is_dir():
            out = wandb.Artifact(f"{group}-results", type="results")
            out.add_dir(str(results))
            run.log_artifact(out)
    finally:
        run.finish()


def rows(experiment: str, cells: pd.DataFrame, col_names: dict[str, str],
         created_at: str | None = None,
         direction: str | None = None,
         params_hash: str | None = None, status: str = "ok",
         pipeline: str = "clean", note: str | None = None) -> list[tuple]:
    """Leaderboard rows for one direction, scored over ``cells``."""
    stamp = created_at or datetime.now().isoformat(timespec="seconds")
    n_series = int(cells["series"].nunique())
    payload = []
    groups = [(str(h), g) for h, g in cells.groupby("horizon")] + [("all", cells)]
    for horizon, group in groups:
        for series, block in group.groupby("series"):
            row = dict(
                experiment=experiment, direction=direction, commodity=series, horizon=horizon,
                aggregate=int(horizon == "all"),
                mape=float(block["mape_cand"].mean()), rmse=None, n=int(len(block)),
                n_series=n_series, cells=int(len(cells)),
                max_ape_pct=float(block["ape_cand_raw"].max()),
                target_series=col_names.get(series, series),
                params_hash=params_hash, params_version=PARAMS_VERSION,
                protocol_version=PROTOCOL_VERSION, evaluator_version=EVALUATOR_VERSION,
                pipeline=pipeline, status=status, removal_reason=note, created_at=stamp,
            )
            check(row)
            payload.append(tuple(row[column] for column in COLUMNS))
    return payload


def write(experiment: str, cells: pd.DataFrame, col_names: dict[str, str],
          direction: str | None = None,
          params_hash: str = PARAMS_HASH, status: str = "ok", results: Path | None = None) -> int:
    """Upsert the candidate rows for ``experiment`` into the W&B ledger (no naive-forecast rows)."""
    if cells.empty:
        return 0
    payload = rows(experiment, cells, col_names, None, direction=direction,
                   params_hash=params_hash, status=status)
    con = connect()
    with con:
        con.executemany(INSERT, payload)
    commit(con, experiment, experiment.split("_")[0], results=results)
    con.close()
    return len(payload)
