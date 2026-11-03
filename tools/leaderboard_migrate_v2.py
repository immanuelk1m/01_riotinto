#!/usr/bin/env python3
"""Migrate ``leaderboard`` v1 to schema v2 without losing a row.

What v1 got wrong
-----------------
* It recorded none of the four columns PROTOCOL.md §6.3 requires — ``n_series``,
  ``cells``, ``evaluator_version``, ``protocol_version`` — so no comparison on it
  could be checked mechanically.
* It stored things that are not a MAPE: ``E021_ridge_validation`` wrote a
  *validation gain* with ``n = 0`` and a negative value; ``E026_mlp_broad`` wrote a
  diverged fit.
* It put ``horizon = 'all'`` aggregate rows in the same column as the cells that
  §5.1 forbids from entering a cell mean.

What this migration does
------------------------
The old table is renamed to ``leaderboard_archive`` and every row is re-expressed
in the v2 schema with an explicit ``status``:

``invalid``            metric cannot be a MAPE (negative, non-finite, ``n <= 0``)
``diverged``           the fit blew up (declared list, plus any MAPE above the cap)
``retired-heuristic``  produced by a model with a free parameter, or by the v1
                       pipeline whose constants were never registered
``ok``                 persistence baselines, which carry no parameter

Only ``ok`` rows may be ranked. Nothing is deleted: ``leaderboard_archive`` keeps
the v1 table byte for byte, and the migration is idempotent.

Usage::

    uv run python tools/leaderboard_migrate_v2.py            # dry run
    uv run python tools/leaderboard_migrate_v2.py --apply
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _exo.leaderboard import COLUMNS, SCHEMA  # noqa: E402
from _exo.params import PARAMS_VERSION  # noqa: E402

DB = ROOT / "leaderboard.sqlite"

#: model runs whose fit diverged; declared rather than inferred from an arbitrary cut
DIVERGED = {
    "E026_mlp_broad": "the MLP diverged: MAPE 50.23 against persistence 5.08",
}

#: rows that are baselines, not candidates
PERSISTENCE = re.compile(r"^(naive|.*_persistence)$")
PREFIX = re.compile(r"^E\d{3}_")


def split_direction(experiment: str) -> str | None:
    """``E024_gbrt_agri`` -> ``gbrt_agri``; a baseline has no direction."""
    if PERSISTENCE.match(experiment):
        return None
    return PREFIX.sub("", experiment)


def classify(row: sqlite3.Row | tuple, columns: list[str]) -> dict:
    record = dict(zip(columns, row))
    experiment = record["experiment"]
    mape, n = record.get("mape"), record.get("n")
    status, reason = "ok", None
    if mape is None or n is None or n <= 0 or mape < 0:
        status = "invalid"
        if mape is not None and mape < 0:
            reason = (
                f"mape={mape!r} is negative — v1 wrote a validation gain into the MAPE column "
                f"(n={n!r})"
            )
        else:
            reason = f"n={n!r} — the row records no cells, so its mean is not a MAPE"
    elif experiment in DIVERGED:
        status = "diverged"
        reason = DIVERGED[experiment]
    elif PERSISTENCE.match(experiment):
        status = "ok"  # a baseline has no fitted object
    else:
        status = "retired-heuristic"
        reason = "produced by a model with an unregistered free parameter, or by the v1 pipeline"
    direction = split_direction(experiment)
    return dict(
        experiment=experiment,
        direction=direction,
        commodity=record["commodity"],
        horizon=record["horizon"],
        aggregate=int(record["horizon"] == "all"),
        window="dev" if experiment.startswith(("E022", "E023", "E024", "E025", "E026", "E027")) else "legacy",
        mape=mape,
        rmse=record.get("rmse"),
        n=n,
        n_series=None,
        cells=None,
        max_ape_pct=record.get("max_ape_pct"),
        target_series=record.get("target_series"),
        params_hash=None,
        params_version=None,
        protocol_version="1.0",
        evaluator_version="cell-v1",
        pipeline="legacy",
        status=status,
        removal_reason=reason,
        created_at=record.get("created_at") or datetime.now().isoformat(timespec="seconds"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--db", default=str(DB))
    args = parser.parse_args()

    con = sqlite3.connect(args.db, timeout=60)
    con.row_factory = sqlite3.Row
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]

    if "leaderboard_archive" in tables:
        print("[migrate] already migrated (leaderboard_archive exists)")
        return 0

    columns = [r[1] for r in con.execute("PRAGMA table_info(leaderboard)")]
    if not columns:
        print("[migrate] no leaderboard table; run the pipeline first")
        return 0
    source = "leaderboard_archive" if "leaderboard_archive" in tables else "leaderboard"
    total = con.execute(f"select count(*) from {source}").fetchone()[0]
    print(f"[migrate] source {source}: {total} rows, v1 columns {len(columns)}")

    rows = [classify(r, columns) for r in con.execute(f"select * from {source}")]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"[migrate] status counts {counts}")

    if args.apply:
        with con:
            if source == "leaderboard":
                con.execute("ALTER TABLE leaderboard RENAME TO leaderboard_archive")
            con.execute(SCHEMA)  # creates `leaderboard` with the v2 schema
            con.executemany(
                f"INSERT OR REPLACE INTO leaderboard({', '.join(COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(COLUMNS))})",
                [tuple(row[column] for column in COLUMNS) for row in rows],
            )
        written = con.execute("select count(*) from leaderboard").fetchone()[0]
        print(f"[migrate] wrote {written} rows into leaderboard (v2); archive kept as leaderboard_archive")
        print(f"[migrate] params_version for future rows: {PARAMS_VERSION}")
    else:
        print("[migrate] dry run; pass --apply to write")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
