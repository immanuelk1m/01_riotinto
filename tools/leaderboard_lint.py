#!/usr/bin/env python3
"""Check that the leaderboard cannot be misread.

The lint enforces the parts of PROTOCOL.md §5 and §6.3 that a table can be checked
against mechanically:

1. every row carries a legal ``status``;
2. an ``ok`` candidate row records ``params_hash``, ``n_series``, ``cells``,
   ``protocol_version`` and ``evaluator_version``;
3. nothing carries a metric a MAPE cannot be (negative, non-finite) or ``n <= 0``;
4. ``aggregate`` is exactly "this is a ``horizon = 'all'`` row", so §5.1's ban on
   aggregates entering a cell mean is a filter, not a memory;
5. no legacy (v1 pipeline) row is back in the ledger;
6. no single experiment reports two different ``n_series`` values, which would mean
   its own rows are being averaged across panels (§5.3).

Usage::

    uv run python tools/leaderboard_lint.py
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _exo.leaderboard import STATUSES, connect  # noqa: E402

REQUIRED_FOR_OK = ("params_hash", "n_series", "cells", "protocol_version", "evaluator_version")


def main() -> int:
    argparse.ArgumentParser().parse_args()

    con = connect()
    con.row_factory = sqlite3.Row
    issues: list[str] = []

    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    if "leaderboard" not in tables:
        print("[leaderboard_lint] no leaderboard table")
        return 1

    def q(sql: str, *params):
        return con.execute(sql, params).fetchall()

    rows = q("select * from leaderboard")
    for row in rows:
        key = f"{row['experiment']}/{row['commodity']}/{row['horizon']}"
        if row["status"] not in STATUSES:
            issues.append(f"{key}: unknown status {row['status']!r}")
        if int(row["aggregate"] or 0) != int(row["horizon"] == "all"):
            issues.append(f"{key}: aggregate flag {row['aggregate']!r} disagrees with horizon")
        if row["status"] == "ok":
            # sanity is only required of rows that claim to be rankable; the
            # ``invalid`` and ``diverged`` statuses exist because v1 broke this rule
            if row["mape"] is None or row["mape"] != row["mape"] or row["mape"] < 0:
                issues.append(f"{key}: ok row has mape {row['mape']!r}")
            if not row["n"] or row["n"] <= 0:
                issues.append(f"{key}: ok row has n={row['n']!r}")
            if row["direction"]:
                for column in REQUIRED_FOR_OK:
                    if row[column] in (None, ""):
                        issues.append(f"{key}: ok candidate row is missing {column}")

    counts = {r[0]: r[1] for r in q("select status, count(*) from leaderboard group by 1")}
    total = sum(counts.values())

    legacy = q("select count(*) from leaderboard where pipeline='legacy'")[0][0]
    if legacy:
        issues.append(f"{legacy} legacy rows are back in the ledger")

    mixed = q(
        "select experiment, count(distinct n_series) k from leaderboard "
        "where status='ok' and n_series is not null group by 1 having k > 1"
    )
    for row in mixed:
        issues.append(f"{row['experiment']}: {row['k']} different n_series in one experiment (§5.3)")

    for issue in issues[:40]:
        print(f"  {issue}")
    if len(issues) > 40:
        print(f"  ... {len(issues) - 40} more")
    print(
        f"[leaderboard_lint] {total} rows {counts}"
        + f" — {len(issues)} violations"
    )
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
