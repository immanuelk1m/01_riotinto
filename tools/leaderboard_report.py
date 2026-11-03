#!/usr/bin/env python3
"""Regenerate ``experiment/LEADERBOARD_exogenous_2026.md`` from the database.

The markdown used to be written by hand, which is how a table ends up carrying
numbers that no query reproduces.  This tool is the only writer: it reads
``leaderboard.sqlite`` and emits the document, so the file is a view of the data
rather than a claim about it.

It also applies PROTOCOL.md §5.1 and §5.3 structurally:

* aggregate rows (``horizon = 'all'``) are never mixed into a cell table;
* a table is emitted per ``n_series``, so a five-commodity panel and a two-commodity
  panel never share a ranking;
* only ``status = 'ok'`` rows are ranked; ``retired-heuristic``, ``invalid`` and
  ``diverged`` rows are counted and labelled, not compared.

Usage::

    uv run python tools/leaderboard_report.py            # write the markdown
    uv run python tools/leaderboard_report.py --check    # fail if the file is stale
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiment" / "LEADERBOARD_exogenous_2026.md"

MEANING = {
    "ok": "rankable: persistence baseline, or a candidate whose model has no free parameter",
    "retired-heuristic": "model carries an unregistered free parameter, or the v1 pipeline produced it "
                         "— kept for the record, never compared",
    "invalid": "the stored metric is not a MAPE (negative value, or `n = 0` cells)",
    "diverged": "the fit blew up; the number is an artefact of divergence",
}


def build(db: Path) -> str:
    con = sqlite3.connect(db, timeout=30)
    con.row_factory = sqlite3.Row
    q = lambda sql, *p: con.execute(sql, *p).fetchall()  # noqa: E731

    counts = {r["status"]: r["n"] for r in q("select status, count(*) n from leaderboard group by 1")}
    total = sum(counts.values())
    lines = [
        "# Leaderboard — exogenous-variable round",
        "",
        f"Generated from `leaderboard.sqlite` by `tools/leaderboard_report.py` at "
        f"{datetime.now().isoformat(timespec='seconds')}. Do not hand-edit.",
        "",
        "## What may be ranked",
        "",
        "Only `status = 'ok'` rows are compared to anything. A row is `ok` when its forecast",
        "comes from a model with no free parameter — persistence, or closed-form ADL — produced",
        "under registered constants (`_exo/params.py`). Every row that a hand-set constant or an",
        "unregistered pipeline produced is kept in the same table under `retired-heuristic` and",
        "is excluded from every table below. See `experiment/RETIRED.md` for the rule.",
        "",
        "## Row accounting",
        "",
        "| status | rows | meaning |",
        "|---|---|---|",
    ]
    for status in ("ok", "retired-heuristic", "invalid", "diverged"):
        lines.append(f"| `{status}` | {counts.get(status, 0)} | {MEANING[status]} |")
    lines += [f"| **total** | **{total}** | |", ""]

    candidates = q(
        "select experiment, direction, n_series, params_hash, count(*) rows_ "
        "from leaderboard where status='ok' and direction is not null group by 1,2,3,4"
    )
    lines += [
        "## Rankable candidates",
        "",
        "| experiment | direction | n_series | params_hash | rows |",
        "|---|---|---|---|---|",
    ]
    if candidates:
        for row in candidates:
            lines.append(
                f"| `{row['experiment']}` | `{row['direction']}` | {row['n_series'] or '—'} "
                f"| `{row['params_hash'] or '—'}` | {row['rows_']} |"
            )
    else:
        lines.append(
            "| — | — | — | — | 0 |"
        )
        lines += [
            "",
            "**No rankable candidate rows exist yet.** The v1 pipeline produced none that survive",
            "the constant-free rule; the E023/E024 re-run under `PARAMS_VERSION = 2.0` is what",
            "creates them. This is the honest state of the program: there is no certified",
            "result to report.",
        ]

    lines += [
        "",
        "## Persistence reference, one panel at a time (§5.3)",
        "",
        "Series sets of different sizes are never averaged together, so each panel is its own",
        "table. These are the only numbers a future candidate is compared against.",
        "",
    ]
    panels = q(
        "select experiment, n_series, count(distinct commodity) series, count(*) rows_ "
        "from leaderboard where status='ok' and direction is null group by 1,2 order by 1"
    )
    for panel in panels:
        lines += [
            f"### `{panel['experiment']}` — {panel['series']} series, n_series="
            f"{panel['n_series'] or '— (v1 row)'}",
            "",
            "| commodity | horizon | MAPE | n |",
            "|---|---|---|---|",
        ]
        for row in q(
            "select commodity, horizon, mape, n from leaderboard "
            "where status='ok' and direction is null and experiment=? and horizon<>'all' "
            "order by commodity, cast(horizon as integer)",
            (panel["experiment"],),
        ):
            lines.append(f"| `{row['commodity']}` | {row['horizon']} | {row['mape']:.3f} | {row['n']} |")
        lines += [
            "",
            "| commodity | all horizons | MAPE | n |",
            "|---|---|---|---|",
        ]
        for row in q(
            "select commodity, mape, n from leaderboard "
            "where status='ok' and direction is null and experiment=? and horizon='all' order by commodity",
            (panel["experiment"],),
        ):
            lines.append(f"| `{row['commodity']}` | all | {row['mape']:.3f} | {row['n']} |")
        lines.append("")

    retired = q(
        "select experiment, count(distinct direction) dirs, count(*) rows_ from leaderboard "
        "where status='retired-heuristic' group by 1 order by 1"
    )
    lines += [
        "## Retired rows, kept for the record",
        "",
        "Counted, never ranked. The numbers are in `leaderboard_archive` exactly as v1 wrote them.",
        "",
        "| experiment | directions | rows |",
        "|---|---|---|",
    ]
    for row in retired:
        lines.append(f"| `{row['experiment']}` | {row['dirs']} | {row['rows_']} |")

    con.close()
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "leaderboard.sqlite"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    rendered = build(Path(args.db))
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        body = "\n".join(line for line in current.splitlines() if not line.startswith("Generated from"))
        fresh = "\n".join(line for line in rendered.splitlines() if not line.startswith("Generated from"))
        if body.strip() != fresh.strip():
            print("[report] stale: re-run tools/leaderboard_report.py")
            return 1
        print("[report] current")
        return 0
    out = Path(args.out)
    out.write_text(rendered, encoding="utf-8")
    print(f"[report] wrote {out} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
