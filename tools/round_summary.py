#!/usr/bin/env python3
"""One table for a registered family: paired difference, interval, one-sided p, Holm.

This replaces the ad-hoc ``/tmp/lb*.py`` scripts that computed the round's p-values
with their own hard-coded block, draw and sealed-window constants. Every number here
comes from ``_exo.params`` and from cells the engine wrote, so the table can be
regenerated identically.

The procedure is PROTOCOL 5.4/5.5:

* paired difference ``d = capped APE(persistence) - capped APE(candidate)``;
* 26-week calendar-block bootstrap of the development window, 1000 draws;
* one-sided ``p = P(mean d <= 0)`` under the bootstrap distribution;
* Holm at the declared family level over the registered statements.

Usage::

    uv run python tools/round_summary.py E023 E024 [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _exo.params import P  # noqa: E402


def bootstrap(dev: pd.DataFrame) -> dict:
    """Calendar-block bootstrap of the mean paired difference and its one-sided p."""
    frame = dev.assign(blk=(dev.origin // int(P.BLOCK)).astype(int))
    groups = [g.d.to_numpy(float) for _, g in frame.groupby("blk")]
    rng = np.random.default_rng(int(P.SEED))
    draws = np.empty(int(P.BOOT))
    for i in range(int(P.BOOT)):
        pick = rng.integers(0, len(groups), size=len(groups))
        draws[i] = np.concatenate([groups[j] for j in pick]).mean()
    return dict(
        mean=float(dev.d.mean()),
        lo=float(np.quantile(draws, P.ALPHA / 2)),
        hi=float(np.quantile(draws, 1 - P.ALPHA / 2)),
        p=float((draws <= 0).mean()),
        blocks=len(groups),
    )


def collect(experiments: list[str]) -> list[dict]:
    rows = []
    for experiment in experiments:
        for path in sorted((ROOT / "experiment" / experiment / "results").glob("*_cells.parquet")):
            cells = pd.read_parquet(path)
            cut = int(cells.origin.max()) - int(P.SEALED_ORIGINS)
            dev, sealed = cells[cells.origin < cut], cells[cells.origin >= cut]
            stats = bootstrap(dev)
            rows.append(dict(
                experiment=experiment,
                direction=str(cells.direction.iloc[0]),
                series=int(dev.series.nunique()),
                origins=int(dev.origin.nunique()),
                blocks=stats["blocks"],
                cells=int(len(dev)),
                mape_pers=float(dev.mape_pers.mean()),
                mape_cand=float(dev.mape_cand.mean()),
                mean_d=stats["mean"],
                ci_low=stats["lo"],
                ci_high=stats["hi"],
                p=stats["p"],
                sealed_d=float(sealed.d.mean()) if len(sealed) else float("nan"),
                sealed_mape_pers=float(sealed.mape_pers.mean()) if len(sealed) else float("nan"),
                sealed_mape_cand=float(sealed.mape_cand.mean()) if len(sealed) else float("nan"),
            ))
    return rows


def holm(rows: list[dict], alpha: float) -> list[dict]:
    ordered = sorted(rows, key=lambda r: r["p"])
    m = len(ordered)
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
        row["holm_alpha"] = alpha / (m - rank + 1)
        row["holm_pass"] = bool(row["p"] <= row["holm_alpha"])
        row["raw_pass"] = bool(row["mean_d"] > 0 and row["ci_low"] > 0)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    rows = holm(collect(args.experiments), float(P.ALPHA))
    header = ("direction", "series", "cells", "org", "blk", "MAPE pers", "MAPE cand",
              "mean d", "95 % CI", "p", "Holm a", "pass")
    print(f"{header[0]:<18}{header[1]:>6}{header[2]:>7}{header[3]:>5}{header[4]:>5}"
          f"{header[5]:>10}{header[6]:>10}{header[7]:>10}{header[8]:>20}{header[9]:>8}"
          f"{header[10]:>9}{header[11]:>7}")
    for row in sorted(rows, key=lambda r: r["direction"]):
        print(f"{row['direction']:<18}{row['series']:>6}{row['cells']:>7}{row['origins']:>5}"
              f"{row['blocks']:>5}{row['mape_pers']:>10.3f}{row['mape_cand']:>10.3f}"
              f"{row['mean_d']:>+10.5f}   [{row['ci_low']:+.5f},{row['ci_high']:+.5f}]"
              f"{row['p']:>8.4f}{row['holm_alpha']:>9.4f}{'PASS' if row['holm_pass'] else 'fail':>7}")
    passed = sum(1 for row in rows if row["holm_pass"])
    print(f"\nm = {len(rows)} registered statements; Holm pass: {passed}")
    print(f"sealed block: " + ", ".join(f"{r['direction']} {r['sealed_d']:+.5f}" for r in rows))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"[summary] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
