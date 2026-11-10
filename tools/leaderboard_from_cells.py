#!/usr/bin/env python3
"""Write ``leaderboard`` rows from ``results/*_cells.parquet`` files.

The ledger is the W&B artifact ``riotinto/ledger``; the ledger step runs on the mini
(W&B credentials in `~/.netrc`) from those cells.  Each direction is one
W&B run, and ``results/`` is uploaded as the ``<EXP>-results`` artifact.  Every TSCV cell is
written; a cells file whose audit was produced under another ``PARAMS_HASH`` is skipped, so a retired
direction's stale cells cannot reach the ledger.

Usage (repository root):

    uv run python tools/leaderboard_from_cells.py E027 [--apply]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _exo.leaderboard import write  # noqa: E402
from _exo.params import PARAMS_HASH  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="experiment directory name, e.g. E027")
    ap.add_argument("--apply", action="store_true", help="write to the W&B ledger")
    args = ap.parse_args()

    base = ROOT / "experiment" / args.experiment
    manifest = json.loads((base / "data" / "manifest.json").read_text())
    col_names = {key: row["col_name"] for key, row in manifest["targets"].items()}
    params_hash = manifest.get("params_hash", PARAMS_HASH)

    paths = sorted((base / "results").glob("*_cells.parquet"))
    if not paths:
        raise SystemExit(f"no cells under {base / 'results'}; run the direction first")

    for path in paths:
        cells = pd.read_parquet(path)
        name = str(cells.direction.iloc[0])
        label = f"{args.experiment}_{name}"
        audit = json.loads(path.with_name(f"{name}_audit.json").read_text())
        if audit.get("params_hash") != PARAMS_HASH:
            print(f"[skip] {label}: cells were produced under params {audit.get('params_hash')}, "
                  f"current is {PARAMS_HASH}")
            continue
        if not args.apply:
            print(f"[dry] {label}: {len(cells)} cells, origins {cells.origin.nunique()}, "
                  f"series {sorted(cells.series.unique())}")
            continue
        written = write(label, cells, col_names, direction=name,
                        params_hash=params_hash, results=base / "results")
        print(f"[ledger] {label}: {written} rows written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
