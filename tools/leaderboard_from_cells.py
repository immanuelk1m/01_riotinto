#!/usr/bin/env python3
"""Write ``leaderboard`` rows from promoted ``*_cells.parquet`` files.

The Colab VM writes its ledger to ``/content/leaderboard.sqlite``, which the harness
does not download, so the ledger step has to run here, on the mini, from the promoted
cells.  Only development-window cells are written (sealed origins stay out), exactly as
``_exo/engine.py`` does at run time.

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
from _exo.params import P, PARAMS_HASH  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="experiment directory name, e.g. E027")
    ap.add_argument("--apply", action="store_true", help="write to leaderboard.sqlite")
    args = ap.parse_args()

    base = ROOT / "experiment" / args.experiment
    manifest = json.loads((base / "data" / "manifest.json").read_text())
    col_names = {key: row["col_name"] for key, row in manifest["targets"].items()}
    params_hash = manifest.get("params_hash", PARAMS_HASH)

    paths = sorted((base / "results").glob("*_cells.parquet"))
    if not paths:
        raise SystemExit(f"no cells under {base / 'results'}; promote the run first")

    for path in paths:
        cells = pd.read_parquet(path)
        name = str(cells.direction.iloc[0])
        cut = int(cells.origin.max()) - P.SEALED_ORIGINS
        dev = cells[cells.origin < cut]
        label = f"{args.experiment}_{name}"
        if not args.apply:
            print(f"[dry] {label}: {len(dev)} dev cells of {len(cells)} "
                  f"(sealed {len(cells) - len(dev)}), series {sorted(dev.series.unique())}")
            continue
        written = write(ROOT / "leaderboard.sqlite", label, dev, col_names,
                        persistence=f"{args.experiment}_persistence",
                        direction=name, params_hash=params_hash)
        print(f"[ledger] {label}: {written} rows written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
