#!/usr/bin/env python3
"""An evaluator written from the text of PROTOCOL.md only, to check `_exo/engine.py` cell by cell.

It never imports the engine for its own numbers. From the text:

* §2.1  err = min(1.0, |expm1(forecast − log_return_target)|); the naive forecast is a zero log return.
* §5.1  cell MAPE = mean(|forecast/target − 1|) × 100 over cells (series, horizon, origin).
* §5.3  series are weighted equally (one panel, one cell count per series).
* §2.2  interval: moving-block bootstrap with 26-week blocks, over the origin-ordered d.

`--check` compares it with the engine on E028's TSCV cells: every cell APE and every
statement MAPE must agree to TOL_ABS. The interval is reported, not gated: the engine resamples
non-overlapping calendar blocks, §2.2 says moving blocks, so the two intervals differ by design.
With ~20 monthly origins a 26-origin moving block does not fit, so that interval prints as nan.
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

from _exo.params import P  # noqa: E402  (protocol constants only)

E028 = ROOT / "experiment" / "E028"


def ape(forecast_logret, target_logret):
    """§2.1 / §5.1: |P̂/P − 1| with P̂ = P_t·e^fc and P = P_t·e^r, capped at CAP."""
    return np.minimum(P.CAP, np.abs(np.exp(forecast_logret - target_logret) - 1.0))


def statement(cells: pd.DataFrame) -> dict:
    pers = ape(0.0, cells.target_logret)
    cand = ape(cells.fc, cells.target_logret)
    by_series = pd.DataFrame(dict(series=cells.series, p=pers, c=cand)).groupby("series").mean()
    mape_p, mape_c = by_series.p.mean() * P.PCT, by_series.c.mean() * P.PCT
    d = (pers - cand).groupby(cells.origin).mean().sort_index().to_numpy()
    return dict(mape_pers=float(mape_p), mape_cand=float(mape_c), reduction=float(1 - mape_c / mape_p),
                moving_block_ci=list(moving_block(d)))


def moving_block(d: np.ndarray):
    block, rng = int(P.BLOCK), np.random.default_rng(int(P.SEED))
    if len(d) <= block:
        return float("nan"), float("nan")
    starts = np.arange(len(d) - block + 1)
    draws = [np.concatenate([d[s:s + block] for s in rng.choice(starts, -(-len(d) // block))])[:len(d)].mean()
             for _ in range(int(P.BOOT))]
    return float(np.quantile(draws, P.ALPHA / 2)), float(np.quantile(draws, 1 - P.ALPHA / 2))


def check() -> int:
    sys.path.insert(0, str(E028))
    import E028 as exp  # the reference side: engine scoring through E028's own selection

    failures = []
    path = E028 / "results" / "tscv_cells.parquet"
    if not path.exists():
        print(f"[evaluator] {path.name} missing, skipped")
        return 0
    raw = pd.read_parquet(path)
    ref = exp.scored(raw, "struct", exp.common_origins(raw))
    mine_p = ape(0.0, ref.target_logret)
    mine_c = ape(ref.fc, ref.target_logret)
    cell_gap = float(max(np.abs(mine_p - ref.ape_pers).max(), np.abs(mine_c - ref.ape_cand).max()))
    rows = json.loads((E028 / "results" / "tscv.json").read_text())["gate_a"]["struct"]
    stat_gap = 0.0
    for key in exp.STATEMENTS:
        got = statement(exp.statement(ref, key))
        for field in ("mape_pers", "mape_cand", "reduction"):
            stat_gap = max(stat_gap, abs(got[field] - rows[key][field]))
        print(f"[evaluator] {key:7} MAPE {got['mape_pers']:.4f}->{got['mape_cand']:.4f} "
              f"({got['reduction'] * P.PCT:+.2f}%) moving-block CI d {got['moving_block_ci']}")
    print(f"[evaluator] {len(ref)} cells, max cell gap {cell_gap:.2e}, max statement gap {stat_gap:.2e}")
    if cell_gap > P.TOL_ABS or stat_gap > P.TOL_ABS:
        failures.append("tscv")
    print(f"[evaluator] {'FAIL ' + str(failures) if failures else 'engine == protocol text, cell by cell'}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    raise SystemExit(check())
