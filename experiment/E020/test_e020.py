"""E020 contracts: direction table, causal windowing, snapshot/split integrity."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import E020 as m


def test_snapshots_and_splits():
    manifest = json.loads((m.FEAT_DIR / "sheet_manifest.json").read_text())
    count = 0
    for commodity in m.TARGETS:
        S = m.load(commodity)
        first, N = int(0.8 * S["N"]), S["N"]
        for cut in range(first, N, m.REFIT):
            available, train, early, gate = m.split_rows(S, cut)
            assert train.max() < cut
            assert early.min() > train.max()
            assert gate.min() > early.max()
            assert available.max() < cut, "mature rows must predate the origin"
            count += 1
    assert count > 0
    assert isinstance(manifest, dict)
    print(f"PASS splits causal for {count} refits")


def test_direction_table():
    assert m.DIRECTIONS == {"base": 0.0, "recent": 0.0, "tta": 0.0}, m.DIRECTIONS
    for direction in m.DIRECTIONS:
        m.configure(direction)
        assert m.DIRECTION == direction and m.PENALTY == 0.0
        assert m.EXP == f"E020_{direction}"
    m.configure("base")
    print("PASS direction table and penalty-zero configuration")


def test_window_rows_is_causal_and_bounded():
    available = np.arange(10, 400)
    rows = m.window_rows(available, m.RECENT_WEEKS)
    assert rows.max() == available.max()
    assert len(rows) == m.RECENT_WEEKS
    assert rows.min() > available.min()
    assert len(m.window_rows(available, 10_000)) == len(available)
    tta = m.window_rows(available, m.TTA_WEEKS)
    assert tta.max() == available.max() and len(tta) == m.TTA_WEEKS
    print("PASS recent/tta windows are trailing and bounded")


def test_output_validation_rejects_short_grid():
    commodity = next(iter(m.TARGETS))
    frame = pd.read_parquet(m.RES_DIR.parent.parent / "E014" / "results" / "erm" / f"predictions_{commodity}.parquet")
    frame = frame.copy()
    frame["y_pred"] = frame["y_cand"]
    frame["deploy"] = True
    frame["kappa"] = 0.0
    m.validate_results(commodity, frame)
    broken = frame.iloc[:-8].copy()
    try:
        m.validate_results(commodity, broken)
    except Exception:
        print("PASS output validation rejects a short grid")
        return
    raise AssertionError("validate_results accepted a short grid")


def test_torch_runs_if_available():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP torch not installed locally")
        return
    rng = np.random.default_rng(0)
    n, seq, feat = 300, m.SEQ, 3
    F = rng.normal(size=(n, seq, feat)).astype(np.float32)
    Z = rng.normal(size=(n, m.H)).astype(np.float32)
    S = {"F": F, "Z": Z, "scale": np.ones((n, m.H))}
    available = np.arange(200, 300)
    for direction in m.DIRECTIONS:
        m.configure(direction)
        model, stats, epochs, history, n_train = m.deployment_model(
            torch, S, available, None, None, 2, "cpu")
        P = m.predict(torch, model, m.sequences(F, available[:2], stats), "cpu")
        assert np.isfinite(P).all()
        assert n_train == (len(available) if direction != "recent" else min(m.RECENT_WEEKS, len(available)))
    m.configure("base")
    print("PASS torch directions train and predict finite values")


if __name__ == "__main__":
    test_snapshots_and_splits()
    test_direction_table()
    test_window_rows_is_causal_and_bounded()
    test_output_validation_rejects_short_grid()
    test_torch_runs_if_available()
    print("ALL E020 CONTRACTS PASS")
