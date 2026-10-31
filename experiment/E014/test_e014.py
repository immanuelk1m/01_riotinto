"""E014 data-pool, split, and optional torch smoke tests."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import E014 as m
import prepare


EXPECTED = {
    "wti_crude_oil": (31, 25, 6),
    "copper": (30, 18, 12),
    "gold": (24, 19, 5),
    "corn": (35, 26, 9),
    "wheat": (30, 22, 8),
    "coffee": (20, 15, 5),
}


def test_weekly_alignment() -> None:
    index = pd.date_range("2026-01-04", periods=3, freq="W-SUN")
    series = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-18"]))
    got = series.resample("W-SUN").last().reindex(index)
    assert got.iloc[0] == 1.0 and got.iloc[1] == 2.0 and got.iloc[2] == 3.0
    print("PASS W-SUN alignment")


def test_snapshot_contract() -> None:
    manifest = json.loads((m.EXP_DIR / "features" / "sheet_manifest.json").read_text())
    assert manifest["experiment"] == "E014_sheet_pool"
    for commodity, (total, approved, forced) in EXPECTED.items():
        tab = manifest["tabs"][commodity]
        assert (tab["feature_count"], tab["approved_count"], tab["forced_count"]) == (total, approved, forced)
        names = [item["col_name"] for item in tab["variables"]]
        feature_names = [item["name"] for item in json.loads(
            (m.EXP_DIR / "features" / f"{commodity}_features.json").read_text()
        )["features"]]
        frame = pd.read_parquet(m.EXP_DIR / "data" / f"{commodity}_X.parquet")
        assert names == sorted(names) == feature_names == list(frame.columns[1:])
        assert frame.columns[0] == "week_end" and not frame.week_end.duplicated().any()
        assert sum(item["unapproved_timing"] for item in tab["variables"]) == forced
        assert all(item["non_null_fraction"] > 0 for item in tab["variables"])
    print("PASS six-tab sheet/data/timing manifest contract")


def test_boundaries() -> None:
    blocks = 0
    for direction in m.core.DIRECTIONS:
        m.configure(direction)
        for commodity in m.TARGETS:
            state = m.core.load(commodity)
            assert state["names"][-2:] == ["tgt_logp", "tgt_r1"]
            expected_features = EXPECTED[commodity][0] + 2
            assert len(state["names"]) == expected_features
            for cut in range(int(0.8 * state["N"]), state["N"], m.core.REFIT):
                available, train, early, gate = m.core.split_rows(state, cut)
                assert train.max() + m.core.H < early.min()
                assert early.max() + m.core.H < gate.min()
                assert available.max() + m.core.H <= cut
                stats = m.core.scaler(state["F"], train.max())
                sequences = m.core.sequences(state["F"], train, stats)
                assert sequences.shape == (len(train), m.core.SEQ, expected_features)
                assert np.isfinite(sequences).all()
                blocks += 1
    print(f"PASS {blocks} direction/commodity split and scaler contracts")


def test_torch() -> None:
    import torch

    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(96, m.core.SEQ, 8)).astype("float32")
    y = rng.normal(size=(96, m.core.H)).astype("float32")
    old_delta = m.core.MIN_DELTA
    m.core.MIN_DELTA = 1e6
    try:
        for direction in m.core.DIRECTIONS:
            m.configure(direction)
            model, epoch, history = m.core.fit_model(torch, x, y, "cpu", 2, (x[:24], y[:24]))
            assert epoch == 0 and len(history) == 3
            assert np.isfinite(m.core.predict(torch, model, x[:2], "cpu")).all()
    finally:
        m.core.MIN_DELTA = old_delta
    print("PASS three-direction torch smoke")


if __name__ == "__main__":
    test_weekly_alignment()
    test_snapshot_contract()
    test_boundaries()
    if "--torch" in sys.argv:
        test_torch()
