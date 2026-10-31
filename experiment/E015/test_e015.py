"""Causal split and deployment-identity tests; --torch exercises real training."""
import copy
import importlib.util
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import E015 as m


def test_snapshots_and_splits():
    manifest = json.loads((m.FEAT_DIR / "sheet_manifest.json").read_text())
    count = 0
    for commodity in m.TARGETS:
        s = m.load(commodity)
        assert len(s["names"]) == manifest["tabs"][commodity]["feature_count"] + 2
        for cut in range(int(.8 * s["N"]), s["N"], m.REFIT):
            available, train, early, gate = m.split_rows(s, cut)
            assert train.max() + m.H < early.min()
            assert early.max() + m.H < gate.min()
            assert available.max() + m.H <= cut
            modified = dict(s, Z=s["Z"].copy())
            # Corrupt every label cell whose target lies after this origin.
            unseen = np.arange(len(s["Z"]))[:, None] + np.arange(1, m.H + 1) > cut
            modified["Z"][unseen] = 1e6
            assert all(np.array_equal(a, b) for a, b in zip(
                m.split_rows(s, cut), m.split_rows(modified, cut)))
            stats = m.scaler(s["F"], train.max())
            assert np.isfinite(m.sequences(s["F"], train, stats)).all()
            count += 1
    print(f"PASS {count} six-commodity split, mature-label mutation and finite-input contracts")


def test_deployment_boundary():
    rng = np.random.default_rng(42)
    s = {"F": rng.normal(size=(160, 4)), "Z": rng.normal(size=(160, m.H))}
    available = np.arange(m.SEQ - 1, 120 - m.H + 1)
    gate_model = object()
    stats = m.scaler(s["F"], 80)
    for direction in m.DIRECTIONS:
        m.configure(direction)
        calls = []
        def fake_fit(torch, x, y, device, epochs, initial_model=None):
            calls.append((x.copy(), y.copy(), epochs, initial_model))
            return initial_model if initial_model is not None else object(), epochs, [{"epoch": epochs}]
        with patch.object(m, "fit_model", side_effect=fake_fit):
            model, new_stats, epochs, history = m.deployment_model(
                None, s, available, gate_model, stats, 3, "cpu")
        if direction == "frozen":
            assert model is gate_model and new_stats is stats and epochs == 0 and history == []
            assert not calls
        else:
            assert calls[0][2] == 3
            np.testing.assert_array_equal(calls[0][1], s["Z"][available])
            if direction == "warm":
                assert model is gate_model and new_stats is stats and calls[0][3] is gate_model
            else:
                assert model is not gate_model and calls[0][3] is None
                assert m.scaler_hash(new_stats) == m.scaler_hash(m.scaler(s["F"], available.max()))
        # Both fit inputs and scaler must remain unchanged if future data change.
        altered = {name: value.copy() for name, value in s.items()}
        altered["F"][121:] = 1e8
        unseen = np.arange(len(s["Z"]))[:, None] + np.arange(1, m.H + 1) > 120
        altered["Z"][unseen] = -1e8
        before = list(calls)
        with patch.object(m, "fit_model", side_effect=fake_fit):
            _, changed_stats, _, _ = m.deployment_model(
                None, altered, available, gate_model, stats, 3, "cpu")
        assert m.scaler_hash(changed_stats) == m.scaler_hash(new_stats)
        if before:
            np.testing.assert_array_equal(before[0][0], calls[-1][0])
            np.testing.assert_array_equal(before[0][1], calls[-1][1])
    print("PASS fresh/frozen/warm identity, scaler and future-data invariance")


def test_torch():
    import torch
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(96, m.SEQ, 4)).astype("float32")
    y = rng.normal(size=(96, m.H)).astype("float32")
    m.configure("fresh")
    model, _, _ = m.fit_model(torch, x, y, "cpu", 2)
    baseline_hash = m.model_hash(model)
    repeated, _, _ = m.fit_model(torch, x, y, "cpu", 2)
    assert m.model_hash(repeated) == baseline_hash
    # Check actual frozen checkpoint identity and real warm weight updates.
    frozen_copy = copy.deepcopy(model)
    m.configure("frozen")
    frozen, _, epochs, _ = m.deployment_model(torch, {}, None, frozen_copy, (), 2, "cpu")
    assert frozen is frozen_copy and epochs == 0 and m.model_hash(frozen) == baseline_hash
    warm, _, history = m.fit_model(torch, x, y, "cpu", 2, initial_model=model)
    assert warm is model and m.model_hash(warm) != baseline_hash
    assert np.isfinite(m.predict(torch, warm, x[:2], "cpu")).all()
    assert all(np.isfinite([row["train_loss"], row["objective"]]).all() for row in history)
    # Fresh fit must remain bit-identical to the original E013 ERM implementation.
    reference = m.EXP_DIR / "reference_e013.py"
    if not reference.exists():
        reference = m.EXP_DIR.parent / "E013" / "E013.py"
    if not reference.exists():
        raise AssertionError("Missing E013 reference for fresh-equivalence smoke")
    spec = importlib.util.spec_from_file_location("reference_e013", reference)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    ref.configure("erm")
    expected, _, _ = ref.fit_model(torch, x, y, "cpu", 2)
    assert m.model_hash(expected) == baseline_hash
    print("PASS torch fresh=E013, frozen checkpoint identity, warm weight update")


def test_output_validation():
    source = m.EXP_DIR.parent / "E014" / "results" / "erm"
    if not source.exists():
        return  # Colab uses the immutable data snapshot; local checks use archived outputs.
    frames = {c: pd.read_parquet(source / f"predictions_{c}.parquet") for c in m.TARGETS}
    for c, frame in frames.items():
        m.validate_results(c, frame)
        for bad in (
            frame.iloc[1:],
            frame[frame.origin != frame.origin.min()],
            pd.concat([frame, frame.iloc[:1]]),
        ):
            try:
                m.validate_results(c, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{c}: malformed prediction grid accepted")
    with tempfile.TemporaryDirectory() as temp:
        with patch.object(m, "LEADERBOARD", Path(temp) / "test.sqlite"):
            m.configure("fresh")
            m.write_leaderboard(frames)
            with sqlite3.connect(m.LEADERBOARD) as connection:
                assert connection.execute("SELECT count(*) FROM leaderboard").fetchone()[0] == 216
                assert connection.execute(
                    "SELECT count(*) FROM leaderboard WHERE experiment='naive'").fetchone()[0] == 0
    print("PASS prediction grids and isolated 216-row namespaced leaderboard")


if __name__ == "__main__":
    test_snapshots_and_splits()
    test_deployment_boundary()
    test_output_validation()
    if "--torch" in sys.argv:
        test_torch()
