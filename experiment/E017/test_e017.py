"""Causality, monotonicity, identity-control, and real snapshot split checks."""
import json
import sys

import numpy as np

import E017 as m


def same_maps(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_mapping():
    rng = np.random.default_rng(42)
    F = rng.normal(size=(400, 7))
    F[:, 1] = 3
    F[:295, 2] = np.nan
    F[201:, 0] += 5
    for window in (52, 104):
        mapping = m.fit_mapping(F, 200, 300, window)
        changed = F.copy()
        changed[301:] = 1e10
        same_maps(mapping, m.fit_mapping(changed, 200, 300, window))
        assert mapping[1] is None and mapping[2] is None
        assert mapping[-1] is None and mapping[-2] is None
        mapped = m.apply_mapping(F, mapping)
        assert np.array_equal(np.isnan(F), np.isnan(mapped))
        assert np.array_equal(F[:, -2:], mapped[:, -2:])
        assert abs(np.mean(mapped[249:301, 0])) < abs(np.mean(F[249:301, 0]))
        for spec in mapping:
            if spec is not None:
                assert np.all(np.diff(spec[0]) > 0)
                assert np.all(np.diff(spec[1]) >= 0)
        assert np.array_equal(
            mapped[:301], m.apply_mapping(changed, mapping)[:301], equal_nan=True
        )
    assert np.array_equal(F, m.apply_mapping(F, m.fit_mapping(F, 200, 300, 0)), equal_nan=True)
    try:
        m.fit_mapping(F, 301, 300, 52)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid mapping chronology accepted")
    print("PASS future-row mutation, identity, sparse/constant, monotonic mapping")


def test_snapshots():
    blocks = 0
    for direction in m.DIRECTIONS:
        m.configure(direction)
        assert m.core.PENALTY == 0
        for commodity in m.TARGETS:
            state = m.load(commodity)
            for cut in range(int(.8 * state["N"]), state["N"], m.REFIT):
                available, train, early, gate = m.split_rows(state, cut)
                assert train.max() + m.H < early.min()
                assert early.max() + m.H < gate.min()
                assert available.max() + m.H <= cut
                for phase, source_end in ((early, train.max()), (gate, train.max()), (np.array([cut]), available.max())):
                    fit_end = phase.min()
                    mapping = m.fit_mapping(state["F"], source_end, fit_end, m.WINDOW)
                    future = state["F"].copy()
                    future[fit_end + 1:] = -1e9
                    same_maps(mapping, m.fit_mapping(future, source_end, fit_end, m.WINDOW))
                    stats = m.scaler(state["F"], source_end)
                    x = m.sequences(m.apply_mapping(state["F"], mapping), phase, stats)
                    assert np.isfinite(x).all()
                    if direction == "identity":
                        assert np.array_equal(x, m.sequences(state["F"], phase, stats))
                blocks += 1
    print(f"PASS {blocks} real-snapshot phase/split/causality contracts")


def test_torch():
    import torch
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    F = rng.normal(size=(400, 7))
    rows = np.arange(300, 324)
    y = rng.normal(size=(len(rows), m.H)).astype("float32")
    for direction in m.DIRECTIONS:
        m.configure(direction)
        mapped = m.apply_mapping(F, m.fit_mapping(F, 200, 300, m.WINDOW))
        x = m.sequences(mapped, rows, m.scaler(F, 200))
        model, epoch, history = m.fit_model(torch, x, y, "cpu", 1, (x, y))
        assert np.isfinite(m.predict(torch, model, x, "cpu")).all()
        json.dumps(history, allow_nan=False)
    print("PASS torch training and mapping smoke")


if __name__ == "__main__":
    test_mapping()
    test_snapshots()
    if "--torch" in sys.argv:
        test_torch()
