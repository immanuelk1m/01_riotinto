"""Boundary checks plus optional torch objective/early-stop smoke."""
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import E013 as m


def test_boundaries():
    data = m.EXP_DIR.parent / 'E008' / 'data'
    features = m.EXP_DIR.parent / 'E008' / 'features'
    if not data.exists():
        data, features = m.EXP_DIR / 'data', m.EXP_DIR / 'features'
    m.DATA_DIR, m.FEAT_DIR = data, features
    blocks = 0
    for commodity in m.TARGETS:
        s = m.load(commodity)
        for cut in range(int(.8 * s['N']), s['N'], m.REFIT):
            available, train, early, gate = m.split_rows(s, cut)
            assert train.max() + m.H < early.min()
            assert early.max() + m.H < gate.min()
            assert available.max() + m.H <= cut
            env = m.environment_ids(len(train))
            assert set(env) == {0, 1, 2} and np.all(np.diff(env) >= 0)
            assert np.bincount(env).max() - np.bincount(env).min() <= 1
            stats = m.scaler(s['F'], train.max())
            x = m.sequences(s['F'], train, stats)
            assert x.shape == (len(train), m.SEQ, len(s['names']))
            assert np.isfinite(x).all()
            changed = s['F'].copy()
            changed[train.max() + 1:] = 1e9
            assert all(np.array_equal(a, b) for a, b in zip(stats, m.scaler(changed, train.max())))
            blocks += 1
    print(f'PASS {blocks} split/environment/scaler contracts')


def test_results():
    source = m.EXP_DIR.parent / 'E010' / 'results' / 'gru_reg'
    if not source.exists():
        return
    frames = {c: pd.read_parquet(source / f'predictions_{c}.parquet') for c in m.TARGETS}
    for c, frame in frames.items():
        m.validate_results(c, frame)
        missing = frame[frame.origin != frame.origin.unique()[2]]
        try:
            m.validate_results(c, missing)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing origin accepted')
        for edge in (frame.origin.min(), frame.origin.max()):
            truncated = frame[frame.origin != edge]
            try:
                m.validate_results(c, truncated)
            except ValueError:
                pass
            else:
                raise AssertionError('Missing edge origin accepted')
    with tempfile.TemporaryDirectory() as temp:
        m.LEADERBOARD = Path(temp) / 'leaderboard.sqlite'
        m.write_leaderboard(frames)
        with sqlite3.connect(m.LEADERBOARD) as con:
            assert con.execute('SELECT COUNT(*) FROM leaderboard').fetchone()[0] == 180
    print('PASS five-commodity output validation and temporary 180-row leaderboard')


def test_torch():
    import torch
    torch.set_num_threads(2)
    losses = torch.tensor([1., 1., 2., 2., 3., 3.], requires_grad=True)
    env = torch.tensor([0, 0, 1, 1, 2, 2])
    objective, pooled, variance, risks = m.environment_objective(torch, losses, env, .1)
    assert torch.allclose(pooled, torch.tensor(2.))
    assert torch.allclose(variance, torch.tensor(2. / 3.))
    assert torch.allclose(objective, torch.tensor(2. + .1 * 2. / 3.))
    objective.backward()
    assert torch.isfinite(losses.grad).all()
    rng = np.random.default_rng(42)
    x = rng.normal(size=(96, m.SEQ, 4)).astype('float32')
    y = rng.normal(size=(96, m.H)).astype('float32')
    old_delta = m.MIN_DELTA
    m.MIN_DELTA = 1e6  # Deliberately prevent any improvement over epoch zero.
    for direction in m.DIRECTIONS:
        m.configure(direction)
        model, epoch, history = m.fit_model(torch, x, y, 'cpu', 2, (x[:24], y[:24]))
        assert epoch == 0 and len(history) == 3
        assert np.isfinite(m.predict(torch, model, x[:2], 'cpu')).all()
        json.dumps(history, allow_nan=False)
        assert all(len(row['environment_losses']) == 3 for row in history)
    m.MIN_DELTA = old_delta
    print('PASS torch objective/gradient/epoch-zero/three-direction smoke')


if __name__ == '__main__':
    test_boundaries()
    test_results()
    if '--torch' in sys.argv:
        test_torch()
