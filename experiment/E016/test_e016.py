"""Causal feedback and archive contracts; temporary outputs only."""
import tempfile
import unittest
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import E016 as m


def synthetic(n=100):
    records = []
    for i, date in enumerate(pd.date_range('2020-01-05', periods=n, freq='W-SUN')):
        for h in range(1, 9):
            raw = .02 * np.sin(i / 7 + h)
            truth = .7 * raw + (.003 if i < 55 else .04)
            records.append(dict(origin=date, target_date=date + pd.Timedelta(weeks=h), horizon=h,
                                y_true=100*np.exp(truth), y_raw=100*np.exp(raw), last_price=100., n_train=100))
    return pd.DataFrame(records)


class Contracts(unittest.TestCase):
    def test_future_labels_do_not_change_past_predictions_or_state(self):
        frame = synthetic()
        cutoff = frame.origin.unique()[65]
        changed = frame.copy()
        changed.loc[changed.target_date > cutoff, 'y_true'] *= 4
        for direction in m.DIRECTIONS:
            a, ta = m.replay(frame, direction)
            b, tb = m.replay(changed, direction)
            keep = a.origin <= cutoff
            np.testing.assert_array_equal(a.loc[keep, ['y_pred', 'y_cand']], b.loc[keep, ['y_pred', 'y_cand']])
            pd.testing.assert_frame_equal(ta.loc[keep], tb.loc[keep])

    def test_release_and_drift(self):
        out, trace = m.replay(synthetic(), 'drift26')
        for h, group in trace.groupby('horizon'):
            np.testing.assert_array_equal(group.n_updates, np.maximum(np.arange(100)-h+1, 0))
        available = trace.last_update_target.notna()
        self.assertTrue((trace.loc[available, 'last_update_target'] <= trace.loc[available, 'origin']).all())
        self.assertTrue(trace.drift.any())
        self.assertTrue(trace.loc[trace.drift, 'window'].eq(26).all())
        self.assertTrue(trace.loc[~trace.drift, 'window'].eq(52).all())
        self.assertTrue((trace.scale.between(0, 1)).all())
        self.assertTrue(trace.bias.abs().le(m.BIAS_CLIP).all())
        self.assertTrue(out.loc[out.n_updates < m.MIN_FIT, 'y_pred'].eq(100).all())

    def test_invalid_grid(self):
        frame = synthetic(20)
        for bad in (frame.iloc[1:], pd.concat([frame, frame.iloc[:1]]), frame[frame.origin != frame.origin.unique()[3]]):
            with self.assertRaises(ValueError):
                m.replay(bad, 'static52')

    def test_real_archives(self):
        for commodity in m.TARGETS:
            source = m.load(commodity)
            out, trace = m.replay(source, 'drift26')
            m.validate(out)
            pd.testing.assert_frame_equal(out[m.KEYS], source[m.KEYS])
            self.assertEqual(len(trace), len(source))
            self.assertTrue(m.diagnostics(out).diagnostic_only.all())

    def test_output_and_leaderboard(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            inputs = base / 'inputs'
            inputs.mkdir()
            for commodity in m.TARGETS:
                synthetic(70).to_parquet(inputs / f'predictions_{commodity}.parquet', index=False)
            with patch.object(m, 'BASE', base):
                for commodity in m.TARGETS:
                    m.run(commodity, 'recent26', inputs)
                m.leaderboard('recent26', base/'scores.sqlite', inputs)
                with closing(sqlite3.connect(base/'scores.sqlite')) as con:
                    self.assertEqual(con.execute('SELECT count(*) FROM leaderboard').fetchone()[0], 108)
                    self.assertEqual(con.execute("SELECT count(*) FROM leaderboard WHERE experiment='naive'").fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
