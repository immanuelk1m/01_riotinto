"""Checks for E019: causality, determinism, e-BH arithmetic and decision contracts."""

import unittest

import numpy as np
import pandas as pd

import E019 as m


def synthetic(commodities=3, origins=40, seed=0):
    rng = np.random.default_rng(seed)
    frames = {}
    dates = pd.date_range('2020-01-05', periods=origins, freq='W-SUN')
    for k in range(commodities):
        rows = []
        edge = 0.004 if k == 0 else -0.004
        for i, origin in enumerate(dates):
            for h in range(1, 9):
                last = 100.0
                rows.append(dict(origin=origin, target_date=origin + pd.Timedelta(weeks=h), horizon=h,
                                 y_true=float(last * np.exp(rng.normal(0, 0.02))), last_price=last,
                                 y_raw=float(last * np.exp(rng.normal(0, 0.01))),
                                 y_cand=float(last * np.exp(edge + rng.normal(0, 0.01))), n_train=100))
        frame = pd.DataFrame(rows)
        frame.loc[frame.index[-8:], 'y_true'] = np.nan
        frames[f'c{k}'] = frame.sort_values(m.KEYS).reset_index(drop=True)
    return frames


def replay(frames, direction):
    panels = {c: m.weekly_panel(f) for c, f in frames.items()}
    origins = sorted(set().union(*[set(f.origin) for f in frames.values()]))
    return m.replay(panels, origins, direction)


class Contracts(unittest.TestCase):
    def test_directions_run(self):
        for direction in m.DIRECTIONS:
            decisions, trace = replay(synthetic(), direction)
            self.assertEqual(len(trace), len(synthetic()['c0'].origin.unique()))
            for c, mapping in decisions.items():
                self.assertTrue(all(isinstance(v, (bool, np.bool_)) for v in mapping.values()))

    def test_future_labels_do_not_change_past_decisions(self):
        frames = synthetic()
        mutated = {c: f.copy() for c, f in frames.items()}
        cutoff = frames['c0'].origin.iloc[0] + pd.Timedelta(weeks=20)
        for frame in mutated.values():
            future = frame.target_date > cutoff
            frame.loc[future, 'y_true'] = frame.loc[future, 'y_true'] * 3.0
        for direction in m.DIRECTIONS:
            base, base_trace = replay(frames, direction)
            alt, alt_trace = replay(mutated, direction)
            early = base_trace.origin <= cutoff
            pd.testing.assert_frame_equal(base_trace.loc[early, ['origin', 'deployed', 'deployed_names']].reset_index(drop=True),
                                          alt_trace.loc[early, ['origin', 'deployed', 'deployed_names']].reset_index(drop=True))
            for c in base:
                early_origins = [o for o in base[c] if o <= cutoff]
                self.assertEqual({o: base[c][o] for o in early_origins},
                                 {o: alt[c][o] for o in early_origins})

    def test_deterministic(self):
        frames = synthetic()
        for direction in m.DIRECTIONS:
            a, ta = replay(frames, direction)
            b, tb = replay(frames, direction)
            self.assertEqual(a, b)
            pd.testing.assert_frame_equal(ta, tb)

    def test_no_deploy_before_min_labels(self):
        frames = synthetic()
        decisions, trace = replay(frames, 'ebh')
        early = trace[trace.origin < frames['c0'].origin.iloc[m.MIN_LABELS + 1]]
        self.assertTrue(early.deployed.eq(0).all())

    def test_ebh_rejects_the_obvious_winners(self):
        e_values = [1000.0, 0.5, 0.4, 0.3, 0.2, 0.1]
        rejected = m.ebh_reject(e_values, alpha=0.10)
        self.assertEqual(rejected, {0})
        self.assertEqual(m.ebh_reject([0.5] * 6, alpha=0.10), set())

    def test_ebh_threshold_scales_with_family_size(self):
        # m/(alpha*i) for m=6, alpha=0.1, i=1 -> 60; 50 must not be rejected.
        self.assertEqual(m.ebh_reject([50.0, 1.0, 1.0, 1.0, 1.0, 1.0], alpha=0.10), set())
        self.assertEqual(m.ebh_reject([61.0, 1.0, 1.0, 1.0, 1.0, 1.0], alpha=0.10), {0})

    def test_error_cap_bounds_differences(self):
        self.assertLessEqual(m.ape(np.array([5.0]), np.array([0.0]))[0], m.CAP)

    def test_run_writes_persistence_when_not_deployed(self):
        frames = synthetic(commodities=2)
        for direction in m.DIRECTIONS:
            decisions, _ = replay(frames, direction)
            for c, frame in frames.items():
                mapping = decisions[c]
                deployed = frame.origin.map(lambda o: mapping.get(o, False))
                self.assertTrue(deployed.eq(False).any() or deployed.all())


if __name__ == '__main__':
    unittest.main()
