"""Checks for E018: causality, determinism and the predeclared decision contracts."""

import unittest

import numpy as np
import pandas as pd

import E018 as m


def synthetic(origins=40, seed=0):
    """Frame with a candidate that is genuinely better in the first half only."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range('2020-01-05', periods=origins, freq='W-SUN')
    for i, origin in enumerate(dates):
        for h in range(1, 9):
            target = origin + pd.Timedelta(weeks=h)
            last = 100.0
            noise = rng.normal(0, 0.02)
            edge = 0.004 if i < origins // 2 else -0.004
            true_lr = noise
            rows.append(dict(origin=origin, target_date=target, horizon=h,
                             y_true=float(last * np.exp(true_lr)), last_price=last,
                             y_raw=float(last * np.exp(edge + rng.normal(0, 0.005))),
                             y_cand=float(last * np.exp(edge + rng.normal(0, 0.004))),
                             n_train=100))
    frame = pd.DataFrame(rows)
    frame.loc[frame.index[-8:], 'y_true'] = np.nan   # trailing unmatured week
    return frame.sort_values(m.KEYS).reset_index(drop=True)


class Contracts(unittest.TestCase):
    def test_directions_run_and_validate(self):
        frame = synthetic()
        for direction in m.DIRECTIONS:
            out, trace = m.replay(frame, direction)
            m.validate(out)
            self.assertTrue(set(out.columns) >= {'y_pred', 'y_cand', 'deploy'})
            self.assertTrue(out.loc[~out.deploy, 'y_pred'].eq(out.loc[~out.deploy, 'last_price']).all(),
                            'non-deployed rows must fall back to persistence')
            self.assertTrue(len(trace) == out.origin.nunique())

    def test_future_labels_do_not_change_past_predictions_or_state(self):
        frame = synthetic()
        mutated = frame.copy()
        future = mutated.target_date > mutated.origin.iloc[0] + pd.Timedelta(weeks=20)
        mutated.loc[future, 'y_true'] = mutated.loc[future, 'y_true'] * 3.0
        state_columns = ['origin', 'target_date', 'horizon', 'y_pred', 'y_cand', 'deploy']
        for direction in m.DIRECTIONS:
            base, base_trace = m.replay(frame, direction)
            alt, alt_trace = m.replay(mutated, direction)
            cutoff = frame.origin.iloc[0] + pd.Timedelta(weeks=20)
            early = base.origin <= cutoff
            pd.testing.assert_frame_equal(base.loc[early, state_columns].reset_index(drop=True),
                                          alt.loc[early, state_columns].reset_index(drop=True))
            early_trace = base_trace.origin <= cutoff
            pd.testing.assert_frame_equal(
                base_trace.loc[early_trace, ['origin', 'deploy', 'n_labels']].reset_index(drop=True),
                alt_trace.loc[early_trace, ['origin', 'deploy', 'n_labels']].reset_index(drop=True))

    def test_deterministic(self):
        frame = synthetic()
        for direction in m.DIRECTIONS:
            a, _ = m.replay(frame, direction)
            b, _ = m.replay(frame, direction)
            pd.testing.assert_frame_equal(a, b)

    def test_min_labels_blocks_early_deploy(self):
        frame = synthetic()
        out, trace = m.replay(frame, 'eproc')
        self.assertTrue(trace[trace.n_labels < m.MIN_LABELS].deploy.eq(False).all())

    def test_e_process_is_a_nonnegative_martingale_like_statistic(self):
        self.assertAlmostEqual(m.eprocess_value([0.1, -0.2, 0.3]), m.eprocess_value([0.1, -0.2, 0.3]))
        self.assertGreater(m.eprocess_value([0.3] * 40), 1.0 / m.ALPHA)
        self.assertGreaterEqual(m.eprocess_value([-0.3] * 40), 0.0)
        self.assertLess(m.eprocess_value([-0.3] * 40), 1.0)

    def test_cs_lower_bound_is_negative_under_noise(self):
        self.assertLess(m.cs_lower([0.01, -0.01] * 30), 0)

    def test_error_cap_bounds_loss_differences(self):
        self.assertLessEqual(m.ape(np.array([5.0]), np.array([0.0]))[0], m.CAP)


if __name__ == '__main__':
    unittest.main()
