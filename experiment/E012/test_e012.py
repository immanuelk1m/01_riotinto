"""Causality and archive alignment tests; no production outputs or DB writes."""
import importlib.util
import tempfile
import unittest
import sqlite3
from contextlib import closing
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("e012", Path(__file__).with_name("E012.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def synthetic(n=90):
    rows = []
    for i, origin in enumerate(pd.date_range("2020-01-05", periods=n, freq="7D")):
        for h in range(1, 9):
            row = dict(origin=origin, target_date=origin + pd.Timedelta(weeks=h), horizon=h,
                       y_true=100 * np.exp(0.01 * np.sin(i + h)), last_price=100., n_train=100)
            row.update({name: 100 * np.exp(0.005 * k * np.cos(i)) for k, name in enumerate(m.EXPERTS)})
            rows.append(row)
    return pd.DataFrame(rows)


class CausalHedgeTests(unittest.TestCase):
    def test_unavailable_labels_cannot_change_issued_forecasts(self):
        frame = synthetic()
        cutoff = frame.origin.unique()[45]
        a, wa = m.aggregate(frame)
        changed = frame.copy()
        changed.loc[changed.target_date > cutoff, "y_true"] *= 5
        b, wb = m.aggregate(changed)
        mask = a.origin <= cutoff
        np.testing.assert_array_equal(a.loc[mask, "y_pred"], b.loc[mask, "y_pred"])
        pd.testing.assert_frame_equal(wa.loc[mask], wb.loc[mask])

    def test_horizon_specific_release_and_convex_weights(self):
        out, weights = m.aggregate(synthetic(20))
        for h, group in out.groupby("horizon"):
            np.testing.assert_array_equal(group.n_updates, np.maximum(np.arange(20) - h + 1, 0))
        valid = weights.last_update_target.notna()
        self.assertTrue((weights.loc[valid, "last_update_target"] <= weights.loc[valid, "origin"]).all())
        w = weights[[f"weight_{name}" for name in m.EXPERTS]].to_numpy()
        np.testing.assert_allclose(w.sum(1), 1)
        self.assertTrue((w > 0).all())
        self.assertTrue((out.y_pred >= out[m.EXPERTS].min(axis=1) - 1e-10).all())
        self.assertTrue((out.y_pred <= out[m.EXPERTS].max(axis=1) + 1e-10).all())

    def test_missing_or_duplicate_forecasts_rejected(self):
        df = synthetic(20)
        for bad in (df.iloc[1:], pd.concat([df, df.iloc[:1]]), df[df.origin != df.origin.unique()[5]]):
            with self.assertRaises(ValueError):
                m.aggregate(bad)

    def test_real_archive_alignment_and_future_transfer(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for exp, direction in (("E009", "gru"), ("E010", "gru_reg")):
                (Path(directory) / exp).symlink_to(root / exp / "results" / direction, target_is_directory=True)
            for commodity in m.TARGETS:
                source = m.load(commodity, directory)
                out, weights = m.aggregate(source)
                self.assertEqual(len(out), len(source))
                self.assertEqual(len(weights), len(source))
                m.validate_frame(out)
                transfer = m.transfer_diagnostics(out)
                self.assertFalse(transfer.empty)
                self.assertTrue(transfer.diagnostic_only.all())

    def test_complete_run_and_atomic_leaderboard(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch.object(m, "RESULTS", base / "results"), patch.object(m, "LOGS", base / "logs"):
                for commodity in m.TARGETS:
                    m.run(commodity)
                m.leaderboard(base / "scores.sqlite")
                with closing(sqlite3.connect(base / "scores.sqlite")) as con:
                    self.assertEqual(con.execute("SELECT count(*) FROM leaderboard").fetchone()[0], 90)
                file = m.RESULTS / "predictions_gold.parquet"
                df = pd.read_parquet(file)
                df.iloc[8:].to_parquet(file, index=False)
                with self.assertRaises(ValueError):
                    m.leaderboard(base / "scores.sqlite")
                with closing(sqlite3.connect(base / "scores.sqlite")) as con:
                    self.assertEqual(con.execute("SELECT count(*) FROM leaderboard").fetchone()[0], 90)


if __name__ == "__main__":
    unittest.main()
