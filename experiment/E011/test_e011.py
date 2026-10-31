import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("e011", Path(__file__).with_name("E011.py"))
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def fixture():
    rows = []
    for i, origin in enumerate(pd.date_range("2020-01-05", periods=50, freq="W-SUN")):
        for h in range(1, 9):
            rows.append(dict(origin=origin, target_date=origin + pd.Timedelta(weeks=h),
                             horizon=h, y_true=100 * np.exp(0.01 * h + 0.001 * i),
                             last_price=100.0, y_raw=100 * np.exp(0.005 * h), n_train=100))
    return pd.DataFrame(rows)


class CausalityTests(unittest.TestCase):
    def test_future_targets_cannot_change_issued_forecasts(self):
        source = fixture()
        cutoff = source.origin.sort_values().unique()[35]
        changed = source.copy()
        changed.loc[changed.target_date > cutoff, "y_true"] *= 10
        first, _ = e.process(source)
        second, _ = e.process(changed)
        pd.testing.assert_frame_equal(first.loc[first.origin <= cutoff, ["y_pred", "y_equal"]],
                                      second.loc[second.origin <= cutoff, ["y_pred", "y_equal"]])

    def test_each_horizon_waits_for_own_target(self):
        source = fixture()
        result, diagnostic = e.process(source)
        for row in diagnostic.itertuples():
            expected = source[(source.horizon == row.horizon) & (source.target_date <= row.origin)]
            self.assertEqual(row.n_updates, len(expected))
        warmup = result.n_updates < e.WARMUP
        np.testing.assert_allclose(result.loc[warmup, "y_pred"], result.loc[warmup, "y_raw"])

    def test_missing_labels_never_update(self):
        source = fixture()
        source["y_true"] = np.nan
        result, diagnostic = e.process(source)
        self.assertTrue(diagnostic.n_updates.eq(0).all())
        np.testing.assert_allclose(result.y_pred, result.y_raw)

    def test_deleted_origin_is_rejected(self):
        source = fixture()
        source = source[source.origin != source.origin.unique()[20]]
        with self.assertRaisesRegex(ValueError, "Missing weekly origins"):
            e.process(source)

    def test_result_must_match_all_source_keys(self):
        source = fixture()
        result, _ = e.process(source)
        with self.assertRaises((AssertionError, ValueError)):
            e.validate(result[result.origin != result.origin.max()], source)


if __name__ == "__main__":
    unittest.main()
