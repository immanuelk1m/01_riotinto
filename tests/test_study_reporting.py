import numpy as np
import pandas as pd
import pytest

from src.pipelines.reporting.nodes import evaluate, run_summary
from src.study_reporting import (
    _moving_block_indices,
    aligned_forecasts,
    annual_comparison,
    compare_forecasts,
    holm_adjust,
)


def forecasts(n=12):
    rows = pd.MultiIndex.from_product(
        [["coffee", "gold"], pd.date_range("2024-01-01", periods=n, freq="MS"), [1, 2]],
        names=["series", "origin", "horizon"],
    ).to_frame(index=False)
    return rows.assign(fc=115.0, actual=100.0, base=110.0)


def test_alignment_is_order_independent_and_does_not_mutate_inputs():
    reference = forecasts()
    candidate = reference.sample(frac=1, random_state=3)
    original = candidate.copy()
    left, right = aligned_forecasts(candidate, reference)
    pd.testing.assert_frame_equal(left, right)
    pd.testing.assert_frame_equal(candidate, original)


@pytest.mark.parametrize("corruption,match", [
    (lambda frame: frame.iloc[:-1], "different cell keys"),
    (lambda frame: pd.concat([frame, frame.iloc[:1]]), "duplicate cell keys"),
    (lambda frame: frame.assign(actual=101.0), "different actual"),
    (lambda frame: frame.assign(base=111.0), "different base"),
    (lambda frame: frame.assign(fc=np.nan), "finite and positive"),
    (lambda frame: frame.assign(fc=-1.0), "finite and positive"),
    (lambda frame: frame.assign(actual=np.inf), "finite and positive"),
    (lambda frame: frame.assign(horizon=0), "positive integers"),
])
def test_alignment_rejects_invalid_comparisons(corruption, match):
    reference = forecasts()
    with pytest.raises(ValueError, match=match):
        aligned_forecasts(corruption(reference), reference)


def test_identical_forecasts_have_zero_difference_and_no_spurious_evidence():
    frame = forecasts()
    result = compare_forecasts(frame, frame, bootstrap_samples=80)
    assert len(result) == (1 + 2 + 4) * 3
    assert (result[["delta_loss_pct", "ci_low", "ci_high", "gain_pct"]] == 0).all(axis=None)
    assert (result.bootstrap_pvalue == 1).all()
    perfect = frame.assign(fc=frame.actual)
    result = compare_forecasts(perfect, perfect, bootstrap_samples=80)
    assert result.gain_pct.isna().all()
    assert (result.bootstrap_pvalue == 1).all()


def test_point_estimates_match_existing_reporting_and_keep_uncapped_worst_loss():
    candidate, reference = forecasts(), forecasts()
    candidate.loc[0, "fc"] = 300.0
    reference["fc"] = reference.base
    result = compare_forecasts(candidate, reference, bootstrap_samples=80, block_lengths=(3,))
    row = result[result.scope == "overall"].iloc[0]
    expected = run_summary(evaluate(candidate, 1.0))
    assert row.mape == pytest.approx(expected["mape"])
    assert row.reference_mape == pytest.approx(expected["mape_naive"])
    assert row.gain_pct == pytest.approx(expected["gain_pct"])
    assert row.max_ape_pct == 200.0
    assert row.n_origins == 12 and row.n_cells == 48


def test_block_indices_preserve_adjacent_origin_runs():
    indices = _moving_block_indices(11, 100, 3, np.random.default_rng(4))
    assert indices.shape == (100, 11)
    for start in (0, 3, 6):
        assert (np.diff(indices[:, start:start + 3], axis=1) == 1).all()
    assert (np.diff(indices[:, 9:11], axis=1) == 1).all()
    assert indices.min() >= 0 and indices.max() < 11


def test_all_series_and_horizons_are_resampled_jointly():
    candidate, reference = forecasts(), forecasts()
    # Opposite contemporaneous changes cancel in the whole origin vector.
    shocks = np.array([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6], dtype=float)
    origin_index = pd.factorize(candidate.origin, sort=True)[0]
    direction = np.where(candidate.series == "coffee", 1, -1)
    candidate["fc"] += shocks[origin_index] * direction * candidate.horizon
    result = compare_forecasts(candidate, reference, bootstrap_samples=300, block_lengths=(3,))
    overall = result[result.scope == "overall"].iloc[0]
    assert overall.delta_loss_pct == pytest.approx(0, abs=1e-12)
    assert overall.ci_low == pytest.approx(0, abs=1e-12)
    assert overall.ci_high == pytest.approx(0, abs=1e-12)
    coffee = result[(result.scope == "horizon") & (result.series == "coffee")].sort_values("horizon")
    assert coffee.iloc[1].ci_low == pytest.approx(2 * coffee.iloc[0].ci_low)
    assert coffee.iloc[1].ci_high == pytest.approx(2 * coffee.iloc[0].ci_high)
    assert coffee.iloc[0].ci_high > coffee.iloc[0].ci_low


def test_bootstrap_is_reproducible_and_one_sided_for_improvement():
    candidate, reference = forecasts(), forecasts()
    candidate["fc"] = 105.0
    first = compare_forecasts(candidate, reference, bootstrap_samples=200)
    pd.testing.assert_frame_equal(first, compare_forecasts(candidate, reference, bootstrap_samples=200))
    assert (first.delta_loss_pct < 0).all()
    assert (first.ci_high < 0).all()
    assert (first.bootstrap_pvalue == 1 / 201).all()
    worse = compare_forecasts(reference, candidate, bootstrap_samples=200)
    assert (worse.bootstrap_pvalue == 1).all()


def test_unbalanced_and_nonmonthly_origin_panels_fail():
    frame = forecasts()
    partial = frame.iloc[:-1]
    with pytest.raises(ValueError, match="incomplete origin panel"):
        compare_forecasts(partial, partial, bootstrap_samples=10)
    missing_pair = frame[~((frame.series == "coffee") & (frame.horizon == 2))]
    with pytest.raises(ValueError, match="incomplete origin panel"):
        compare_forecasts(missing_pair, missing_pair, bootstrap_samples=10)
    missing_month = frame[frame.origin != pd.Timestamp("2024-03-01")]
    with pytest.raises(ValueError, match="consecutive calendar month"):
        compare_forecasts(missing_month, missing_month, bootstrap_samples=10)


def test_short_panels_clip_and_deduplicate_effective_block_lengths():
    frame = forecasts(1)
    result = compare_forecasts(frame, frame, bootstrap_samples=10)
    assert result.block_length.unique().tolist() == [1]
    assert (result.n_origins == 1).all()


def test_annual_comparison_separates_years_and_series():
    candidate, reference = forecasts(24), forecasts(24)
    candidate.loc[candidate.origin.dt.year == 2024, "fc"] = 105.0
    candidate.loc[candidate.origin.dt.year == 2025, "fc"] = 125.0
    result = annual_comparison(candidate, reference)
    assert len(result) == 6
    overall = result[result.scope == "overall"].set_index("year")
    assert overall.loc[2024, "mape"] == pytest.approx(5)
    assert overall.loc[2025, "mape"] == pytest.approx(25)
    assert overall.loc[2024, "gain_pct"] == pytest.approx(100 * (1 - 5 / 15))
    assert (overall.n_origins == 12).all()
    assert (overall.n_cells == 48).all()


def test_holm_adjust_known_values_order_and_missing_tests():
    np.testing.assert_allclose(holm_adjust([0.04, 0.001, 0.03, 0.2]), [0.09, 0.004, 0.09, 0.2])
    np.testing.assert_allclose(holm_adjust([0.01, np.nan, 0.02]), [0.03, np.nan, 0.04], equal_nan=True)
    assert holm_adjust([]).size == 0
    with pytest.raises(ValueError, match="within"):
        holm_adjust([1.1])
