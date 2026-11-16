import numpy as np
import pandas as pd
import pytest

from analysis import causal_common as cc


def panel_data():
    rng = np.random.default_rng(12)
    weeks = pd.date_range(cc.START, periods=400, freq="W-SUN")
    values = {
        "price": np.exp(5 + np.cumsum(rng.normal(0, .02, len(weeks)))),
        "weekly": np.exp(rng.normal(2, .2, len(weeks))),
        "rain": rng.uniform(1, 20, len(weeks)),
        "step": np.repeat(rng.uniform(5, 15, 100), 4),
        "late": np.r_[np.full(300, np.nan), rng.normal(size=100)],
    }
    return pd.concat([pd.DataFrame({"week": weeks, "series": "coffee", "col_name": c,
                                   "value": v, "code": cc.PRECIP if c == "rain" else "FND"})
                      for c, v in values.items()], ignore_index=True), weeks


def test_future_cannot_change_training_transform_or_columns(monkeypatch):
    raw, weeks = panel_data()
    cutoff = weeks[300]
    monkeypatch.setattr(cc, "raw", raw)
    before, info = cc.make_panel("coffee", fit_end=cutoff)
    changed = raw.copy()
    changed.loc[changed.week >= cutoff, "value"] = -10000
    monkeypatch.setattr(cc, "raw", changed)
    after, other = cc.make_panel("coffee", fit_end=cutoff)
    pd.testing.assert_frame_equal(before.loc[:weeks[299]], after.loc[:weeks[299]])
    pd.testing.assert_frame_equal(info, other)
    assert before.columns.tolist() == after.columns.tolist()
    assert "late" not in before
    np.testing.assert_allclose(before.loc[:weeks[299]].mean(), 0, atol=1e-12)
    assert info.set_index("변수").loc["step", "변환"].startswith("계단형")


def test_training_start_controls_scaling_and_preserves_missing_weeks(monkeypatch):
    raw, weeks = panel_data()
    raw = raw[raw.week != weeks[200]]
    monkeypatch.setattr(cc, "raw", raw)
    X, _ = cc.make_panel("coffee", fit_end=weeks[300], train_start=weeks[100])
    assert X.index.equals(weeks)
    assert np.isnan(X.loc[weeks[200], "price"])
    assert np.isnan(X.loc[weeks[201], "price"])
    np.testing.assert_allclose(X.loc[weeks[100]:weeks[299]].mean(), 0, atol=1e-12)


def test_legacy_default_is_unchanged(monkeypatch):
    raw, _ = panel_data()
    monkeypatch.setattr(cc, "raw", raw)
    a, ai = cc.make_panel("coffee")
    b, bi = cc.make_panel("coffee", fit_end=None)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ai, bi)
    # Frozen legacy weekly-variable calculation: transformation uses full-sample log positivity.
    v = raw[raw.col_name == "weekly"].set_index("week").value
    expected = np.log(v).diff()
    expected = (expected - expected.mean()) / expected.std()
    np.testing.assert_allclose(a.weekly, expected, equal_nan=True)
    with pytest.raises(ValueError, match="requires fit_end"):
        cc.make_panel("coffee", train_start="2020")


def test_event_scaling_ignores_future(monkeypatch):
    raw, weeks = panel_data()
    monkeypatch.setattr(cc, "raw", raw)
    X, info = cc.make_panel("coffee", fit_end=weeks[300])
    wide = raw.pivot(index="week", columns="col_name", values="value")
    monkeypatch.setattr(cc, "lagged_values", lambda s: wide)
    a = cc.event_panel("coffee", X, info, fit_end=weeks[300])
    wide.loc[weeks[300]:, "step"] *= 10000
    b = cc.event_panel("coffee", X, info, fit_end=weeks[300])
    pd.testing.assert_frame_equal(a.loc[:weeks[299]], b.loc[:weeks[299]])


def test_link_assumptions_requested_lags():
    links = cc.link_assumptions(np.array([False, True]), tau_max=8)
    assert (1, -8) in links[0]
    assert (1, -9) not in links[0]
    assert links[0][1, 0] == "-?>"
