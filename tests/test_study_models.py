import numpy as np
import pandas as pd
import pytest

from src.features import exog, lags
from src.models.statistical.drift import Drift
from src.pipelines.data_processing.nodes import make_targets
from src.pipelines.feature_engineering.nodes import build_features, make_tabular
from src.pipelines.modeling.nodes import fit, make_model, predict, select_input


H = 8
WEEKS = pd.date_range("2020-01-05", periods=320, freq="W-SUN")
EMPTY_EXOG = pd.DataFrame(columns=["series", "col_name", "week", "value", "code", "drivers", "ml"])


def prices(values=None, series="x"):
    if values is None:
        values = np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, len(WEEKS))))
    return pd.DataFrame({"series": series, "week": WEEKS, "value": values})


def driver(series, col, values=None, first=0):
    values = np.sin(np.arange(len(WEEKS))) if values is None else values
    return pd.DataFrame({"series": series, "week": WEEKS[first:], "col_name": col,
                         "value": values[first:], "code": "PFM.FX", "drivers": True, "ml": True})


def inputs(spec, raw=None, external=EMPTY_EXOG):
    targets = make_targets(prices() if raw is None else raw, H, "2024-01-01")
    features, columns = build_features(targets, external, {"e": spec}, {}, H)
    return select_input(make_tabular(features, targets), columns, spec, H)


def test_offsets_preserve_lag_meaning_and_default():
    logp = pd.Series(np.arange(70, dtype=float) ** 2, index=pd.date_range("2020-01-05", periods=70, freq="W-SUN"))
    chosen = lags.build(logp, H, offsets=[0, 1, 3, 11, 51])
    assert list(chosen) == ["lag0", "lag1", "lag3", "lag11", "lag51"]
    assert chosen.iloc[-1].tolist() == [137.0, 135.0, 131.0, 115.0, 35.0]
    pd.testing.assert_frame_equal(lags.build(logp, H), lags.build(logp, H, offsets=list(range(8))))


@pytest.mark.parametrize("offsets", [[], [-1], [0, 0], [0.5], [True]])
def test_invalid_offsets_fail(offsets):
    with pytest.raises(ValueError, match="offsets"):
        lags.build(pd.Series([0.0, 1.0]), H, offsets=offsets)


def test_scoped_columns_do_not_leak_from_shared_driver_table():
    targets = make_targets(pd.concat([prices(series="gold"), prices(series="coffee")]), H, "2024-01-01")
    external = pd.concat([driver("gold", "usd"), driver("coffee", "usd")])
    scoped = {"model": "xgb", "features": ["lags", "exog"],
              "feature_params": {"exog": {"columns_by_series": {"gold": ["usd"], "coffee": []}}}}
    full = {"model": "xgb", "features": ["lags", "exog"]}
    features, columns = build_features(targets, external, {"scoped": scoped, "full": full}, {}, H)
    selected = select_input(make_tabular(features, targets), columns, scoped, H)
    scoped_cols = [c for c in selected if c.startswith("usd")]
    assert len(scoped_cols) == 1 and "__scope_" in scoped_cols[0]
    assert selected.loc[selected.series == "coffee", scoped_cols].isna().all(axis=None)
    assert selected.loc[selected.series == "gold", scoped_cols].notna().all(axis=None)
    assert features.loc[features.series == "coffee", "usd"].notna().all()


def test_scoped_columns_are_exact_and_missing_selection_fails():
    logp = pd.Series(0.0, index=WEEKS)
    external = pd.concat([driver("x", "usd"), driver("x", "vix")])
    external.attrs["series"] = "x"
    selected = exog.build(logp, H, external, columns_by_series={"x": ["vix"]}, diff=4)
    assert len(selected.columns) == 1 and selected.columns[0].startswith("vix__scope_")
    assert selected.columns[0].endswith("__d4")
    np.testing.assert_allclose(selected.iloc[4:, 0], np.sin(np.arange(320))[4:] - np.sin(np.arange(320))[:-4])
    with pytest.raises(ValueError, match="요청한 외생변수"):
        exog.build(logp, H, external, columns_by_series={"x": ["missing"]})
    with pytest.raises(ValueError, match="명시적 열 목록"):
        exog.build(logp, H, external, columns_by_series={"other": []})
    assert exog.build(logp, H, external, columns_by_series={"x": []}).shape == (320, 0)


def test_rolling_uses_calendar_cutoff_mature_labels_and_start():
    raw = prices()
    raw.loc[240, "value"] = np.nan
    spec = {"model": "drift", "train_window_weeks": 52, "train_start": str(WEEKS[200].date()),
            "lineage": {"root": "base_xgb"}}
    frame = inputs(spec, raw)
    fitted = fit(make_model(spec), frame, spec, H, {})
    assert fitted
    ycols = [f"y{h}" for h in range(1, H + 1)]
    for (_, origin), model in fitted.items():
        eligible = frame[(frame.week >= origin - pd.Timedelta(weeks=52))
                         & (frame.week <= origin - pd.Timedelta(weeks=H))
                         & (frame.week >= WEEKS[200])].dropna(subset=ycols)
        info = model.study_training_info
        assert info["n_train"] == len(eligible)
        assert info["training_start"] == str(eligible.week.min().date())
        assert info["training_end"] == str(eligible.week.max().date())
        assert info["cutoff"] == str((origin - pd.Timedelta(weeks=H)).date())
        np.testing.assert_allclose(model.mean_return_, eligible[ycols].mean().to_numpy(), atol=1e-14)
    expected_origins = frame.loc[frame.is_origin & frame[ycols].notna().all(axis=1), "week"]
    assert {origin for _, origin in fitted} == set(expected_origins)


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_invalid_rolling_window_fails(window):
    spec = {"model": "drift", "train_window_weeks": window}
    with pytest.raises(ValueError, match="train_window_weeks"):
        fit(make_model(spec), inputs(spec), spec, H, {})


def test_drift_is_horizon_mean_and_ignores_missing_features():
    X = pd.DataFrame({"unused": [np.nan, np.nan, np.nan]})
    Y = np.array([[0.01, 0.05], [0.02, 0.04], [0.03, 0.03]])
    model = Drift().fit(X, Y)
    np.testing.assert_allclose(model.predict(X.iloc[:2]), [[0.02, 0.04], [0.02, 0.04]])
    with pytest.raises(ValueError, match="타겟"):
        Drift().fit(X.iloc[:0], Y[:0])


@pytest.mark.parametrize("model", ["ols", "xgb"])
def test_study_fit_and_feature_selection_ignore_future_prices_and_exog(model):
    raw = prices()
    external = pd.concat([driver("x", "a"), driver("x", "future", first=260)])
    changed_raw = raw.copy()
    changed_raw.loc[250:, "value"] *= 4
    changed_exog = external.copy()
    changed_exog.loc[changed_exog.week >= WEEKS[250], "value"] = np.nan
    spec = {"model": model, "features": ["lags", "exog"], "train_window_weeks": 104,
            "lineage": {"root": "base_xgb"}}
    if model == "xgb":
        spec["model_params"] = {"n_estimators": 5, "max_depth": 2, "random_state": 0}
    outputs = []
    models = []
    cutoff = WEEKS[249]
    for source, drivers in [(raw, external), (changed_raw, changed_exog)]:
        frame = inputs(spec, source, drivers)
        frame.loc[frame.week > cutoff, "is_origin"] = False
        fitted = fit(make_model(spec), frame, spec, H, {})
        outputs.append(predict(fitted, frame, spec, H))
        models.append(fitted)
    before, after = [out.loc[out.origin <= cutoff, ["series", "origin", "horizon", "fc"]] for out in outputs]
    assert len(before) > 0
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
    for key in models[0]:
        if key[1] <= cutoff:
            assert "future" not in models[0][key].study_feature_columns
            assert models[0][key].study_feature_columns == models[1][key].study_feature_columns
