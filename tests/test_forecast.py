import numpy as np
import pandas as pd
import pytest

from src.pipelines.data_processing.nodes import make_targets
from src.pipelines.feature_engineering.nodes import build_features, make_tabular, make_timeseries
from src.pipelines.modeling.nodes import fit, make_model, predict, select_input
from src.pipelines.reporting.nodes import _features, evaluate, run_config, run_id
from src.utils import KEYS

H, START = 8, "2024-01-01"
NAIVE = {"model": "naive"}
SEASONAL = {"model": "seasonal_naive", "features": ["seasonal"]}
WEEKS = pd.date_range("2020-01-05", periods=300, freq="W-SUN")
NO_EXOG = pd.DataFrame(columns=["series", "col_name", "week", "value", "code", "drivers", "ml"])


def exog_rows(series: str, col: str, code: str, source: str, first: int = 0) -> pd.DataFrame:
    """`first` 번째 주부터 값이 있는 외생변수 하나."""
    weeks = WEEKS[first:]
    return pd.DataFrame({"series": series, "col_name": col, "week": weeks, "value": np.sin(np.arange(len(weeks))),
                         "code": code, "drivers": source == "drivers", "ml": source == "ml"})


def model_input(spec: dict, targets: pd.DataFrame, exog: pd.DataFrame = NO_EXOG) -> pd.DataFrame:
    features, columns = build_features(targets, exog, {"e": spec}, {}, H)
    return select_input(make_tabular(features, targets), columns, spec, H)


def run(spec: dict, value: np.ndarray):
    weeks = pd.date_range("2020-01-05", periods=len(value), freq="W-SUN")
    targets = make_targets(pd.DataFrame({"series": "x", "week": weeks, "value": value}), H, START)
    frame = model_input(spec, targets)
    forecasts = predict(fit(make_model(spec), frame, H), frame, H)
    return forecasts, evaluate(forecasts, 1.0)


def test_experiments_share_one_table_and_pick_their_own_columns():
    targets = make_targets(pd.DataFrame({"series": "x", "week": WEEKS, "value": 7.0}), H, START)
    lags = lambda k: {"model": "ols", "features": ["lags"], "feature_params": {"lags": {"n": k}}}  # noqa: E731
    features, columns = build_features(targets, NO_EXOG, {"a": lags(2), "b": lags(3), "c": NAIVE}, {}, H)
    tabular = make_tabular(features, targets)
    picked = lambda spec: list(_features(select_input(tabular, columns, spec, H), H).columns[2:])  # noqa: E731
    assert [picked(lags(2)), picked(lags(3)), picked(NAIVE)] == [["lag0", "lag1"], ["lag0", "lag1", "lag2"], []]
    with pytest.raises(ValueError, match="없는 피처"):
        select_input(tabular, columns, lags(4), H)
    seasonal = lambda k: {"model": "ols", "features": ["seasonal"], "feature_params": {"seasonal": {"season": k}}}  # noqa: E731
    with pytest.raises(ValueError, match="다른 값"):           # 같은 열 이름 s1.. 에 다른 값
        build_features(make_targets(pd.DataFrame({"series": "x", "week": WEEKS, "value": np.arange(1.0, 301)}), H, START),
                       NO_EXOG, {"a": seasonal(52), "b": seasonal(26)}, {}, H)


def test_timeseries_input_is_the_calendar_grid_in_long_format():
    targets = make_targets(pd.DataFrame({"series": "x", "week": WEEKS, "value": 7.0}), H, START)
    spec = {"model": "ols", "features": ["lags"], "feature_params": {"lags": {"n": 2}}}
    ts = make_timeseries(build_features(targets, NO_EXOG, {"e": spec}, {}, H)[0], targets)
    assert list(ts.columns) == ["unique_id", "ds", "y", "lag0", "lag1"]
    assert len(ts) == len(targets) and np.array_equal(ts.y, targets.logp) and (ts.ds == targets.week).all()


def test_naive_is_exact_on_constant_price():
    _, metrics = run(NAIVE, np.full(300, 7.0))
    assert len(metrics) == H and np.allclose(metrics.mape, 0)


def test_seasonal_naive_is_exact_on_periodic_price():
    value = 10 + np.sin(2 * np.pi * np.arange(300) / 52)
    _, seasonal = run(SEASONAL, value)
    assert np.allclose(seasonal.mape, 0, atol=1e-9) and seasonal.mape_naive.mean() > 1
    assert np.allclose(run(NAIVE, value)[1].gain_pct, 0)


def test_origins_are_first_sunday_of_each_month_from_start():
    forecasts, _ = run(NAIVE, np.full(300, 7.0))
    origins = pd.DatetimeIndex(forecasts.origin.unique())
    assert (origins.day_of_week == 6).all() and (origins.day <= 7).all()
    assert origins.min() == pd.Timestamp("2024-01-07")
    assert origins.to_period("M").is_unique and len(origins) == len(pd.period_range(origins.min(), origins.max(), freq="M"))


def test_ols_learns_a_linear_rule():
    value = np.exp(0.05 * (np.arange(300) % 2))         # 수익률이 매주 부호만 바뀜: y_h 는 lag0 의 선형함수
    _, metrics = run({"model": "ols", "features": ["lags"], "feature_params": {"lags": {"n": 1}}}, value)
    assert np.allclose(metrics.mape, 0, atol=1e-6) and metrics.mape_naive.mean() > 1


def test_forecast_at_origin_ignores_the_future():
    rng = np.random.default_rng(0)
    value = np.exp(np.cumsum(rng.normal(0, 0.03, 300)))
    changed = value.copy()
    changed[250:] *= 3                                  # 250주차(2024-10-20) 이후만 바꾼다
    spec = {"model": "ols", "features": ["lags", "seasonal"]}
    before, after = (run(spec, v)[0] for v in (value, changed))
    cutoff = pd.Timestamp("2020-01-05") + pd.Timedelta(weeks=249)
    past = lambda f: f[f.origin <= cutoff].fc.to_numpy()  # noqa: E731
    assert len(past(before)) > 0 and np.array_equal(past(before), past(after))


def test_run_id_follows_spec_and_data():
    weeks = pd.date_range("2020-01-05", periods=300, freq="W-SUN")
    frame = lambda v: make_targets(pd.DataFrame({"series": "x", "week": weeks, "value": v}), H, START)  # noqa: E731
    params = {"horizon": H, "experiments": {"ignored": {}}}
    rid = lambda spec, v: run_id(  # noqa: E731
        run_config("e", spec, params, frame(v), _features(model_input(spec, frame(v)), H)))
    flat, bumped = np.full(300, 7.0), np.r_[np.full(299, 7.0), 8.0]
    assert rid(NAIVE, flat) == rid(dict(NAIVE), flat.copy())
    assert len({rid(NAIVE, flat), rid(SEASONAL, flat), rid(NAIVE, bumped)}) == 3


def test_exog_selects_by_source_and_code_and_respects_start():
    targets = make_targets(pd.DataFrame({"series": "x", "week": WEEKS, "value": 7.0}), H, START)
    exog = pd.concat([
        exog_rows("x", "a", "MAC.INF.PRODUCER", "drivers"),
        exog_rows("x", "b", "PFM.FX.BILATERAL", "ml"),
        exog_rows("x", "late", "MAC.ACT.OUTPUT", "ml", first=100),   # start 에 아직 값이 없다 → 빠진다
        exog_rows("other", "c", "MAC.INF.PRODUCER", "ml"),           # 다른 시리즈의 변수
    ])
    def cols(**params):
        spec = {"model": "ols", "features": ["exog"], "feature_params": {"exog": params}}
        features, _ = build_features(targets, exog, {"e": spec}, {"x": {"start": str(WEEKS[50].date())}}, H)
        assert features.loc[features.week < WEEKS[50]].drop(columns=KEYS).isna().all(axis=None)
        assert features.loc[features.week >= WEEKS[50]].notna().all(axis=None)
        return set(features.columns) - set(KEYS)

    assert cols() == {"a", "b"}
    assert cols(source="drivers") == {"a"} and cols(source="ml") == {"b"}
    assert cols(codes=["MAC"]) == {"a"} and cols(codes=["PFM.FX"]) == {"b"}
    assert cols(exclude=["MAC.INF.PRODUCER"]) == {"b"}


def test_series_with_different_exog_columns_all_get_forecasts():
    value = np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.03, 300)))
    targets = make_targets(
        pd.concat([pd.DataFrame({"series": s, "week": WEEKS, "value": value}) for s in ("x", "y")]), H, START)
    exog = pd.concat([exog_rows("x", "a", "MAC.INF.PRODUCER", "ml"), exog_rows("y", "b", "PFM.FX.BILATERAL", "ml")])
    spec = {"model": "ols", "features": ["exog"]}
    frame = model_input(spec, targets, exog)
    forecasts = predict(fit(make_model(spec), frame, H), frame, H)
    origins = run(NAIVE, value)[0].origin.nunique()              # 외생변수 없이 얻는 origin 수
    assert origins > 0 and forecasts.groupby("series").origin.nunique().to_dict() == {"x": origins, "y": origins}
