import pickle

import numpy as np
import pandas as pd
import pytest

from src.models.ml.xgb import XGB
from src.models.preprocessing import ScaledRegressor
from src.pipelines.modeling.nodes import fit, make_model, predict, prepare_input


H = 2
ORIGIN = 220


def frame():
    weeks = pd.date_range("2020-01-05", periods=230, freq="W-SUN")
    logp = pd.Series(4 + np.arange(len(weeks)) * 0.001)
    data = pd.DataFrame({"series": "coffee", "week": weeks, "logp": logp,
                         "is_origin": np.arange(len(weeks)) == ORIGIN,
                         "x": np.arange(len(weeks), dtype=float),
                         "z": np.sin(np.arange(len(weeks)) / 10)})
    for h in range(1, H + 1):
        data[f"y{h}"] = logp.shift(-h) - logp
    data.loc[:2, "x"] = np.nan
    data.loc[[12, 34, ORIGIN], "x"] = np.nan
    return data


def spec(model="ols"):
    params = {"xgb": {"n_estimators": 2, "max_depth": 2, "n_jobs": 1},
              "gaussian_process": {"optimizer": None}}
    return {"model": model, "model_params": params.get(model, {}), "target": "level",
            "lineage": {"root": "base_xgb"},
            "preprocessing": {"missing": "ffill", "scaling": "standard"}}


def test_ffill_is_chronological_per_series_preserving_targets_and_row_identity():
    data = frame().iloc[:6].copy()
    other = data.assign(series="gold", x=[100., np.nan, np.nan, 103., np.nan, 105.])
    data = pd.concat([data, other]).sample(frac=1, random_state=7)
    original = data.copy(deep=True)
    result = prepare_input(data, spec(), H)
    pd.testing.assert_frame_equal(data, original)
    pd.testing.assert_frame_equal(result.drop(columns=["x", "z"]), original.drop(columns=["x", "z"]))
    coffee = result[result.series == "coffee"].sort_values("week")
    gold = result[result.series == "gold"].sort_values("week")
    assert coffee.x.iloc[:3].isna().all()
    np.testing.assert_allclose(coffee.x.iloc[3:], [3., 4., 5.])
    np.testing.assert_allclose(gold.x, [100., 100., 100., 103., 103., 105.])


def test_native_preprocessing_preserves_xgb_missing_rows_and_factory_behavior():
    data = frame()
    native = {"model": "xgb", "model_params": {"n_estimators": 2, "n_jobs": 1}}
    assert prepare_input(data, native, H) is data
    assert isinstance(make_model(native), XGB)
    fitted = fit(make_model(native), data, native, H, {})
    model = next(iter(fitted.values()))
    assert model.study_training_info["n_train"] == ORIGIN - H + 1
    assert model.study_training_info["filled_cells"] == 0
    assert np.isfinite(predict(fitted, data, native, H).fc).all()


def test_ffill_unresolved_inference_missing_fails_without_backfill():
    data = frame()
    # A future-only feature remains unknown at this origin in the legacy input path.
    data["late"] = np.nan
    data.loc[ORIGIN + 1:, "late"] = 42.
    config = spec("xgb")
    config.pop("lineage")
    with pytest.raises(ValueError, match="ffill 후 추론 피처 결측"):
        fit(make_model(config), data, config, H, {})


def test_gp_and_xgb_use_identical_mature_rows_and_training_only_scalers():
    data = frame()
    expected = prepare_input(data, spec(), H).iloc[3:ORIGIN - H + 1][["x", "z"]]
    models = []
    for name in ["xgb", "gaussian_process"]:
        config = spec(name)
        fitted = fit(make_model(config), data, config, H, {})
        model = next(iter(fitted.values()))
        assert isinstance(model, ScaledRegressor)
        info = model.study_training_info
        assert info["n_train"] == len(expected)
        assert info["excluded_missing_rows"] == 3
        assert info["training_end"] == str(data.week.iloc[ORIGIN - H].date())
        np.testing.assert_allclose(model.scaler_.mean_, expected.mean())
        np.testing.assert_allclose(model.scaler_.scale_, expected.std(ddof=0))
        forecast = predict(fitted, data, config, H)
        assert np.isfinite(forecast.fc).all()
        np.testing.assert_allclose(forecast.actual, np.exp(data.logp.iloc[ORIGIN + 1:ORIGIN + H + 1]))
        np.testing.assert_allclose(forecast.base, np.exp(data.logp.iloc[ORIGIN]))
        models.append(model)
    assert models[0].study_training_info["training_features_sha"] == models[1].study_training_info["training_features_sha"]
    np.testing.assert_array_equal(models[0].scaler_.mean_, models[1].scaler_.mean_)


def test_future_features_cannot_change_scaler_fitted_model_or_forecast():
    original = frame()
    changed = original.copy()
    changed.loc[ORIGIN + 1:, ["x", "z"]] = 1e9
    config = spec()
    results = [fit(make_model(config), data, config, H, {}) for data in [original, changed]]
    left, right = [next(iter(result.values())) for result in results]
    np.testing.assert_array_equal(left.scaler_.mean_, right.scaler_.mean_)
    np.testing.assert_array_equal(left.model.beta, right.model.beta)
    assert left.study_training_info == right.study_training_info
    pd.testing.assert_frame_equal(predict(results[0], original, config, H),
                                  predict(results[1], changed, config, H))


def test_scaled_wrapper_pickle_roundtrip_preserves_predictions():
    data = frame()
    config = spec()
    fitted = fit(make_model(config), data, config, H, {})
    restored = pickle.loads(pickle.dumps(fitted))
    pd.testing.assert_frame_equal(predict(fitted, data, config, H), predict(restored, data, config, H))


def test_origin_checkpoint_resumes_without_refit_and_rejects_corruption(tmp_path, monkeypatch):
    data = frame()
    config = spec("gaussian_process")
    fitted = fit(make_model(config), data, config, H, {}, checkpoint_dir=tmp_path)

    def unexpected_fit(*args, **kwargs):
        pytest.fail("an intact origin checkpoint must avoid refitting")

    monkeypatch.setattr(ScaledRegressor, "fit", unexpected_fit)
    resumed = fit(make_model(config), data, config, H, {}, checkpoint_dir=tmp_path)
    pd.testing.assert_frame_equal(predict(fitted, data, config, H), predict(resumed, data, config, H))
    checkpoint = next(tmp_path.glob("*.pkl"))
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checkpoint 손상"):
        fit(make_model(config), data, config, H, {}, checkpoint_dir=tmp_path)
