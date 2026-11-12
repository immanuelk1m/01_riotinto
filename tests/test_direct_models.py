import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.models.direct import DirectRegressor
from src.models.ml.lightgbm import LightGBM
from src.models.ml.xgb import XGB
from src.models.preprocessing import ScaledRegressor
from src.pipelines.modeling.nodes import fit, make_model


def inputs():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=list('abc'))
    Y = X.to_numpy() @ rng.normal(size=(3, 8))
    return X, Y


@pytest.mark.parametrize('estimator', [LinearRegression(),
    XGB(n_estimators=3, max_depth=2, n_jobs=1, random_state=0),
    LightGBM(n_estimators=3, min_child_samples=3, num_leaves=4, n_jobs=1,
             random_state=0, verbosity=-1, deterministic=True, force_col_wise=True)])
def test_independent_scalar_heads_order_pickle_and_audit(estimator):
    X, Y = inputs()
    model = DirectRegressor(estimator).fit(X, Y)
    assert len({id(head) for head in model.estimators_}) == 8
    assert all(head is not estimator for head in model.estimators_)
    prediction = model.predict(X)
    assert prediction.shape == (50, 8)
    for h, head in enumerate(model.estimators_):
        np.testing.assert_array_equal(prediction[:, h], head.predict(X))
    changed = Y.copy()
    changed[:, 3] += 100
    other = DirectRegressor(estimator).fit(X, changed)
    np.testing.assert_array_equal(prediction[:, [0, 1, 2, 4, 5, 6, 7]],
                                  other.predict(X)[:, [0, 1, 2, 4, 5, 6, 7]])
    assert not np.allclose(prediction[:, 3], other.predict(X)[:, 3])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(X), prediction)
    heads = model.study_model_info['heads']
    assert [head['horizon'] for head in heads] == list(range(1, 9))
    assert len({head['training_features_sha'] for head in heads}) == 1
    assert len({head['training_targets_sha'] for head in heads}) == 8
    assert all(head['n_train'] == 50 and head['n_features'] == 3 for head in heads)
    json.dumps(model.study_model_info, allow_nan=False)
    assert not hasattr(model, 'get_booster')


def test_factory_preserves_native_and_wraps_before_scaling():
    native = {'model': 'xgb', 'model_params': {'multi_strategy': 'multi_output_tree'}}
    assert isinstance(make_model(native), XGB)
    with pytest.raises(ValueError, match='multi_strategy'):
        make_model({**native, 'forecast_strategy': 'direct'})
    direct = {'model': 'lightgbm', 'forecast_strategy': 'direct'}
    assert make_model(direct).allow_nan
    scaled = make_model({**direct, 'preprocessing': {'scaling': 'standard'}})
    assert isinstance(scaled, ScaledRegressor)
    assert isinstance(scaled.model, DirectRegressor)
    assert isinstance(scaled.model.base_estimator, LightGBM)
    assert not scaled.allow_nan


@pytest.mark.parametrize('Y', [np.zeros(50), np.zeros((49, 8)), np.zeros((50, 0)), np.full((50, 8), np.nan)])
def test_invalid_targets_rejected(Y):
    X, _ = inputs()
    with pytest.raises(ValueError, match='targets'):
        DirectRegressor(LinearRegression()).fit(X, Y)


def test_direct_common_maturity_ffill_and_future_invariance():
    H, origin = 8, 40
    weeks = pd.date_range('2020-01-05', periods=55, freq='W-SUN')
    logp = pd.Series(4 + np.sin(np.arange(55) / 10) * .1)
    data = pd.DataFrame({'series': 'coffee', 'week': weeks, 'logp': logp,
                         'is_origin': np.arange(55) == origin,
                         'x': np.arange(55, dtype=float)})
    for h in range(1, H + 1):
        data[f'y{h}'] = logp.shift(-h) - logp
    data.loc[:1, 'x'] = np.nan
    data.loc[10, 'x'] = np.nan
    shared = {'target': 'logdiff', 'preprocessing': {'missing': 'ffill'},
              'lineage': {'root': 'base_xgb'}}
    specs = [{**shared, 'model': 'xgb', 'model_params': {'n_estimators': 2, 'n_jobs': 1,
               'random_state': 0, 'multi_strategy': 'multi_output_tree', 'tree_method': 'hist'}},
             {**shared, 'model': 'xgb', 'forecast_strategy': 'direct',
               'model_params': {'n_estimators': 2, 'n_jobs': 1, 'random_state': 0}},
             {**shared, 'model': 'lightgbm', 'forecast_strategy': 'direct',
               'model_params': {'n_estimators': 2, 'n_jobs': 1, 'verbosity': -1, 'random_state': 0}}]
    models = [next(iter(fit(make_model(spec), data, spec, H, {}).values())) for spec in specs]
    audits = [model.study_training_info for model in models]
    assert len({a['training_features_sha'] for a in audits}) == 1
    assert all(a['n_train'] == origin - H + 1 - 2 for a in audits)
    assert all(pd.Timestamp(a['training_end']) + pd.Timedelta(weeks=H) <= weeks[origin] for a in audits)
    for model, spec in zip(models[1:], specs[1:]):
        assert all(head['training_features_sha'] == model.study_training_info['training_features_sha']
                   and head['n_train'] == model.study_training_info['n_train']
                   for head in model.study_model_info['heads'])
        future = data.copy()
        future.loc[future.week > weeks[origin], ['x', 'logp', *[f'y{h}' for h in range(1, H + 1)]]] = 999
        refit = next(iter(fit(make_model(spec), future, spec, H, {}).values()))
        np.testing.assert_array_equal(model.predict(data.loc[[origin], ['x']]),
                                      refit.predict(data.loc[[origin], ['x']]))
