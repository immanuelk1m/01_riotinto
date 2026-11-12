import json

import numpy as np
import pytest

from src.models import MODELS
from src.models.ml.gaussian_process import GaussianProcess


def test_multioutput_predictions_are_finite_deterministic_and_audited():
    X = np.linspace(-2, 2, 24)[:, None]
    Y = 100 + np.sin(X) * np.arange(1, 9)[None, :]
    first = MODELS["gaussian_process"](optimizer=None).fit(X, Y)
    second = GaussianProcess(optimizer=None).fit(X, Y)
    prediction = first.predict(X[:3])
    assert prediction.shape == (3, 8)
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(prediction, second.predict(X[:3]))
    info = first.study_model_info
    assert info["n_train"] == 24 and info["n_targets"] == 8
    assert info["computational_threads"] == 1
    assert info["fit_seconds"] >= 0
    assert "Matern" in info["learned_kernel"]
    assert info["warnings"] == []
    json.dumps(info)


def test_constant_horizons_preserve_original_target_units():
    X = np.linspace(-1, 1, 12)[:, None]
    constants = np.arange(1, 9) * 1500.0
    model = GaussianProcess(optimizer=None).fit(X, np.tile(constants, (len(X), 1)))
    np.testing.assert_allclose(model.predict([[0.2], [8.0]]), np.tile(constants, (2, 1)))


def test_target_normalization_is_reversed_without_rescaling_inputs():
    X = np.linspace(10, 20, 20)[:, None]
    Y = np.sin(X) * np.arange(1, 9)[None, :]
    scales = np.arange(1, 9)[None, :] * 100.0
    offsets = np.arange(1, 9)[None, :] * 1000.0
    standard = GaussianProcess(optimizer=None).fit(X, Y)
    priced = GaussianProcess(optimizer=None).fit(X, Y * scales + offsets)
    np.testing.assert_allclose(priced.predict(X[:4]), standard.predict(X[:4]) * scales + offsets)
    np.testing.assert_array_equal(priced.model_.X_train_, X)


def test_default_optimizer_and_single_horizon_output():
    X = np.linspace(-1, 1, 12)[:, None]
    model = GaussianProcess().fit(X, np.sin(X))
    assert model.predict(X[:2]).shape == (2, 1)
    assert np.isfinite(model.predict(X[:2])).all()
    assert model.study_model_info["optimizer"] == "fmin_l_bfgs_b"
    assert model.model_.kernel_.k1.k2.nu == 1.5
    assert model.model_.alpha == 1e-6
    json.dumps(model.study_model_info)


@pytest.mark.parametrize("threads", [0, -1, True, 1.5])
def test_invalid_thread_count_fails(threads):
    with pytest.raises(ValueError, match="computational_threads"):
        GaussianProcess(computational_threads=threads)
