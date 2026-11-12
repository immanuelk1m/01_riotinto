"""Exact Gaussian Process with a shared kernel across output horizons."""

import logging
from time import perf_counter
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from threadpoolctl import threadpool_limits

log = logging.getLogger(__name__)


class GaussianProcess:
    """MIMO regression; input preprocessing belongs to the common model pipeline."""

    def __init__(
        self,
        constant_value=1.0,
        constant_value_bounds=(1e-3, 1e3),
        length_scale=1.0,
        length_scale_bounds=(1e-2, 1e2),
        nu=1.5,
        noise_level=0.1,
        noise_level_bounds=(1e-5, 1e1),
        normalize_y=True,
        alpha=1e-6,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        random_state=0,
        computational_threads=1,
    ):
        self.kernel = ConstantKernel(constant_value, constant_value_bounds) * Matern(
            length_scale, length_scale_bounds, nu=nu
        ) + WhiteKernel(noise_level, noise_level_bounds)
        self.normalize_y = normalize_y
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        if isinstance(computational_threads, bool) or not isinstance(computational_threads, int) or computational_threads < 1:
            raise ValueError("computational_threads must be a positive integer")
        self.computational_threads = computational_threads

    def fit(self, X, Y):
        self.model_ = GaussianProcessRegressor(
            kernel=self.kernel,
            normalize_y=self.normalize_y,
            alpha=self.alpha,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with threadpool_limits(limits=self.computational_threads):
                self.model_.fit(X, Y)
        self.study_model_info = {
            "kernel": str(self.kernel),
            "kernel_log_bounds": self.kernel.bounds.tolist(),
            "learned_kernel": str(self.model_.kernel_),
            "learned_kernel_theta": self.model_.kernel_.theta.tolist(),
            "normalize_y": self.normalize_y,
            "alpha": self.alpha,
            "optimizer": self.optimizer,
            "n_restarts_optimizer": self.n_restarts_optimizer,
            "random_state": self.random_state,
            "computational_threads": self.computational_threads,
            "n_train": len(X),
            "n_features": self.model_.n_features_in_,
            "n_targets": 1 if np.ndim(Y) == 1 else np.shape(Y)[1],
            "fit_seconds": perf_counter() - started,
            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
        }
        log.info(
            "GP_FIT n=%s seconds=%.3f kernel=%s warnings=%s",
            len(X), self.study_model_info["fit_seconds"], self.model_.kernel_,
            self.study_model_info["warnings"],
        )
        return self

    def predict(self, X):
        prediction = self.model_.predict(X)
        return prediction[:, None] if prediction.ndim == 1 else prediction
