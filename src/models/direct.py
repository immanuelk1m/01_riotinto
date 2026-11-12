"""Independent scalar regressors on the same mature training rows."""
import copy
import hashlib
import json
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported model audit parameter: {type(value).__name__}")


class DirectRegressor:
    """Fit one fresh estimator per Y column, preserving horizon order."""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    @property
    def allow_nan(self):
        return getattr(self.base_estimator, "allow_nan", False)

    def fit(self, X, Y):
        Y = np.asarray(Y)
        if Y.ndim != 2 or not Y.shape[1] or not len(Y) or len(X) != len(Y):
            raise ValueError("Direct targets must have shape (n_samples, n_horizons) matching X")
        if not np.isfinite(Y).all():
            raise ValueError("Direct targets must be finite")
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        features_sha = hashlib.sha256(pd.util.hash_pandas_object(frame).to_numpy().tobytes()).hexdigest()
        estimators, heads = [], []
        for index in range(Y.shape[1]):
            estimator = copy.deepcopy(self.base_estimator)
            started = perf_counter()
            # Heads run sequentially without changing the parent's model parameters.
            with threadpool_limits(limits=1):
                estimator.fit(X, Y[:, index])
            elapsed = perf_counter() - started
            parameters = estimator.get_params()
            backend = type(estimator).__module__
            if hasattr(estimator, "get_booster"):
                backend = "xgboost"
                effective = json.loads(estimator.get_booster().save_config())
            elif hasattr(estimator, "booster_"):
                backend = "lightgbm"
                effective = estimator.booster_.params
            else:
                effective = parameters
            heads.append({
                "horizon": index + 1, "n_train": len(Y), "n_features": frame.shape[1],
                "training_features_sha": features_sha,
                "training_targets_sha": hashlib.sha256(np.ascontiguousarray(Y[:, index]).tobytes()).hexdigest(),
                "parameters": _json_safe(parameters), "effective_parameters": _json_safe(effective),
                "backend": backend, "fit_seconds": elapsed, "computational_threads": 1,
            })
            estimators.append(estimator)
        self.estimators_ = estimators
        self.study_model_info = {"forecast_strategy": "direct", "n_heads": len(heads), "heads": heads}
        return self

    def predict(self, X):
        return np.column_stack([estimator.predict(X) for estimator in self.estimators_])
