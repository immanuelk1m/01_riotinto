import numpy as np


def _design(X) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])


class OLS:
    """절편 포함 다중출력 최소제곱(MIMO: H개 horizon 을 한 번에). 랭크 부족은 최소노름 해."""

    def fit(self, X, Y):
        self.beta, *_ = np.linalg.lstsq(_design(X), Y, rcond=None)
        return self

    def predict(self, X):
        return _design(X) @ self.beta
