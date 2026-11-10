import numpy as np


class Naive:
    """naive forecast: 로그수익률 0, 즉 `P_t` 유지."""

    def fit(self, X, Y):
        self.horizon = Y.shape[1]
        return self

    def predict(self, X):
        return np.zeros((len(X), self.horizon))
