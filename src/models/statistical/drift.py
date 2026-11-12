import numpy as np


class Drift:
    """성숙한 학습 행의 horizon별 평균 로그수익률을 예측한다."""

    target = "logdiff"
    allow_nan = True

    def fit(self, X, Y):
        values = np.asarray(Y, dtype=float)
        if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("drift는 비어 있지 않은 유한한 2차원 타겟을 필요로 한다")
        self.mean_return_ = values.mean(axis=0)
        return self

    def predict(self, X):
        return np.tile(self.mean_return_, (len(X), 1))
