from .naive import Naive


class SeasonalNaive(Naive):
    """seasonal naive: `P_{t+h}` 의 예측 = `P_{t+h-season}`. 피처 블록 `seasonal` 이 필요하다."""

    def predict(self, X):
        return X[[f"s{h}" for h in range(1, self.horizon + 1)]].to_numpy()
