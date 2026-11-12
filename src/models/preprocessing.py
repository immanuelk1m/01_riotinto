"""학습 행으로만 적합하는 공통 입력 스케일러."""
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ScaledRegressor:
    allow_nan = False

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "model"), name)

    def fit(self, X, Y):
        self.scaler_ = StandardScaler().fit(X)
        self.model.fit(self._transform(X), Y)
        return self

    def _transform(self, X):
        return pd.DataFrame(self.scaler_.transform(X), index=X.index, columns=X.columns)

    def predict(self, X):
        return self.model.predict(self._transform(X))
