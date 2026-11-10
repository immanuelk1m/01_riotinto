"""모델 레지스트리. 인터페이스: `fit(X: DataFrame, Y: ndarray(n, H)) -> self`, `predict(X) -> ndarray(n, H)`.

Y 와 예측은 로그수익률 `log(P_{t+h} / P_t)`.
새 모델: 카테고리 폴더(`statistical/`, `ml/`, `dl/`)에 파일 하나 추가 + 아래 dict 에 한 줄.
"""
from .statistical.naive import Naive
from .statistical.ols import OLS
from .statistical.seasonal_naive import SeasonalNaive

MODELS = {
    "naive": Naive,
    "seasonal_naive": SeasonalNaive,
    "ols": OLS,
}
