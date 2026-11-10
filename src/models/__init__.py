"""모델 레지스트리. 인터페이스: `fit(X: DataFrame, Y: ndarray(n, H)) -> self`, `predict(X) -> ndarray(n, H)`.

Y 와 예측은 실험의 `target` 공간(`src.utils.TARGETS`: level·diff·log·logdiff, 기본 logdiff).
출력 공간이 정해진 모델은 클래스 속성 `target` 을 적는다(다른 target 실험은 거부된다).
새 모델: 카테고리 폴더(`statistical/`, `ml/`, `dl/`)에 파일 하나 추가 + 아래 dict 에 한 줄.
"""
from .ml.xgb import XGB
from .statistical.naive import Naive
from .statistical.ols import OLS
from .statistical.seasonal_naive import SeasonalNaive

MODELS = {
    "naive": Naive,
    "seasonal_naive": SeasonalNaive,
    "ols": OLS,
    "xgb": XGB,
}
