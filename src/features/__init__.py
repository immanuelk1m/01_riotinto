"""피처 블록 레지스트리. 블록 = `build(logp, horizon, exog, **params) -> DataFrame`.

`logp` 는 한 시리즈의 주간 로그가격(달력 주 인덱스), `exog` 는 그 시리즈의 외생변수 long 프레임
(col_name, week, value, code, drivers, ml — 시작 날짜 규칙 적용 후). 반환 행 t 에는 **t 까지의 정보만** 들어가야 한다.
새 블록: 파일 하나 추가 + 아래 dict 에 한 줄.
"""
from . import exog, lags, momentum, price_lags, seasonal

FEATURES = {
    "exog": exog.build,
    "lags": lags.build,
    "momentum": momentum.build,
    "price_lags": price_lags.build,
    "seasonal": seasonal.build,
}
