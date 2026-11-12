from numbers import Integral

import pandas as pd


def build(logp: pd.Series, horizon: int, exog=None, n: int = 8,
          offsets: list[int] | None = None) -> pd.DataFrame:
    """주간 로그수익률. lag0 = t 주; offsets가 없으면 최근 n개를 쓴다."""
    if offsets is None:
        if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0:
            raise ValueError("lags n은 양의 정수여야 한다")
        offsets = list(range(n))
    if (not offsets or any(isinstance(k, bool) or not isinstance(k, Integral) or k < 0 for k in offsets)
            or len(set(offsets)) != len(offsets)):
        raise ValueError("lags offsets는 중복 없는 비음수 정수 목록이어야 한다")
    ret = logp.diff()
    return pd.DataFrame({f"lag{k}": ret.shift(k) for k in offsets})
