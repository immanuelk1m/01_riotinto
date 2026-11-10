import pandas as pd


def build(logp: pd.Series, horizon: int, exog=None, n: int = 8) -> pd.DataFrame:
    """최근 n개 주간 로그수익률. lag0 = t 주의 수익률."""
    ret = logp.diff()
    return pd.DataFrame({f"lag{k}": ret.shift(k) for k in range(n)})
