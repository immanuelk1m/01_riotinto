import pandas as pd


def build(logp: pd.Series, horizon: int, exog=None, season: int = 52) -> pd.DataFrame:
    """한 계절 전 같은 주의 가격까지의 로그수익률: s{h} = log(P_{t+h-season} / P_t)."""
    return pd.DataFrame({f"s{h}": logp.shift(season - h) - logp for h in range(1, horizon + 1)})
