import pandas as pd


def build(logp: pd.Series, horizon: int, exog=None, windows=(1, 2, 4, 12, 52)) -> pd.DataFrame:
    """k주 누적 로그수익률: mom{k} = log(P_t / P_{t-k})."""
    return pd.DataFrame({f"mom{k}": logp - logp.shift(k) for k in windows})
