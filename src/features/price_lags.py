import numpy as np
import pandas as pd


def build(logp: pd.Series, horizon: int, exog=None, lags=(1, 2, 4, 12, 52)) -> pd.DataFrame:
    """가격 수준 lag. 첫 타겟 P_{t+1} 기준이라 plag1 = P_t, plag{k} = P_{t+1-k}."""
    price = np.exp(logp)
    return pd.DataFrame({f"plag{k}": price.shift(k - 1) for k in lags})
