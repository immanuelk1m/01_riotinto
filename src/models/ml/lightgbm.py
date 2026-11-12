"""Scalar LightGBM regression, combined across horizons by DirectRegressor."""
from lightgbm import LGBMRegressor


class LightGBM(LGBMRegressor):
    allow_nan = True
