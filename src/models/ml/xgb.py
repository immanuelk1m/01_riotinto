from xgboost import XGBRegressor


class XGB(XGBRegressor):
    """XGBoost 다중출력 회귀(MIMO). 전략은 model_params 의 `multi_strategy`(`tree_method: hist` 필요)."""
