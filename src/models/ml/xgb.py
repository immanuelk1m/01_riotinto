from xgboost import XGBRegressor


class XGB(XGBRegressor):
    """XGBoost 다중출력 회귀(MIMO). 전략은 model_params 의 `multi_strategy`(`tree_method: hist` 필요)."""

    allow_nan = True    # 결측은 그대로 둔다 — XGBoost 가 분기마다 기본 방향으로 보낸다
