"""여러 파이프라인이 함께 쓰는 열 규칙."""
import json

import numpy as np
import pandas as pd

KEYS = ["series", "week"]

# 실험의 `target`: 모델이 학습·예측하는 공간. `(encode(P_t, P_{t+h}) -> y, decode(P_t, y) -> P̂)`.
TARGETS = {
    "level": (lambda p0, p: p, lambda p0, y: y),
    "diff": (lambda p0, p: p - p0, lambda p0, y: p0 + y),
    "log": (lambda p0, p: np.log(p), lambda p0, y: np.exp(y)),
    "logdiff": (lambda p0, p: np.log(p / p0), lambda p0, y: p0 * np.exp(y)),
}
DEFAULT_TARGET = "logdiff"


def block_keys(spec: dict) -> dict[str, tuple[str, dict]]:
    """실험이 쓰는 피처 블록: `{"블록:인자JSON": (블록, 인자)}`. 키가 같으면 같은 열이다."""
    params = spec.get("feature_params", {})
    return {f"{name}:{json.dumps(params.get(name, {}), sort_keys=True)}": (name, params.get(name, {}))
            for name in spec.get("features", [])}


def xy_columns(tabular: pd.DataFrame, horizon: int):
    ycols = [f"y{h}" for h in range(1, horizon + 1)]
    return [c for c in tabular.columns if c not in {*KEYS, "logp", "is_origin", *ycols}], ycols
