"""여러 파이프라인이 함께 쓰는 열 규칙."""
import json

import pandas as pd

KEYS = ["series", "week"]


def block_keys(spec: dict) -> dict[str, tuple[str, dict]]:
    """실험이 쓰는 피처 블록: `{"블록:인자JSON": (블록, 인자)}`. 키가 같으면 같은 열이다."""
    params = spec.get("feature_params", {})
    return {f"{name}:{json.dumps(params.get(name, {}), sort_keys=True)}": (name, params.get(name, {}))
            for name in spec.get("features", [])}


def xy_columns(tabular: pd.DataFrame, horizon: int):
    ycols = [f"y{h}" for h in range(1, horizon + 1)]
    return [c for c in tabular.columns if c not in {*KEYS, "logp", "is_origin", *ycols}], ycols
