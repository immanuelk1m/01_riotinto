"""model → learning → inference. Y 와 예측은 로그수익률 `log(P_{t+h} / P_t)`."""
import copy

import numpy as np
import pandas as pd

from src.models import MODELS
from src.utils import KEYS, block_keys, xy_columns


def select_input(tabular: pd.DataFrame, feature_columns: dict[str, list[str]], spec: dict, horizon: int) -> pd.DataFrame:
    """공유 tabular 에서 이 실험의 피처 열만 남긴다(타겟 격자 열은 그대로)."""
    if missing := sorted(set(block_keys(spec)) - set(feature_columns)):
        raise ValueError(f"04_model_input 에 없는 피처 {missing} — 실험 목록이 바뀌었으면 `uv run kedro run` 으로 전체를 다시 돌린다")
    xcols, _ = xy_columns(tabular, horizon)
    mine = dict.fromkeys(col for key in block_keys(spec) for col in feature_columns[key])
    return tabular[[*(col for col in tabular.columns if col not in xcols), *mine]]


def make_model(spec: dict):
    return MODELS[spec["model"]](**spec.get("model_params", {}))


def fit(model, model_input: pd.DataFrame, horizon: int) -> dict:
    """walk-forward: origin 마다 expanding window 로 다시 학습한 `{(series, origin): 모델}`.

    origin t 의 학습 행은 라벨이 t 시점에 이미 성숙한 것(`week + H <= t`)뿐이다 — 미래 누수 없음.
    """
    all_xcols, ycols = xy_columns(model_input, horizon)
    fitted = {}
    for series, group in model_input.groupby("series"):
        group = group.reset_index(drop=True)                     # 달력 격자라 위치 차 = 주 차
        position = np.arange(len(group))
        xcols = group[all_xcols].dropna(axis=1, how="all").columns.tolist()   # 다른 시리즈 전용 열은 뺀다
        has_x = group[xcols].notna().all(axis=1).to_numpy()
        complete = has_x & group[ycols].notna().all(axis=1).to_numpy()
        for t in np.flatnonzero(group.is_origin.to_numpy() & has_x):
            train = group[complete & (position + horizon <= t)]
            fitted[(series, group.week[t])] = copy.deepcopy(model).fit(train[xcols], train[ycols].to_numpy())
    return fitted


def predict(fitted: dict, model_input: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """origin × horizon 셀. H개 타겟이 모두 성숙한 origin 만 쓴다."""
    xcols, ycols = xy_columns(model_input, horizon)
    rows = model_input[model_input.is_origin & model_input[ycols].notna().all(axis=1)].set_index(KEYS)
    keys = [key for key in fitted if key in rows.index]
    return pd.DataFrame({
        "series": np.repeat([series for series, _ in keys], horizon),
        "origin": np.repeat(pd.DatetimeIndex([week for _, week in keys]), horizon),
        "horizon": np.tile(np.arange(1, horizon + 1), len(keys)),
        "fc": np.concatenate([fitted[key].predict(rows.loc[[key], xcols].dropna(axis=1))[0] for key in keys]),
        "target_logret": rows.loc[keys, ycols].to_numpy().ravel(),
    })
