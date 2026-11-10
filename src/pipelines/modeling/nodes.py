"""model → learning → inference. 모델의 Y·예측은 실험의 `target` 공간, 저장하는 예측은 가격."""
import copy

import numpy as np
import pandas as pd

from src.models import MODELS
from src.utils import DEFAULT_TARGET, KEYS, TARGETS, block_keys, xy_columns


def _prices(rows: pd.DataFrame, ycols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """공유 격자(logp, y_h = logdiff)에서 `P_t` (n, 1) 와 `P_{t+h}` (n, H)."""
    p0 = np.exp(rows.logp.to_numpy())[:, None]
    return p0, p0 * np.exp(rows[ycols].to_numpy())


def select_input(tabular: pd.DataFrame, feature_columns: dict[str, list[str]], spec: dict, horizon: int) -> pd.DataFrame:
    """공유 tabular 에서 이 실험의 피처 열만 남긴다(타겟 격자 열은 그대로)."""
    if missing := sorted(set(block_keys(spec)) - set(feature_columns)):
        raise ValueError(f"04_model_input 에 없는 피처 {missing} — 실험 목록이 바뀌었으면 `uv run kedro run` 으로 전체를 다시 돌린다")
    xcols, _ = xy_columns(tabular, horizon)
    mine = dict.fromkeys(col for key in block_keys(spec) for col in feature_columns[key])
    return tabular[[*(col for col in tabular.columns if col not in xcols), *mine]]


def make_model(spec: dict):
    return MODELS[spec["model"]](**spec.get("model_params", {}))


def fit(model, model_input: pd.DataFrame, spec: dict, horizon: int) -> dict:
    """walk-forward: origin 마다 expanding window 로 다시 학습한 `{(series, origin): 모델}`.

    origin t 의 학습 행은 라벨이 t 시점에 이미 성숙한 것(`week + H <= t`)뿐이다 — 미래 누수 없음.
    """
    encode = TARGETS[spec.get("target", DEFAULT_TARGET)][0]
    all_xcols, ycols = xy_columns(model_input, horizon)
    fitted = {}
    for series, group in model_input.groupby("series"):
        group = group.reset_index(drop=True)                     # 달력 격자라 위치 차 = 주 차
        position = np.arange(len(group))
        xcols = group[all_xcols].dropna(axis=1, how="all").columns.tolist()   # 다른 시리즈 전용 열은 뺀다
        has_x = group[xcols].notna().all(axis=1).to_numpy()
        complete = has_x & group[ycols].notna().all(axis=1).to_numpy()
        if "train_start" in spec:                                # 학습 구간을 맞춘 대조군용
            complete &= (group.week >= pd.Timestamp(spec["train_start"])).to_numpy()
        for t in np.flatnonzero(group.is_origin.to_numpy() & has_x):
            train = group[complete & (position + horizon <= t)]
            fitted[(series, group.week[t])] = copy.deepcopy(model).fit(train[xcols], encode(*_prices(train, ycols)))
    return fitted


def predict(fitted: dict, model_input: pd.DataFrame, spec: dict, horizon: int) -> pd.DataFrame:
    """origin × horizon 셀, 가격 공간: `fc` = P̂_{t+h}, `actual` = P_{t+h}, `base` = P_t.

    H개 타겟이 모두 성숙한 origin 만 쓴다.
    """
    decode = TARGETS[spec.get("target", DEFAULT_TARGET)][1]
    xcols, ycols = xy_columns(model_input, horizon)
    rows = model_input[model_input.is_origin & model_input[ycols].notna().all(axis=1)].set_index(KEYS)
    keys = [key for key in fitted if key in rows.index]
    p0, actual = _prices(rows.loc[keys], ycols)
    y = np.vstack([fitted[key].predict(rows.loc[[key], xcols].dropna(axis=1)) for key in keys]).reshape(-1, horizon)
    return pd.DataFrame({
        "series": np.repeat([series for series, _ in keys], horizon),
        "origin": np.repeat(pd.DatetimeIndex([week for _, week in keys]), horizon),
        "horizon": np.tile(np.arange(1, horizon + 1), len(keys)),
        "fc": decode(p0, y).ravel(),
        "actual": actual.ravel(),
        "base": np.repeat(p0.ravel(), horizon),
    })
