"""model → learning → inference. 모델의 Y·예측은 실험의 `target` 공간, 저장하는 예측은 가격."""
import copy
import hashlib
import pickle
from pathlib import Path
from numbers import Integral

import numpy as np
import pandas as pd

from src.models import MODELS
from src.models.direct import DirectRegressor
from src.models.preprocessing import ScaledRegressor
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
    if (spec.get("forecast_strategy") == "direct" and spec["model"] == "xgb"
            and "multi_strategy" in spec.get("model_params", {})):
        raise ValueError("Direct XGB requires scalar model_params without multi_strategy")
    model = MODELS[spec["model"]](**spec.get("model_params", {}))
    if spec.get("forecast_strategy") == "direct":
        model = DirectRegressor(model)
    return ScaledRegressor(model) if spec.get("preprocessing", {}).get("scaling") == "standard" else model


def prepare_input(model_input: pd.DataFrame, spec: dict, horizon: int) -> pd.DataFrame:
    """X만 시리즈별 순방향으로 채우고 라벨·가격·원래 행 순서는 유지한다."""
    if spec.get("preprocessing", {}).get("missing") != "ffill":
        return model_input
    xcols, _ = xy_columns(model_input, horizon)
    out = model_input.copy()
    # stable positions prevent dependence on incoming row order or duplicate index labels.
    ordered = out.reset_index(drop=True).sort_values(["series", "week"])
    ordered[xcols] = ordered.groupby("series", sort=False)[xcols].ffill()
    out[xcols] = ordered.sort_index()[xcols].to_numpy()
    return out


def _series_xcols(group: pd.DataFrame, xcols: list[str]) -> list[str]:
    """그 시리즈에서 한 번이라도 값이 있는 열 — 다른 시리즈 전용 열은 뺀다."""
    return group[xcols].dropna(axis=1, how="all").columns.tolist()


def fit(model, model_input: pd.DataFrame, spec: dict, horizon: int, exog_spec: dict,
        checkpoint_dir: str | Path | None = None) -> dict:
    """walk-forward로 학습한 `{(series, origin): 모델}`. 기본 expanding, 선택적으로 최근 달력 W주.

    origin t 의 학습 행은 라벨이 t 시점에 이미 성숙한 것(`week + H <= t`)뿐이다 — 미래 누수 없음.
    `allow_nan` 모델(xgb)은 X 결측을 그대로 받고, 아니면 X 가 모두 있는 행만 쓴다.
    `exog` 실험의 학습 구간은 그 원자재의 `start` 부터다.
    계보가 있는 연구는 평가 가능한 origin만 적합하고, 피처 열도 해당 origin까지의 관측으로 고른다.
    """
    raw_input = model_input
    model_input = prepare_input(model_input, spec, horizon)
    ffill = spec.get("preprocessing", {}).get("missing") == "ffill"
    encode = TARGETS[spec.get("target", DEFAULT_TARGET)][0]
    all_xcols, ycols = xy_columns(model_input, horizon)
    window = spec.get("train_window_weeks")
    if window is not None and (isinstance(window, bool) or not isinstance(window, Integral) or window <= 0):
        raise ValueError("train_window_weeks는 양의 정수여야 한다")
    is_study = "lineage" in spec
    fitted = {}
    for series, group in model_input.groupby("series"):
        group = group.reset_index(drop=True)                     # 달력 격자라 위치 차 = 주 차
        xcols = _series_xcols(group, all_xcols)
        complete = group[ycols].notna().all(axis=1).to_numpy(copy=True)
        eligible = group.is_origin.to_numpy(copy=True)
        if is_study:
            eligible &= complete
        starts = [spec.get("train_start")]                       # 학습 구간을 맞춘 대조군용
        if "exog" in spec.get("features", []):
            starts.append(exog_spec.get(series, {}).get("start"))
        for start in filter(None, starts):
            complete &= (group.week >= pd.Timestamp(start)).to_numpy()
        for t in np.flatnonzero(eligible):
            origin = group.week[t]
            checkpoint = Path(checkpoint_dir) / f"{series}_{origin.date()}.pkl" if checkpoint_dir else None
            if checkpoint is not None and checkpoint.exists() and checkpoint.with_suffix('.sha256').exists():
                payload = checkpoint.read_bytes()
                if hashlib.sha256(payload).hexdigest() != checkpoint.with_suffix('.sha256').read_text():
                    raise ValueError(f"origin checkpoint 손상: {checkpoint}")
                fitted[(series, origin)] = pickle.loads(payload)
                continue
            if is_study:
                xcols = _series_xcols(group.iloc[:t + 1], all_xcols)
            has_x = (np.ones(len(group), bool) if getattr(model, "allow_nan", False) and not ffill
                     else group[xcols].notna().all(axis=1).to_numpy())
            if not has_x[t]:
                if ffill:
                    raise ValueError(f"{series} {origin.date()}: ffill 후 추론 피처 결측")
                continue
            cutoff = origin - pd.Timedelta(weeks=horizon)
            train_mask = complete & has_x & (group.week <= cutoff).to_numpy()
            window_start = origin - pd.Timedelta(weeks=window) if window is not None else None
            if window_start is not None:
                train_mask &= (group.week >= window_start).to_numpy()
            train = group[train_mask]
            if train.empty:
                raise ValueError(f"{series} {origin.date()}: 성숙한 학습 행이 없다")
            fitted_model = copy.deepcopy(model).fit(train[xcols], encode(*_prices(train, ycols)))
            fitted_model.study_feature_columns = xcols.copy()
            fitted_model.study_training_info = {
                "n_train": len(train),
                "training_start": str(train.week.min().date()),
                "training_end": str(train.week.max().date()),
                "cutoff": str(cutoff.date()),
                "origin": str(origin.date()),
                "window_start": str(window_start.date()) if window_start is not None else None,
                "train_window_weeks": window,
            }
            raw_group = raw_input[raw_input.series == series].reset_index(drop=True)
            past = group.week <= origin
            raw_x = raw_group.loc[past, xcols]
            prepared_x = group.loc[past, xcols]
            fingerprint = lambda x: hashlib.sha256(pd.util.hash_pandas_object(x).to_numpy().tobytes()).hexdigest()
            fitted_model.study_training_info.update(
                raw_features_sha=fingerprint(raw_x), processed_features_sha=fingerprint(prepared_x),
                training_features_sha=fingerprint(train[xcols]),
                filled_cells=int((raw_x.isna() & prepared_x.notna()).sum().sum()),
                remaining_missing_cells=int(prepared_x.isna().sum().sum()),
                excluded_missing_rows=int((complete & ~has_x & (group.week <= cutoff).to_numpy()
                                           & ((group.week >= window_start).to_numpy() if window_start is not None else True)).sum()),
            )
            if isinstance(fitted_model, ScaledRegressor):
                fitted_model.study_training_info.update(scaler_mean=fitted_model.scaler_.mean_.tolist(),
                                                        scaler_scale=fitted_model.scaler_.scale_.tolist())
            if hasattr(fitted_model, "study_model_info"):
                fitted_model.study_training_info["model_info"] = fitted_model.study_model_info
            fitted[(series, origin)] = fitted_model
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                payload = pickle.dumps(fitted_model, protocol=pickle.HIGHEST_PROTOCOL)
                temporary = checkpoint.with_suffix('.tmp')
                temporary.write_bytes(payload)
                temporary.replace(checkpoint)
                checkpoint.with_suffix('.sha256').write_text(hashlib.sha256(payload).hexdigest())
            if spec.get('model') == 'gaussian_process':
                info = fitted_model.study_model_info
                print(f"GP_ORIGIN {series} {origin.date()} n={len(train)} seconds={info['fit_seconds']:.2f}", flush=True)
    return fitted


def predict(fitted: dict, model_input: pd.DataFrame, spec: dict, horizon: int) -> pd.DataFrame:
    """origin × horizon 셀, 가격 공간: `fc` = P̂_{t+h}, `actual` = P_{t+h}, `base` = P_t.

    H개 타겟이 모두 성숙한 origin 만 쓴다.
    """
    model_input = prepare_input(model_input, spec, horizon)
    decode = TARGETS[spec.get("target", DEFAULT_TARGET)][1]
    xcols, ycols = xy_columns(model_input, horizon)
    rows = model_input[model_input.is_origin & model_input[ycols].notna().all(axis=1)].set_index(KEYS)
    keys = [key for key in fitted if key in rows.index]
    p0, actual = _prices(rows.loc[keys], ycols)
    cols = {series: _series_xcols(group, xcols) for series, group in model_input.groupby("series")}
    y = np.vstack([fitted[key].predict(rows.loc[[key], getattr(fitted[key], "study_feature_columns", cols[key[0]])])
                   for key in keys]).reshape(-1, horizon)
    return pd.DataFrame({
        "series": np.repeat([series for series, _ in keys], horizon),
        "origin": np.repeat(pd.DatetimeIndex([week for _, week in keys]), horizon),
        "horizon": np.tile(np.arange(1, horizon + 1), len(keys)),
        "fc": decode(p0, y).ravel(),
        "actual": actual.ravel(),
        "base": np.repeat(p0.ravel(), horizon),
    })
