"""evaluation + 실험 원장. 실험마다 마지막에 `log_run` 이 W&B 에 run 하나를 남긴다(끄기: `WANDB_MODE=disabled`)."""
import hashlib
import json

import numpy as np
import pandas as pd

from src.utils import xy_columns

WANDB_PROJECT = "riotinto"


def evaluate(forecasts: pd.DataFrame, ape_cap: float) -> pd.DataFrame:
    """series × horizon 별 mape(캡 적용 평균, %), max_ape_pct(캡 없는 최악 셀, %), naive forecast 대비 개선.

    `mape_naive` 는 **같은 셀**에서의 naive forecast(`P_t` 유지) MAPE 라서 실험끼리 셀집합이 달라도 공정하다.
    """
    ape = np.abs(forecasts.fc / forecasts.actual - 1)            # |P̂ / P − 1|
    cells = forecasts.assign(
        ape=ape,
        capped=ape.clip(upper=ape_cap),
        naive=np.abs(forecasts.base / forecasts.actual - 1).clip(upper=ape_cap),   # |P_t / P_{t+h} − 1|
    )
    grouped = cells.groupby(["series", "horizon"])
    metrics = pd.DataFrame({
        "mape": grouped.capped.mean() * 100,
        "max_ape_pct": grouped.ape.max() * 100,
        "mape_naive": grouped.naive.mean() * 100,
        "n": grouped.ape.size(),
    }).reset_index()
    metrics["gain_pct"] = (1 - metrics.mape / metrics.mape_naive) * 100   # >0: naive forecast 보다 좋음
    return metrics


def summarize(names: list[str], *metrics: pd.DataFrame) -> pd.DataFrame:
    """모든 실험의 평가표를 한 장으로."""
    return pd.concat([m.assign(experiment=name) for name, m in zip(names, metrics)], ignore_index=True)


def _features(model_input: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """이 실험의 실제 피처 값(`features_sha` 용). 열 순서는 예전 run id 와 맞춘 것이다."""
    return model_input[["week", "series", *xy_columns(model_input, horizon)[0]]]


def _sha(frame: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(frame).to_numpy().tobytes()).hexdigest()[:12]


def run_config(name: str, spec: dict, parameters: dict, targets: pd.DataFrame, features: pd.DataFrame) -> dict:
    """run 을 규정하는 전부: 실험 spec + 공통 평가 설정 + 데이터 스냅샷 지문 + 실제 피처 지문.

    외생변수 설정(`exogenous`)은 config 에 넣지 않는다 — 쓰는 실험에서만 `features_sha` 로 드러난다.
    """
    return {
        **{key: value for key, value in parameters.items() if key not in ("experiments", "exogenous")},
        **spec,
        "experiment": name,
        "data_sha": _sha(targets),
        "features_sha": _sha(features),
        "data_last_week": str(targets.week.max().date()),
    }


def run_id(config: dict) -> str:
    """같은 실험·설정·데이터면 같은 id → 재실행은 같은 run 을 덮어쓰고, 하나라도 바뀌면 새 run."""
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_summary(metrics: pd.DataFrame) -> dict:
    gain = lambda m: float((1 - m.mape.sum() / m.mape_naive.sum()) * 100)  # noqa: E731
    summary = {
        "mape": float(metrics.mape.mean()),
        "mape_naive": float(metrics.mape_naive.mean()),
        "gain_pct": gain(metrics),
        "max_ape_pct": float(metrics.max_ape_pct.max()),
    }
    for series, group in metrics.groupby("series"):
        summary[f"mape/{series}"] = float(group.mape.mean())
        summary[f"gain_pct/{series}"] = gain(group)
    return summary


def log_run(name: str, spec: dict, parameters: dict, targets: pd.DataFrame, model_input: pd.DataFrame,
            forecasts: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """W&B 원장에 run 하나: config, 요약 지표, 평가표, 예측 셀. 파일(parquet·모델)은 올리지 않는다."""
    import wandb

    config = run_config(name, spec, parameters, targets, _features(model_input, parameters["horizon"]))
    with wandb.init(project=WANDB_PROJECT, id=run_id(config), name=name, group=spec.get("study", "default"),
                    config=config, resume="allow", reinit="create_new") as run:
        run.log({
            "metrics": wandb.Table(dataframe=metrics),
            "cells": wandb.Table(dataframe=forecasts.assign(origin=forecasts.origin.dt.strftime("%Y-%m-%d"))),
        })
        run.summary.update(run_summary(metrics))
