"""feature → model input. 실험들이 쓰는 (블록, 인자) 조합을 한 번씩 계산해 공유 입력 두 장을 만든다."""
import logging

import pandas as pd

from src.features import FEATURES
from src.utils import KEYS, block_keys

log = logging.getLogger(__name__)


def _exog_from(exogenous: pd.DataFrame, series: str, start: str | None) -> pd.DataFrame:
    """한 시리즈의 외생변수. `start` 가 있으면 그 주부터의 행만, 그리고 `start` 에 이미 값이 있는 변수만 남긴다.

    늦게 시작하는 변수를 빼 두어야 학습 구간이 `start` 로 고정된다(변수를 넣고 빼도 같은 구간).
    """
    exog = exogenous[exogenous.series == series]
    if start is None:
        return exog
    first = exog.groupby("col_name").week.transform("min")
    kept = exog[(first <= pd.Timestamp(start)) & (exog.week >= pd.Timestamp(start))]
    log.info("%s: 외생변수 %d개 중 %d개 사용 (start=%s)", series, exog.col_name.nunique(), kept.col_name.nunique(), start)
    return kept


def build_features(targets: pd.DataFrame, exogenous: pd.DataFrame, experiments: dict[str, dict], exog_spec: dict,
                   horizon: int) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """모든 실험이 쓰는 (블록, 인자) 조합을 한 번씩 계산해 옆으로 붙인 넓은 표와 `{블록 키: 열 목록}`.

    열 이름이 같으면 값도 같아야 한다 — 같은 이름에 다른 값을 내는 조합이 섞이면 멈춘다.
    """
    wanted = {key: block for spec in experiments.values() for key, block in block_keys(spec).items()}
    frames, columns = [], {key: [] for key in wanted}
    for series, group in targets.groupby("series"):
        logp = group.set_index("week").logp
        exog = _exog_from(exogenous, series, exog_spec.get(series, {}).get("start"))
        wide = {}
        for key, (name, params) in wanted.items():
            block = FEATURES[name](logp, horizon, exog, **params)
            for col in block:
                if col in wide and not wide[col].equals(block[col]):
                    raise ValueError(f"피처 열 {col!r} 이 {key} 에서 다른 값으로 다시 나왔다 — 인자가 값을 바꾸면 열 이름에 드러내라")
                wide[col] = block[col]
            columns[key] += [col for col in block if col not in columns[key]]
        frames.append(pd.DataFrame({"series": series, **wide}, index=logp.index).reset_index())
    return pd.concat(frames, ignore_index=True), columns


def make_tabular(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """tabular 입력: 행 = (series, week), 타겟 격자(`logp`, `is_origin`, y1..yH) + 피처 열."""
    return targets.merge(features, on=KEYS)


def make_timeseries(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """timeseries 입력(Nixtla long 형식): `unique_id`, `ds`, `y`(= `logp`) + 피처 열(과거 외생).

    달력 격자 그대로라 빠진 주는 `y` 가 NaN 이다. origin t 의 학습에는 `ds <= t` 행만 쓴다.
    """
    return targets[[*KEYS, "logp"]].merge(features, on=KEYS).rename(
        columns={"series": "unique_id", "week": "ds", "logp": "y"})
