import hashlib
import json
import logging

import pandas as pd

log = logging.getLogger(__name__)
SOURCES = ("drivers", "ml", "all")


def _under(code: pd.Series, prefixes: list[str]) -> pd.Series:
    """code 가 접두어(대 `MAC` / 중 `MAC.INF` / 소 `MAC.INF.PRODUCER`) 중 하나에 속하는가."""
    return code.map(lambda c: any(c == p or c.startswith(p + ".") for p in prefixes))


def build(logp: pd.Series, horizon: int, exog: pd.DataFrame, source: str = "all",
          codes: list[str] | None = None, exclude: list[str] | None = None, diff: int = 0,
          columns_by_series: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """외생변수 수준값(week × col_name). 스냅샷이 이미 발표 지연을 적용해 행 t 는 t 까지의 정보다.

    `source`: drivers(주간 원인 변수) | ml(`ml.<원자재>` 패널 변수) | all.
    `codes` / `exclude`: `value_manifest.code` 접두어로 넣을 / 뺄 그룹.
    `diff`: 0 이면 수준값, k 면 k주 변화량(열 이름 `<col>__d<k>`).
    `columns_by_series`: 시리즈별 정확한 열 목록. 빈 목록은 그 시리즈에서 외생변수를 쓰지 않는다.
    """
    if source not in SOURCES:
        raise ValueError(f"source={source!r} (허용: {SOURCES})")
    chosen = exog if source == "all" else exog[exog[source]]
    if codes:
        chosen = chosen[_under(chosen.code, codes)]
    if exclude:
        chosen = chosen[~_under(chosen.code, exclude)]
    if columns_by_series is not None:
        series = exog.attrs.get("series")
        if series not in columns_by_series:
            raise ValueError(f"columns_by_series에 시리즈 {series!r}의 명시적 열 목록이 없다")
        columns = columns_by_series[series]
        if (not isinstance(columns, (list, tuple)) or any(not isinstance(c, str) for c in columns)
                or len(set(columns)) != len(columns)):
            raise ValueError(f"columns_by_series[{series!r}]는 중복 없는 열 이름 목록이어야 한다")
        if not columns:
            return pd.DataFrame(index=logp.index)
        if missing := sorted(set(columns) - set(chosen.col_name)):
            raise ValueError(f"시리즈 {series!r}에서 요청한 외생변수 열이 없다: {missing}")
        chosen = chosen[chosen.col_name.isin(columns)]
    if chosen.empty:
        log.warning("고른 외생변수가 없는 시리즈가 있다 (source=%s, codes=%s, exclude=%s)", source, codes, exclude)
    wide = chosen.pivot(index="week", columns="col_name", values="value").reindex(logp.index)
    if columns_by_series is not None:
        wide = wide.reindex(columns=columns)
        scope = hashlib.sha256(json.dumps(columns_by_series, sort_keys=True).encode()).hexdigest()[:12]
        wide = wide.add_suffix(f"__scope_{scope}")
    return wide.diff(diff).add_suffix(f"__d{diff}") if diff else wide
