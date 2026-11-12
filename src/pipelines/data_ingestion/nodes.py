"""data: Postgres `commodity` → data/01_raw 스냅샷."""
import logging

import pandas as pd

log = logging.getLogger(__name__)


def load_prices(targets: dict[str, str]) -> pd.DataFrame:
    """Postgres `ml.price_week` → (series, week, value) 주간 가격.

    뷰가 끝난 `W-SUN` 주의 마지막 양수 관측값을 준다(앞값 채움·발표 지연 없음, 빠진 주는 행이 없다).
    """
    import psycopg

    with psycopg.connect("dbname=commodity") as con:
        rows = con.execute(
            "select col_name, week_end, value from ml.price_week where col_name = any(%s) order by week_end",
            (list(targets.values()),),
        ).fetchall()
    raw = pd.DataFrame(rows, columns=["col_name", "week", "value"])
    raw["series"] = raw.col_name.map({col: key for key, col in targets.items()})
    if missing := set(targets) - set(raw.series):
        raise ValueError(f"public.value 에 없는 시리즈: {sorted(missing)}")
    return raw.assign(week=pd.to_datetime(raw.week), value=raw.value.astype(float))[["series", "week", "value"]]


def load_exogenous(targets: dict[str, str], exogenous: dict[str, dict]) -> pd.DataFrame:
    """Postgres `ml.series_week_lagged` → (series, col_name, week, value, code, drivers, ml) 주간 외생변수.

    뷰가 일요일 마감 주간 값·앞값 채움·발표 지연을 이미 적용했다 — 행 t 의 값은 t 시점에 알려진 값이다.
    `ml` = 그 원자재의 `ml.feature` 변수, `drivers` = `parameters_exogenous.yml` 의 주간 원인 변수.
    """
    import psycopg

    drivers = pd.DataFrame(
        [(series, col) for series, spec in exogenous.items() for col in spec["drivers"]], columns=["series", "col_name"]
    )
    with psycopg.connect("dbname=commodity") as con:
        ml = pd.DataFrame(
            con.execute("select target_id, col_name from ml.feature where target_id = any(%s)", (list(exogenous),)).fetchall(),
            columns=["series", "col_name"],
        )
        members = ml.assign(ml=True).merge(drivers.assign(drivers=True), how="outer", on=["series", "col_name"])
        members = members[members.col_name != members.series.map(targets)]      # 타겟 자기 가격은 외생변수가 아니다
        values = pd.DataFrame(
            con.execute(
                "select l.col_name, l.week_end, l.value, m.code from ml.series_week_lagged l "
                "join public.value_manifest m using (col_name) where l.col_name = any(%s) and l.value is not null",
                (members.col_name.unique().tolist(),),
            ).fetchall(),
            columns=["col_name", "week", "value", "code"],
        )
    if missing := sorted(set(members.col_name) - set(values.col_name)):
        log.warning("ml.series_week_lagged 에 없어 건너뛴 외생변수 %d개: %s", len(missing), missing)
    members[["drivers", "ml"]] = members[["drivers", "ml"]].notna()
    out = members.merge(values, on="col_name").assign(week=lambda f: pd.to_datetime(f.week))
    return out[["series", "col_name", "week", "value", "code", "drivers", "ml"]]
