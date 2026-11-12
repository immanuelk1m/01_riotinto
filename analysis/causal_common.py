"""인과 탐색 노트북(RCBNB-MB, PCMCI-LPCMCI)이 함께 쓰는 데이터·변환·답지. 여기를 바꾸면 두 노트북의 조건이 같이 바뀐다."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parent.parent
TAU = 4              # 최대 시차(주) — 한 달
ALPHA = 0.05         # 모든 검정의 유의수준
START = "2015-01-04" # parameters_exogenous.yml 의 start 와 같다
WEEKLY_FRAC = 0.5    # 주간 변화 비율이 이 이상이면 주간 변수, 아니면 계단형(월·분기)
MIN_CHANGES = 30     # 구간 내 값이 바뀐 횟수가 이보다 적으면 검정할 정보가 없다
RATE = r"MAC\.MON|Pct|Percent|_yoy|_mom|_qoq|_pmi|unemployment"  # 금리·비율: 로그를 취하지 않는다
MISSING = 999.0      # tigramite missing_flag
SERIES = {"coffee": "커피", "copper": "구리", "gold": "금"}  # 답지 시트 이름
KEY = ROOT / "docs/commodity_weekly_drivers.xlsx"

prices = pd.read_parquet(ROOT / "data/02_intermediate/weekly_prices.parquet")
exo = pd.read_parquet(ROOT / "data/01_raw/exogenous_nodelay.parquet")


def make_panel(series):
    """가격 로그 수익률(0번 열 `price`) + drivers 전부를 변환해 z-score 한 주간 패널과 변환 표."""
    price = prices[prices.series == series].set_index("week").value
    weeks = pd.date_range(START, price.index.max(), freq="W-SUN")
    g = exo[(exo.series == series) & exo.drivers]
    wide, code = g.pivot(index="week", columns="col_name", values="value"), g.groupby("col_name").code.first()
    cols, rows = {"price": np.log(price).diff()}, []
    for c in wide:
        x = wide[c].dropna()
        log = bool((x > 0).all()) and not re.search(RATE, f"{c} {code[c]}")
        base = np.log(x) if log else x
        d = base.diff()
        moved = d.abs() > 0
        n_chg, frac = int(moved[START:].sum()), moved[START:].mean()
        if n_chg < MIN_CHANGES:
            kind = "제외 (변화 부족)"
        elif frac >= WEEKLY_FRAC:
            kind, cols[c] = "주간 변화", d.where(d.index <= moved[moved].index.max())  # 갱신이 끊긴 꼬리는 결측
        elif adfuller(base[moved].values)[1] < ALPHA:
            kind, cols[c] = "계단형: 수준", base
        else:
            kind, cols[c] = "계단형: 발표 변화 유지", d.where(moved).ffill()
        rows.append((c, code[c], kind, "로그" if log else "", n_chg, round(frac, 2), x.index.min().date()))
    X = pd.DataFrame(cols).reindex(weeks)
    info = pd.DataFrame(rows, columns=["변수", "code", "변환", "로그", "변화 횟수", "주간 변화 비율", "시작"])
    return (X - X.mean()) / X.std(), info


def load_key(series):
    """답지(주별 원인표): index = 주 일요일, type = 요인 유형, cited = 원인으로 인용된 col_name 집합."""
    k = pd.read_excel(KEY, sheet_name=SERIES[series]).dropna(subset=["주 일요일"]).set_index("주 일요일")
    cited = k["커버 시계열 (col_name)"].fillna("").str.split(r"[\n,;]+").map(lambda v: {x.strip() for x in v} - {""})
    return pd.DataFrame({"type": k["요인 유형"].fillna("기타"), "cited": cited})
