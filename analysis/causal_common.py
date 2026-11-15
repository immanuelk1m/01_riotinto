"""인과 탐색 노트북(RCBNB-MB, PCMCI-LPCMCI)이 함께 쓰는 데이터·변환·답지·채점. 여기를 바꾸면 두 노트북의 조건이 같이 바뀐다.

`uv run python analysis/causal_common.py` — Postgres `commodity` 에서 분석 입력을 다시 받고(snapshot) 자체 점검을 돌린다.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import comb
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parent.parent
TAU = 4              # 최대 시차(주) — 한 달
ALPHA = 0.05         # 모든 검정의 유의수준
START = "2015-01-04" # parameters_exogenous.yml 의 start 와 같다
WEEKLY_FRAC = 0.5    # 주간 변화 비율이 이 이상이면 주간 변수, 아니면 계단형(월·분기)
MIN_CHANGES = 30     # 구간 내 값이 바뀐 횟수가 이보다 적으면 검정할 정보가 없다
RATE = r"MAC\.MON|Pct|Percent|_yoy|_mom|_qoq|_pmi|unemployment"  # 금리·비율: 로그를 취하지 않는다
PRECIP, PRECIP_WEEKS = "EXT.CLIM.PRECIPITATION", 4  # 강수: 주간 변화가 아니라 최근 4주의 평년 대비 편차로 본다
REACTIVE = r"PFM\.POS|FND\.INV"  # 포지션·재고·ETF 보유량: 가격에 반응할 수 있어 같은 주 방향을 가정하지 않는다
MISSING = 999.0      # tigramite missing_flag
SERIES = {"coffee": "커피", "copper": "구리", "gold": "금"}  # 답지 시트 이름
KEY = ROOT / "docs/commodity_weekly_drivers.xlsx"
PANEL = ROOT / "data/01_raw/causal_panel.parquet"
COFFEE_BRL = "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_BRL_Per_60kg_Bag"  # 커피 답지가 설명하는 가격은 헤알 가격이다
RELEASE_FILE = ROOT / "data/01_raw/causal_releases.parquet"
WEATHER_FILE = ROOT / "data/01_raw/causal_weather.parquet"
WEATHER_EXTRA = {  # 강수 열 → 같은 지점의 기온 표현 {이름: (col_name, 주간 집계)}. 답지는 폭염·서리도 강수 열로 인용한다
    "nasa_power_daily_minas_br_precip": {"최고기온": ("nasa_power_daily_minas_br_tmax", "mean"), "최저기온": ("nasa_power_daily_minas_br_tmin", "min")},
}
RELEASES = {  # 답지가 인용하는 열 → 그 지표의 발표 사건(Investing.com 발표일 기준 계열). 여럿이면 표준화한 값의 평균
    "Com_Econ_US_nonfarm_payrolls": ["Com_Econ_US_nonfarm_payrolls"],
    "Com_Econ_US_unemployment": ["Com_Econ_US_unemployment"],
    "Com_BLS_CPI_All_Items_US_SA_Index": ["Com_Econ_US_cpi_mom", "Com_Econ_US_core_cpi_mom"],
    "Com_BLS_Core_CPI_All_Items_Less_Food_Energy_US_YoY_Pct": ["Com_Econ_US_core_cpi_mom"],
    "Com_Econ_US_ppi_yoy": ["Com_Econ_US_ppi_yoy"],
    "Com_Econ_US_ism_pmi": ["Com_Econ_US_ism_pmi"],
    "Com_Econ_US_ism_pmi_nonmfg": ["Com_Econ_US_ism_pmi_nonmfg"],
    "Com_Econ_US_retail_sales_mom": ["Com_Econ_US_retail_sales_mom"],
    "Com_Econ_US_real_gdp_qoq_annual_rate": ["Com_Econ_US_gdp_qoq"],
    "Com_Econ_CN_pmi_mfg": ["Com_Econ_CN_pmi_mfg"],
    "Com_Econ_CN_gdp_yoy": ["Com_Econ_CN_gdp_yoy"],
    "Com_Econ_CN_industrial_prod_yoy": ["Com_Econ_CN_industrial_prod_yoy"],
}


def snapshot():
    """Postgres → PANEL: (series, col_name, week, value, code), 가격은 col_name = "price". 발표 지연 없음, 앞값 채움.

    변수 = parameters_exogenous.yml 의 drivers. 값 = `ml.series_week_live`(일간은 주 평균, 월·분기는 기준 기간 끝 주).
    단 일간 시장 변수(환율·금리·VIX·재고 등, code 가 EXT. 가 아닌 것)는 주 마지막 값으로 바꾼다 — 가격이 주 마지막 값이라,
    주 평균과 짝지으면 "가격이 다음 주 변수를 이끈다"는 가짜 시차가 생긴다. 강수·뉴스 지수(EXT.)는 주 평균 그대로 둔다.
    """
    import psycopg
    import yaml

    conf = lambda name: yaml.safe_load((ROOT / f"conf/base/{name}.yml").read_text())
    drivers = {s: spec["drivers"] for s, spec in conf("parameters_exogenous")["exogenous"].items()}
    targets = {**conf("parameters")["targets"], "coffee": COFFEE_BRL}
    cols = sorted({c for v in drivers.values() for c in v})
    with psycopg.connect("dbname=commodity") as con:
        q = lambda sql, arg: pd.DataFrame(con.execute(sql, (arg,)).fetchall(), columns=["col_name", "week", "value"])
        meta = pd.DataFrame(con.execute("select col_name, update_freq, code from public.value_manifest where col_name = any(%s)",
                                        (cols,)).fetchall(), columns=["col_name", "freq", "code"]).set_index("col_name")
        market = meta.index[(meta.freq == "D") & ~meta.code.str.startswith("EXT.")].tolist()
        live = q("select col_name, week_end, value from ml.series_week_live where col_name = any(%s)", cols)
        last = q("select distinct on (col_name, ml.week_end(dt)) col_name, ml.week_end(dt), value from public.value "
                 "where col_name = any(%s) and value = value and ml.week_end(dt) <= ml.completed_sunday() "
                 "order by col_name, ml.week_end(dt), dt desc", market)
        price = q("select col_name, week_end, value from ml.price_week where col_name = any(%s)", list(targets.values()))
        end = con.execute("select ml.completed_sunday()").fetchone()[0]
        rel = pd.DataFrame(con.execute("select col_name, dt, value from public.value where col_name like %s and dt <= ml.completed_sunday()",
                                       (r"%\_\_release\_%",)).fetchall(), columns=["col_name", "dt", "value"])
        temp = pd.DataFrame(con.execute("select col_name, ml.week_end(dt), avg(value), min(value) from public.value where col_name = any(%s) "
                                        "and value = value and ml.week_end(dt) <= ml.completed_sunday() group by 1, 2",
                                        ([c for v in WEATHER_EXTRA.values() for c, _ in v.values()],)).fetchall(), columns=["col_name", "week", "mean", "min"])
    temp.assign(week=pd.to_datetime(temp.week)).to_parquet(WEATHER_FILE, index=False)
    rel = pd.concat([rel.col_name.str.extract(r"(?P<event>.+)__release_(?P<field>.+)"), pd.to_datetime(rel.dt), rel.value], axis=1)
    rel.to_parquet(RELEASE_FILE, index=False)
    print(f"{RELEASE_FILE.relative_to(ROOT)}: 발표 {len(rel) // 2}건, 사건 {rel.event.nunique()}개")
    wide = lambda f: f.assign(week=pd.to_datetime(f.week)).pivot(index="week", columns="col_name", values="value")
    exo = wide(pd.concat([live[~live.col_name.isin(market)], last]))
    exo = exo.reindex(pd.date_range(exo.index.min(), end, freq="W-SUN")).ffill()
    if missing := sorted(set(cols) - set(exo.columns)):
        raise ValueError(f"DB 에 없는 drivers: {missing}")
    price = wide(price)
    for s in [s for s in drivers if targets[s] not in price or price[targets[s]].first_valid_index() > pd.Timestamp(START)]:
        print(f"경고: {s} 가격 {targets[s]} 이 DB 에 {START} 부터 있지 않다 → {s} 는 스냅샷에서 뺀다 (이력을 넣고 다시 실행)")
        del drivers[s]
    long = lambda f: f.rename_axis(index="week", columns="col_name").stack().rename("value").reset_index()
    out = pd.concat([
        pd.concat([long(exo[drivers[s]]).assign(code=lambda f: f.col_name.map(meta.code)),
                   long(price[[targets[s]]]).assign(col_name="price", code="")]).assign(series=s)
        for s in drivers])
    out[["series", "col_name", "week", "value", "code"]].to_parquet(PANEL, index=False)
    print(f"{PANEL.relative_to(ROOT)}: {len(out)}행, ~{end}, 주 마지막 값으로 바꾼 일간 시장 변수 {len(market)}개")


raw = pd.read_parquet(PANEL) if PANEL.exists() else None  # 없으면 먼저 snapshot()
prices = None if raw is None else raw[raw.col_name == "price"]
if raw is not None:  # 스냅샷에 없는 원자재(가격 이력이 아직 DB 에 없다)는 분석하지 않는다
    SERIES = {s: name for s, name in SERIES.items() if s in set(raw.series)}


def make_panel(series, precip_weeks=PRECIP_WEEKS, fit_end=None, train_start=None):
    """가격 로그 수익률(0번 열 `price`) + drivers 전부를 변환해 z-score 한 주간 패널과 변환 표."""
    if fit_end is not None:
        return _training_panel(series, precip_weeks, fit_end, train_start)
    if train_start is not None:
        raise ValueError("train_start requires fit_end")
    g = raw[raw.series == series]
    wide, code = g.pivot(index="week", columns="col_name", values="value"), g.groupby("col_name").code.first()
    price = wide.pop("price").dropna()
    weeks = pd.date_range(START, price.index.max(), freq="W-SUN")
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
        elif code[c] == PRECIP:
            m = x.rolling(precip_weeks).mean()  # 평년 = 같은 주차의 전체 기간 평균
            kind, cols[c] = f"강수: {precip_weeks}주 평균의 평년 대비 편차", m - m.groupby(m.index.isocalendar().week).transform("mean")
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


def _training_rows(index, fit_end, train_start=None):
    """Fit interval is [train_start or START, fit_end)."""
    start = pd.Timestamp(train_start or START)
    end = pd.Timestamp(fit_end)
    if start >= end:
        raise ValueError("train_start must precede fit_end")
    return (index >= start) & (index < end)


def _training_panel(series, precip_weeks, fit_end, train_start):
    """Fit transformations on past weeks; preserve calendar gaps when transforming."""
    g = raw[raw.series == series]
    wide = g.pivot(index="week", columns="col_name", values="value").sort_index()
    code = g.groupby("col_name").code.first()
    weeks = pd.date_range(START, wide.price.last_valid_index(), freq="W-SUN")
    # Keep pre-START observations for the first weekly difference, without closing gaps.
    wide = wide.reindex(pd.date_range(min(wide.index.min(), weeks.min()), weeks.max(), freq="W-SUN"))
    fit = _training_rows(wide.index, fit_end, train_start)
    if fit.sum() < 2:
        raise ValueError("Insufficient training weeks")
    cols = {"price": np.log(wide.price.where(wide.price > 0)).diff()}
    rows = []
    for c in wide.columns.drop("price"):
        x = wide[c]
        past = x[fit].dropna()
        log = bool(len(past) and (past > 0).all()) and not re.search(RATE, f"{c} {code[c]}")
        base = np.log(x.where(x > 0)) if log else x
        d = base.diff()
        moved = d.abs() > 0
        n_chg, frac = int(moved[fit].sum()), float(moved[fit].mean())
        kind = "제외 (변화 부족)"
        if n_chg >= MIN_CHANGES:
            if code[c] == PRECIP:
                m = x.rolling(precip_weeks).mean()
                climate = m[fit].groupby(m[fit].index.isocalendar().week).mean()
                normal = pd.Series(m.index.isocalendar().week.map(climate).to_numpy(), index=m.index)
                kind, cols[c] = f"강수: {precip_weeks}주 평균의 평년 대비 편차", m - normal
            elif frac >= WEEKLY_FRAC:
                kind, cols[c] = "주간 변화", d
            else:
                release = base[fit & moved].dropna()
                try:
                    stationary = adfuller(release.values)[1] < ALPHA
                except ValueError:
                    stationary = False
                kind = "계단형: 수준" if stationary else "계단형: 발표 변화 유지"
                cols[c] = base if stationary else d.where(moved).ffill().where(x.notna())
        rows.append((c, code[c], kind, "로그" if log else "", n_chg, round(frac, 2),
                     past.index.min().date() if len(past) else None))
    X = pd.DataFrame(cols).reindex(weeks)
    training = X.loc[_training_rows(X.index, fit_end, train_start)]
    scale = training.std()
    # No future observations can revive a column with no training variation.
    keep = [c for c in X if np.isfinite(scale[c]) and scale[c] > 0]
    if "price" not in keep:
        raise ValueError("Price has no finite training variation")
    info = pd.DataFrame(rows, columns=["변수", "code", "변환", "로그", "변화 횟수", "주간 변화 비율", "시작"])
    info.loc[~info.변수.isin(keep), "변환"] = "제외 (학습 변화 부족)"
    return (X[keep] - training[keep].mean()) / scale[keep], info


def lagged_values(series):
    """발표 지연이 반영된 주간 값(파이프라인 스냅샷 `exogenous.parquet`, 읽기만): 주 × 변수. 값이 바뀐 주 = 그 값이 알려진 주."""
    lag = pd.read_parquet(ROOT / "data/01_raw/exogenous.parquet", filters=[("series", "==", series)])
    return lag.pivot(index="week", columns="col_name", values="value")


def event_panel(series, X, info, yoy=False, fit_end=None, train_start=None):
    """X 에서 계단형(월·분기) 열만 **발표 주 사건**으로 바꾼 패널. 주간·강수 열과 가격은 그대로다.

    사건 = 발표 지연이 반영된 값이 바뀐 주의 변화(그 변수의 변화들로 표준화), 나머지 주는 0.
    답지는 CPI·고용 같은 지표를 발표된 주에 인용한다 — 한 달 내내 같은 값을 들고 있으면 그 주를 가려낼 수 없다.
    yoy=True 면 물량 계열(로그를 취하는 열)의 크기를 52주 전 대비 로그 변화로 바꾼다 — 답지는 수출·재고를 전년 동월 대비로 인용한다.
    """
    L, E, logs = lagged_values(series).reindex(X.index), X.copy(), set(info.변수[info.로그 == "로그"])
    for c in info.변수[info.변환.str.startswith("계단형")]:
        d = L[c].diff()
        moved = d.fillna(0) != 0
        if yoy and c in logs:
            d = np.log(L[c]).diff(52)
        fit = moved if fit_end is None else moved & _training_rows(d.index, fit_end, train_start)
        E[c] = ((d - d[fit].mean()) / d[fit].std()).where(moved, 0.0)
    return E


def _views(x, shift=0):
    """주간 강수 한 줄 → 표현 네 가지(평년 대비 편차, 주차별 평균·표준편차로 표준화). shift 주만큼 뒤로 민다."""
    def anom(w):
        by = w.groupby(w.index.isocalendar().week.values)
        return (w - by.transform("mean")) / by.transform("std")
    a = anom(x)
    return pd.DataFrame({"지난주": a.shift(1), "같은 주": a, "다음 주": a.shift(-1), "8주 평균": anom(x.rolling(8).mean())}).shift(shift)


def weather_views(series, shift=0):
    """기상 변수(강수 열) → 그 변수의 표현들(주 × 표현). 답지가 기상을 인용하는 모습에 맞춘 것이다.

    지난주 = 월요일에 나오는 주간 강우 보고, 다음 주 = 예보의 대용(**미래 값** — 사후 설명에만 쓴다), 8주 평균 = 누적 상태.
    WEATHER_EXTRA 에 있는 열은 같은 지점의 기온 편차(같은 주)를 더한다. shift=52 는 위약: 같은 계절의 다른 해.
    """
    g = raw[(raw.series == series) & (raw.code == PRECIP)]
    wide, temp = g.pivot(index="week", columns="col_name", values="value"), pd.read_parquet(WEATHER_FILE)
    out = {}
    for c in wide:
        v = _views(wide[c].dropna())
        for name, (col, agg) in WEATHER_EXTRA.get(c, {}).items():
            v[name] = _views(temp[temp.col_name == col].set_index("week")[agg])["같은 주"]
        out[c] = v.shift(shift)
    return out


def release_panel(E, kind, fit_end, rel=None, robust=False):
    """E(event_panel)에서 RELEASES 의 열만 **실제 발표 주**의 사건으로 바꾼 패널. 나머지 열은 그대로다.

    kind = "change": 직전 발표 대비 실제치 변화(기대 = 지난번 값), "surprise": 실제 − 시장 예상(기대 = 컨센서스).
    사건마다 fit_end 이전 발표들의 표준편차로 나눈다. 평균은 빼지 않는다 — 0 은 "기대와 같다"는 뜻이어야 한다.
    발표가 없는 주는 0. 한 주에 같은 사건이 두 번 발표되면 나중 것.
    robust=True 면 표준편차 대신 1.4826 × |값|의 중앙값으로 나누고 ±5 에서 자른다 — 2020년 고용처럼 몇 건이 척도를 정하는 것을 막는다.
    """
    rel = pd.read_parquet(RELEASE_FILE) if rel is None else rel
    r, out = rel.pivot(index=["event", "dt"], columns="field", values="value").sort_index(), E.copy()
    for c, events in RELEASES.items():
        if c not in E:
            continue
        zs = []
        for e in events:
            v = r.loc[e].actual - r.loc[e].forecast if kind == "surprise" else r.loc[e].actual.diff()
            past = v[v.index < fit_end]
            v = (v / (1.4826 * past.abs().median())).clip(-5, 5) if robust else v / past.std()
            zs.append(v.groupby(v.index.to_period("W-SUN").end_time.normalize()).last())
        out[c] = pd.concat(zs, axis=1).mean(axis=1).reindex(E.index).fillna(0.0)
    return out


def load_key(series):
    """답지(주별 원인표): index = 주 일요일, type = 요인 유형, cited = 원인으로 인용된 col_name 집합, ret = 그 주 등락률."""
    k = pd.read_excel(KEY, sheet_name=SERIES[series]).dropna(subset=["주 일요일"]).set_index("주 일요일")
    cited = k["커버 시계열 (col_name)"].fillna("").str.split(r"[\n,;]+").map(lambda v: {x.strip() for x in v} - {""})
    return pd.DataFrame({"type": k["요인 유형"].fillna("기타"), "cited": cited, "ret": k["등락률(%)"].astype(float)})


# ---- 같은 주 방향 가정 ----

def sinks(X, info):
    """같은 주 링크를 `변수 → 가격` 으로 가정하는 변수(열 순서의 bool). 거시·기상·수급 변수는 한 원자재 가격의 같은 주 결과가 아니라고 본다."""
    code = info.set_index("변수").code
    return np.array([c != "price" and not re.search(REACTIVE, code[c]) for c in X.columns])


def link_assumptions(sink, pag=False, tau_max=TAU):
    """tigramite `link_assumptions`: 모든 링크는 있을 수도 없을 수도 있고, sink 변수와 가격(0번)의 같은 주 링크만 방향을 정한다.

    pag=False (PCMCI+): `변수 -?> 가격`. pag=True (LPCMCI): `변수 o?> 가격` — 가격은 그 변수의 원인이 아니다(잠재 교란은 허용).
    """
    d = len(sink)
    lagged, into, back = ("o?>", "o?>", "<?o") if pag else ("-?>", "-?>", "<?-")
    la = {j: {(i, -t): lagged if t else "o?o" for i in range(d) for t in range(tau_max + 1) if t or i != j} for j in range(d)}
    for i in map(int, np.flatnonzero(sink)):
        la[0][i, 0], la[i][0, 0] = into, back
    return la


# ---- 답지 채점 ----

def chance_hit(n_cited, m, N):
    """N 개 중 무작위 m 개가 인용 변수 n_cited 개 가운데 하나 이상을 포함할 확률(초기하분포)."""
    return 1 - comb(N - n_cited, m) / comb(N, m)


def _cited(X, key):
    """답지 기간의 주별 인용 변수 중 패널에 있는 것. 반환: (전체 주의 Series, 입력 밖 인용 건수)."""
    cols = set(X.columns) - {"price"}
    key = key[key.index.isin(X.index)]
    return key.cited.map(lambda v: v & cols), int(key.cited.map(lambda v: len(v - cols)).sum())


def score_sets(S, X, key):
    """기간 전체의 변수 집합 채점. S = 변수 집합 하나, 또는 주마다 다른 집합의 Series(레짐).

    포함률 = 인용 변수가 패널에 있는 주 중 그 주의 집합이 인용 변수를 하나 이상 포함한 비율. 시차·방향·그 주의 크기는 보지 않는다.
    우연 = 같은 크기의 무작위 변수 집합의 기대 포함률.
    """
    cited, _ = _cited(X, key)
    cited, N = cited[cited.map(bool)], X.shape[1] - 1
    S = S.reindex(cited.index) if isinstance(S, pd.Series) else pd.Series([S] * len(cited), index=cited.index)
    return {"집합 크기": S.map(len).mean(), "답지 주 수": len(cited),
            "포함률": np.mean([bool(c & s) for c, s in zip(cited, S)]),
            "우연": np.mean([chance_hit(len(c), len(s), N) for c, s in zip(cited, S)])}


def attribution(X, parents, rows=None):
    """가격을 부모 항 {(변수, 시차 τ ≥ 0)} 에 회귀해 주별 기여 Σ_τ β·x(t−τ) 를 변수별로 낸다(주 × 가격 외 변수, 부모가 아니면 0).

    rows = 계수를 적합할 주(bool, 기본 전체). 가격 자기 시차는 통제 변수로만 쓰고 기여에서 뺀다.
    """
    # ponytail: 결측 항은 0(z-score 평균 = 기여 없음)으로 둔다. 늦게 시작하는 변수의 계수가 0 쪽으로 눌린다 — 문제가 되면 변수 시작 뒤 구간만 적합.
    Z = X.fillna(0)
    D = pd.DataFrame({(c, t): Z[c].shift(t).fillna(0) for c, t in sorted(parents)}, index=X.index)
    ok = (X.price.notna() & (True if rows is None else rows)).values
    beta = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), D.values[ok]]), X.price.values[ok], rcond=None)[0][1:]
    out = pd.DataFrame(0.0, index=X.index, columns=X.columns.drop("price"))
    for (c, _), b in zip(D.columns, beta):
        if c != "price":
            out[c] += b * D[c, _]
    return out


def score_weeks(contrib, X, key, ks=(1, 3)):
    """주별 귀속 채점 — 답지의 질문("이번 주 가격을 움직인 것")에 맞춘다.

    top-k 적중률 = 그 주 |기여| 상위 k 개(기여 0 은 제외)에 답지 인용 변수가 있는 주의 비율. 우연 = 무작위 k 개의 기대 적중률.
    분모는 인용 변수가 패널에 있는 주. `(전체 주)` 는 답지 기간의 모든 주가 분모다(입력 밖 인용·인용 없는 주는 실패).
    방향 일치 = top-k(가장 큰 k) 적중 주에서 인용 변수 기여 합의 부호가 답지 등락률의 부호와 같은 비율.
    """
    cited, outside = _cited(X, key)
    n_all, cited, N = len(cited), cited[cited.map(bool)], contrib.shape[1]
    out = {"답지 주 수": len(cited), "전체 주 수": n_all, "입력 밖 인용": outside}
    for k in ks:
        hit = week_hits(contrib, X, key, k)
        out |= {f"top{k}": hit.mean(), f"top{k} 우연": np.mean([chance_hit(len(c), k, N) for c in cited]),
                f"top{k} (전체 주)": hit.sum() / n_all}
    top, ret = _top(contrib, cited.index, k), key.ret.reindex(cited.index)
    push = pd.Series([contrib.loc[w, sorted(c & set(top.columns[t]))].sum() for w, c, t in zip(cited.index, cited, top.values)], index=cited.index)
    judged = hit & (ret != 0) & (push != 0)
    out[f"방향 일치 (top{k} 적중 주)"] = (np.sign(push) == np.sign(ret))[judged].mean() if judged.any() else np.nan
    return out


def _top(contrib, weeks, k):
    """주별 |점수| 상위 k 개(점수 0 은 제외, 동점은 열 순서)."""
    a = contrib.loc[weeks].abs()
    return (a.rank(axis=1, ascending=False, method="first") <= k) & (a > 0)


def week_hits(contrib, X, key, k):
    """주별 적중 여부: 그 주 상위 k 개에 답지 인용 변수가 있는가. index = 인용 변수가 패널에 있는 답지 주."""
    cited, _ = _cited(X, key)
    cited = cited[cited.map(bool)]
    top = _top(contrib, cited.index, k)
    return pd.Series([bool(c & set(top.columns[t])) for c, t in zip(cited, top.values)], index=cited.index)


def prior_scores(X, key):
    """빈도 사전확률: 주 × 변수 = 그 주 **이전** 답지 주들에서 그 변수가 인용된 횟수(확장 창). 그 주의 답은 쓰지 않는다."""
    cols = X.columns.drop("price")
    cited, _ = _cited(X, key)
    seen = pd.DataFrame([[c in v for c in cols] for v in cited], index=cited.index, columns=cols, dtype=float)
    return seen.cumsum().shift(1).fillna(0)


def paired_block_ci(a, b, block=4, n=2000, seed=0):
    """같은 주끼리 짝지은 적중 여부 a, b 의 적중률 차이 (a − b) 와 95% 구간. 연속한 block 주를 한 덩어리로 다시 뽑는다(주간 자기상관)."""
    d = (a.astype(float) - b.astype(float)).values
    blocks = [d[i:i + block] for i in range(0, len(d), block)]
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]).mean() for _ in range(n)]
    return d.mean(), *np.percentile(draws, [2.5, 97.5])


def pcmci_parents(X, sink):
    """PCMCI+ (같은 주 방향 가정 포함) 가 찾은 가격의 부모 항 {(변수, 시차 τ ≥ 0)}. 가격 자기 시차는 뺀다."""
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    frame = pp.DataFrame(np.nan_to_num(X.values, nan=MISSING), missing_flag=MISSING)
    g = PCMCI(frame, ParCorr(significance="analytic"), verbosity=0).run_pcmciplus(
        tau_min=0, tau_max=TAU, pc_alpha=ALPHA, link_assumptions=link_assumptions(sink))["graph"]
    return {(X.columns[i], int(t)) for i, t in zip(*np.nonzero(g[:, 0, :] == "-->")) if i}


def demo():
    """채점 자체 점검: 변수 a 만 가격을 움직이는 합성 패널."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=["price", "a", "b", "c"], index=pd.date_range("2020-01-05", periods=300, freq="W-SUN"))
    X["price"] = 0.8 * X.a.shift(1).fillna(0)  # 잡음 없음: 기여의 부호 = 가격의 부호
    key = lambda v: pd.DataFrame({"cited": [{v, "밖"}] * 100, "ret": X.price[-100:].values}, index=X.index[-100:])
    c = attribution(X, {("a", 1), ("price", 1)})
    good, bad = score_weeks(c, X, key("a")), score_weeks(c, X, key("b"))
    assert good["top1"] == 1 and good["방향 일치 (top3 적중 주)"] == 1 and bad["top3"] == 0 and good["입력 밖 인용"] == 100, (good, bad)
    assert abs(good["top1 우연"] - 1 / 3) < 1e-12 and chance_hit(3, 10, 10) == 1 and chance_hit(3, 0, 10) == 0
    assert score_sets({"a"}, X, key("a"))["포함률"] == 1 and score_sets({"b"}, X, key("a"))["포함률"] == 0
    h = week_hits(c, X, key("a"), 1)
    assert h.mean() == good["top1"] and paired_block_ci(h, h) == (0, 0, 0)
    pr = prior_scores(X, key("a"))
    assert (pr.iloc[0] == 0).all() and pr.a.iloc[-1] == 99 and pr.b.iloc[-1] == 0  # 그 주의 답은 세지 않는다
    assert week_hits(pr, X, key("a"), 1).tolist() == [False] + [True] * 99
    la = link_assumptions(np.array([False, True, False, False]))
    assert la[0][1, 0] == "-?>" and la[1][0, 0] == "<?-" and la[0][2, 0] == "o?o" and la[2][1, -1] == "-?>" and (0, 0) not in la[0]
    col, fri = "Com_Econ_US_nonfarm_payrolls", pd.to_datetime(["2020-02-07", "2020-03-06", "2020-04-05"])  # 금·금·일요일
    rel = pd.DataFrame({"event": col, "dt": [*fri, *fri], "field": ["actual"] * 3 + ["forecast"] * 3, "value": [100, 150, 170, 120, 120, 150]})
    R = {k: release_panel(X.rename(columns={"a": col}), k, "2021", rel)[col] for k in ("surprise", "change")}
    at = lambda k: R[k][R[k] != 0].round(6).to_dict()
    weeks = list(pd.to_datetime(["2020-02-09", "2020-03-08", "2020-04-05"]))  # 일요일 발표는 그 주에 남는다
    assert list(at("surprise")) == weeks and np.sign(list(at("surprise").values())).tolist() == [-1, 1, 1]
    assert list(at("change")) == weeks[1:] and min(at("change").values()) > 0  # 첫 발표는 직전 값이 없다
    wk = pd.Series(rng.normal(size=1040), index=pd.date_range("2000-01-02", periods=1040, freq="W-SUN"))  # 20년: 한 주의 편차가 충분히 커질 수 있다
    wk["2018-03-04"] = 50.0
    spike = lambda shift: _views(wk, shift)[["지난주", "같은 주", "다음 주"]].idxmax().dt.strftime("%Y-%m-%d").tolist()
    assert spike(0) == ["2018-03-11", "2018-03-04", "2018-02-25"] and spike(52)[1] == "2019-03-03"  # 위약은 52주 뒤
    print("자체 점검 통과")


if __name__ == "__main__":
    snapshot()
    demo()
