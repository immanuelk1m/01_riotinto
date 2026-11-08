"""E028 stage 0 — predictability ceiling for the six fixed targets.

No model is fitted and no leaderboard row is written. The question is only how large a
MAPE reduction over persistence the structure of each target allows.

    uv run --with xlrd python experiment/E028/E028.py prepare          # Mac mini: Postgres + EIA -> parquet
    uv run python -m _colab.job run --exp E028 --cmd ceiling --include _exo

Identity per target (log, h weeks):  r = f + b,  f = same-contract futures change (efficient,
unpredictable by assumption), b = everything else (basis / premium / carry / roll).
Strict bound for any forecaster whose information makes f a martingale difference:
    R2_oos <= E[(b + f_known)^2] / E[r^2],   f_known = the part of f already traded before the
    origin's information cutoff (async allowance, declared in CEILING.md).
`prepare` truncates every series at CUTOFF = last common target week - (SEALED_ORIGINS + H)
weeks, so neither the sealed block nor its targets reach the VM.
"""
import hashlib
import json
import pathlib
import sys
import urllib.request

import numpy as np
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent
for candidate in (BASE, BASE.parent, BASE.parent.parent):
    sys.path.insert(0, str(candidate))

from _exo import engine  # noqa: E402
from _exo.params import P, PARAMS_HASH  # noqa: E402

TARGETS = {
    "gold": "Com_LBMA_Gold_Price_PM_USD_V_0",
    "copper": "Com_LME_Cu_Cash_Bid",
    "corn": "Com_Barchart_Cash_National_Corn_Index",
    "wheat": "Com_Barchart_Cash_Soft_Red_Wheat_Index",
    "coffee": "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag",
    "wti": "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel",
}
COMPONENTS = {
    "gold_am": "Com_LBMA_Gold_Price_AM_USD_V_0",
    "effr": "Macro_NYFed_Effective_Federal_Funds_Rate_Pct",
    "cu_3m": "Com_LME_Cu_3M_Bid",
    "corn_basis": "Com_Barchart_US_Corn_Basis_Index",
    "brent": "Com_EIA_Daily_Europe_Brent_Spot_Price_FOB_USD_Per_Barrel",
}
#: NYMEX CL contracts 1-2 by position, free from EIA (publication stopped 2024-04)
EIA = {f"cl{k}": f"https://www.eia.gov/dnav/pet/hist_xls/RCLC{k}d.xls" for k in "12"}
#: series whose futures component is measurable from the DB + EIA; the rest are bounded at 100 %
MEASURED = ("gold", "copper", "corn", "wti")
#: async allowance in trading days of f already traded at the origin (declared; swing in CEILING.md)
ASYNC_DAYS = (0.5, 1.0, 2.0)
PRIMARY_ASYNC = 1.0
#: OOS R2 granted to the futures component itself (0 = efficient; 0.25 = best published, B&K 2012)
FUTURES_R2 = (0.0, 0.05, 0.25)


# --------------------------------------------------------------------------- prepare (mini)

def prepare(base: pathlib.Path, full: bool = False) -> None:
    """dev snapshot (default): everything after CUTOFF dropped. `full`: the whole DB history, written to
    data/full only when the sealed block is opened (stage 4) or a forward forecast is issued; the
    campaign is DB-only, so `full` skips the EIA files."""
    import psycopg

    raw, daily = base / "data" / "raw", base / "data" / ("full" if full else "daily")
    raw.mkdir(parents=True, exist_ok=True)
    daily.mkdir(parents=True, exist_ok=True)
    frames, sha = {}, {}
    with psycopg.connect("dbname=commodity") as con, con.cursor() as cur:
        for key, col in {**TARGETS, **COMPONENTS}.items():
            cur.execute("select dt, value from public.value where col_name = %s order by dt", (col,))
            frames[key] = pd.DataFrame(cur.fetchall(), columns=["dt", "value"])
    for key, url in ({} if full else EIA).items():
        path = raw / f"{key}.xls"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        sha[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        sheet = pd.read_excel(path, sheet_name="Data 1", skiprows=2)
        frames[key] = sheet.set_axis(["dt", "value"], axis=1)

    for key, frame in frames.items():
        frame["dt"] = pd.to_datetime(frame["dt"])
        frame["value"] = frame["value"].astype(float)
        frames[key] = frame.dropna().sort_values("dt")
    last_common = min(frames[k].dt.max() for k in TARGETS)
    last_week = last_common.to_period("W-SUN").end_time.normalize()
    cutoff = last_week - pd.Timedelta(weeks=int(P.SEALED_ORIGINS) + int(P.H))

    manifest = dict(last_common_target=str(last_common.date()), cutoff=str(cutoff.date()),
                    params_hash=PARAMS_HASH, eia_sha256=sha, series={})
    for key, frame in frames.items():
        kept = frame if full else frame[frame.dt <= cutoff]
        kept.to_parquet(daily / f"{key}.parquet", index=False)
        manifest["series"][key] = dict(
            col_name={**TARGETS, **COMPONENTS}.get(key, EIA.get(key)), rows=len(kept),
            first=str(kept.dt.min().date()), last=str(kept.dt.max().date()),
            dropped_after_cutoff=int(len(frame) - len(kept)),
            sha256=hashlib.sha256((daily / f"{key}.parquet").read_bytes()).hexdigest(),
        )
        print(f"[prepare] {key}: {len(kept)} rows {manifest['series'][key]['first']}"
              f"..{manifest['series'][key]['last']}", flush=True)
    (base / "data" / ("manifest_full.json" if full else "manifest.json")).write_text(
        json.dumps(manifest, indent=2))
    print(f"[prepare] cutoff {cutoff.date()} (last common target {last_common.date()})", flush=True)


# --------------------------------------------------------------------------- ceiling (Colab)

def _load(base: pathlib.Path, sub: str = "daily") -> dict:
    out = {}
    for path in sorted((base / "data" / sub).glob("*.parquet")):
        frame = pd.read_parquet(path)
        out[path.stem] = frame.assign(dt=frame.dt.astype("datetime64[ns]")).sort_values("dt")
    return out


def _weekly(frame: pd.DataFrame) -> pd.DataFrame:
    """W-SUN last of a daily series, keeping the date the value was fixed on."""
    week = frame.dt.dt.to_period("W-SUN").dt.end_time.dt.normalize().astype("datetime64[ns]")
    return frame.assign(week=week).groupby("week")[["dt", "value"]].last()


def _asof(dates: pd.Series, frame: pd.DataFrame, strict: bool = False) -> np.ndarray:
    """Value of `frame` on the last day at or before each date (strictly before if `strict`)."""
    left = pd.DataFrame({"dt": dates.to_numpy()}).sort_values("dt")
    joined = pd.merge_asof(left, frame.sort_values("dt"), on="dt", direction="backward",
                           allow_exact_matches=not strict)
    return joined.set_index(left.index).sort_index()["value"].to_numpy()


def _cl_expiries(cl1: pd.DataFrame) -> np.ndarray:
    """NYMEX CL last trade: 3rd business day before the 25th of the prior month (4th if the 25th
    is not a business day). Business days = the dates EIA reports a settle on."""
    days = pd.DatetimeIndex(cl1.dt)
    out = []
    for month in pd.period_range(days.min(), days.max(), freq="M"):
        d25 = month.start_time + pd.Timedelta(days=24)
        before = days[days < d25]
        rank = 3 if d25 in days else 4
        if len(before) >= rank:
            out.append(before[-rank])
    return np.array(out, dtype="datetime64[ns]")


def _spliced(ret: pd.Series) -> pd.DataFrame:
    """Cumulative log index of daily same-contract returns: a futures series with no roll jumps."""
    return ret.fillna(0).cumsum().rename("value").rename_axis("dt").reset_index()


def _cl_index(cl1: pd.DataFrame, cl2: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Roll-adjusted CL front. Each month the roll day is the one of expiry+0..2 trading days on
    which CL1 best continues yesterday's CL2; that day's return is log(CL1 / CL2[-1])."""
    j = pd.concat([cl1.set_index("dt").value.rename("c1"), cl2.set_index("dt").value.rename("c2")],
                  axis=1, join="inner")
    j = j[(j > 0).all(axis=1)]
    stay, lag = np.log(j.c1 / j.c1.shift(1)), np.log(j.c1 / j.c2.shift(1))
    gain, ret = (stay.abs() - lag.abs()).to_numpy(), stay.to_numpy().copy()
    picked, wide = [], []
    calm = np.nanmedian(stay.abs())   # a spread below one median daily move cannot be located
    for e in _cl_expiries(cl1):
        i = j.index.searchsorted(e)
        cand = [k for k in (i, i + 1, i + 2) if 0 < k < len(j)]
        if cand:
            k = max(cand, key=lambda c: gain[c])
            ret[k] = lag.iloc[k]
            picked.append(k - i)
            wide.append(abs(np.log(j.c2.iloc[i] / j.c1.iloc[i])) > calm)
    hit = np.array(picked) == 1
    return _spliced(pd.Series(ret, index=j.index)), float(hit[np.array(wide)].mean())


def _corn_index(cash: pd.DataFrame, basis: pd.DataFrame) -> pd.DataFrame:
    """Nearby CBOT corn = cash - basis (cents/bu; basis is USD/bu). On a roll day the futures
    return is taken from cash, i.e. the basis is held unchanged across the roll."""
    j = pd.concat([cash.set_index("dt").value.rename("cash"),
                   basis.set_index("dt").value.rename("basis")], axis=1, join="inner")
    ret = np.log(j.cash - j.basis * 100).diff()
    month = j.index.month.to_numpy()
    roll = np.r_[False, month[1:] != month[:-1]] & np.isin(month, P.CORN_MONTHS)
    ret[roll] = np.log(j.cash).diff()[roll]
    return _spliced(ret)


def cells(data: dict) -> tuple[pd.DataFrame, dict]:
    """One row per (series, origin week, h) with r, the measurable non-futures component b, and f."""
    rows = []
    cl_index, cl_rule = _cl_index(data["cl1"], data["cl2"])
    corn_index = _corn_index(data["corn"], data["corn_basis"])
    for series in TARGETS:
        weekly = _weekly(data[series])
        weekly = weekly[weekly.value > 0]
        logp = np.log(weekly.value.to_numpy())
        dates = weekly.dt
        comp = {}
        if series == "gold":
            comp["effr"] = _asof(dates, data["effr"])
        if series == "copper":
            comp["log3m"] = np.log(_asof(dates, data["cu_3m"]))
        if series == "corn":
            comp["G"] = _asof(dates, corn_index)
        if series == "wti":
            comp["G"] = _asof(dates, cl_index)
            comp["cl_last"] = np.datetime64(cl_index.dt.max())
        n = len(logp)
        for h in range(1, int(P.H) + 1):
            t = np.arange(n - h)
            r = logp[t + h] - logp[t]
            b = np.full(len(t), np.nan)
            if series == "gold":  # spot = futures less carry; full EFFR carry is an upper bound
                b = comp["effr"][t] / P.PCT * h * 7 / 365
            if series == "copper":  # 3M is a rolling prompt: add back the curve roll over h weeks
                slope = (comp["log3m"] - logp) / 13
                b = r - (comp["log3m"][t + h] - comp["log3m"][t]) + slope[t + h] * h
            if series in ("corn", "wti"):  # f = roll-adjusted futures index change
                b = r - (comp["G"][t + h] - comp["G"][t])
            if series == "wti":  # EIA stopped in 2024-04: no as-of past the last settle
                b = np.where(dates.to_numpy()[t + h] <= comp["cl_last"], b, np.nan)
            rows.append(pd.DataFrame(dict(series=series, h=h, week=weekly.index[t], r=r, b=b)))
    out = pd.concat(rows, ignore_index=True)
    out["f"] = out.r - out.b
    out["ape_pers"] = np.minimum(P.CAP, np.abs(np.expm1(-out.r)))        # PROTOCOL §5.1
    out["ape_pers_engine"] = np.minimum(P.CAP, np.abs(np.expm1(out.r)))  # _exo/engine.py:288
    out["ape_oracle_b"] = np.minimum(P.CAP, np.abs(np.expm1(-out.f)))    # knows b exactly
    return out, dict(cl_roll_at_rule_plus1_wide_spread=cl_rule)


def bounds(c: pd.DataFrame, async_days: float, futures_r2: float = 0.0) -> pd.DataFrame:
    """Per (series, h): variance shares and the MAPE ceiling."""
    g = c.groupby(["series", "h"])
    t = pd.DataFrame(dict(
        n=g.size(), mape=g.ape_pers.mean() * P.PCT,
        m2r=g.r.apply(lambda x: (x ** 2).mean()),
        m2b=g.b.apply(lambda x: (x ** 2).mean()),
        m2f=g.f.apply(lambda x: (x ** 2).mean()),
        oracle=g.ape_oracle_b.mean() * P.PCT,
    )).reset_index()
    measured = t.m2b.notna()
    t["m2f"] = t.m2f.fillna(t.m2r)
    t["share_b"] = t.m2b / t.m2r
    known = async_days / (5 * t.h)                      # share of f traded before the cutoff
    t["share_f_pred"] = (known + futures_r2 * (1 - known)) * t.m2f / t.m2r
    t["r2_max"] = np.minimum(1.0, (np.sqrt(t.share_b) + np.sqrt(t.share_f_pred)) ** 2)
    t["c_gauss"] = 1 - np.sqrt(1 - t.r2_max)
    t["c_oracle"] = 1 - t.oracle / t.mape
    t["ceiling"] = np.where(measured, np.fmax(t.c_gauss, t.c_oracle), np.nan)
    # unmeasured series: strict = perfect forecast; indicative = most basis-heavy measured series at h
    proxy = t[measured].groupby("h").ceiling.max()
    t["ceiling_strict"] = t.ceiling.fillna(1.0)
    t["ceiling_indicative"] = t.ceiling.fillna(t.h.map(proxy))
    t["measured"] = measured
    return t


def pooled(t: pd.DataFrame) -> dict:
    """Equal weight per series and per h: reduction = sum(mape * c) / sum(mape).
    `need_unmeasured` = reduction wheat and coffee would each need for the pool to reach 50 %."""
    total = t.mape.sum()
    out = {k: float((t.mape * t[f"ceiling_{k}"]).sum() / total) for k in ("strict", "indicative")}
    measured = float((t.mape * t.ceiling.fillna(0)).sum() / total)
    weight = float(t.mape[~t.measured].sum() / total)
    return dict(out, measured_part=measured, unmeasured_weight=weight,
                need_unmeasured=(0.5 - measured) / weight)


def r2_required(c: pd.DataFrame) -> pd.DataFrame:
    """R2 that halves MAPE if the residual is a scaled copy of the return (scale family)."""
    from scipy.optimize import brentq

    out = []
    for (series, h), x in c.groupby(["series", "h"]):
        r = x.r.to_numpy()
        target = np.minimum(P.CAP, np.abs(np.expm1(-r))).mean() / 2
        s = brentq(lambda k: np.minimum(P.CAP, np.abs(np.expm1(-k * r))).mean() - target, 0, 1)
        out.append(dict(series=series, h=h, scale=s, r2_required=1 - s * s))
    return pd.DataFrame(out)


def run_ceiling(base: pathlib.Path) -> int:
    data = _load(base)
    manifest = json.loads((base / "data" / "manifest.json").read_text())
    allc, diag = cells(data)
    # an origin counts only when all H horizons exist (and, for measured series, all H b's)
    usable = allc[~(allc.series.isin(MEASURED) & allc.b.isna())]
    full = usable.groupby(["series", "week"]).h.transform("size") == int(P.H)
    usable = usable[full]
    span_by = usable[usable.series.isin(MEASURED)].groupby("series").week
    start, end_t = span_by.min().max(), span_by.max().min()
    common = usable[(usable.week >= start) & (usable.week <= end_t)]

    results = dict(manifest=manifest, window=dict(first_origin=str(start.date()),
                   last_origin=str(end_t.date())),
                   cells=common.groupby("series").size().to_dict(),
                   **diag, swing={}, swing_futures_r2={})
    tables = {}
    for days in ASYNC_DAYS:
        t = bounds(common, days)
        tables[days] = t
        results["swing"][days] = pooled(t)
    for r2f in FUTURES_R2:
        results["swing_futures_r2"][r2f] = pooled(bounds(common, PRIMARY_ASYNC, r2f))
    t = tables[PRIMARY_ASYNC]
    results["pooled"] = results["swing"][PRIMARY_ASYNC]
    results["pooled_mape_persistence"] = float(t.groupby("series").mape.mean().mean())
    results["series_weight"] = (t.groupby("series").mape.sum() / t.mape.sum()).to_dict()

    # every 52-origin window inside the common window (the sealed block's length)
    weeks = np.sort(common.week.unique())
    span = int(P.SEALED_ORIGINS)
    roll = []
    for i in range(len(weeks) - span + 1):
        w = common[(common.week >= weeks[i]) & (common.week <= weeks[i + span - 1])]
        roll.append(dict(start=str(pd.Timestamp(weeks[i]).date()), **pooled(bounds(w, PRIMARY_ASYNC))))
    roll = pd.DataFrame(roll)
    results["rolling_52"] = dict(
        windows=len(roll),
        strict_max=float(roll.strict.max()), strict_max_at=roll["start"][roll.strict.idxmax()],
        indicative_max=float(roll.indicative.max()),
        indicative_max_at=roll["start"][roll.indicative.idxmax()],
        strict_median=float(roll.strict.median()), indicative_median=float(roll.indicative.median()),
        worst=roll.loc[roll.strict.idxmax()].to_dict(),
    )

    # gold fix-time component: share of daily variance between the AM (10:30) and PM (15:00) fixes
    fix = pd.merge(data["gold"], data["gold_am"], on="dt", suffixes=("_pm", "_am"))
    fix = fix[fix.dt >= start]
    intraday = np.log(fix.value_pm / fix.value_am)
    daily = np.log(fix.value_pm).diff()
    results["gold_am_pm_share_of_daily_var"] = float((intraday ** 2).mean() / (daily ** 2).mean())

    persist = common.groupby("series")[["ape_pers", "ape_pers_engine"]].mean() * P.PCT
    results["persistence_formula_gap"] = persist.to_dict()
    req = r2_required(common)
    t = t.merge(req, on=["series", "h"])
    out = base / "results"
    out.mkdir(exist_ok=True)
    t.to_csv(out / "ceiling_by_series_h.csv", index=False)
    roll.to_csv(out / "ceiling_rolling52.csv", index=False)
    (out / "ceiling.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str), flush=True)
    cols = ["series", "h", "n", "mape", "share_b", "share_f_pred", "r2_max", "c_gauss", "c_oracle",
            "ceiling_strict", "ceiling_indicative", "r2_required"]
    print(t[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    return 0


# =========================================================================== campaign (stages 1-5)
# DB-only (user instruction 2026-11-08). Registration: README.md. Every forecast is a closed-form OLS of
# the DB-observable component b on its own level and calendar month, fitted on labels already mature
# at the origin; the market component f is forecast as zero.

MODELS = ("fut", "lvl", "mon", "struct")
UNMEASURED = ("wheat", "coffee")          # no DB component: every model is persistence
STATEMENTS = ("pooled", "gold", "copper", "corn", "wti")


def components(data: dict) -> dict:
    """Weekly frame per target: log price and the DB components of its decomposition."""
    corn_index = _corn_index(data["corn"], data["corn_basis"])
    out = {}
    for series in TARGETS:
        weekly = _weekly(data[series])
        weekly = weekly[weekly.value > 0]
        dates = weekly.dt
        frame = pd.DataFrame({"date": dates.to_numpy(), "logp": np.log(weekly.value.to_numpy())},
                             index=weekly.index)
        if series == "gold":  # EFFR is published the next business day: never the same day
            frame["effr"] = _asof(dates, data["effr"], strict=True)
        if series == "copper":
            frame["logc"] = np.log(_asof(dates, data["cu_3m"]))
        if series == "corn":
            frame["logc"] = _asof(dates, corn_index)
            frame["basis"] = _asof(dates, data["corn_basis"])
        if series == "wti":
            frame["logc"] = np.log(_asof(dates, data["brent"]))
        out[series] = frame
    return out


def labels(series: str, frame: pd.DataFrame) -> dict:
    """r, b (component the model may predict) and fut (futures-implied forecast) as (n, H); level x (n,)."""
    logp = frame.logp.to_numpy()
    n, horizons = len(logp), np.arange(1, int(P.H) + 1)
    ahead = np.arange(n)[:, None] + horizons[None, :]
    mature = ahead < n

    def at(values):
        return np.where(mature, values[np.minimum(ahead, n - 1)], np.nan)

    r = at(logp) - logp[:, None]
    out = dict(r=r, b=np.full_like(r, np.nan), fut=np.full_like(r, np.nan), x=np.full(n, np.nan))
    if series == "gold":
        carry = frame.effr.to_numpy()[:, None] / P.PCT * P.DAYS_PER_WEEK * horizons / P.CARRY_DAYCOUNT
        out.update(b=carry, fut=carry)
    if series in ("copper", "corn", "wti"):
        logc = frame.logc.to_numpy()
        out["b"] = r - (at(logc) - logc[:, None])
    if series == "copper":  # 3M is a rolling prompt: the fixed contract rolls down the curve
        x = logc - logp
        out.update(b=out["b"] + at(x / P.WEEKS_3M) * horizons, x=x,
                   fut=x[:, None] / P.WEEKS_3M * horizons)
    if series == "corn":  # cash cents/bu, basis USD/bu; no deferred contract in the DB, so fut = 0
        out.update(x=logp - np.log(np.exp(logp) - frame.basis.to_numpy() * 100), fut=np.zeros_like(r))
    if series == "wti":
        out["x"] = logp - logc
    return out


def forecast(series: str, model: str, lab: dict, frame: pd.DataFrame, leak=None) -> np.ndarray:
    """(n, H) log-return forecasts. OLS per h, refit every REFIT_EVERY weeks on mature labels only."""
    n = len(frame)
    if series in UNMEASURED:
        return np.zeros((n, int(P.H)))
    if model == "fut" or series == "gold":  # gold: b is the carry itself, nothing to estimate
        return np.nan_to_num(lab["fut"])     # no curve in the DB -> futures-implied = persistence
    month = frame.date.dt.month.to_numpy()
    cols = []
    if model in ("lvl", "struct"):
        cols.append(lab["x"][:, None])
    if model in ("mon", "struct"):
        cols.append((month[:, None] == np.unique(month)[None, 1:]).astype(float))
    if leak is not None:
        cols.append(leak[:, None])
    X = np.column_stack(cols)
    usable = np.isfinite(X).all(1)
    rows = np.arange(n)
    fc = np.full((n, int(P.H)), np.nan)
    for k in range(int(P.H)):
        y = lab["b"][:, k]
        good = usable & np.isfinite(y)
        for start in range(0, n, int(P.REFIT_EVERY)):
            train = good & (rows + k + 1 <= start)   # label of row j at horizon k+1 matures at j+k+1
            if train.sum() < int(P.MIN_TRAIN):
                continue
            block = slice(start, start + int(P.REFIT_EVERY))
            pred = engine._apply(engine._ols(X[train], y[train]), np.nan_to_num(X[block]))
            fc[block, k] = np.where(usable[block], pred, np.nan)
    return fc


def campaign_cells(data: dict, models=MODELS, synth=None, leak: bool = False) -> pd.DataFrame:
    rows = []
    for series, frame in components(data).items():
        if synth is not None:
            frame = synth(series, frame)
        lab = labels(series, frame)
        origin = frame.index.to_period("W-SUN").asi8
        for model in models:
            fc = forecast(series, model, lab, frame, lab["r"][:, 0] if leak else None)
            for k in range(int(P.H)):
                rows.append(pd.DataFrame(dict(
                    series=series, model=model, origin=origin, week=frame.index, horizon=k + 1,
                    logp=frame.logp.to_numpy(), target_logret=lab["r"][:, k], fc=fc[:, k],
                    b=lab["b"][:, k])))
    return pd.concat(rows, ignore_index=True)


def common_origins(cells: pd.DataFrame, model: str = "struct", series=tuple(TARGETS)) -> np.ndarray:
    """Origins where every listed series has all H mature cells for `model` (equal weight per series)."""
    c = cells[(cells.model == model) & cells.series.isin(series)].dropna(subset=["target_logret", "fc"])
    full = c.groupby(["series", "origin"]).horizon.size().eq(int(P.H))
    per = full[full].reset_index().groupby("origin").series.nunique()
    return per[per == len(series)].index.to_numpy()


def scored(cells: pd.DataFrame, model: str, origins, series=tuple(TARGETS)) -> pd.DataFrame:
    c = cells[(cells.model == model) & cells.origin.isin(origins) & cells.series.isin(series)]
    return engine.score(c.dropna(subset=["target_logret", "fc"]))


def summary(sc: pd.DataFrame) -> dict:
    boot = engine.block_bootstrap(sc)
    pers, cand = float(sc.mape_pers.mean()), float(sc.mape_cand.mean())
    return dict(cells=int(len(sc)), origins=int(sc.origin.nunique()), mean_d=boot["mean"],
                ci_low=boot["lo"], ci_high=boot["hi"], blocks=boot["blocks"], mape_pers=pers,
                mape_cand=cand, reduction=1 - cand / pers,
                reduction_ci=[boot["lo"] * P.PCT / pers, boot["hi"] * P.PCT / pers])


def statement(sc: pd.DataFrame, name: str) -> pd.DataFrame:
    return sc if name == "pooled" else sc[sc.series == name]


# ------------------------------------------------------------------------------- statistics

def _lrv(x: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett (Newey-West) long-run covariance of the rows of x."""
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    x = x - x.mean(0)
    n = len(x)
    v = x.T @ x / n
    for lag in range(1, lags + 1):
        g = x[lag:].T @ x[:-lag] / n
        v = v + (1 - lag / (lags + 1)) * (g + g.T)
    return v


def dm_hln(d: np.ndarray, h: int) -> dict:
    """Diebold-Mariano on d = loss_pers - loss_cand, Harvey-Leybourne-Newbold correction; H1: d > 0."""
    from scipy import stats

    n = len(d)
    stat = d.mean() / np.sqrt(_lrv(d, h - 1)[0, 0] / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dict(stat=float(stat), p=float(stats.t.sf(stat, n - 1)), n=n)


def clark_west(sc: pd.DataFrame) -> dict:
    """Clark-West for a model nesting persistence, on log-return MSE, origin-level, HAC lag H-1."""
    from scipy import stats

    f = sc.target_logret ** 2 - (sc.target_logret - sc.fc) ** 2 + sc.fc ** 2
    ft = f.groupby(sc.origin).mean().to_numpy()
    stat = ft.mean() / np.sqrt(_lrv(ft, int(P.H) - 1)[0, 0] / len(ft))
    return dict(stat=float(stat), p=float(stats.norm.sf(stat)), n=len(ft))


def giacomini_white(d: np.ndarray) -> dict:
    """Conditional predictive ability, instruments (1, d_{t-H}): d_{t-H} is mature at t."""
    from scipy import stats

    lag = int(P.H)
    z = np.column_stack([d[lag:], d[:-lag] * d[lag:]])
    zbar = z.mean(0)
    stat = len(z) * zbar @ np.linalg.solve(_lrv(z, lag - 1), zbar)
    return dict(stat=float(stat), p=float(stats.chi2.sf(stat, z.shape[1])), n=len(z))


def ct_r2(sc: pd.DataFrame) -> float:
    """Campbell-Thompson out-of-sample R2 against the no-change (zero) forecast, log returns."""
    return float(1 - ((sc.target_logret - sc.fc) ** 2).sum() / (sc.target_logret ** 2).sum())


def holm(pvalues: dict) -> dict:
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = running
    return out


def losses(cells: pd.DataFrame, origins, series) -> pd.DataFrame:
    """Origin x model loss matrix (capped APE averaged over h and series); duplicate models dropped."""
    cols = {"pers": None}
    for model in MODELS:
        sc = scored(cells, model, origins, series)
        cols[model] = sc.groupby("origin").ape_cand.mean()
        cols["pers"] = sc.groupby("origin").ape_pers.mean()
    frame = pd.DataFrame(cols)
    return frame.T.drop_duplicates().T


def spa_mcs(loss: pd.DataFrame) -> dict:
    from arch.bootstrap import MCS, SPA

    out = {}
    if loss.shape[1] < 2:
        return dict(note="one distinct model")
    spa = SPA(loss["pers"], loss.drop(columns="pers"), block_size=int(P.BLOCK), reps=int(P.BOOT),
              seed=int(P.SEED))
    spa.compute()
    out["spa"] = {k: float(v) for k, v in spa.pvalues.items()}
    mcs = MCS(loss, size=float(P.ALPHA), block_size=int(P.BLOCK), reps=int(P.BOOT), seed=int(P.SEED))
    mcs.compute()
    out["mcs_included"] = [str(c) for c in mcs.included]
    out["mcs_pvalues"] = {str(k): float(v) for k, v in mcs.pvalues.iloc[:, 0].items()}
    return out


def pbo(perf: np.ndarray) -> float:
    """CSCV probability of backtest overfitting (Bailey et al. 2014); perf is T x N, higher better."""
    import itertools

    splits = int(P.CSCV_SPLITS)
    groups = np.array_split(np.arange(len(perf)), splits)
    count = perf.shape[1]
    hits = []
    for chosen in itertools.combinations(range(splits), splits // 2):
        ins = np.concatenate([groups[g] for g in chosen])
        oos = np.concatenate([groups[g] for g in range(splits) if g not in chosen])
        best = perf[ins].mean(0).argmax()
        rel = ((perf[oos].mean(0) < perf[oos].mean(0)[best]).sum() + 1) / (count + 1)
        hits.append(np.log(rel / (1 - rel)) <= 0)
    return float(np.mean(hits))


def mde(sc: pd.DataFrame, origins: int) -> dict:
    """Minimum detectable pooled reduction at `origins` sealed origins (one-sided Holm step 1, 80 % power)."""
    from scipy import stats

    d = sc.groupby("origin").d.mean().to_numpy()
    se = np.sqrt(_lrv(d, int(P.H) - 1)[0, 0] / origins)
    z = stats.norm.isf(P.ALPHA / len(STATEMENTS)) + stats.norm.isf(1 - 0.8)
    return dict(mde_d=float(z * se), mde_reduction=float(z * se * P.PCT / sc.mape_pers.mean()))


# ------------------------------------------------------------------------------- detectors

def _rw(rng, frame):
    """Random-walk log price with the target's weekly volatility."""
    steps = rng.normal(0.0, np.nanstd(np.diff(frame.logp.to_numpy())), len(frame))
    steps[0] = 0.0
    return frame.assign(logp=frame.logp.iloc[0] + np.cumsum(steps))


def _season(frame):
    month = frame.date.dt.month.to_numpy().astype(float)
    return (month - month.mean()) / month.std()


def run_detectors(base: pathlib.Path) -> int:
    data = _load(base)
    out = {}
    rng = np.random.default_rng(int(P.SEED))

    # D1: random-walk targets, real components -> nothing to find
    null = campaign_cells(data, ("struct",), synth=lambda s, f: _rw(rng, f))
    o = common_origins(null)
    out["D1"] = summary(scored(null, "struct", o))
    out["D1"]["pass"] = bool(out["D1"]["ci_low"] <= 0)

    # D2 (v2): a calendar drift planted on the REAL price of the regression series, sized so that it
    # explains PLANTED_R2 of one-week variance -> must be recovered. v1 planted it on a random walk,
    # which made the basis level non-stationary (results/detectors_v1 in RESULTS.md).
    planted = ("copper", "corn", "wti")
    oracle = {}

    def plant(series, frame):
        if series not in planted:
            return frame
        s = _season(frame)
        gamma = np.nanstd(np.diff(frame.logp.to_numpy())) * np.sqrt(P.PLANTED_R2 / (1 - P.PLANTED_R2))
        syn = frame.assign(logp=frame.logp + np.cumsum(gamma * np.r_[0.0, s[:-1]]))
        csum = np.r_[0.0, np.cumsum(s)]
        n = len(s)
        ahead = np.minimum(np.arange(n)[:, None] + np.arange(1, int(P.H) + 1)[None, :], n)
        known = gamma * (csum[ahead] - csum[np.arange(n)][:, None])   # E[r_h] given the calendar
        oracle[series] = pd.DataFrame(known, index=frame.index).stack().rename("oracle")
        return syn

    syn = campaign_cells(data, ("struct",), synth=plant)
    o = common_origins(syn, series=planted)
    rec = summary(scored(syn, "struct", o, planted))
    known = pd.concat(oracle, names=["series", "week", "k"]).reset_index()
    known["horizon"] = known.pop("k") + 1
    orc = syn[syn.model == "struct"].merge(known, on=["series", "week", "horizon"])
    orc = summary(scored(orc.assign(fc=orc.oracle), "struct", o, planted))
    out["D2"] = dict(recovered=rec, oracle=orc,
                     pass_=bool(rec["ci_low"] > 0 and rec["reduction"] >= orc["reduction"] / 2))

    # D3: perturb every daily value after each block-start cutoff; earlier forecasts must not move.
    # D4: the same check on a variant that is given next week's realised return must FAIL.
    base_cells = campaign_cells(data, ("struct",))
    leak_cells = campaign_cells(data, ("struct",), leak=True)
    o = common_origins(base_cells)
    by_origin = dict(zip(base_cells.origin, base_cells.week))
    cuts = sorted({by_origin[x] for x in o if x % int(P.BLOCK) == 0})
    moved, leak_moved = [], []
    for cut in cuts:
        mutated = {k: v.assign(value=np.where(v.dt > cut, v.value * np.exp(rng.normal(size=len(v))),
                                              v.value)) for k, v in data.items()}
        for ref, kwargs, sink in ((base_cells, {}, moved), (leak_cells, {"leak": True}, leak_moved)):
            alt = campaign_cells(mutated, ("struct",), **kwargs)
            keep = ref.week <= cut
            a = ref.loc[keep, "fc"].to_numpy()
            b = alt.set_index(["series", "week", "horizon"]).fc.reindex(
                pd.MultiIndex.from_frame(ref.loc[keep, ["series", "week", "horizon"]])).to_numpy()
            sink.append(float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else 0.0)
            sink[-1] = sink[-1] if np.array_equal(np.isnan(a), np.isnan(b)) else float("inf")
    out["D3"] = dict(cutoffs=[str(c.date()) for c in cuts], max_abs_change=max(moved),
                     pass_=bool(max(moved) == 0.0))
    leak = summary(scored(leak_cells, "struct", common_origins(leak_cells)))
    out["D4"] = dict(max_abs_change=max(leak_moved), detected=bool(max(leak_moved) > 0),
                     leaky_dev_reduction=leak["reduction"], pass_=bool(max(leak_moved) > 0))
    _dump(base, "detectors.json", out)
    return 0


def _dump(base: pathlib.Path, name: str, obj) -> None:
    out = base / "results"
    out.mkdir(exist_ok=True)
    text = json.dumps(obj, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    (out / name).write_text(text)
    print(text, flush=True)


# ------------------------------------------------------------------------------- dev / sealed

def run_dev(base: pathlib.Path) -> int:
    data = _load(base)
    cells = campaign_cells(data)
    o = common_origins(cells)
    out = dict(origins=dict(n=len(o), first=str(cells[cells.origin == o.min()].week.iloc[0].date()),
                            last=str(cells[cells.origin == o.max()].week.iloc[0].date())), gate_a={})
    for model in MODELS:
        sc = scored(cells, model, o)
        out["gate_a"][model] = {name: summary(statement(sc, name)) for name in STATEMENTS}
    struct = scored(cells, "struct", o)
    out["mde_at_sealed"] = mde(struct, int(P.SEALED_ORIGINS))
    loss = losses(cells, o, tuple(TARGETS))
    out["pbo"] = dict(models=list(loss.columns), pbo=pbo(-loss.to_numpy()))
    keep = cells[cells.origin.isin(o)]
    (base / "results").mkdir(exist_ok=True)
    keep.to_parquet(base / "results" / "dev_cells.parquet", index=False)
    _dump(base, "dev.json", out)
    return 0


def _cells_sha(sc: pd.DataFrame) -> str:
    cols = ["series", "origin", "horizon", "target_logret", "fc"]
    frame = sc[cols].sort_values(cols[:3]).reset_index(drop=True)
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def _pnl(sc: pd.DataFrame, series: str, sealed) -> dict:
    """h=1 timing trade on the sealed origins: position = sign(forecast - mean of earlier forecasts)."""
    c = sc[(sc.series == series) & (sc.horizon == 1)].sort_values("origin")
    prior = c.fc.expanding().mean().shift(1)
    pos = np.sign(c.fc - prior).fillna(0.0)
    turn = pos.diff().abs().fillna(0.0)
    m = c.origin.isin(sealed).to_numpy()
    out = {}
    for leg, ret in (("futures", c.target_logret - c.b), ("basis", c.b)):
        for cost in P.COST_SWING:
            pnl = (pos * ret - turn * cost / 1e4)[m]
            sd = pnl.std()
            out[f"{leg}_{cost}bp"] = dict(
                mean_weekly=float(pnl.mean()),
                sharpe=float(pnl.mean() / sd * np.sqrt(P.WEEKS_PER_YEAR)) if sd > 0 else float("nan"))
    return out


def run_sealed(base: pathlib.Path) -> int:
    sys.path.insert(0, str(base / "experiment"))   # where --include puts it on the VM
    import transfer_contract

    dev_manifest = json.loads((base / "data" / "manifest.json").read_text())
    full_manifest = json.loads((base / "data" / "manifest_full.json").read_text())
    cutoff = pd.Timestamp(dev_manifest["cutoff"])
    last = pd.Timestamp(dev_manifest["last_common_target"]).to_period("W-SUN").end_time.normalize()
    data = {k: v[v.dt <= last] for k, v in _load(base, "full").items()}
    cells = campaign_cells(data)
    allo = common_origins(cells)
    week_of = dict(zip(cells.origin, cells.week))
    sealed = np.array([x for x in allo if cutoff < week_of[x] <= cutoff + pd.Timedelta(weeks=int(P.SEALED_ORIGINS))])
    out = dict(cutoff=str(cutoff.date()), last_target_week=str(last.date()), n_sealed=len(sealed),
               first=str(week_of[sealed.min()].date()), last=str(week_of[sealed.max()].date()),
               full_manifest_cutoff_check=full_manifest["last_common_target"], statements={})
    struct = scored(cells, "struct", sealed)
    for name in STATEMENTS:
        sc = statement(struct, name)
        d = sc.groupby("origin").d.mean().to_numpy()
        series = tuple(TARGETS) if name == "pooled" else (name,)
        out["statements"][name] = dict(
            summary(sc), dm_hln=dm_hln(d, int(P.H)), clark_west=clark_west(sc),
            giacomini_white=giacomini_white(d), ct_r2=ct_r2(sc),
            by_h={int(h): dict(mape_pers=float(g.mape_pers.mean()), mape_cand=float(g.mape_cand.mean()),
                              ct_r2=ct_r2(g)) for h, g in sc.groupby("horizon")},
            spa_mcs=spa_mcs(losses(cells, sealed, series)),
            fut=summary(statement(scored(cells, "fut", sealed), name)))
    out["holm_round"] = holm({k: v["dm_hln"]["p"] for k, v in out["statements"].items()})
    out["pnl"] = {s: _pnl(scored(cells, "struct", allo), s, sealed) for s in ("gold", "copper", "corn", "wti")}

    def frames(model):
        sc = scored(cells, model, sealed)
        return {s: pd.DataFrame(dict(origin=pd.to_datetime(g.week), horizon=g.horizon,
                                     last_price=np.exp(g.logp), y_true=np.exp(g.logp + g.target_logret),
                                     y_pred=np.exp(g.logp + g.fc))).reset_index(drop=True)
                for s, g in sc.groupby("series")}

    out["gate_e"] = transfer_contract.evaluate_frame(frames("struct"), control=frames("fut"))
    out["cells_sha256"] = _cells_sha(struct)
    (base / "results").mkdir(exist_ok=True)
    cells[cells.origin.isin(sealed)].to_parquet(base / "results" / "sealed_cells.parquet", index=False)
    _dump(base, "sealed.json", out)
    return 0


def run_forward(base: pathlib.Path) -> int:
    """Issue one forecast per series at its latest DB week; append-only log, evaluated once in 2028."""
    data = _load(base, "full")
    cells = campaign_cells(data, ("fut", "struct"))
    issued = pd.Timestamp.now(tz="UTC")
    week_now = issued.tz_localize(None).to_period("W-SUN").end_time.normalize()
    code = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    snap = hashlib.sha256(json.dumps(json.loads(
        (base / "data" / "manifest_full.json").read_text())["series"], sort_keys=True).encode()).hexdigest()
    last = cells.groupby("series").week.transform("max")
    rows = cells[cells.week == last].pivot_table(index=["series", "week", "horizon"], columns="model",
                                                  values="fc").reset_index()
    rows["issued_utc"] = issued.isoformat(timespec="seconds")
    rows["stale_weeks"] = ((week_now - rows.week).dt.days // P.DAYS_PER_WEEK).astype(int)
    rows["code_sha256"], rows["data_sha256"], rows["params_hash"] = code, snap, PARAMS_HASH
    out = base / "results"
    out.mkdir(exist_ok=True)
    rows.to_csv(out / f"forward_{issued.date()}.csv", index=False)
    print(rows.to_string(index=False), flush=True)
    return 0


def run_ledger(base: pathlib.Path) -> int:
    """Mini, seconds: write the E028 rows into leaderboard.sqlite and apply the program-wide family."""
    import sqlite3

    from _exo import leaderboard

    db = base.parents[1] / "leaderboard.sqlite"
    con = sqlite3.connect(db)
    prior = con.execute(
        "select count(distinct experiment || '|' || coalesce(direction, '')) from leaderboard "
        "where experiment not like 'E028%' and experiment not like '%persist%' "
        "and experiment not like '%naive%'").fetchone()[0]
    family = int(prior) + len(MODELS)
    names = {k: v for k, v in TARGETS.items()}
    payload = []
    for window, path in (("dev", "dev_cells.parquet"), ("sealed", "sealed_cells.parquet")):
        cells = pd.read_parquet(base / "results" / path)
        sc = scored(cells, "struct", common_origins(cells))
        payload += leaderboard.rows("E028_struct", sc, names, direction="struct", window=window,
                                    params_hash=PARAMS_HASH)
        payload += leaderboard.rows("E028_struct_persistence", sc, names, kind="persistence",
                                    window=window, params_hash=PARAMS_HASH)
    with con:
        leaderboard.ensure_schema(con)
        con.execute("delete from leaderboard where experiment like 'E028%'")
        con.executemany(leaderboard.INSERT, payload)
    con.close()
    sealed = json.loads((base / "results" / "sealed.json").read_text())
    p = sealed["statements"]["pooled"]["dm_hln"]["p"]
    program = dict(prior_directions=int(prior), family=family, pooled_p_raw=p,
                   pooled_p_bonferroni=min(1.0, p * family), rows_written=len(payload))
    _dump(base, "program_family.json", program)
    return 0


COMMANDS = dict(ceiling=run_ceiling, detectors=run_detectors, dev=run_dev, sealed=run_sealed,
                forward=run_forward, ledger=run_ledger)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", *COMMANDS):
        raise SystemExit(f"usage: E028.py prepare [full] | {' | '.join(COMMANDS)}")
    if sys.argv[1] == "prepare":
        prepare(BASE, full=sys.argv[2:] == ["full"])
    else:
        raise SystemExit(COMMANDS[sys.argv[1]](BASE))
