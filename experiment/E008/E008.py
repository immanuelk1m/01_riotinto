"""E008: transfer-first MIMO LightGBM — validation that replays the test protocol.

E007 lesson: purged k-fold CV gains did not predict realised test gains (Spearman val/test ~0 or
negative in 19/25 cells). Same task as E001-E007 (expanding walk-forward from the 80% point, step 1
week, y_h = log(P_{t+h}/P_t), E004 common MI top-30 features). Shared core for every direction:
  * MIMO: one LightGBM per origin, rows stacked over h=1..8 with h as a feature (E004 ablation)
  * vol-scaled target z_h = y_h / (sigma_52w * sqrt(h)) so horizons and eras weigh alike
    (standardized anomalies, MIXSAMOS arXiv 2412.09583; lead-time-continuous postprocessing,
    Wessel et al. 2024 QJRMS)
  * validation = the candidate's own forward track record: pre-test forecasts refit every 26 weeks
    on labels realised before each pseudo-origin, then the live test forecasts themselves; never
    data after the forecast (Cerqueira, Torgo & Mozetic 2020; Schnaubelt 2019: forward schemes
    beat CV under non-stationarity)
  * every RETUNE origins: one kappa in [0, 1] pooled over h on the last VAL_WEEKS of the track
    record (kappa = 0 -> naive); deploy only if the gain over naive has HAC t >= T_GATE
  * log (validation gain, realised test gain) per retune block x horizon -> transfer rate
Directions (one tmux pane each):
  global   cross-learning: one model pooled over the 5 commodities on commodity-agnostic features
           (scaled momentum, drawdown, common macro z-scores, commodity id); kappa and gate pooled
           over commodities (Montero-Manso & Hyndman 2021; Oikonomou et al. 2025 Resources Policy;
           Zhao & Abolghasemi arXiv 2411.06394)
  dro      era-invariant boosting: GroupDRO over calendar-year eras, era weights raised by their
           relative loss between boosting chunks (DeLise arXiv 2309.14496 era splitting; Sagawa et
           al. arXiv 1911.08731). ponytail: directional era splitting itself needs a custom split
           criterion (erasplit is an sklearn fork, sdist only); DRO reweighting is the LightGBM graft
  avg      no selection: config x window members, drop the worst third on the track record, average
           the rest after winsorizing each member's forecasts (Elliott & Liao 2025 JBES; Frazier et
           al. arXiv 2308.05263; Mekelburg & Strauss 2024 JEF pooling and winsorizing)
  huber    robust objective: Huber loss on winsorized z (Shen & Xiu 2024 weak signals; heavy tails)
  anytime  no retune window: deploy at every origin iff a time-uniform confidence sequence on the
           full track record (predictable kappa) says the model beats naive (Choe & Ramdas arXiv
           2110.00115; Waudby-Smith et al. arXiv 2103.06476; Arnold et al. arXiv 2404.18678)

Runs on Colab from parquet snapshots:
  colab : python E008.py <direction> [commodity ...]
  local : uv run experiment/E008/E008.py leaderboard [direction ...]
"""

import json
import logging
import math
import sqlite3
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

EXP = "E008"  # set to E008_<direction> in main
ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
LOG_DIR, RES_DIR, DATA_DIR, FEAT_DIR = (EXP_DIR / d for d in ("logs", "results", "data", "features"))
LEADERBOARD = ROOT / "leaderboard.sqlite"

TARGETS = {
    "copper": "Com_LME_Cu_Cash_Bid",
    "gold": "Com_LBMA_Gold_Price_PM_USD_V_0",
    "corn": "Com_Barchart_Cash_National_Corn_Index",
    "coffee": "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag",
    "wheat": "Com_Barchart_Cash_Soft_Red_Wheat_Index",
}
COMMON = ["Com_BLS_CPI_All_Items_US_YoY_Pct_lag_3", "Com_CBOE_VIX_Index_lag_0", "Com_EPU_Gpr_Gpr_lag_1",
          "Com_Econ_CN_exports_yoy_lag_2", "Com_Econ_CN_imports_yoy_lag_2",
          "Com_Home_Treasury_Gov_Daily_Treasury_Par_Yield_Curve_Rates_2_Year_Treasury_lag_0",
          "Com_ICE_US_Dollar_Index_DXY_lag_1", "Macro_FRED_US_10Y_Treasury_Yield_lag_0",
          "Macro_Policyuncertainty_US_Daily_Trade_Policy_Uncertainty_Index_lag_0",
          "customs_china_summary_trade_balance_usd_million_lag_3", "fed_h10_nominal_broad_dollar_index_daily_lag_0"]
H = 8
HS = np.arange(1, H + 1)
TRAIN_FRAC = 0.8
SEED = 42
RETUNE, VAL_WEEKS, REFIT_VAL, T_GATE = 26, 156, 26, 1.0
VOL_WINDOW, MIN_TRAIN = 52, 260
DRO_CHUNKS, DRO_ETA, DRO_PRIOR = 10, 20.0, 52.0
HUBER_ALPHA, WINSOR = 1.345, (5, 95)
CS_ALPHA, CS_BLOCK, CS_NSTAR = 0.1, 8, 50
BASE = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, force_col_wise=True,
            n_jobs=1)  # = E004
GRID = {
    "e004": BASE,
    "shallow": dict(BASE, num_leaves=4, max_depth=2, learning_rate=0.02, n_estimators=200, min_child_samples=50),
    "rf": dict(BASE, boosting_type="rf", subsample=0.7, colsample_bytree=0.7),
}
DIRECTIONS = ("global", "dro", "avg", "huber", "anytime")

log = logging.getLogger(EXP)


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(LOG_DIR / f"{EXP}.log"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


# ---------------------------------------------------------------- data
def load(commodity):
    X = pd.read_parquet(DATA_DIR / f"{commodity}_X.parquet")
    px = pd.read_parquet(DATA_DIR / f"{commodity}_price.parquet")
    X["week_end"] = pd.to_datetime(X["week_end"])
    X = X.set_index("week_end").astype(float)
    px = px.set_index(pd.to_datetime(px["dt"]))["value"].astype(float)
    weekly = px.resample("W-SUN").last()  # no ffill: weeks without a quote stay NaN
    start = weekly.first_valid_index()
    idx = pd.date_range(max(start, X.index[0]), weekly.index[-1], freq="W-SUN")
    return X.reindex(idx), weekly.reindex(idx)


def series(commodity, direction):
    """Everything one commodity contributes: weeks, features, Y (log returns h=1..8), vol scale."""
    Xdf, price = load(commodity)
    pff = price.ffill()
    lp = np.log(pff)
    r1 = lp.diff()
    sig = r1.rolling(VOL_WINDOW, min_periods=26).std()
    if direction == "global":
        M = Xdf.reindex(columns=COMMON).ffill()
        roll = M.rolling(156, min_periods=52)
        F = pd.DataFrame({**{f"mom{k}": lp.diff(k) / (sig * math.sqrt(k)) for k in (1, 4, 13, 26, 52)},
                          "z52": (lp - lp.rolling(52).mean()) / sig, "dd52": (lp - lp.rolling(52).max()) / sig,
                          "vratio": r1.rolling(13).std() / sig}).join((M - roll.mean()) / roll.std())
    else:
        names = [f["name"] for f in json.loads((FEAT_DIR / f"{commodity}_features.json").read_text())["features"]]
        F = Xdf.assign(tgt_logp=lp, tgt_r1=r1)[names].ffill()
    last_week = min(Xdf.dropna(how="all").index[-1], price.last_valid_index())
    Y = np.column_stack([(np.log(price).shift(-h) - lp).values for h in HS])
    return {"id": list(TARGETS).index(commodity), "weeks": Xdf.index, "X": F.values, "Y": Y, "sig": sig.values,
            "price": price, "pff": pff, "N": int((Xdf.index <= last_week).sum())}


def cutoff(S, T):
    """Row of the last week <= T (labels with s + h <= cutoff are realised at T)."""
    return int(S["weeks"].searchsorted(T, side="right")) - 1


def train_rows(S, cut, window=None):
    """(row, h) pairs with s + h <= cut, a known y_h and a known vol scale."""
    r = np.concatenate([np.arange(0, cut - h + 1) for h in HS])
    hs = np.concatenate([np.full(max(cut - h + 1, 0), h) for h in HS])
    ok = ~np.isnan(S["Y"][r, hs - 1]) & ~np.isnan(S["sig"][r])
    if window:
        ok &= r > cut - window
    return r[ok], hs[ok]


def design(S, r, hs, direction):
    cols = [S["X"][r], hs] + ([np.full(len(r), S["id"])] if direction == "global" else [])
    return np.column_stack(cols)


# ---------------------------------------------------------------- models
def fit(direction, params, X, z, era):
    """List of boosters whose summed predict() is the z forecast."""
    if direction == "huber":
        z = np.clip(z, *np.percentile(z, [1, 99]))
        params = dict(params, objective="huber", alpha=HUBER_ALPHA)
    if direction != "dro":
        cat = [X.shape[1] - 1] if direction == "global" else "auto"
        return [lgb.LGBMRegressor(**params).fit(X, z, categorical_feature=cat)]
    _, inv = np.unique(era, return_inverse=True)
    cnt = np.bincount(inv)
    logq, f, models = np.zeros(len(cnt)), np.zeros(len(z)), []
    for _ in range(DRO_CHUNKS):
        q = np.exp(logq - logq.max())  # exponentiated-gradient weights in log space: no exp overflow
        q /= q.sum()
        m = lgb.LGBMRegressor(**dict(params, n_estimators=params["n_estimators"] // DRO_CHUNKS)).fit(
            X, z, sample_weight=(q / cnt)[inv] * len(z), init_score=f)
        f += m.predict(X)
        models.append(m)
        # vs naive per era, + DRO_PRIOR pseudo-rows at ratio 1 so partial (January) eras cannot dominate
        loss = (np.bincount(inv, (z - f) ** 2) + DRO_PRIOR) / (np.bincount(inv, z ** 2) + DRO_PRIOR)
        logq += DRO_ETA * loss
    return models


def forecast(direction, pool, spec, T, qdates):
    """Fit on every pooled series at cutoff date T; z forecasts {(commodity, row): (H,)} for qdates."""
    params, window = spec
    Xs, zs, eras = [], [], []
    for S in pool.values():
        r, hs = train_rows(S, cutoff(S, T), window)
        Xs.append(design(S, r, hs, direction))
        zs.append(S["Y"][r, hs - 1] / (S["sig"][r] * np.sqrt(hs)))
        eras.append(S["id"] * 10000 + S["weeks"].year.values[r])
    models = fit(direction, params, np.vstack(Xs), np.concatenate(zs), np.concatenate(eras))
    out = {}
    for c, S in pool.items():
        q = S["weeks"].get_indexer(qdates)
        q = q[q >= 0]
        if len(q):
            P = sum(m.predict(design(S, np.repeat(q, H), np.tile(HS, len(q)), direction)) for m in models)
            out.update({(c, int(a)): p for a, p in zip(q, P.reshape(len(q), H))})
    return out, sum(len(z) for z in zs)


# ---------------------------------------------------------------- track record + decisions
def gather(pool, stream, T, n_weeks=None):
    """Track-record keys before T (all, or the last n_weeks) with z targets masked to labels realised by T."""
    keys, Z, dates = [], [], []
    for c, S in pool.items():
        cut = cutoff(S, T)
        lo = 0 if n_weeks is None else int(S["weeks"].searchsorted(T - pd.Timedelta(weeks=n_weeks)))
        for q in range(lo, cut):
            if (c, q) in stream:
                z = S["Y"][q] / (S["sig"][q] * np.sqrt(HS))
                z[HS > cut - q] = np.nan
                keys.append((c, q)), Z.append(z), dates.append(S["weeks"][q])
    return keys, np.array(Z), pd.DatetimeIndex(dates)


def hac_t(d, lag):
    """t-stat of mean(d) with Bartlett/Newey-West variance (lag = horizon - 1 overlap)."""
    n = len(d)
    if n < 10:
        return 0.0
    dc = d - d.mean()
    v = dc @ dc / n + 2 * sum((1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lag + 1))
    return float(d.mean() / math.sqrt(max(v, 1e-300) / n))


def score(P, Z, dates):
    """kappa, HAC t of the gain over naive (per-date mean over commodities), relative MSE per h."""
    m = ~np.isnan(Z) & ~np.isnan(P)
    P0, Z0 = np.where(m, P, 0), np.where(m, Z, 0)
    kappa = float(np.clip((P0 * Z0).sum() / max((P0 ** 2).sum(), 1e-300), 0, 1))
    D = np.where(m, Z0 ** 2 - (Z0 - kappa * P0) ** 2, np.nan)
    d = pd.Series(np.nanmean(D, 1), index=dates).dropna().groupby(level=0).mean().values
    rel = ((Z0 - kappa * P0) ** 2).sum(0) / np.maximum((Z0 ** 2).sum(0), 1e-300) - 1
    return kappa, hac_t(d, H - 1), rel


def combine(streams, keep, bands, keys):
    """Equal-weight mean of the kept members' z forecasts, each clipped to its winsor band."""
    return np.mean([np.clip(np.vstack([streams[m][k] for k in keys]), *bands[m]) for m in keep], axis=0)


def retune(pool, streams, T):
    """Members to keep, their bands, kappa and gate from the last VAL_WEEKS of the track record."""
    keys, Z, dates = gather(pool, streams[next(iter(streams))], T, VAL_WEEKS)
    keep, bands = list(streams), {m: (-np.inf, np.inf) for m in streams}
    if len(streams) > 1:  # avg: drop the worst third, winsorize the rest
        P = {m: np.vstack([streams[m][k] for k in keys]) for m in streams}
        mse = {m: np.nanmean((P[m] - Z) ** 2) for m in streams}
        keep = sorted(streams, key=mse.get)[: math.ceil(2 * len(streams) / 3)]
        bands = {m: tuple(np.percentile(P[m], WINSOR)) for m in streams}
    kappa, t, rel = score(combine(streams, keep, bands, keys), Z, dates)
    return {"keep": keep, "bands": bands, "kappa": kappa, "t": t, "rel": rel, "deploy": t >= T_GATE and kappa > 0}


def cs_lower(b):
    """Lower end of a time-uniform asymptotic confidence sequence for mean(b) (Waudby-Smith et al.)."""
    n = len(b)
    if n < 2:
        return -np.inf
    v, rho2 = b.var(ddof=1), (-2 * math.log(CS_ALPHA) + math.log(1 - 2 * math.log(CS_ALPHA))) / CS_NSTAR
    return float(b.mean() - math.sqrt(2 * (n * v * rho2 + 1) / (n ** 2 * rho2)
                                      * math.log(math.sqrt(n * v * rho2 + 1) / CS_ALPHA)))


def anytime_gate(pool, stream, T):
    """Deploy iff the CS on the full track record's gain (kappa predictable: from rows realised before
    each forecast) is above 0; kappa for deployment uses the whole realised record."""
    keys, Z, _ = gather(pool, stream, T)
    P = np.vstack([stream[k] for k in keys])
    m = ~np.isnan(Z)
    pz, pp = np.where(m, P * Z, 0).sum(1), np.where(m, P ** 2, 0).sum(1)
    cpz, cpp = np.concatenate([np.zeros(H), np.cumsum(pz)[:-H]]), np.concatenate([np.zeros(H), np.cumsum(pp)[:-H]])
    kq = np.clip(cpz / np.maximum(cpp, 1e-300), 0, 1)
    d = np.nanmean(np.where(m, Z ** 2 - (Z - kq[:, None] * P) ** 2, np.nan), 1)
    d = d[~np.isnan(d)]
    lo = cs_lower(d[: len(d) // CS_BLOCK * CS_BLOCK].reshape(-1, CS_BLOCK).mean(1))
    kappa = float(np.clip(pz.sum() / max(pp.sum(), 1e-300), 0, 1))
    return lo > 0 and kappa > 0, kappa, lo


# ---------------------------------------------------------------- experiment
def run(commodity, direction):
    t0 = time.time()
    pool = {c: series(c, direction) for c in (TARGETS if direction == "global" else [commodity])}
    S = pool[commodity]
    weeks, N = S["weeks"], S["N"]
    first = int(math.floor(TRAIN_FRAC * N))
    log.info("[%s] %s weeks=%s..%s N=%d first_origin=%s (idx %d) origins=%d pool=%s features=%d", commodity,
             direction, weeks[0].date(), weeks[N - 1].date(), N, weeks[first].date(), first, N - first,
             list(pool), S["X"].shape[1])

    members = ({f"{c}_{w or 'exp'}": (GRID[c], w) for c in GRID for w in (None, 520)} if direction == "avg"
               else {"main": (BASE, None)})
    streams = {m: {} for m in members}
    start = MIN_TRAIN if direction == "anytime" else first - VAL_WEEKS
    for p in range(start, first, REFIT_VAL):  # pre-test track record, refit every REFIT_VAL weeks
        for m, spec in members.items():
            streams[m].update(forecast(direction, pool, spec, weeks[p], weeks[p:min(p + REFIT_VAL, first)])[0])
    log.info("[%s] %s pre-test track record %d forecasts x %d members (%.0fs)", commodity, direction,
             len(streams["main" if "main" in streams else next(iter(streams))]), len(members), time.time() - t0)

    rows, trace, sel = [], [], None
    for i in range(first, N):
        T, k = weeks[i], (i - first) // RETUNE
        n_train = 0
        for m, spec in members.items():
            out, n_train = forecast(direction, pool, spec, T, weeks[i:i + 1])
            streams[m].update(out)
        if (i - first) % RETUNE == 0:
            sel = retune(pool, streams, T)
            if direction == "anytime":
                sel["deploy"], sel["kappa"], sel["t"] = anytime_gate(pool, streams["main"], T)
            trace += [{"retune": k, "origin": str(T.date()), "h": h, "kappa": sel["kappa"], "t": sel["t"],
                       "deploy": sel["deploy"], "val_rel": float(sel["rel"][h - 1]), "keep": "+".join(sel["keep"])}
                      for h in HS]
        elif direction == "anytime":
            sel["deploy"], sel["kappa"], _ = anytime_gate(pool, streams["main"], T)
        scale = S["sig"][i] * np.sqrt(HS)
        cand = sel["kappa"] * np.nan_to_num(combine(streams, sel["keep"], sel["bands"], [(commodity, i)])[0] * scale)
        pred = cand if sel["deploy"] else np.zeros(H)
        for h in HS:
            tgt = T + pd.Timedelta(weeks=int(h))
            rows.append({"origin": T, "target_date": tgt, "horizon": int(h), "retune": k,
                         "y_true": S["price"].get(tgt, np.nan), "y_pred": S["pff"][T] * math.exp(pred[h - 1]),
                         "y_cand": S["pff"][T] * math.exp(cand[h - 1]), "last_price": S["pff"][T], "n_train": n_train,
                         "deploy": bool(sel["deploy"])})
        if (i - first) % 25 == 0:
            log.info("[%s] %s origin %d/%d %s deploy=%s kappa=%.2f (%.0fs)", commodity, direction, i - first + 1,
                     N - first, T.date(), sel["deploy"], sel["kappa"], time.time() - t0)
    df = pd.DataFrame(rows)
    trace = transfer(df, pd.DataFrame(trace))
    trace.to_csv(LOG_DIR / f"{commodity}_{direction}_transfer.csv", index=False)
    df.drop(columns="retune").to_parquet(RES_DIR / f"predictions_{commodity}.parquet", index=False)
    assert df["origin"].min() == weeks[first] and df["origin"].nunique() == N - first
    log.info("[%s] saved %d predictions, transfer %s (%.0fs)", commodity, len(df), transfer_summary(trace),
             time.time() - t0)
    render_mp4(commodity, df, S["pff"], S["price"])


def transfer(df, trace):
    """Attach the realised test gain (deployed forecast, and the candidate before the gate) per block x h."""
    d = df.dropna(subset=["y_true"])
    e0 = np.log(d.y_true / d.last_price) ** 2
    g0 = e0.groupby([d.retune, d.horizon]).mean()
    for name, col in (("test_rel", "y_pred"), ("cand_test_rel", "y_cand")):
        g = (np.log(d[col] / d.y_true) ** 2).groupby([d.retune, d.horizon]).mean() / g0 - 1
        trace[name] = [g.get((r, h), np.nan) for r, h in zip(trace.retune, trace.h)]
    return trace


def transfer_summary(tr):
    """Does the validation gain predict the realised test gain? Over (retune block, horizon) pairs."""
    dep = tr[tr.deploy]
    ok = tr.dropna(subset=["val_rel", "cand_test_rel"])
    rho = spearmanr(ok.val_rel, ok.cand_test_rel).statistic if len(ok) > 2 else np.nan
    return {"blocks": len(tr), "deployed": len(dep),
            "deploy_win_rate": float((dep.test_rel < 0).mean()) if len(dep) else np.nan,
            "deploy_mean_test_rel": float(dep.test_rel.mean()) if len(dep) else np.nan,
            "spearman_val_cand": float(rho),
            "sign_agree": float((np.sign(ok.val_rel) == np.sign(ok.cand_test_rel)).mean()) if len(ok) else np.nan}


# ---------------------------------------------------------------- outputs
def metrics(d, pred_col):
    d = d.dropna(subset=["y_true"])
    err = d[pred_col] - d["y_true"]
    return float((err.abs() / d["y_true"].abs()).mean() * 100), float(np.sqrt((err ** 2).mean())), len(d)


def write_leaderboard(commodity, df):
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(LEADERBOARD, timeout=60) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS leaderboard(
            experiment TEXT, commodity TEXT, horizon TEXT, mape REAL, rmse REAL, n INTEGER,
            target_series TEXT, created_at TEXT, PRIMARY KEY(experiment, commodity, horizon))""")
        for exp, col in ((EXP, "y_pred"), ("naive", "last_price")):
            groups = [(str(h), g) for h, g in df.groupby("horizon")] + [("all", df)]
            for h, g in groups:
                mape, rmse, n = metrics(g, col)
                assert np.isfinite(mape) and np.isfinite(rmse)
                con.execute("INSERT OR REPLACE INTO leaderboard VALUES (?,?,?,?,?,?,?,?)",
                            (exp, commodity, h, mape, rmse, n, TARGETS[commodity], now))
                if exp == EXP or h == "all":
                    log.info("[%s] %-15s h=%-3s MAPE=%.3f%% RMSE=%.4f n=%d", commodity, exp, h, mape, rmse, n)


def render_mp4(commodity, df, pff, price):
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    origins = sorted(df["origin"].unique())
    by_origin = {o: g.sort_values("horizon") for o, g in df.groupby("origin")}
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

    def draw(k):
        ax.clear()
        o = origins[k]
        g = by_origin[o]
        hist = pff.loc[o - pd.Timedelta(weeks=104):o]
        fut = price.loc[o:o + pd.Timedelta(weeks=H)].dropna()
        ax.plot(hist.index, hist.values, color="#444", lw=1.4, label="actual")
        ax.plot(fut.index, fut.values, color="#2a9d8f", lw=1.4, ls="--", label="actual (future)")
        ax.plot([o, *g["target_date"]], [g["last_price"].iloc[0], *g["y_pred"]], color="#e76f51",
                marker="o", ms=3, lw=1.6, label="forecast h=1..8")
        ax.axvline(o, color="#999", lw=0.8)
        ax.set_title(f"{EXP} {commodity} — origin {pd.Timestamp(o).date()}  ({k + 1}/{len(origins)})")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

    anim = animation.FuncAnimation(fig, draw, frames=len(origins))
    anim.save(RES_DIR / f"{commodity}.mp4", writer=animation.FFMpegWriter(fps=10))
    plt.close(fig)
    log.info("[%s] saved %s.mp4 (%d frames)", commodity, commodity, len(origins))


def main():
    global EXP, RES_DIR
    cmd = sys.argv[1]
    if cmd == "leaderboard":
        setup_logging()
        for direction in sys.argv[2:] or DIRECTIONS:
            EXP = f"E008_{direction}"
            for f in sorted((RES_DIR / direction).glob("predictions_*.parquet")):
                c = f.stem.removeprefix("predictions_")
                write_leaderboard(c, pd.read_parquet(f))
                tr = LOG_DIR / f"{c}_{direction}_transfer.csv"
                if tr.exists():
                    log.info("[%s] %s transfer %s", c, EXP, transfer_summary(pd.read_csv(tr)))
        return
    assert cmd in DIRECTIONS, f"command must be leaderboard or one of {DIRECTIONS}"
    EXP, RES_DIR = f"E008_{cmd}", RES_DIR / cmd
    setup_logging()
    names = sys.argv[2:] or list(TARGETS)
    log.info("%s start: %s", EXP, names)
    for c in names:
        run(c, cmd)
    log.info("%s done", EXP)


if __name__ == "__main__":
    main()
