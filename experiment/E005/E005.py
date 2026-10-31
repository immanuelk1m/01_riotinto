"""E005: low-SNR linear forecasts for the 8-week price forecast — combination vs Ridge.

Same protocol as E001-E004 (expanding TSCV from 80% point, step 1 week), zero-drift forecasts
anchored on the no-change price. Candidates: panel columns + target features, levels and
4-week changes, fixed on data before the first TSCV origin.
  combo  equal-weight mean of univariate OLS forecasts b_j (x_j - mean_j) (Rapach, Strauss &
         Zhou 2010), with a sign restriction against the slope at the first origin and
         truncation to the training 5-95% range of y_h (Campbell & Thompson 2008)
  ridge  Ridge on all standardised candidates, alpha by TimeSeriesSplit CV re-tuned every
         RETUNE origins (Shen & Xiu 2025)

Runs on Colab from parquet snapshots (no DB):
  local : uv run experiment/E005/E005.py export            # DB -> data/*.parquet
  colab : python E005.py <combo|ridge> [commodity ...]     # predictions, mp4, logs
  local : uv run experiment/E005/E005.py leaderboard       # results -> leaderboard.sqlite
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

warnings.filterwarnings("ignore")

EXP = "E005"  # set to E005_<method> in main
ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
LOG_DIR, RES_DIR, DATA_DIR = EXP_DIR / "logs", EXP_DIR / "results", EXP_DIR / "data"
LEADERBOARD = ROOT / "leaderboard.sqlite"
DSN = "dbname=commodity"

# commodity -> (ml panel, daily price series in public.value)
TARGETS = {
    "copper": ("copper", "Com_LME_Cu_Cash_Bid"),
    "gold": ("gold", "Com_LBMA_Gold_Price_PM_USD_V_0"),
    "corn": ("corn", "Com_Barchart_Cash_National_Corn_Index"),
    "coffee": ("coffee", "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag"),
    "wheat": ("wheat", "Com_Barchart_Cash_Soft_Red_Wheat_Index"),
}
H = 8
TRAIN_FRAC = 0.8
SEED = 42
NAN_MAX, CORR_DUP, CHG = 0.5, 0.95, 4
MIN_OBS, CLIP_Q = 52, (0.05, 0.95)
ALPHAS, RETUNE = np.logspace(-1, 6, 29), 13
METHODS = ("combo", "ridge")

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
def export():
    """Snapshot ml.<panel> and the daily price series to data/<commodity>_{X,price}.parquet."""
    import psycopg

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        for c, (panel, price_col) in TARGETS.items():
            cur.execute(f'SELECT * FROM ml."{panel}" ORDER BY week_end')
            X = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
            cur.execute("SELECT dt, value FROM public.value WHERE col_name=%s ORDER BY dt", (price_col,))
            px = pd.DataFrame(cur.fetchall(), columns=["dt", "value"])
            X.to_parquet(DATA_DIR / f"{c}_X.parquet", index=False)
            px.to_parquet(DATA_DIR / f"{c}_price.parquet", index=False)
            print(c, X.shape, px.shape)


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


def candidates(base, pre):
    """Levels + CHG-week changes (ffilled, backward-looking); drop sparse/constant/near-duplicate on pre."""
    lvl = base.ffill()
    chg = lvl.drop(columns=["tgt_logp", "tgt_r1"]).diff(CHG).add_prefix(f"chg{CHG}_")
    F = pd.concat([lvl, chg], axis=1)
    P = F.loc[pre]
    F = F.loc[:, (P.isna().mean() <= NAN_MAX) & (P.std() > 0)]
    corr = F.loc[pre].corr().abs().values
    keep = []
    for j in range(F.shape[1]):
        if all(not corr[j, k] > CORR_DUP for k in keep):
            keep.append(j)
    return F.iloc[:, keep]


# ---------------------------------------------------------------- forecasters
def realised(Y, i, h):
    """Training rows s with s + h <= i and a known y_h."""
    tr = np.arange(0, i - h + 1)
    return tr[~np.isnan(Y[tr, h - 1])]


def univariate_slopes(Xtr, y):
    """NaN-aware OLS slope and x mean for each column (pairwise complete rows)."""
    m = ~np.isnan(Xtr)
    n = m.sum(0)
    x = np.where(m, Xtr, 0.0)
    xm = x.sum(0) / np.maximum(n, 1)
    ym = (m * y[:, None]).sum(0) / np.maximum(n, 1)
    dx = np.where(m, Xtr - xm, 0.0)
    b = (dx * (y[:, None] - ym)).sum(0) / np.maximum((dx ** 2).sum(0), 1e-12)
    b[n < MIN_OBS] = np.nan
    return b, xm


def combo_forecast(Xtr, y, x_now, ref_sign):
    b, xm = univariate_slopes(Xtr, y)
    f = b * (x_now - xm)
    f[np.sign(b) != ref_sign] = 0.0  # sign restriction: slope must keep its first-origin sign
    f = f[~np.isnan(b) & ~np.isnan(x_now)]
    if not len(f):
        return 0.0
    return float(np.clip(f.mean(), *np.quantile(y, CLIP_Q)))


def standardise(Xtr, x_now):
    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
    sd[~(sd > 0)] = 1.0
    z = lambda a: np.nan_to_num((a - mu) / sd)
    return z(Xtr), z(x_now)


def ridge_alpha(Ztr, y):
    gs = GridSearchCV(Ridge(), {"alpha": ALPHAS}, cv=TimeSeriesSplit(5), scoring="neg_mean_squared_error")
    return float(gs.fit(Ztr, y).best_params_["alpha"])


# ---------------------------------------------------------------- experiment
def run(commodity, method):
    t0 = time.time()
    X, price = load(commodity)
    pff = price.ffill()
    last_week = min(X.dropna(how="all").index[-1], price.last_valid_index())
    weeks = X.index[X.index <= last_week]
    N = len(weeks)
    first = int(math.floor(TRAIN_FRAC * N))
    log.info("[%s] %s weeks=%s..%s N=%d first_origin=%s (idx %d) origins=%d", commodity, method,
             weeks[0].date(), weeks[-1].date(), N, weeks[first].date(), first, N - first)

    base = X.copy()
    base["tgt_logp"] = np.log(pff)
    base["tgt_r1"] = base["tgt_logp"].diff()
    lp = np.log(price)
    y = {h: lp.shift(-h) - np.log(pff) for h in range(1, H + 1)}  # y_h(t) = log(P_{t+h}/P_t)
    Y = np.column_stack([y[h].values for h in range(1, H + 1)])

    F = candidates(base, weeks[:first])
    Fv = F.values
    log.info("[%s] candidates: %d (levels + chg%d) from %d raw", commodity, F.shape[1], CHG, base.shape[1])

    ref_sign = {h: np.sign(univariate_slopes(Fv[realised(Y, first, h)], Y[realised(Y, first, h), h - 1])[0])
                for h in range(1, H + 1)}
    alpha, info = {}, {}
    if method == "combo":
        info = {h: {"n_pos": int((ref_sign[h] > 0).sum()), "n_neg": int((ref_sign[h] < 0).sum())} for h in ref_sign}

    rows = []
    for i in range(first, N):
        t = weeks[i]
        for h in range(1, H + 1):
            tr = realised(Y, i, h)  # s + h <= i : label already realised at origin
            assert weeks[tr[-1]] + pd.Timedelta(weeks=h) <= t
            if method == "combo":
                p = combo_forecast(Fv[tr], Y[tr, h - 1], Fv[i], ref_sign[h])
            else:
                Ztr, z_now = standardise(Fv[tr], Fv[i])
                if (i - first) % RETUNE == 0:
                    alpha[h] = ridge_alpha(Ztr, Y[tr, h - 1])
                    info.setdefault(h, []).append({"origin": str(t.date()), "alpha": alpha[h]})
                p = float(Ridge(alpha=alpha[h]).fit(Ztr, Y[tr, h - 1]).coef_ @ z_now)  # zero drift
            tgt = t + pd.Timedelta(weeks=h)
            rows.append({"origin": t, "target_date": tgt, "horizon": h,
                         "y_true": price.get(tgt, np.nan), "y_pred": pff[t] * math.exp(p),
                         "last_price": pff[t], "n_train": len(tr)})
        if (i - first) % 25 == 0:
            log.info("[%s] %s origin %d/%d %s (%.0fs)", commodity, method, i - first + 1, N - first,
                     t.date(), time.time() - t0)
    (LOG_DIR / f"{commodity}_{method}.json").write_text(json.dumps(
        {"commodity": commodity, "method": method, "n_candidates": F.shape[1], "horizons": info},
        indent=2, ensure_ascii=False))
    df = pd.DataFrame(rows)
    df.to_parquet(RES_DIR / f"predictions_{commodity}.parquet", index=False)
    assert df["origin"].min() == weeks[first] and df["origin"].nunique() == N - first
    log.info("[%s] saved %d predictions (%.0fs)", commodity, len(df), time.time() - t0)
    render_mp4(commodity, df, pff, price)
    return df


def metrics(d, pred_col):
    d = d.dropna(subset=["y_true"])
    err = d[pred_col] - d["y_true"]
    return float((err.abs() / d["y_true"].abs()).mean() * 100), float(np.sqrt((err ** 2).mean())), len(d)


def write_leaderboard(commodity, df):
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(LEADERBOARD) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS leaderboard(
            experiment TEXT, commodity TEXT, horizon TEXT, mape REAL, rmse REAL, n INTEGER,
            target_series TEXT, created_at TEXT, PRIMARY KEY(experiment, commodity, horizon))""")
        for exp, col in ((EXP, "y_pred"), ("naive", "last_price")):
            groups = [(str(h), g) for h, g in df.groupby("horizon")] + [("all", df)]
            for h, g in groups:
                mape, rmse, n = metrics(g, col)
                assert np.isfinite(mape) and np.isfinite(rmse)
                con.execute("INSERT OR REPLACE INTO leaderboard VALUES (?,?,?,?,?,?,?,?)",
                            (exp, commodity, h, mape, rmse, n, TARGETS[commodity][1], now))
                if exp == EXP or h == "all":
                    log.info("[%s] %-11s h=%-3s MAPE=%.3f%% RMSE=%.4f n=%d", commodity, exp, h, mape, rmse, n)


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
    if cmd == "export":
        return export()
    if cmd == "leaderboard":
        setup_logging()
        for method in METHODS:
            EXP = f"E005_{method}"
            for f in sorted((RES_DIR / method).glob("predictions_*.parquet")):
                write_leaderboard(f.stem.removeprefix("predictions_"), pd.read_parquet(f))
        return
    assert cmd in METHODS, f"command must be export, leaderboard or one of {METHODS}"
    EXP, RES_DIR = f"E005_{cmd}", RES_DIR / cmd
    setup_logging()
    names = sys.argv[2:] or list(TARGETS)
    log.info("%s start: %s", EXP, names)
    for c in names:
        run(c, cmd)
    log.info("%s done", EXP)


if __name__ == "__main__":
    main()
