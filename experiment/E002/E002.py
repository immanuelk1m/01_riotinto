"""E002: LightGBM direct 8-week price forecast with Mutual Information top-30 feature selection.

Same protocol as E001 (expanding TSCV from 80% point, step 1 week) with ELATE removed:
per horizon h, rank raw panel columns + target features by MI with y_h on labels realised
before the first TSCV origin, keep the top 30, fixed for all origins.

Usage: uv run experiment/E002/E002.py [commodity ...]
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
import psycopg
from matplotlib import animation
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")

EXP = "E002"
ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
LOG_DIR, RES_DIR = EXP_DIR / "logs", EXP_DIR / "results"
LEADERBOARD = ROOT / "leaderboard.sqlite"
DSN = "dbname=commodity"

# commodity -> (ml panel, daily price series in public.value)
TARGETS = {
    "copper": ("copper", "Com_LME_Cu_Cash_Bid"),
    "gold": ("gold", "Com_LBMA_Gold_Price_PM_USD_V_0"),
    "corn": ("corn", "Com_Barchart_Cash_National_Corn_Index"),
    "coffee": ("coffee", "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag"),
}
H = 8
TRAIN_FRAC = 0.8
SEED = 42
TOP_K = 30
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                  verbose=-1, force_col_wise=True, n_jobs=2)  # 2 threads: commodities run as parallel processes

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
def load(panel, price_col):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(f'SELECT * FROM ml."{panel}" ORDER BY week_end')
        cols = [d.name for d in cur.description]
        X = pd.DataFrame(cur.fetchall(), columns=cols)
        cur.execute("SELECT dt, value FROM public.value WHERE col_name=%s ORDER BY dt", (price_col,))
        px = pd.DataFrame(cur.fetchall(), columns=["dt", "value"])
    if px.empty:
        return None, None
    X["week_end"] = pd.to_datetime(X["week_end"])
    X = X.set_index("week_end").astype(float)
    px = px.set_index(pd.to_datetime(px["dt"]))["value"].astype(float)
    weekly = px.resample("W-SUN").last()  # no ffill: weeks without a quote stay NaN
    start = weekly.first_valid_index()
    idx = pd.date_range(max(start, X.index[0]), weekly.index[-1], freq="W-SUN")
    return X.reindex(idx), weekly.reindex(idx)


# ---------------------------------------------------------------- MI feature selection
def mi_select(base, y_h, idx, k=TOP_K):
    """Top-k columns of base by MI with y_h on idx (pairwise NaN drop)."""
    scores = {}
    for c in base.columns:
        d = pd.concat([base.loc[idx, c], y_h.loc[idx]], axis=1).dropna()
        if len(d) < 100 or d.iloc[:, 0].std() == 0:
            continue
        scores[c] = float(mutual_info_regression(d.iloc[:, [0]].values, d.iloc[:, 1].values, random_state=SEED)[0])
    return sorted(scores.items(), key=lambda kv: -kv[1])[:k]


def fit_predict(Xtr, ytr, Xte):
    m = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xtr, ytr)
    return m, m.predict(Xte)


# ---------------------------------------------------------------- experiment
def run(commodity):
    panel, price_col = TARGETS[commodity]
    t0 = time.time()
    X, price = load(panel, price_col)
    if X is None:
        log.warning("[%s] no rows for %s in public.value — skipped", commodity, price_col)
        return None
    pff = price.ffill()
    last_week = min(X.dropna(how="all").index[-1], price.last_valid_index())
    weeks = X.index[X.index <= last_week]
    N = len(weeks)
    first = int(math.floor(TRAIN_FRAC * N))
    log.info("[%s] price=%s weeks=%s..%s N=%d first_origin=%s (idx %d) origins=%d",
             commodity, price_col, weeks[0].date(), weeks[-1].date(), N, weeks[first].date(), first, N - first)

    base = X.copy()
    base["tgt_logp"] = np.log(pff)
    base["tgt_r1"] = base["tgt_logp"].diff()
    lp = np.log(price)
    y = {h: lp.shift(-h) - np.log(pff) for h in range(1, H + 1)}  # y_h(t) = log(P_{t+h}/P_t)

    # per-horizon selection on labels realised before the first origin: s + h <= first
    sel, Xv = {}, {}
    for h in range(1, H + 1):
        sel[h] = mi_select(base, y[h], weeks[: first - h + 1])
        Xv[h] = base[[c for c, _ in sel[h]]].values
        log.info("[%s] h=%d MI top%d (max=%.4f min=%.4f): %s", commodity, h, len(sel[h]),
                 sel[h][0][1], sel[h][-1][1], ", ".join(c for c, _ in sel[h][:5]))
    (LOG_DIR / f"{commodity}_mi.json").write_text(json.dumps(
        {"commodity": commodity, "n_candidates": base.shape[1],
         "features": {h: [{"name": c, "mi": s} for c, s in sel[h]] for h in sel}},
        indent=2, ensure_ascii=False))

    rows = []
    for i in range(first, N):
        t = weeks[i]
        for h in range(1, H + 1):
            yh = y[h].values
            tr = np.arange(0, i - h + 1)  # s + h <= i : label already realised at origin
            tr = tr[~np.isnan(yh[tr])]
            assert weeks[tr[-1]] + pd.Timedelta(weeks=h) <= t
            _, p = fit_predict(Xv[h][tr], yh[tr], Xv[h][i:i + 1])
            tgt = t + pd.Timedelta(weeks=h)
            truth = price.get(tgt, np.nan)
            rows.append({"origin": t, "target_date": tgt, "horizon": h,
                         "y_true": truth, "y_pred": pff[t] * math.exp(p[0]),
                         "last_price": pff[t], "n_train": len(tr)})
        if (i - first) % 25 == 0:
            log.info("[%s] origin %d/%d %s (%.0fs)", commodity, i - first + 1, N - first, t.date(), time.time() - t0)
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
    setup_logging()
    names = sys.argv[1:] or list(TARGETS)
    log.info("%s start: %s", EXP, names)
    for c in names:
        df = run(c)
        if df is not None:
            write_leaderboard(c, df)
    log.info("%s done", EXP)


if __name__ == "__main__":
    main()
