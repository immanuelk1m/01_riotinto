"""E004: LightGBM multi-step strategy ablation for the 8-week price forecast.

Same protocol as E001-E003 (expanding TSCV from 80% point, step 1 week). One common feature
set for all strategies: top 30 by MI averaged over h=1..8, on labels realised before the
first TSCV origin. Strategies (Taieb et al. 2012, arXiv 1108.3259):
  direct     one model per horizon
  mimo       one model, rows stacked over h=1..8 with h as a feature
  dirmo2/4   MIMO per block of 2 / 4 horizons
  recursive  one 1-step model iterated 8 times, target features updated with predictions
  dirrec     per-horizon models chained on earlier-horizon targets (truth in training,
             predictions at origin)

Usage: uv run experiment/E004/E004.py <strategy> [commodity ...]
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

EXP = "E004"  # set to E004_<strategy> in main
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
    "wheat": ("wheat", "Com_Barchart_Cash_Soft_Red_Wheat_Index"),
}
H = 8
TRAIN_FRAC = 0.8
SEED = 42
TOP_K = 30
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                  verbose=-1, force_col_wise=True, n_jobs=1)  # 1 thread: commodity x strategy run as parallel processes
BLOCK_SIZE = {"direct": 1, "dirmo2": 2, "dirmo4": 4, "mimo": 8}
STRATEGIES = [*BLOCK_SIZE, "recursive", "dirrec"]

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
def mi_scores(base, y_h, idx):
    """MI of each column of base with y_h on idx (pairwise NaN drop)."""
    scores = {}
    for c in base.columns:
        d = pd.concat([base.loc[idx, c], y_h.loc[idx]], axis=1).dropna()
        if len(d) < 100 or d.iloc[:, 0].std() == 0:
            continue
        scores[c] = float(mutual_info_regression(d.iloc[:, [0]].values, d.iloc[:, 1].values, random_state=SEED)[0])
    return scores


def common_select(base, y, weeks, first, k=TOP_K):
    """Top-k by MI averaged over horizons; horizon h uses labels realised before the first origin."""
    per_h = [mi_scores(base, y[h], weeks[: first - h + 1]) for h in range(1, H + 1)]
    mean = {c: float(np.mean([s.get(c, 0.0) for s in per_h])) for c in base.columns}
    return sorted(mean.items(), key=lambda kv: -kv[1])[:k]


def fit_predict(Xtr, ytr, Xte):
    m = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xtr, ytr)
    return m, m.predict(Xte)


def realised(Y, i, h):
    """Training rows s with s + h <= i and a known y_h."""
    tr = np.arange(0, i - h + 1)
    return tr[~np.isnan(Y[tr, h - 1])]


def predict_origin(strategy, X, Y, i, tgt_pos):
    """8 log-return forecasts (h=1..8) at origin row i."""
    if strategy in BLOCK_SIZE:
        out = []
        for b in range(0, H, BLOCK_SIZE[strategy]):
            hs = range(b + 1, b + BLOCK_SIZE[strategy] + 1)
            trs = {h: realised(Y, i, h) for h in hs}
            Xtr = np.vstack([np.column_stack([X[trs[h]], np.full(len(trs[h]), h)]) for h in hs])
            ytr = np.concatenate([Y[trs[h], h - 1] for h in hs])
            Xte = np.column_stack([np.repeat(X[i:i + 1], len(hs), 0), list(hs)])
            out.extend(fit_predict(Xtr, ytr, Xte)[1])
        return np.array(out)
    if strategy == "recursive":
        # ponytail: exogenous features held at their origin value; only tgt_logp / tgt_r1 roll forward
        tr = realised(Y, i, 1)
        m = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[tr], Y[tr, 0])
        x, cum, out = X[i].copy(), 0.0, []
        for _ in range(H):
            r = m.predict(x[None])[0]
            cum += r
            out.append(cum)
            if "tgt_logp" in tgt_pos:
                x[tgt_pos["tgt_logp"]] += r
            if "tgt_r1" in tgt_pos:
                x[tgt_pos["tgt_r1"]] = r
        return np.array(out)
    # dirrec
    out = []
    for h in range(1, H + 1):
        tr = realised(Y, i, h)
        Xtr = np.column_stack([X[tr], Y[tr, : h - 1]])
        out.append(fit_predict(Xtr, Y[tr, h - 1], np.concatenate([X[i], out])[None])[1][0])
    return np.array(out)


# ---------------------------------------------------------------- experiment
def run(commodity, strategy):
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

    sel = common_select(base, y, weeks, first)
    names = [c for c, _ in sel]
    log.info("[%s] %s common MI top%d (max=%.4f min=%.4f): %s", commodity, strategy, len(sel),
             sel[0][1], sel[-1][1], ", ".join(names[:5]))
    (LOG_DIR / f"{commodity}_features.json").write_text(json.dumps(
        {"commodity": commodity, "n_candidates": base.shape[1],
         "features": [{"name": c, "mean_mi": s} for c, s in sel]}, indent=2, ensure_ascii=False))
    Xv = base[names].values
    Y = np.column_stack([y[h].values for h in range(1, H + 1)])
    tgt_pos = {c: names.index(c) for c in ("tgt_logp", "tgt_r1") if c in names}

    rows = []
    for i in range(first, N):
        t = weeks[i]
        for h in range(1, H + 1):
            tr = realised(Y, i, h)  # s + h <= i : label already realised at origin
            assert weeks[tr[-1]] + pd.Timedelta(weeks=h) <= t
        preds = predict_origin(strategy, Xv, Y, i, tgt_pos)
        for h in range(1, H + 1):
            tgt = t + pd.Timedelta(weeks=h)
            truth = price.get(tgt, np.nan)
            rows.append({"origin": t, "target_date": tgt, "horizon": h,
                         "y_true": truth, "y_pred": pff[t] * math.exp(preds[h - 1]),
                         "last_price": pff[t], "n_train": len(realised(Y, i, h))})
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
        if "max_ape_pct" not in [row[1] for row in con.execute("PRAGMA table_info(leaderboard)")]:
            con.execute("ALTER TABLE leaderboard ADD COLUMN max_ape_pct REAL")
        for exp, col in ((EXP, "y_pred"), ("naive", "last_price")):
            groups = [(str(h), g) for h, g in df.groupby("horizon")] + [("all", df)]
            for h, g in groups:
                mape, rmse, n = metrics(g, col)
                scored = g.dropna(subset=["y_true"])
                peak = float(((scored[col] - scored.y_true).abs() / scored.y_true.abs() * 100).max())
                assert np.isfinite(mape) and np.isfinite(rmse)
                con.execute("INSERT OR REPLACE INTO leaderboard(experiment, commodity, horizon, mape, rmse, n, target_series, created_at, max_ape_pct) VALUES (?,?,?,?,?,?,?,?,?)",
                            (exp, commodity, h, mape, rmse, n, TARGETS[commodity][1], now, peak))
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
    strategy = sys.argv[1]
    assert strategy in STRATEGIES, f"strategy must be one of {STRATEGIES}"
    EXP, RES_DIR = f"E004_{strategy}", RES_DIR / strategy
    setup_logging()
    names = sys.argv[2:] or list(TARGETS)
    log.info("%s start: %s", EXP, names)
    for c in names:
        df = run(c, strategy)
        if df is not None:
            write_leaderboard(c, df)
    log.info("%s done", EXP)


if __name__ == "__main__":
    main()
