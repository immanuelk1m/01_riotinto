"""E001: LightGBM direct 8-week price forecast with ELATE-style feature engineering.

ELATE (arXiv 2508.14667) with the LLM generator replaced by rule-based mutation:
temperature-softmax parent sampling, Granger+MI evaluators, corr + SHAP pruning,
generation-level validation RMSE selection. Feature search runs only on data
before the first TSCV origin.

Usage: uv run experiment/E001/E001.py [commodity ...]
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
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

EXP = "E001"
ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
LOG_DIR, RES_DIR = EXP_DIR / "logs", EXP_DIR / "results"
LEADERBOARD = ROOT / "leaderboard.sqlite"
DSN = "dbname=commodity"

# commodity -> (ml panel, daily price series in public.value)
TARGETS = {
    "wti_crude_oil": ("wti_crude_oil", "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel"),
    "copper": ("copper", "Com_LME_Cu_Cash_Bid"),
    "gold": ("gold", "Com_LBMA_Gold_Price_PM_USD_V_0"),
    "corn": ("corn", "Com_Barchart_Cash_National_Corn_Index"),
    "coffee": ("coffee", "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag"),
}
H = 8
TRAIN_FRAC = 0.8
SEED = 42
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                  verbose=-1, force_col_wise=True, n_jobs=2)  # 2 threads: commodities run as parallel processes
# ELATE hyperparameters (paper defaults)
N_MAX, N_KEEP, GENERATIONS = 100, 50, 10
T0, K_DECAY, EPS = 10.0, 5.0, 0.1
CORR_MAX, PRUNE_FRAC = 0.9, 0.1
WINDOWS, LAGS = (4, 8, 13, 26), (1, 2, 4, 8)

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


# ---------------------------------------------------------------- ELATE features
# A feature is a nested tuple spec: ("raw", col) | ("lag", k, f) | ("rmean", w, f) |
# ("rstd", w, f) | ("diff", k, f) | ("pct", k, f) | ("z", w, f) | ("sub", f, g) | ("ratio", f, g).
# Every op only looks backwards in time.
def compute(spec, base, cache):
    if spec in cache:
        return cache[spec]
    op = spec[0]
    if op == "raw":
        s = base[spec[1]]
    elif op in ("sub", "ratio"):
        a, b = compute(spec[1], base, cache), compute(spec[2], base, cache)
        s = a - b if op == "sub" else a / b.where(b.abs() > 1e-12)
    else:
        n, f = spec[1], compute(spec[2], base, cache).ffill()
        s = {"lag": lambda: f.shift(n),
             "rmean": lambda: f.rolling(n, min_periods=n // 2).mean(),
             "rstd": lambda: f.rolling(n, min_periods=n // 2).std(),
             "diff": lambda: f.diff(n),
             "pct": lambda: f.pct_change(n, fill_method=None),
             "z": lambda: (f - f.rolling(n, min_periods=n // 2).mean()) / f.rolling(n, min_periods=n // 2).std()}[op]()
    s = s.replace([np.inf, -np.inf], np.nan)
    cache[spec] = s
    return s


def name_of(spec):
    if spec[0] == "raw":
        return spec[1]
    return f"{spec[0]}(" + ",".join(name_of(a) if isinstance(a, tuple) else str(a) for a in spec[1:]) + ")"


def evaluate(x, r_now, y_next):
    """ELATE evaluator: mean of Granger (linear, temporal) and MI (non-linear) scores in [0, 1]."""
    d = pd.concat([r_now, x], axis=1).dropna()
    if len(d) < 100 or d.iloc[:, 1].std() == 0:
        return 0.0
    try:
        res = grangercausalitytests(d.values, maxlag=4, verbose=False)
        g = 1 - min(v[0]["ssr_ftest"][1] for v in res.values())
    except Exception:
        g = 0.0
    m = pd.concat([x, y_next], axis=1).dropna()
    mi = mutual_info_regression(m.iloc[:, [0]].values, m.iloc[:, 1].values, random_state=SEED)[0]
    return float((g + (1 - math.exp(-2 * mi))) / 2)  # MI mapped to [0,1] (information coefficient)


def fit_predict(Xtr, ytr, Xte, n_estimators=None):
    params = dict(LGB_PARAMS, **({"n_estimators": n_estimators} if n_estimators else {}))
    m = lgb.LGBMRegressor(**params).fit(Xtr, ytr)
    return m, m.predict(Xte)


def shap_prune(db, F, Xb, y, tr_idx):
    """Corr filter, then drop the worst 10% by fold-normalised TreeSHAP until N_KEEP remain."""
    names = sorted(db, key=lambda n: -db[n]["score"])
    keep = []
    corr = F.loc[tr_idx, names].corr().abs()
    for n in names:
        if all(corr.loc[n, k] <= CORR_MAX or np.isnan(corr.loc[n, k]) for k in keep):
            keep.append(n)
    folds = np.array_split(np.arange(len(tr_idx)), 4)  # walk-forward: train on <k, score on k
    while len(keep) > N_KEEP:
        imp = np.zeros(len(keep))
        for k in range(1, 4):
            tr, te = np.concatenate(folds[:k]), folds[k]
            Xall = pd.concat([Xb.loc[tr_idx], F.loc[tr_idx, keep]], axis=1).values
            ok = ~np.isnan(y.loc[tr_idx].values)
            trm, tem = tr[ok[tr]], te[ok[te]]
            m = lgb.LGBMRegressor(**dict(LGB_PARAMS, n_estimators=100)).fit(Xall[trm], y.loc[tr_idx].values[trm])
            contrib = np.abs(m.booster_.predict(Xall[tem], pred_contrib=True)[:, :-1]).mean(0)
            contrib = contrib[Xb.shape[1]:]
            imp += contrib / (contrib.sum() or 1)
        n_drop = min(max(1, int(len(keep) * PRUNE_FRAC)), len(keep) - N_KEEP)
        for i in np.argsort(imp)[:n_drop]:
            keep[i] = None
        keep = [k for k in keep if k]
    return {n: db[n] for n in keep}


def elate(base, raw_cols, y1, r_now, tr_idx, va_idx, commodity):
    rng = np.random.default_rng(SEED)
    cache, F = {}, {}
    Xb = base[raw_cols]
    y_tr, r_tr = y1.loc[tr_idx], r_now.loc[tr_idx]

    def add(spec, db):
        nm = name_of(spec)
        if nm in db or len(nm) > 400:
            return
        s = compute(spec, base, cache)
        if s.loc[tr_idx].notna().sum() < 100 or s.loc[tr_idx].std() == 0:
            return
        F[nm] = s
        db[nm] = {"spec": spec, "score": evaluate(s.loc[tr_idx], r_tr, y_tr)}

    def val_rmse(db):
        cols = list(db)
        Xa = pd.concat([Xb, pd.DataFrame({c: F[c] for c in cols})], axis=1)
        ok_tr, ok_va = y1.loc[tr_idx].notna().values, y1.loc[va_idx].notna().values
        _, p = fit_predict(Xa.loc[tr_idx].values[ok_tr], y1.loc[tr_idx].values[ok_tr],
                           Xa.loc[va_idx].values[ok_va], n_estimators=150)
        return float(np.sqrt(np.mean((p - y1.loc[va_idx].values[ok_va]) ** 2)))

    def mutate(parents):
        op = rng.choice(["lag", "rmean", "rstd", "diff", "pct", "z", "sub", "ratio", "raw_unary"])
        p = parents[0]
        if op in ("sub", "ratio"):
            q = parents[1] if rng.random() < 0.5 else ("raw", rng.choice(raw_cols))
            return (op, p, q)
        if op == "raw_unary":  # explore base data D_B
            p, op = ("raw", rng.choice(raw_cols)), rng.choice(["lag", "rmean", "diff", "pct", "z"])
        n = int(rng.choice(LAGS if op in ("lag", "diff", "pct") else WINDOWS))
        return (op, n, p)

    # initial features X_I (paper seeds with 2)
    db = {}
    add(("raw", "tgt_r1"), db)
    add(("rmean", 4, ("raw", "tgt_r1")), db)
    best, best_rmse = dict(db), val_rmse(db)
    history = [{"generation": 0, "n": len(db), "val_rmse": best_rmse}]
    log.info("[%s] ELATE gen0 seed rmse=%.5f", commodity, best_rmse)
    for g in range(1, GENERATIONS + 1):
        tries = 0
        while len(db) < N_MAX and tries < 2000:
            tries += 1
            names = list(db)
            T = T0 * math.exp(-K_DECAY * len(db) / N_MAX) + EPS
            sc = np.array([db[n]["score"] for n in names]) / T
            pr = np.exp(sc - sc.max()); pr /= pr.sum()
            picks = rng.choice(len(names), size=min(2, len(names)), replace=len(names) < 2, p=pr)
            add(mutate([db[names[i]]["spec"] for i in picks]), db)
        db = shap_prune(db, pd.DataFrame(F), Xb, y1, tr_idx)
        rmse = val_rmse(db)
        improved = rmse < best_rmse
        if improved:
            best, best_rmse = dict(db), rmse
        else:
            db = dict(best)
        history.append({"generation": g, "n": len(db), "val_rmse": rmse, "kept": improved})
        log.info("[%s] ELATE gen%d n=%d val_rmse=%.5f best=%.5f %s", commodity, g, len(db), rmse,
                 best_rmse, "improved" if improved else "reverted")
    out = {"commodity": commodity, "best_val_rmse": best_rmse, "history": history,
           "features": [{"name": n, "score": v["score"]} for n, v in sorted(best.items(), key=lambda kv: -kv[1]["score"])]}
    (LOG_DIR / f"{commodity}_elate.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return pd.DataFrame({n: F[n] for n in best})


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
    raw_cols = list(base.columns)
    lp = np.log(price)
    y = {h: lp.shift(-h) - np.log(pff) for h in range(1, H + 1)}  # y_h(t) = log(P_{t+h}/P_t)

    # ELATE on pre-origin data only: last target used is y1 at first-2 -> realised at first-1
    pre = weeks[: first - 1]
    cut = int(len(pre) * 0.9)
    feats = elate(base, raw_cols, y[1], base["tgt_r1"], pre[:cut], pre[cut:], commodity)
    full = pd.concat([base, feats], axis=1)
    Xv = full.values
    log.info("[%s] features: base=%d elate=%d total=%d (%.0fs)", commodity, len(raw_cols), feats.shape[1],
             full.shape[1], time.time() - t0)

    rows = []
    for i in range(first, N):
        t = weeks[i]
        for h in range(1, H + 1):
            yh = y[h].values
            tr = np.arange(0, i - h + 1)  # s + h <= i : label already realised at origin
            tr = tr[~np.isnan(yh[tr])]
            assert weeks[tr[-1]] + pd.Timedelta(weeks=h) <= t
            _, p = fit_predict(Xv[tr], yh[tr], Xv[i:i + 1])
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
