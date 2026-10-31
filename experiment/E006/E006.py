"""E006: model-agnostic wrappers around LightGBM for the 8-week price forecast.

Same protocol as E001-E005 (expanding TSCV from 80% point, step 1 week, direct strategy,
y_h = log(P_{t+h}/P_t)). LightGBM base learner = E004_direct (same params, E004 common MI
top-30 features from features/<commodity>_features.json). Wrappers:
  ecw    full-history LGBM (exp. time-decay weights) + recent-window LGBM, combined by
         inverse recent squared error (Liu, Godahewa, Bandara & Bergmeir, arXiv 2304.01512)
  rewts  one LGBM per 104-week chunk, members + naive weighted on the simplex by least
         squares on the last 52 realised rounds (Johnsen et al., arXiv 2403.02150)
  aalf   naive almost always; LGBM only when a selector classifier is confident the LGBM
         beats naive (Jakobs & Liebig, arXiv 2409.10142)
  boa    Bernstein Online Aggregation over naive + every saved E001-E006 forecast, with
         h-week delayed feedback (Wintenberger 2024; Pfitzner et al., arXiv 2506.15217)

Runs on Colab from parquet snapshots (no DB):
  colab : python E006.py <ecw|rewts|aalf> [commodity ...]   then   python E006.py boa
  local : uv run experiment/E006/E006.py leaderboard
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
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

EXP = "E006"  # set to E006_<method> in main
ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
LOG_DIR, RES_DIR, DATA_DIR = EXP_DIR / "logs", EXP_DIR / "results", EXP_DIR / "data"
FEAT_DIR, EXPERT_DIR = EXP_DIR / "features", EXP_DIR / "experts"
LEADERBOARD = ROOT / "leaderboard.sqlite"

TARGETS = {
    "copper": "Com_LME_Cu_Cash_Bid",
    "gold": "Com_LBMA_Gold_Price_PM_USD_V_0",
    "corn": "Com_Barchart_Cash_National_Corn_Index",
    "coffee": "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag",
}
H = 8
TRAIN_FRAC = 0.8
SEED = 42
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                  verbose=-1, force_col_wise=True, n_jobs=1)  # = E004, 1 thread per process
PARTIAL_WEEKS, HALF_LIFE, ECW_WINDOW, ECW_MIN = 156, 260, 13, 4  # ecw
CHUNK, LOOKBACK = 104, 52                                          # rewts
INNER_BLOCKS, SEL_EVERY, SEL_FRAC = 6, 13, 0.1                     # aalf
SEL_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=1)
METHODS = ("ecw", "rewts", "aalf", "boa")

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


def realised(Y, i, h):
    """Training rows s with s + h <= i and a known y_h."""
    tr = np.arange(0, i - h + 1)
    return tr[~np.isnan(Y[tr, h - 1])]


def fit(Xtr, ytr, w=None, params=LGB_PARAMS):
    return lgb.LGBMRegressor(**params).fit(Xtr, ytr, sample_weight=w)


# ---------------------------------------------------------------- wrappers (state kept in closures)
def ecw(X, Y, first, N, info):
    hist = {h: [] for h in range(1, H + 1)}  # (origin, p_all, p_partial)

    def step(i):
        out = []
        for h in range(1, H + 1):
            tr = realised(Y, i, h)
            part = tr[tr > i - h - PARTIAL_WEEKS]
            p_all = fit(X[tr], Y[tr, h - 1], 0.5 ** ((i - tr) / HALF_LIFE)).predict(X[i:i + 1])[0]
            p_part = fit(X[part], Y[part, h - 1]).predict(X[i:i + 1])[0]
            done = [(s, a, b) for s, a, b in hist[h] if s + h <= i and not np.isnan(Y[s, h - 1])][-ECW_WINDOW:]
            w_part = 0.5
            if len(done) >= ECW_MIN:
                y = np.array([Y[s, h - 1] for s, _, _ in done])
                e_all = np.mean((np.array([a for _, a, _ in done]) - y) ** 2)
                e_part = np.mean((np.array([b for _, _, b in done]) - y) ** 2)
                w_part = e_all / (e_all + e_part)  # inverse-error weight of the recent model
            hist[h].append((i, p_all, p_part))
            info.setdefault(h, []).append(w_part)
            out.append(w_part * p_part + (1 - w_part) * p_all)
        return out
    return step


def simplex_lsq(P, y):
    """argmin ||P w - y||^2 s.t. w >= 0, sum w = 1."""
    k = P.shape[1]
    res = minimize(lambda w: np.sum((P @ w - y) ** 2), np.full(k, 1 / k), jac=lambda w: 2 * P.T @ (P @ w - y),
                   bounds=[(0, 1)] * k, constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP")
    return res.x


def rewts(X, Y, first, N, info):
    preds = {}  # (chunk, h) -> chunk model forecasts for every row

    def chunk_preds(c, h):
        if (c, h) not in preds:
            rows = np.arange(c * CHUNK, (c + 1) * CHUNK)
            rows = rows[~np.isnan(Y[rows, h - 1])]
            preds[(c, h)] = fit(X[rows], Y[rows, h - 1]).predict(X) if len(rows) >= 2 * LGB_PARAMS["min_child_samples"] else None
        return preds[(c, h)]

    def step(i):
        out = []
        for h in range(1, H + 1):
            L = realised(Y, i, h)
            L = L[L > i - h - LOOKBACK]
            # chunks fully realised and ending before the look-back window (no in-sample weighting)
            members = [p for c in range(L[0] // CHUNK) if (p := chunk_preds(c, h)) is not None]
            if not members:
                out.append(0.0)
                continue
            P = np.column_stack([np.zeros(len(L)), *[p[L] for p in members]])  # column 0 = naive
            w = simplex_lsq(P, Y[L, h - 1])
            info.setdefault(h, []).append({"w_naive": float(w[0]), "n_chunks": len(members)})
            out.append(float(w[1:] @ np.array([p[i] for p in members])))
        return out
    return step


def aalf(X, Y, first, N, info):
    oos = {h: {} for h in range(1, H + 1)}  # row -> out-of-sample LGBM forecast
    for h in range(1, H + 1):  # walk-forward OOS forecasts on pre-origin rows to seed the selector
        for b in np.array_split(np.arange(first), INNER_BLOCKS)[1:]:
            tr = realised(Y, b[0], h)
            for s, p in zip(b, fit(X[tr], Y[tr, h - 1]).predict(X[b])):
                oos[h][s] = p
    sel = {}

    def selector(i, h):
        rows = np.array([s for s in oos[h] if s + h <= i and not np.isnan(Y[s, h - 1])])
        p = np.array([oos[h][s] for s in rows])
        y = Y[rows, h - 1]
        F = np.column_stack([X[rows], np.abs(p)])
        clf = lgb.LGBMClassifier(**SEL_PARAMS).fit(F, ((p - y) ** 2 < y ** 2).astype(int))
        # ponytail: threshold = in-sample (1-SEL_FRAC) score quantile; calibrate on held-out rows if it drifts
        return clf, np.quantile(clf.predict_proba(F)[:, 1], 1 - SEL_FRAC)

    def step(i):
        out = []
        for h in range(1, H + 1):
            if (i - first) % SEL_EVERY == 0:
                sel[h] = selector(i, h)
            tr = realised(Y, i, h)
            p = fit(X[tr], Y[tr, h - 1]).predict(X[i:i + 1])[0]
            oos[h][i] = p
            clf, tau = sel[h]
            use = clf.predict_proba(np.append(X[i], abs(p))[None])[0, 1] >= tau
            info.setdefault(h, []).append(int(use))
            out.append(p if use else 0.0)
        return out
    return step


WRAPPERS = {"ecw": ecw, "rewts": rewts, "aalf": aalf}


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
    Y = np.column_stack([(lp.shift(-h) - np.log(pff)).values for h in range(1, H + 1)])
    names = [f["name"] for f in json.loads((FEAT_DIR / f"{commodity}_features.json").read_text())["features"]]
    Xv = base[names].values

    info = {}
    step = WRAPPERS[method](Xv, Y, first, N, info)
    rows = []
    for i in range(first, N):
        t = weeks[i]
        for h in range(1, H + 1):
            assert weeks[realised(Y, i, h)[-1]] + pd.Timedelta(weeks=h) <= t
        preds = step(i)
        for h in range(1, H + 1):
            tgt = t + pd.Timedelta(weeks=h)
            rows.append({"origin": t, "target_date": tgt, "horizon": h,
                         "y_true": price.get(tgt, np.nan), "y_pred": pff[t] * math.exp(preds[h - 1]),
                         "last_price": pff[t], "n_train": len(realised(Y, i, h))})
        if (i - first) % 25 == 0:
            log.info("[%s] %s origin %d/%d %s (%.0fs)", commodity, method, i - first + 1, N - first,
                     t.date(), time.time() - t0)
    (LOG_DIR / f"{commodity}_{method}.json").write_text(json.dumps(
        {"commodity": commodity, "method": method, "features": names, "trace": info}, ensure_ascii=False))
    finish(commodity, pd.DataFrame(rows), pff, price, weeks[first], N - first, t0)


def finish(commodity, df, pff, price, first_origin, n_origins, t0):
    df.to_parquet(RES_DIR / f"predictions_{commodity}.parquet", index=False)
    assert df["origin"].min() == first_origin and df["origin"].nunique() == n_origins
    log.info("[%s] saved %d predictions (%.0fs)", commodity, len(df), time.time() - t0)
    render_mp4(commodity, df, pff, price)


# ---------------------------------------------------------------- BOA over saved forecasts
def boa_weights(Xe, y, h):
    """BOA (gradient trick, adaptive per-expert rates) with h-round delayed feedback.
    Xe: (T, E) expert log-return forecasts, y: (T,) realised log-returns (NaN = unknown)."""
    T, E = Xe.shape
    log_w0 = np.full(E, -np.log(E))
    R, V, B = np.zeros(E), np.zeros(E), 0.0
    w, agg, W = np.full(E, 1 / E), np.zeros(T), np.zeros((T, E))
    for k in range(T):
        s = k - h  # round whose target is realised at origin k
        if s >= 0 and not np.isnan(y[s]):
            r = 2 * (agg[s] - y[s]) * (agg[s] - Xe[s])  # linearised instantaneous regret
            V += r ** 2
            B = max(B, np.abs(r).max(), 1e-12)
            eta = np.minimum(1 / (2 * B), np.sqrt(-log_w0 / np.maximum(V, 1e-300)))
            R += r - eta * r ** 2
            z = log_w0 + np.log(eta) + eta * R
            w = np.exp(z - z.max())
            w /= w.sum()
        W[k] = w
        agg[k] = w @ Xe[k]
    return agg, W


def run_boa(commodity):
    t0 = time.time()
    files = {f.parent.name: f for f in sorted(EXPERT_DIR.glob(f"*/predictions_{commodity}.parquet"))}
    files |= {f"E006_{m}": RES_DIR.parent / m / f"predictions_{commodity}.parquet" for m in WRAPPERS}
    frames = {n: pd.read_parquet(f).astype({"origin": "datetime64[ns]"}).sort_values(["horizon", "origin"])
              .reset_index(drop=True) for n, f in files.items()}  # local (ms) vs Colab (ns) parquet units
    ref = frames[next(iter(frames))]
    for n, d in frames.items():
        assert d[["origin", "horizon"]].equals(ref[["origin", "horizon"]]), n
    names = ["naive", *frames]
    _, price = load(commodity)
    pff = price.ffill()
    out, info = [], {}
    for h, g in ref.groupby("horizon"):
        idx = g.index
        y = np.log(g["y_true"] / g["last_price"]).values
        Xe = np.column_stack([np.zeros(len(g))] + [np.log(frames[n].loc[idx, "y_pred"] / g["last_price"]).values
                                                   for n in frames])
        agg, W = boa_weights(Xe, y, h)
        out.append(g.assign(y_pred=g["last_price"].values * np.exp(agg)))
        info[int(h)] = {"final": dict(zip(names, W[-1].round(4).tolist())),
                        "mean": dict(zip(names, W.mean(0).round(4).tolist()))}
        top = sorted(info[int(h)]["mean"].items(), key=lambda kv: -kv[1])[:3]
        log.info("[%s] boa h=%d mean weights top3: %s", commodity, h, top)
    (LOG_DIR / f"{commodity}_boa.json").write_text(json.dumps({"experts": names, "weights": info}, indent=2))
    df = pd.concat(out).sort_values(["origin", "horizon"]).reset_index(drop=True)
    finish(commodity, df, pff, price, ref["origin"].min(), ref["origin"].nunique(), t0)


# ---------------------------------------------------------------- outputs
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
                            (exp, commodity, h, mape, rmse, n, TARGETS[commodity], now))
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
    if cmd == "leaderboard":
        setup_logging()
        for method in METHODS:
            EXP = f"E006_{method}"
            for f in sorted((RES_DIR / method).glob("predictions_*.parquet")):
                write_leaderboard(f.stem.removeprefix("predictions_"), pd.read_parquet(f))
        return
    assert cmd in METHODS, f"command must be leaderboard or one of {METHODS}"
    EXP, RES_DIR = f"E006_{cmd}", RES_DIR / cmd
    setup_logging()
    names = sys.argv[2:] or list(TARGETS)
    log.info("%s start: %s", EXP, names)
    for c in names:
        run_boa(c) if cmd == "boa" else run(c, cmd)
    log.info("%s done", EXP)


if __name__ == "__main__":
    main()
