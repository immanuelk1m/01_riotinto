"""E007: transfer-first LightGBM — internal validation must predict the test outcome.

Same protocol as E001-E006 (expanding TSCV from 80% point, step 1 week, direct strategy,
y_h = log(P_{t+h}/P_t), E004 common MI top-30 features). Shared core for every direction:
  * every RETUNE origins, purged/embargoed blocked k-fold on the last CV_BLOCKS x BLOCK_WEEKS
    realised weeks (rows whose h-week label overlaps the test block are purged, h rows after it
    embargoed; Arian, Norouzi & Seco 2024 CPCV; Soloviov purged-CV quantification)
  * shrink factor kappa in [0, 1] on the pooled out-of-fold forecasts (kappa = 0 -> naive)
  * deploy only if the out-of-fold loss gain over naive has HAC t >= T_GATE, else naive
  * log (validation gain, realised test gain) per retune block -> transfer rate
Directions (one tmux pane each):
  purged   LGBM config grid {E004, shallow, rf}, select by mean purged-CV loss
  robust   same grid, lexicographic mean -> worst-block loss with per-config reshuffled block
           offsets, conservative kappa (HyperTime arXiv 2305.18421; Nagler et al. arXiv 2405.15393)
  anomaly  rolling standardized-anomaly features + persistence predictors (climate preprocessing
           practice arXiv 2508.07062; MIXSAMOS arXiv 2412.09583; persistence arXiv 2410.11916)
  shallow  additive depth-1/2 LGBM with monotone constraints from sign-stable pre-period slopes
           (arXiv 2410.19098; shape constraints arXiv 2409.17084)
  tsfm     Chronos-Bolt zero-shot anchor + LGBM residual on covariates (TFMAdapter arXiv
           2509.13906; ChronosX arXiv 2503.12107)

Runs on Colab from parquet snapshots:
  colab : python E007.py <direction> [commodity ...]
  local : uv run experiment/E007/E007.py leaderboard [direction ...]
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

EXP = "E007"  # set to E007_<direction> in main
ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
LOG_DIR, RES_DIR, DATA_DIR, FEAT_DIR = (EXP_DIR / d for d in ("logs", "results", "data", "features"))
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
RETUNE, CV_BLOCKS, BLOCK_WEEKS, T_GATE = 26, 5, 52, 1.0
ROBUST_TOL, RESHUFFLE_MAX = 0.01, 13
ANOM_WINDOW, CHRONOS_CONTEXT = 156, 512
BASE = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, force_col_wise=True,
            n_jobs=1)  # = E004
GRID = {
    "e004": BASE,
    "shallow": dict(BASE, num_leaves=4, max_depth=2, learning_rate=0.02, n_estimators=200, min_child_samples=50),
    "rf": dict(BASE, boosting_type="rf", subsample=0.7, colsample_bytree=0.7),
}
DIRECTIONS = ("purged", "robust", "anomaly", "shallow", "tsfm")

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


def hac_t(d, lag):
    """t-stat of mean(d) with Bartlett/Newey-West variance (lag = horizon - 1 overlap)."""
    n = len(d)
    if n < 10:
        return 0.0
    dc = d - d.mean()
    v = dc @ dc / n + 2 * sum((1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lag + 1))
    return float(d.mean() / math.sqrt(max(v, 1e-300) / n))


# ---------------------------------------------------------------- features per direction
def features(direction, commodity, base, pff, first):
    names = [f["name"] for f in json.loads((FEAT_DIR / f"{commodity}_features.json").read_text())["features"]]
    F = base[names].ffill()
    if direction == "anomaly":
        roll = F.rolling(ANOM_WINDOW, min_periods=ANOM_WINDOW // 3)
        F = ((F - roll.mean()) / roll.std()).add_prefix("anom_")
        lp = np.log(pff)
        F = F.assign(persist_r1=lp.diff(1), persist_r4=lp.diff(4), persist_r13=lp.diff(13))
    return F


def monotone_signs(X, Y, first):
    """Per (h, feature): sign of the pre-period univariate slope if equal in both halves, else 0."""
    out = {}
    halves = np.array_split(np.arange(first), 2)
    for h in range(1, H + 1):
        signs = []
        for rows in halves:
            rows = rows[~np.isnan(Y[rows, h - 1])]
            x, y = X[rows], Y[rows, h - 1]
            m = ~np.isnan(x)
            xm = np.where(m, x, 0).sum(0) / np.maximum(m.sum(0), 1)
            dx = np.where(m, x - xm, 0)
            signs.append(np.sign((dx * (y - y.mean())[:, None]).sum(0)))
        out[h] = np.where(signs[0] == signs[1], signs[0], 0).astype(int)
    return out


def chronos_anchor(pff):
    """Chronos-Bolt zero-shot median log-return forecasts c[t, h-1] for every week t."""
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu",
                                               torch_dtype=torch.float32)
    v = pff.values
    C = np.full((len(v), H), np.nan)
    ts = [t for t in range(len(v)) if t >= 52 and not np.isnan(v[t])]
    for b in range(0, len(ts), 256):
        batch = ts[b:b + 256]
        ctx = [torch.tensor(v[max(0, t - CHRONOS_CONTEXT + 1):t + 1], dtype=torch.float32) for t in batch]
        q, _ = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.5])
        C[batch] = np.log(q[:, :, 0].numpy() / v[batch][:, None])
    return C


# ---------------------------------------------------------------- purged CV + selection
def cv_oos(params, X, y, i, h, offset=0):
    """Purged/embargoed blocked k-fold over the last CV_BLOCKS*BLOCK_WEEKS realised rows.
    Returns (rows, out-of-fold forecasts, block id)."""
    end = i - h + 1 - offset  # rows < end are realised at origin i
    start = end - CV_BLOCKS * BLOCK_WEEKS
    usable = np.arange(0, i - h + 1)
    usable = usable[~np.isnan(y[usable])]
    rows, preds, blk = [], [], []
    for b in range(CV_BLOCKS):
        a, e = start + b * BLOCK_WEEKS, start + (b + 1) * BLOCK_WEEKS
        test = usable[(usable >= a) & (usable < e)]
        train = usable[(usable < a - h) | (usable >= e + h)]  # purge label overlap, embargo h after
        if len(test) == 0 or len(train) < 100:
            continue
        p = lgb.LGBMRegressor(**params).fit(X[train], y[train]).predict(X[test])
        rows.append(test), preds.append(p), blk.append(np.full(len(test), b))
    return np.concatenate(rows), np.concatenate(preds), np.concatenate(blk)


def score(p, y, blk, h, conservative=False):
    """kappa, validation gain (relative MSE vs naive, <0 = better), HAC t of the gain, worst block."""
    kappa = float(np.clip((p @ y) / max(p @ p, 1e-300), 0, 1))
    if conservative:  # most cautious per-block kappa
        kappa = min(float(np.clip((p[blk == b] @ y[blk == b]) / max(p[blk == b] @ p[blk == b], 1e-300), 0, 1))
                    for b in np.unique(blk))
    d = y ** 2 - (y - kappa * p) ** 2
    rel = float(np.mean((y - kappa * p) ** 2) / np.mean(y ** 2) - 1)
    worst = max(float(np.mean((y[blk == b] - kappa * p[blk == b]) ** 2) / np.mean(y[blk == b] ** 2) - 1)
                for b in np.unique(blk))
    return {"kappa": kappa, "val_rel": rel, "t": hac_t(d, h - 1), "worst": worst}


def select(direction, X, Y, i, h, mono, offset_y=None, retune_id=0):
    """Pick (config, kappa) for horizon h at origin i; kappa = 0 unless the CV gain passes the gate."""
    y = Y[:, h - 1] if offset_y is None else Y[:, h - 1] - offset_y
    grid = {"e004": GRID["e004"], "shallow": GRID["shallow"]} if direction == "tsfm" else GRID
    if direction == "shallow":
        grid = {"stump": dict(BASE, num_leaves=2, max_depth=1, learning_rate=0.02, n_estimators=300,
                              min_child_samples=50, monotone_constraints=mono[h].tolist()),
                "depth2_mono": dict(GRID["shallow"], monotone_constraints=mono[h].tolist())}
    rng = np.random.default_rng(SEED + retune_id * 100 + h)
    res = {}
    for name, params in grid.items():
        off = int(rng.integers(0, RESHUFFLE_MAX + 1)) if direction == "robust" else 0
        rows, p, blk = cv_oos(params, X, y, i, h, off)
        res[name] = dict(score(p, y[rows], blk, h, conservative=direction == "robust"), params=params)
    if direction == "robust":
        best_mean = min(r["val_rel"] for r in res.values())
        name = min((n for n, r in res.items() if r["val_rel"] <= best_mean + ROBUST_TOL), key=lambda n: res[n]["worst"])
    else:
        name = min(res, key=lambda n: res[n]["val_rel"])
    r = res[name]
    deploy = r["t"] >= T_GATE and r["kappa"] > 0
    return {"config": name, "params": r["params"], "kappa": r["kappa"] if deploy else 0.0, "deploy": deploy,
            "val_rel": r["val_rel"], "t": r["t"], "worst": r["worst"]}


# ---------------------------------------------------------------- experiment
def run(commodity, direction):
    t0 = time.time()
    Xdf, price = load(commodity)
    pff = price.ffill()
    last_week = min(Xdf.dropna(how="all").index[-1], price.last_valid_index())
    weeks = Xdf.index[Xdf.index <= last_week]
    N = len(weeks)
    first = int(math.floor(TRAIN_FRAC * N))
    log.info("[%s] %s weeks=%s..%s N=%d first_origin=%s (idx %d) origins=%d", commodity, direction,
             weeks[0].date(), weeks[-1].date(), N, weeks[first].date(), first, N - first)

    base = Xdf.copy()
    base["tgt_logp"] = np.log(pff)
    base["tgt_r1"] = base["tgt_logp"].diff()
    lp = np.log(price)
    Y = np.column_stack([(lp.shift(-h) - np.log(pff)).values for h in range(1, H + 1)])
    F = features(direction, commodity, base, pff, first)
    C = chronos_anchor(pff) if direction == "tsfm" else None
    mono = monotone_signs(F.values, Y, first) if direction == "shallow" else None
    log.info("[%s] %s features=%d (%.0fs)", commodity, direction, F.shape[1], time.time() - t0)

    rows, trace, sel, anchor = [], [], {}, {}
    for i in range(first, N):
        t = weeks[i]
        k = (i - first) // RETUNE
        preds = np.zeros(H)
        for h in range(1, H + 1):
            tr = realised(Y, i, h)
            assert weeks[tr[-1]] + pd.Timedelta(weeks=h) <= t
            if (i - first) % RETUNE == 0:
                if direction == "tsfm":  # stage 1: shrink the anchor itself with the same gate
                    rows_a = tr[~np.isnan(C[tr, h - 1])][-CV_BLOCKS * BLOCK_WEEKS:]
                    sa = score(C[rows_a, h - 1], Y[rows_a, h - 1], np.zeros(len(rows_a)), h)
                    anchor[h] = sa["kappa"] if sa["t"] >= T_GATE else 0.0
                    Xh = np.column_stack([F.values, C[:, h - 1]])
                    sel[h] = select(direction, Xh, Y, i, h, mono, offset_y=anchor[h] * np.nan_to_num(C[:, h - 1]),
                                    retune_id=k)
                    sel[h]["anchor_kappa"], sel[h]["anchor_t"] = anchor[h], sa["t"]
                else:
                    sel[h] = select(direction, F.values, Y, i, h, mono, retune_id=k)
                trace.append({"retune": k, "origin": str(t.date()), "h": h,
                              **{a: b for a, b in sel[h].items() if a != "params"}})
            s = sel[h]
            p = 0.0
            if direction == "tsfm":
                p = anchor[h] * np.nan_to_num(C[i, h - 1])
                Xh = np.column_stack([F.values, C[:, h - 1]])
                if s["deploy"]:
                    yres = Y[tr, h - 1] - anchor[h] * np.nan_to_num(C[tr, h - 1])
                    p += s["kappa"] * lgb.LGBMRegressor(**s["params"]).fit(Xh[tr], yres).predict(Xh[i:i + 1])[0]
            elif s["deploy"]:
                p = s["kappa"] * lgb.LGBMRegressor(**s["params"]).fit(F.values[tr], Y[tr, h - 1]).predict(
                    F.values[i:i + 1])[0]
            preds[h - 1] = p
        for h in range(1, H + 1):
            tgt = t + pd.Timedelta(weeks=h)
            rows.append({"origin": t, "target_date": tgt, "horizon": h, "retune": k,
                         "y_true": price.get(tgt, np.nan), "y_pred": pff[t] * math.exp(preds[h - 1]),
                         "last_price": pff[t], "n_train": len(realised(Y, i, h))})
        if (i - first) % 25 == 0:
            log.info("[%s] %s origin %d/%d %s (%.0fs)", commodity, direction, i - first + 1, N - first,
                     t.date(), time.time() - t0)
    df = pd.DataFrame(rows)
    trace = transfer(df, pd.DataFrame(trace))
    trace.to_csv(LOG_DIR / f"{commodity}_{direction}_transfer.csv", index=False)
    df.drop(columns="retune").to_parquet(RES_DIR / f"predictions_{commodity}.parquet", index=False)
    assert df["origin"].min() == weeks[first] and df["origin"].nunique() == N - first
    log.info("[%s] saved %d predictions (%.0fs)", commodity, len(df), time.time() - t0)
    render_mp4(commodity, df, pff, price)


def transfer(df, trace):
    """Attach the realised test gain of each retune block to its validation gain."""
    d = df.dropna(subset=["y_true"]).assign(
        e_m=lambda x: np.log(x.y_pred / x.last_price) - np.log(x.y_true / x.last_price),
        e_0=lambda x: -np.log(x.y_true / x.last_price))
    g = d.groupby(["retune", "horizon"]).apply(lambda x: (x.e_m ** 2).mean() / (x.e_0 ** 2).mean() - 1)
    trace["test_rel"] = [g.get((r, h), np.nan) for r, h in zip(trace.retune, trace.h)]
    return trace


def transfer_summary(tr):
    """Transfer of internal validation to test, over (retune block, horizon) pairs."""
    dep = tr[tr.deploy]
    rho = spearmanr(tr.val_rel, tr.test_rel, nan_policy="omit").statistic if len(tr) > 2 else np.nan
    return {"blocks": len(tr), "deployed": len(dep), "deploy_win_rate": float((dep.test_rel < 0).mean()) if len(dep) else np.nan,
            "deploy_mean_test_rel": float(dep.test_rel.mean()) if len(dep) else np.nan, "spearman_val_test": float(rho)}


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
            EXP = f"E007_{direction}"
            for f in sorted((RES_DIR / direction).glob("predictions_*.parquet")):
                c = f.stem.removeprefix("predictions_")
                write_leaderboard(c, pd.read_parquet(f))
                tr = LOG_DIR / f"{c}_{direction}_transfer.csv"
                if tr.exists():
                    log.info("[%s] %s transfer %s", c, EXP, transfer_summary(pd.read_csv(tr)))
        return
    assert cmd in DIRECTIONS, f"command must be leaderboard or one of {DIRECTIONS}"
    EXP, RES_DIR = f"E007_{cmd}", RES_DIR / cmd
    setup_logging()
    names = sys.argv[2:] or list(TARGETS)
    log.info("%s start: %s", EXP, names)
    for c in names:
        run(c, cmd)
    log.info("%s done", EXP)


if __name__ == "__main__":
    main()
