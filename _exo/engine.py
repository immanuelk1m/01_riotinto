"""Shared engine for the E022-E026 exogenous-variable round.

Runs on the Colab VM (fit/predict) and on the Mac mini (``prepare`` only, which
needs the local ``commodity`` Postgres database).  Every direction is a
pre-registered combination of (target panel, exogenous set, model family); the
engine only *audits* candidates against persistence (Gate A), it does not
deploy anything.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .leaderboard import write as write_leaderboard
else:  # executed as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from leaderboard import write as write_leaderboard

CAP = 1.0
H = 8
LAGS = 4
TRAIN_FRAC = 0.70
MIN_TRAIN = 100
MIN_WEEKS = 200
BLOCK = 26
BOOT = 1000
REFIT_EVERY = 4
MAX_WEEKS = 1040
SEALED_ORIGINS = 52
ALPHA = 0.05
BIG = 1e9

#: Default evaluation grid.  A round may override it through a "design" object in its
#: config.json; the keys are the only ones changed between rounds, so a re-run without
#: a "design" block reproduces the E022-E026 settings exactly.
DESIGN = dict(train_frac=TRAIN_FRAC, refit_every=REFIT_EVERY, block=BLOCK,
              sealed_origins=SEALED_ORIGINS)


# --------------------------------------------------------------------------- io

def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)[-90:]


def _weekly(frame: pd.DataFrame) -> pd.Series:
    frame = frame.assign(dt=pd.to_datetime(frame["dt"]))
    series = frame.set_index("dt")["value"].astype(float).sort_index()
    return series.resample("W-SUN").last().dropna()


def _pick(cur, pattern: str, min_rows: int = 120):
    """Pick the longest, most recent series matching a regex.

    Two steps on purpose: the regex is evaluated on the (small) manifest only, so a
    pattern that matches nothing cannot trigger a nested-loop scan of ``value``.
    """
    cur.execute(
        """
        select col_name, name_en, ingest_lag_days
        from public.value_manifest
        where lower(coalesce(name_en, col_name)) ~ %s
        limit 200
        """,
        (pattern,),
    )
    candidates = cur.fetchall()
    if not candidates:
        return None
    names = [row[0] for row in candidates]
    cur.execute(
        """
        select col_name, count(*) n, min(dt), max(dt)
        from public.value
        where col_name = any(%s)
        group by 1
        having count(*) >= %s
        """,
        (names, min_rows),
    )
    stats = {row[0]: row[1:] for row in cur.fetchall()}
    rows = [
        (col, name_en, lag, *stats[col])
        for col, name_en, lag in candidates
        if col in stats
    ]
    if not rows:
        return None
    rows = [r for r in rows if str(r[5]) >= '2024-06-01'] or rows
    rows.sort(key=lambda r: (r[3], r[5]), reverse=True)
    return rows[0]


def prepare(base: Path) -> dict:
    """Build weekly parquet panels from Postgres.  Runs on the Mac mini only."""
    import psycopg

    cfg = json.loads((base / "config.json").read_text())
    data = base / "data"
    (data / "targets").mkdir(parents=True, exist_ok=True)
    (data / "exog").mkdir(parents=True, exist_ok=True)
    manifest = {"targets": {}, "exog": {}}
    with psycopg.connect("dbname=commodity") as con, con.cursor() as cur:
        for kind, store in (("targets", "targets"), ("exog", "exog")):
            for key, pattern in cfg[kind].items():
                row = _pick(cur, pattern)
                if row is None:
                    print(f"[prepare] MISS {kind}:{key} pattern={pattern}", flush=True)
                    continue
                col, name_en, lag_days, n, first, last = row
                cur.execute(
                    "select dt, value from public.value where col_name = %s order by dt",
                    (col,),
                )
                frame = pd.DataFrame(cur.fetchall(), columns=["dt", "value"])
                series = _weekly(frame)
                if len(series) < MIN_WEEKS and kind == "targets":
                    print(f"[prepare] SHORT {kind}:{key} weeks={len(series)}", flush=True)
                    continue
                if kind == "targets" and (series <= 0).any():
                    print(f"[prepare] NONPOS {kind}:{key}", flush=True)
                    continue
                out = data / store / f"{_safe(key)}.parquet"
                series.rename("value").reset_index().rename(columns={"dt": "week"}).to_parquet(
                    out, index=False
                )
                manifest[store][key] = dict(
                    col_name=col,
                    name_en=name_en,
                    ingest_lag_days=lag_days,
                    lag_weeks=int(math.ceil((lag_days if lag_days is not None else 3) / 7.0)),
                    weeks=int(len(series)),
                    first=str(series.index[0].date()),
                    last=str(series.index[-1].date()),
                    file=out.name,
                )
                print(
                    f"[prepare] {kind}:{key} <- {col} weeks={len(series)} "
                    f"lag={lag_days}d ({manifest[store][key]['lag_weeks']}w)",
                    flush=True,
                )
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load(base: Path):
    data = base / "data"
    manifest = json.loads((data / "manifest.json").read_text())
    targets, exog = {}, {}
    for key, row in manifest["targets"].items():
        frame = pd.read_parquet(data / "targets" / row["file"])
        series = pd.Series(
            frame.value.to_numpy(), index=pd.DatetimeIndex(frame.week), name=key
        ).iloc[-MAX_WEEKS:]
        targets[key] = series
    for key, row in manifest["exog"].items():
        frame = pd.read_parquet(data / "exog" / row["file"])
        exog[key] = (
            pd.Series(frame.value.to_numpy(), index=pd.DatetimeIndex(frame.week), name=key),
            int(row["lag_weeks"]),
        )
    return targets, exog


# ----------------------------------------------------------------------- design

def _zscore(values: np.ndarray, window: int = 52) -> np.ndarray:
    series = pd.Series(values)
    mean = series.rolling(window, min_periods=max(12, window // 2)).mean()
    std = series.rolling(window, min_periods=max(12, window // 2)).std()
    return ((series - mean) / std.replace(0.0, np.nan)).to_numpy()


def features(price: pd.Series, exog: dict, keys: list[str]):
    """Causal design matrix on the target's weekly grid (availability lag applied)."""
    index = price.index
    logp = np.log(price.to_numpy(dtype=float))
    ret = np.diff(logp, prepend=np.nan)

    cols, names = [], []
    for lag in range(1, LAGS + 1):
        cols.append(np.roll(ret, lag))
        names.append(f"ret_lag{lag}")
    cols.append(_zscore(logp))
    names.append("level_z52")

    for key in keys:
        if key not in exog:
            continue
        series, lag_weeks = exog[key]
        aligned = series.reindex(index).ffill(limit=8).shift(lag_weeks)
        values = aligned.to_numpy(dtype=float)
        safe = _safe(key)
        cols.append(_zscore(values))
        names.append(f"{safe}_z52")
        cols.append(np.diff(values, prepend=np.nan))
        names.append(f"{safe}_d1")
        cols.append(pd.Series(values).diff(4).to_numpy())
        names.append(f"{safe}_d4")

    X = np.column_stack(cols) if cols else np.zeros((len(index), 0))
    return X, names


def targets(logp: np.ndarray) -> np.ndarray:
    return np.column_stack([np.roll(logp, -h) - logp for h in range(1, H + 1)])


# ----------------------------------------------------------------------- models

def _ols(X: np.ndarray, Y: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    gram = X1.T @ X1 + lam * np.eye(X1.shape[1])
    gram[0, 0] -= lam
    return np.linalg.solve(gram, X1.T @ Y)


def _apply(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def _pcr3(X: np.ndarray, Y: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    comp = Z @ Vt[:k].T
    beta = _ols(comp, Y)
    return comp, beta, (mu, sd, Vt[:k], beta)


def _pcr3_apply(state, X: np.ndarray) -> np.ndarray:
    mu, sd, V, beta = state
    Z = (X - mu) / sd
    return _apply(beta, Z @ V.T)


def _model_fit_predict(name: str, Xtr, Ytr, Xte):
    """Return (1, H) predictions for the next origin."""
    if name in ("adl", "ols", "arimax", "varx", "midas"):
        return _apply(_ols(Xtr, Ytr), Xte)
    if name == "ridge":
        return _apply(_ols(Xtr, Ytr, lam=1.0), Xte)
    if name == "pcr":
        _, _, state = _pcr3(Xtr, Ytr)
        return _pcr3_apply(state, Xte)
    if name == "lasso":
        from sklearn.linear_model import Lasso

        preds = []
        for h in range(Ytr.shape[1]):
            model = Lasso(alpha=1e-3, max_iter=3000, fit_intercept=True)
            model.fit(Xtr, Ytr[:, h])
            preds.append(model.predict(Xte))
        return np.column_stack(preds)
    if name == "gbrt":
        from sklearn.ensemble import HistGradientBoostingRegressor

        preds = []
        for h in range(Ytr.shape[1]):
            model = HistGradientBoostingRegressor(
                max_iter=100, learning_rate=0.06, max_depth=3,
                min_samples_leaf=20, l2_regularization=1.0, random_state=0,
            )
            model.fit(Xtr, Ytr[:, h])
            preds.append(model.predict(Xte))
        return np.column_stack(preds)
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=80, max_depth=5, min_samples_leaf=10,
            n_jobs=2, random_state=0,
        )
        model.fit(Xtr, Ytr)
        return model.predict(Xte)
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor

        model = MLPRegressor(
            hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400,
            early_stopping=True, random_state=0,
        )
        model.fit(Xtr, Ytr)
        return model.predict(Xte)
    if name == "combo":
        parts = [
            _model_fit_predict("adl", Xtr, Ytr, Xte),
            _model_fit_predict("gbrt", Xtr, Ytr, Xte),
            _model_fit_predict("rf", Xtr, Ytr, Xte),
        ]
        return np.mean(parts, axis=0)
    raise ValueError(f"unknown model {name}")


# ------------------------------------------------------------------- walk and audit

def walk_forward(price: pd.Series, exog: dict, keys: list[str], model: str,
                 train_frac: float = TRAIN_FRAC, refit_every: int = REFIT_EVERY):
    logp = np.log(price.to_numpy(dtype=float))
    X, names = features(price, exog, keys)
    Y = targets(logp)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1) & np.isfinite(logp)
    valid = np.flatnonzero(ok)
    if len(valid) < MIN_TRAIN + 40:
        return None
    n = len(logp)
    first = int(math.floor(train_frac * valid.max()))
    rows = []
    for t in range(first, n, int(refit_every)):
        if t not in set(valid.tolist()):
            continue
        train = valid[valid < t]
        if len(train) < MIN_TRAIN:
            continue
        try:
            pred = _model_fit_predict(model, X[train], Y[train], X[t : t + 1])[0]
        except Exception as exc:  # noqa: BLE001 - one bad fit must not kill a direction
            print(f"[walk] fit failed t={t}: {exc}", flush=True)
            continue
        for h in range(1, H + 1):
            if t + h >= n:
                continue
            rows.append(
                dict(origin=int(t), horizon=h, target_logret=float(Y[t, h - 1]),
                     fc=float(np.clip(pred[h - 1], -0.5, 0.5)))
            )
    if not rows:
        return None
    return pd.DataFrame(rows)


def score(cells: pd.DataFrame) -> pd.DataFrame:
    cells = cells.copy()
    cells["ape_pers"] = np.minimum(CAP, np.abs(np.expm1(cells.target_logret)))
    cells["ape_cand"] = np.minimum(CAP, np.abs(np.expm1(cells.fc - cells.target_logret)))
    cells["d"] = cells.ape_pers - cells.ape_cand          # >0 means candidate better
    cells["mape_pers"] = cells.ape_pers * 100.0
    cells["mape_cand"] = cells.ape_cand * 100.0
    # uncapped twins, so the leaderboard can report the worst single cell instead of
    # a maximum clipped at 100 % (see _exo/leaderboard.py)
    cells["ape_pers_raw"] = np.abs(np.expm1(cells.target_logret)) * 100.0
    cells["ape_cand_raw"] = np.abs(np.expm1(cells.fc - cells.target_logret)) * 100.0
    return cells


def block_bootstrap(dev: pd.DataFrame, block: int = BLOCK, draws: int = BOOT, seed: int = 0):
    """Resample 26-week *calendar* blocks of origins and pool their cells.

    The earlier version counted distinct origins and sliced them as if they were
    consecutive weeks.  On a sparse origin grid (refit every 4 weeks, series with
    different start dates) 171 distinct origins span 602 weeks, so that version
    reported 6 blocks where 18 exist and returned a CI that was too narrow - enough
    to turn gbrt_agri into a false PASS in its ``*_audit.md``.  Blocking has to use the
    calendar index (``origin // block``), which is also what
    ``experiment/LEADERBOARD_exogenous_2026.md`` recomputes.
    """
    frame = dev.assign(_blk=(dev.origin // block).astype(int))
    groups = [g.d.to_numpy(dtype=float) for _, g in frame.groupby("_blk")]
    n_blocks = len(groups)
    if n_blocks < 2:
        values = dev.d.to_numpy(dtype=float)
        return dict(mean=float(values.mean()), lo=float("nan"), hi=float("nan"), blocks=n_blocks)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for b in range(draws):
        pick = rng.integers(0, n_blocks, size=n_blocks)
        means[b] = np.concatenate([groups[i] for i in pick]).mean()
    return dict(
        mean=float(dev.d.mean()),
        lo=float(np.quantile(means, ALPHA / 2)),
        hi=float(np.quantile(means, 1 - ALPHA / 2)),
        blocks=n_blocks,
    )


def audit(cells: pd.DataFrame, sealed_cut: int, block: int = BLOCK) -> dict:
    dev = cells[cells.origin < sealed_cut]
    sealed = cells[cells.origin >= sealed_cut]
    boot = block_bootstrap(dev, block=block)
    blocks = dev.assign(block=(dev.origin // block)).groupby("block").d.mean()
    per_series = dev.groupby("series").agg(
        n=("d", "size"), mean_d=("d", "mean"),
        mape_pers=("mape_pers", "mean"), mape_cand=("mape_cand", "mean"),
    )
    out = dict(
        n_cells=int(len(dev)),
        n_origins=int(dev.origin.nunique()),
        n_series=int(dev.series.nunique()),
        mean_d=float(dev.d.mean()),
        ci_low=boot["lo"],
        ci_high=boot["hi"],
        blocks=boot["blocks"],
        block_win_rate=float((blocks > 0).mean()) if len(blocks) else float("nan"),
        mape_pers=float(dev.mape_pers.mean()),
        mape_cand=float(dev.mape_cand.mean()),
        frac_series_positive=float((per_series.mean_d > 0).mean()),
        worst_series_delta=float(per_series.mean_d.min()),
        eligible=bool(dev.d.mean() > 0 and boot["lo"] > 0),
        sealed=dict(
            n_cells=int(len(sealed)),
            mean_d=float(sealed.d.mean()) if len(sealed) else float("nan"),
            mape_pers=float(sealed.mape_pers.mean()) if len(sealed) else float("nan"),
            mape_cand=float(sealed.mape_cand.mean()) if len(sealed) else float("nan"),
        ),
        per_series=per_series.round(6).to_dict(orient="index"),
    )
    return out


def markdown(direction: str, audit_row: dict) -> str:
    rows = audit_row["per_series"]
    head = (
        f"# {direction}\n\n"
        f"- cells: {audit_row['n_cells']} (origins {audit_row['n_origins']}, series {audit_row['n_series']})\n"
        f"- mean d (capped APE, persistence - candidate): **{audit_row['mean_d']:.6f}** "
        f"[{audit_row['ci_low']:.6f}, {audit_row['ci_high']:.6f}] ({audit_row['blocks']} blocks)\n"
        f"- MAPE persistence {audit_row['mape_pers']:.3f} vs candidate {audit_row['mape_cand']:.3f}\n"
        f"- block win rate {audit_row['block_win_rate']:.3f}; series positive "
        f"{audit_row['frac_series_positive']:.3f}; worst series {audit_row['worst_series_delta']:.6f}\n"
        f"- **Gate A: {'PASS' if audit_row['eligible'] else 'FAIL'}**\n"
        f"- sealed block (not used for eligibility): mean d {audit_row['sealed']['mean_d']:.6f}, "
        f"MAPE {audit_row['sealed']['mape_pers']:.3f} vs {audit_row['sealed']['mape_cand']:.3f}\n\n"
        "| series | n | mean d | MAPE pers | MAPE cand |\n|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {series} | {row['n']} | {row['mean_d']:+.6f} | {row['mape_pers']:.3f} | {row['mape_cand']:.3f} |\n"
        for series, row in sorted(rows.items(), key=lambda kv: -kv[1]["mean_d"])
    )
    return head + body


# ------------------------------------------------------------------------ runner

def run(base: Path, spec: str) -> int:
    cfg = json.loads((base / "config.json").read_text())
    parts = spec.split()
    name = parts[0]
    limit = None
    for part in parts[1:]:
        if part.startswith("--limit"):
            limit = int(part.split("=", 1)[1]) if "=" in part else None
    direction = next((d for d in cfg["directions"] if d["name"] == name), None)
    if direction is None:
        raise SystemExit(f"unknown direction {name}; known: {[d['name'] for d in cfg['directions']]}")

    # A round may pre-register its evaluation grid (E027 widens the walk-forward to buy
    # blocks).  Absent a "design" block the round keeps the E022-E026 settings exactly.
    design = {**DESIGN, **cfg.get("design", {})}
    block = int(design["block"])
    sealed_origins = int(design["sealed_origins"])

    targets_all, exog_all = load(base)
    keys = [k for k in direction["targets"] if k in targets_all]
    if limit:
        keys = keys[:limit]
    if not keys:
        raise SystemExit("no target panels available")

    exog_keys = [k for k in direction["exog"] if k in exog_all]
    missing = [k for k in direction["exog"] if k not in exog_all]
    if missing:
        print(f"[run] missing exog (skipped): {missing}", flush=True)

    frames = []
    for key in keys:
        cells = walk_forward(targets_all[key], exog_all, exog_keys, direction["model"],
                             train_frac=float(design["train_frac"]),
                             refit_every=int(design["refit_every"]))
        if cells is None or cells.empty:
            print(f"[run] no cells for {key}", flush=True)
            continue
        cells["series"] = key
        frames.append(cells)
        print(f"[run] {key}: {len(cells)} cells", flush=True)
    if not frames:
        raise SystemExit("no cells produced")

    cells = score(pd.concat(frames, ignore_index=True))
    cells["direction"] = name
    sealed_cut = int(cells.origin.max()) - sealed_origins
    row = audit(cells, sealed_cut, block=block)
    row.update(direction=name, model=direction["model"], exog=exog_keys,
               targets=keys, experiment=cfg["experiment"], design=design)

    out = base / "results"
    out.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(out / f"{name}_cells.parquet", index=False)
    (out / f"{name}_audit.json").write_text(json.dumps(row, indent=2, default=float))
    (out / f"{name}_audit.md").write_text(markdown(name, row))

    dev = cells[cells.origin < sealed_cut]
    ledger_db = Path(os.environ.get("EXO_LEADERBOARD", base.parents[1] / "leaderboard.sqlite"))
    if limit and "EXO_LEADERBOARD" not in os.environ:
        print("[ledger] skipped (--limit: partial panel)", flush=True)
    else:
        col_names = {key: meta["col_name"] for key, meta in
                     json.loads((base / "data" / "manifest.json").read_text())["targets"].items()}
        written = write_leaderboard(ledger_db, f"{cfg['experiment']}_{name}", dev, col_names,
                                    persistence=f"{cfg['experiment']}_persistence")
        print(f"[ledger] {written} rows -> {ledger_db}", flush=True)
    print(json.dumps({k: v for k, v in row.items() if k != "per_series"}, indent=2, default=float), flush=True)
    return 0
