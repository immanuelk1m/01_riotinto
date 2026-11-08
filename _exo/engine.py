"""Shared engine for the exogenous-variable rounds — v2, constant-free.

What changed against v1 and why
-------------------------------
Every literal that could change a fitted object is gone; the survivors live in
``_exo/params.py`` with a source and a rationale, and ``tools/params_lint.py``
fails the build if a new one appears.  Concretely, v1 decided results with:

* ``_pick(pattern, min_rows=120)`` — a regex plus a recency filter
  (``>= '2024-06-01'``) plus a ``(rows, last_dt)`` tie-break that *chose which
  physical series represented a commodity*.  That is a research degree of freedom
  larger than any hyper-parameter, so v2 refuses to guess: ``prepare`` accepts
  only an explicit ``col_name`` per key.
* ``ridge(lam=1.0)``, ``lasso(alpha=1e-3)``, ``pcr(k=3)``, ``gbrt`` (five
  settings), ``rf`` (three), ``mlp`` (three plus a random early-stopping split)
  and ``combo`` (an unvalidated equal-weight blend).  Those models are deleted;
  ``RETAINED_MODELS`` is the complete list of what may run.
* ``np.clip(pred, ±0.5)`` — a silent ±65 % return cap that only ever helped the
  candidate.  The protocol's capped APE already bounds the loss, so v2 removes it
  and reports the uncapped worst cell instead.
* ``ffill(limit=8)`` — an arbitrary carry horizon.  A forecaster holds the last
  published value until the next print, which is ``ffill()`` without a limit and
  therefore needs no constant.
* ``_zscore(window=52)`` — a one-year rolling window.  v2 uses an expanding
  z-score, which has no window length at all.
* ``_ols(lam=1e-6)`` — a numerical jitter with a magic size.  v2 uses
  ``np.linalg.lstsq``, which is parameter-free and handles rank deficiency by
  minimum norm instead of by a hand-picked ridge.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .leaderboard import write as write_leaderboard
    from .params import P, PARAMS_HASH, PARAMS_VERSION
else:  # executed as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from leaderboard import write as write_leaderboard
    from params import P, PARAMS_HASH, PARAMS_VERSION

#: the complete set of models that may run.  ``adl`` is a closed-form OLS on a
#: declared design matrix; ``persistence`` forecasts a zero log-return.  Neither
#: carries a free parameter, so neither can be tuned into a win.
RETAINED_MODELS = ("persistence", "adl")

#: keys a round must not override; the evaluation grid is registered, not per-round
FORBIDDEN_KEYS = ("design",)


# --------------------------------------------------------------------------- io

def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)[-90:]


def _weekly(frame: pd.DataFrame) -> pd.Series:
    frame = frame.assign(dt=pd.to_datetime(frame["dt"]))
    series = frame.set_index("dt")["value"].astype(float).sort_index()
    return series.resample("W-SUN").last().dropna()


def _col_name(entry) -> str | None:
    """A config entry must name its series; a pattern is documentation only."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("col_name") or None
    return None


def prepare(base: Path) -> dict:
    """Build weekly parquet panels from Postgres.  Runs on the Mac mini only."""
    import psycopg

    cfg = json.loads((base / "config.json").read_text())
    data = base / "data"
    (data / "targets").mkdir(parents=True, exist_ok=True)
    (data / "exog").mkdir(parents=True, exist_ok=True)
    manifest = {"targets": {}, "exog": {}, "params_hash": PARAMS_HASH}
    with psycopg.connect("dbname=commodity") as con, con.cursor() as cur:
        for kind, store in (("targets", "targets"), ("exog", "exog")):
            for key, entry in cfg[kind].items():
                col = _col_name(entry)
                if not col:
                    raise SystemExit(
                        f"[prepare] {kind}:{key} has no explicit col_name. "
                        f"Run tools/resolve_series.py once and paste the column into config.json; "
                        f"the v1 regex selector is gone on purpose."
                    )
                cur.execute(
                    "select name_en, ingest_lag_days from public.value_manifest where col_name = %s",
                    (col,),
                )
                meta = cur.fetchone()
                if meta is None:
                    print(f"[prepare] MISS {kind}:{key} col_name={col}", flush=True)
                    continue
                name_en, lag_days = meta
                # An unknown ingest lag is not zero: assuming same-week availability can only
                # manufacture an edge. v1 silently defaulted to 3 days; v2 refuses and asks for
                # the lag to be declared on the series itself.
                declared_lag_weeks = entry.get("lag_weeks") if isinstance(entry, dict) else None
                if lag_days is None and declared_lag_weeks is None:
                    raise SystemExit(
                        f"[prepare] {kind}:{key} ({col}) has no ingest_lag_days and no declared "
                        f"lag_weeks. Add \"lag_weeks\": <n> with a rationale to config.json — an "
                        f"unknown publication lag cannot be treated as instantaneous."
                    )
                lag_weeks = (
                    int(declared_lag_weeks)
                    if declared_lag_weeks is not None
                    else int(np.ceil(lag_days / 7.0))
                )
                cur.execute(
                    "select dt, value from public.value where col_name = %s order by dt", (col,)
                )
                frame = pd.DataFrame(cur.fetchall(), columns=["dt", "value"])
                series = _weekly(frame)
                if kind == "targets" and len(series) < P.MIN_WEEKS:
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
                    lag_weeks=lag_weeks,
                    lag_source="config" if declared_lag_weeks is not None else "manifest",
                    weeks=int(len(series)),
                    first=str(series.index[0].date()),
                    last=str(series.index[-1].date()),
                    file=out.name,
                )
                print(
                    f"[prepare] {kind}:{key} <- {col} weeks={len(series)} "
                    f"lag={lag_days}d ({lag_weeks}w, {manifest[store][key]['lag_source']})",
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
        ).iloc[-int(P.MAX_WEEKS):]
        targets[key] = series
    for key, row in manifest["exog"].items():
        frame = pd.read_parquet(data / "exog" / row["file"])
        exog[key] = (
            pd.Series(frame.value.to_numpy(), index=pd.DatetimeIndex(frame.week), name=key),
            int(row["lag_weeks"]),
        )
    return targets, exog


# ----------------------------------------------------------------------- design

def _expanding_z(values: np.ndarray) -> np.ndarray:
    """z-score on all history to date; no window length to choose."""
    series = pd.Series(np.asarray(values, dtype=float))
    mean = series.expanding(min_periods=P.Z_MIN_OBS).mean()
    std = series.expanding(min_periods=P.Z_MIN_OBS).std()
    return ((series - mean) / std.replace(0.0, np.nan)).to_numpy()


def features(price: pd.Series, exog: dict, keys: list[str], lags: int | None = None):
    """Causal design matrix on the target's weekly grid (availability lag applied)."""
    index = price.index
    logp = np.log(price.to_numpy(dtype=float))
    ret = np.diff(logp, prepend=np.nan)

    lags = int(P.LAGS if lags is None else lags)
    cols, names = [], []
    for lag in range(1, lags + 1):
        cols.append(np.roll(ret, lag))
        names.append(f"ret_lag{lag}")
    cols.append(_expanding_z(logp))
    names.append("level_z")

    for key in keys:
        if key not in exog:
            continue
        series, lag_weeks = exog[key]
        # the last published value is what a forecaster holds until the next print, so
        # carry it forward without a horizon and then delay it by the availability lag
        carried = series.reindex(index).ffill().shift(lag_weeks)
        values = carried.to_numpy(dtype=float)
        safe = _safe(key)
        cols.append(_expanding_z(values))
        names.append(f"{safe}_z")
        cols.append(np.diff(values, prepend=np.nan))
        names.append(f"{safe}_d1")

    X = np.column_stack(cols) if cols else np.zeros((len(index), 0))
    return X, names


def targets(logp: np.ndarray) -> np.ndarray:
    return np.column_stack([np.roll(logp, -h) - logp for h in range(1, int(P.H) + 1)])


# ----------------------------------------------------------------------- models

def _ols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Minimum-norm least squares: no penalty to tune and no jitter to pick."""
    design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(design, Y, rcond=None)
    return beta


def _apply(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def fit_predict(model: str, Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """Return (n, H) predictions for the next origin(s)."""
    if model == "persistence":
        return np.zeros((len(Xte), Ytr.shape[1]))
    if model == "adl":
        return _apply(_ols(Xtr, Ytr), Xte)
    raise ValueError(
        f"model {model!r} is not retained; allowed: {RETAINED_MODELS}. "
        f"A model with a free parameter cannot enter this pipeline (see _exo/params.py)."
    )


# ------------------------------------------------------------------- walk and audit

def walk_forward(price: pd.Series, exog: dict, keys: list[str], model: str,
                 lags: int | None = None):
    logp = np.log(price.to_numpy(dtype=float))
    X, names = features(price, exog, keys, lags=lags)
    Y = targets(logp)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1) & np.isfinite(logp)
    valid = np.flatnonzero(ok)
    # a training window plus one complete horizon set, nothing more
    if len(valid) < int(P.MIN_TRAIN) + int(P.H):
        return None
    n = len(logp)
    usable = set(valid.tolist())
    first = int(valid[min(len(valid) - 1, int(round(P.TRAIN_FRAC * len(valid))))])
    rows = []
    for t in range(first, n, int(P.REFIT_EVERY)):
        if t not in usable:
            continue
        train = valid[valid < t]
        if len(train) < int(P.MIN_TRAIN):
            continue
        try:
            pred = fit_predict(model, X[train], Y[train], X[t : t + 1])[0]
        except Exception as exc:  # noqa: BLE001 - one bad fit must not kill a direction
            print(f"[walk] fit failed t={t}: {exc}", flush=True)
            continue
        for h in range(1, int(P.H) + 1):
            if t + h >= n:
                continue
            rows.append(
                dict(origin=int(t), horizon=h, target_logret=float(Y[t, h - 1]),
                     fc=float(pred[h - 1]))
            )
    if not rows:
        return None
    return pd.DataFrame(rows)


def score(cells: pd.DataFrame) -> pd.DataFrame:
    cells = cells.copy()
    # persistence forecasts P_t, so its APE is |P_t / P_{t+h} - 1| = |expm1(-r)| (§5.1); cell-v2
    # used |expm1(r)| = |P_{t+h} / P_t - 1|, a different denominator from the candidate's
    cells["ape_pers"] = np.minimum(P.CAP, np.abs(np.expm1(-cells.target_logret)))
    cells["ape_cand"] = np.minimum(P.CAP, np.abs(np.expm1(cells.fc - cells.target_logret)))
    cells["d"] = cells.ape_pers - cells.ape_cand          # >0 means candidate better
    cells["mape_pers"] = cells.ape_pers * P.PCT
    cells["mape_cand"] = cells.ape_cand * P.PCT
    # uncapped twins, so the leaderboard can report the worst single cell
    cells["ape_pers_raw"] = np.abs(np.expm1(-cells.target_logret)) * P.PCT
    cells["ape_cand_raw"] = np.abs(np.expm1(cells.fc - cells.target_logret)) * P.PCT
    return cells


def block_bootstrap(dev: pd.DataFrame, block: int | None = None, draws: int | None = None,
                    seed: int | None = None):
    """Resample calendar blocks of origins and pool their cells.

    Blocking uses the calendar index (``origin // block``), not the count of
    distinct origins: on a sparse grid (refit every 4 weeks) counting origins
    reported 6 blocks where 18 exist and produced intervals narrow enough to turn
    a false result into a PASS.
    """
    block = int(P.BLOCK if block is None else block)
    draws = int(P.BOOT if draws is None else draws)
    seed = int(P.SEED if seed is None else seed)
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
        lo=float(np.quantile(means, P.ALPHA / 2)),
        hi=float(np.quantile(means, 1 - P.ALPHA / 2)),
        blocks=n_blocks,
    )


def audit(cells: pd.DataFrame, sealed_cut: int, block: int | None = None) -> dict:
    dev = cells[cells.origin < sealed_cut]
    sealed = cells[cells.origin >= sealed_cut]
    boot = block_bootstrap(dev, block=block)
    block_size = int(P.BLOCK if block is None else block)
    blocks = dev.assign(block=(dev.origin // block_size)).groupby("block").d.mean()
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
        f"MAPE {audit_row['sealed']['mape_pers']:.3f} vs {audit_row['sealed']['mape_cand']:.3f}\n"
        f"- params: v{PARAMS_VERSION} `{PARAMS_HASH}`\n\n"
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
    lags = None
    for part in parts[1:]:
        if part.startswith("--limit"):
            limit = int(part.split("=", 1)[1]) if "=" in part else None
        if part.startswith("--lags"):
            lags = int(part.split("=", 1)[1])
    swing = lags is not None and lags != int(P.LAGS)
    if lags is not None and lags not in P.SENSITIVITY_LAGS:
        raise SystemExit(
            f"--lags={lags} is not on the registered swing grid {P.SENSITIVITY_LAGS}. "
            f"A lag count outside the grid is an unregistered degree of freedom."
        )

    for key in FORBIDDEN_KEYS:
        if key in cfg:
            raise SystemExit(
                f"config.json carries a {key!r} block. The evaluation grid is registered in "
                f"_exo/params.py, not per round; register it and bump PARAMS_VERSION instead."
            )
    direction = next((d for d in cfg["directions"] if d["name"] == name), None)
    if direction is None:
        raise SystemExit(f"unknown direction {name}; known: {[d['name'] for d in cfg['directions']]}")
    if direction["model"] not in RETAINED_MODELS:
        raise SystemExit(
            f"direction {name} asks for model {direction['model']!r}, which is not retained "
            f"({RETAINED_MODELS})."
        )

    targets_all, exog_all = load(base)
    # Silently dropping a declared target or regressor is a research degree of freedom:
    # v1 skipped a missing exog with a log line, which is how E023's "basis" regressor
    # never loaded and how a panel shrank without anyone noticing. Refuse instead.
    missing_targets = [k for k in direction["targets"] if k not in targets_all]
    missing_exog = [k for k in direction["exog"] if k not in exog_all]
    if missing_targets or missing_exog:
        raise SystemExit(
            f"direction {name} declares series that the panel does not contain: "
            f"targets={missing_targets} exog={missing_exog}. Run prepare again, or change the "
            f"registration — a direction may not run on a different information set than the one "
            f"it declares."
        )
    keys = list(direction["targets"])
    if limit:
        keys = keys[:limit]
    if not keys:
        raise SystemExit("no target panels available")

    exog_keys = list(direction["exog"])

    frames = []
    for key in keys:
        cells = walk_forward(targets_all[key], exog_all, exog_keys, direction["model"], lags=lags)
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
    sealed_cut = int(cells.origin.max()) - int(P.SEALED_ORIGINS)
    row = audit(cells, sealed_cut)
    row.update(direction=name, model=direction["model"], exog=exog_keys,
               targets=keys, experiment=cfg["experiment"], lags=int(P.LAGS if lags is None else lags),
               sensitivity=swing,
               params_version=PARAMS_VERSION, params_hash=PARAMS_HASH)

    out = base / "results" / (f"sensitivity/lags_{lags}" if swing else Path())
    out.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(out / f"{name}_cells.parquet", index=False)
    (out / f"{name}_audit.json").write_text(json.dumps(row, indent=2, default=float))
    (out / f"{name}_audit.md").write_text(markdown(name, row))

    dev = cells[cells.origin < sealed_cut]
    ledger_db = Path(os.environ.get("EXO_LEADERBOARD", base.parents[1] / "leaderboard.sqlite"))
    if swing:
        print(f"[ledger] skipped (sensitivity swing LAGS={lags})", flush=True)
    elif limit and "EXO_LEADERBOARD" not in os.environ:
        print("[ledger] skipped (--limit: partial panel)", flush=True)
    else:
        col_names = {key: meta["col_name"] for key, meta in
                     json.loads((base / "data" / "manifest.json").read_text())["targets"].items()}
        # the baseline is scoped to the direction, not to the experiment: two directions in one
        # experiment can score different panels, and a shared `_persistence` label then mixes a
        # 2-series and a 3-series panel under one name (§5.3)
        written = write_leaderboard(ledger_db, f"{cfg['experiment']}_{name}", dev, col_names,
                                    persistence=f"{cfg['experiment']}_{name}_persistence",
                                    direction=name)
        print(f"[ledger] {written} rows -> {ledger_db}", flush=True)
    print(json.dumps({k: v for k, v in row.items() if k != "per_series"}, indent=2, default=float), flush=True)
    return 0
