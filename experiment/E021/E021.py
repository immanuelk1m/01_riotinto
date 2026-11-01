"""E021 panel expansion: does a larger cross-section certify an edge the six-commodity panel cannot?

E018 and E019 showed the binding constraint is the size of the evidence base: a
0.02-0.12 percentage-point effect against ~1.5 pp weekly noise cannot be certified
from 236-384 weeks, and the one rule that beat persistence failed its
deployed-worse contract. E021 attacks the constraint directly instead of changing
the rule again: every weekly-resolvable price series in the local `commodity`
database with enough history is put through the same protocol.

Protocol per series (identical to the repository's expanding walk-forward):
  * weekly log price from the daily series, ``W-SUN`` last observation
  * internal validation = three expanding folds inside the first 80%, scored as the
    capped-APE gain of the candidate over persistence
  * outer evaluation = expanding refits from the 80% point, step one week, h=1..8
  * candidate = multi-output ridge on four lagged weekly returns and a 52-week level
    z-score, clipped to the persistence range (cheap, CPU, no GPU, no sheet features)

Directions (one pane each):
  ridge      per-series anytime-valid e-process on the outer differences
  `ebh`      cross-series e-BH over the same e-values at ``ALPHA_FDR``
  pooled     one pooled e-process across all series

The headline diagnostic is the rank correlation between the internal validation
gain and the realised outer gain across series: with six commodities that
correlation was unmeasurable, and E021 asks whether a larger cross-section changes
that. Preparation reads Postgres and therefore runs on the Mac mini; replay runs on
CPU anywhere.
"""

import argparse
import json
import logging
import math
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
DIRECTIONS = ('ridge', 'ebh', 'pooled')
H = 8
LAGS = 4
TRAIN_FRAC = 0.8
MIN_WEEKS = 250
ALPHA = 0.05
ALPHA_FDR = 0.10
MIN_LABELS = 13
CAP = 1.0
LAMBDA_MAX = 0.5
RIDGE_LAMBDA = 1.0
SERIES_QUERY = """
select col_name, dt, value
from public.value
where col_name ilike any(array['%Price%','%Index%','%Spot%','%Cash%'])
  and value is not null
order by col_name, dt
"""
log = logging.getLogger('E021')


def prepare(directory=DATA):
    """Build weekly panels from Postgres. Runs on the Mac mini; writes one parquet per series."""
    import psycopg

    directory.mkdir(parents=True, exist_ok=True)
    frames = {}
    with psycopg.connect('dbname=commodity') as con:
        frame = pd.read_sql(SERIES_QUERY, con, parse_dates=['dt'])
    for name, group in frame.groupby('col_name'):
        series = group.set_index('dt').value.astype(float).resample('W-SUN').last().dropna()
        if len(series) < MIN_WEEKS:
            continue
        if (series <= 0).any():
            continue
        frames[name] = series
    index = []
    for name, series in frames.items():
        safe = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in name)[-80:]
        series.rename('price').reset_index().rename(columns={'dt': 'week'}).to_parquet(directory / f'{safe}.parquet', index=False)
        index.append(dict(name=name, file=f'{safe}.parquet', weeks=int(len(series)),
                          first=str(series.index[0].date()), last=str(series.index[-1].date())))
    (directory / 'index.json').write_text(json.dumps(index, indent=2))
    log.info('prepared %d series', len(index))
    return index


def panels(directory=DATA):
    index = json.loads((directory / 'index.json').read_text())
    out = {}
    for row in index:
        frame = pd.read_parquet(directory / row['file'])
        out[row['name']] = pd.Series(frame.price.to_numpy(), index=pd.DatetimeIndex(frame.week), name=row['name'])
    return out


def features(price):
    """Causal design matrix: four lagged weekly returns plus a 52-week level z-score."""
    logp = np.log(price.to_numpy(dtype=float))
    ret = np.diff(logp, prepend=np.nan)
    cols, names = [], []
    for lag in range(1, LAGS + 1):
        cols.append(np.roll(ret, lag))
        names.append(f'ret_lag{lag}')
    rolling = pd.Series(logp).rolling(52, min_periods=26)
    level = (logp - rolling.mean().to_numpy()) / np.maximum(rolling.std().to_numpy(), 1e-9)
    cols.append(level)
    names.append('level_z52')
    X = np.column_stack(cols)
    X[:LAGS + 1] = np.nan
    return X, names


def targets(logp):
    return np.column_stack([np.roll(logp, -h) - logp for h in range(1, H + 1)])


def capped_ape(forecast, target):
    return np.minimum(CAP, np.abs(np.expm1(forecast - target)))


def ridge_fit(X, Y, lam=RIDGE_LAMBDA):
    X1 = np.column_stack([np.ones(len(X)), X])
    gram = X1.T @ X1 + lam * np.eye(X1.shape[1])
    gram[0, 0] -= lam
    return np.linalg.solve(gram, X1.T @ Y)


def ridge_predict(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def walk_forward(price):
    """Internal expanding folds plus the outer expanding walk-forward for one series."""
    logp = np.log(price.to_numpy(dtype=float))
    X, names = features(price)
    Y = targets(logp)
    ok = ~np.isnan(X).any(1) & ~np.isnan(Y).any(1)
    valid = np.flatnonzero(ok)
    N = len(logp)
    first = int(math.floor(TRAIN_FRAC * N))
    # internal validation: three expanding folds over the in-sample region
    inner = valid[valid < first - H]
    folds = np.array_split(inner, 4)
    val_gains = []
    for k in range(1, 4):
        train, score = np.concatenate(folds[:k]), folds[k]
        beta = ridge_fit(X[train], Y[train])
        pred = ridge_predict(beta, X[score])
        val_gains.append(float(np.mean(capped_ape(np.zeros_like(pred), Y[score]) - capped_ape(pred, Y[score]))))
    val_gain = float(np.mean(val_gains))
    # outer: expanding refits, one step per origin
    rows = []
    for t in range(first, N):
        train = valid[valid < t]
        if len(train) < 100:
            continue
        beta = ridge_fit(X[train], Y[train])
        pred = ridge_predict(beta, X[t:t + 1])[0]
        for h in range(1, H + 1):
            if t + h >= N:
                continue
            rows.append(dict(origin=t, horizon=h, target_logret=float(Y[t, h - 1]),
                             cand=float(np.clip(pred[h - 1], -0.5, 0.5))))
    outer = pd.DataFrame(rows)
    if outer.empty:
        return val_gain, outer, names
    outer['d'] = capped_ape(np.zeros(len(outer)), outer.target_logret) - capped_ape(outer.cand, outer.target_logret)
    return val_gain, outer, names


def hedge_bet(history):
    if not history:
        return 0.0
    d = np.asarray(history, dtype=float)
    mean, var = float(d.mean()), float(d.var())
    if var <= 0:
        return LAMBDA_MAX
    return float(np.clip(2 * mean / (var + mean ** 2), 0.0, LAMBDA_MAX))


def eprocess_value(history):
    log_e, seen = 0.0, []
    for d in history:
        log_e += float(np.log1p(hedge_bet(seen) * d))
        seen.append(d)
    return float(np.exp(log_e))


def ebh_reject(e_values, alpha=ALPHA_FDR):
    m = len(e_values)
    order = np.argsort(-np.asarray(e_values))
    best = 0
    for i, idx in enumerate(order, start=1):
        if e_values[idx] >= m / (alpha * i):
            best = i
    return {int(idx) for idx in order[:best]}


def weekly_differences(outer):
    """One bounded observation per target week (horizons averaged), as in E018/E019."""
    return outer.groupby(outer.origin).d.mean()


def replay(results, direction):
    """Cross-series decision replay. results: {series: (val_gain, outer, names)}."""
    if direction not in DIRECTIONS:
        raise ValueError(direction)
    names = list(results)
    panels_by_series = {s: weekly_differences(results[s][1]) for s in names}
    origins = sorted(set().union(*[set(p.index) for p in panels_by_series.values()]))
    decisions = {s: {} for s in names}
    trace = []
    for t in origins:
        hist = {s: panels_by_series[s][panels_by_series[s].index <= t].to_numpy() for s in names}
        e_vals = [eprocess_value(list(hist[s])) for s in names]
        deploy = {s: False for s in names}
        if all(len(hist[s]) >= MIN_LABELS for s in names):
            if direction == 'ridge':
                deploy = {s: e_vals[i] >= 1.0 / ALPHA for i, s in enumerate(names)}
            elif direction == 'ebh':
                rejected = ebh_reject(e_vals, ALPHA_FDR)
                deploy = {s: (i in rejected) for i, s in enumerate(names)}
            else:
                pooled = float(np.mean([np.log(max(v, 1e-300)) for v in e_vals]))
                if pooled >= np.log(1.0 / ALPHA):
                    deploy = {s: e_vals[i] >= len(names) / ALPHA for i, s in enumerate(names)}
        for s in names:
            decisions[s][t] = deploy[s]
        trace.append(dict(origin=t, deployed=sum(deploy.values()),
                          deployed_names=','.join(s for s in names if deploy[s]),
                          max_e=float(max(e_vals)) if e_vals else float('nan')))
    return decisions, pd.DataFrame(trace)


def evaluate(results, decisions, prefix):
    """MAPE of deployed, candidate and persistence per series, pooled by horizon cells."""
    records = []
    for series, (val_gain, outer, _) in results.items():
        frames = outer.groupby('origin')
        for (origin, group), deployed in ((kv, decisions[series].get(kv[0], False)) for kv in frames):
            target = group.target_logret.to_numpy()
            cand = group.cand.to_numpy()
            choice = cand if deployed else np.zeros_like(cand)
            for h, tgt, c, p in zip(group.horizon, target, cand, choice):
                records.append(dict(experiment=f'{prefix}_deployed', series=series, horizon=int(h),
                                    mape=float(capped_ape(np.array([p]), np.array([tgt]))[0] * 100),
                                    max_ape_pct=float(np.abs(np.expm1(p - tgt)) * 100)))
                records.append(dict(experiment=f'{prefix}_candidate', series=series, horizon=int(h),
                                    mape=float(capped_ape(np.array([c]), np.array([tgt]))[0] * 100),
                                    max_ape_pct=float(np.abs(np.expm1(c - tgt)) * 100)))
                records.append(dict(experiment=f'{prefix}_persistence', series=series, horizon=int(h),
                                    mape=float(capped_ape(np.array([0.0]), np.array([tgt]))[0] * 100),
                                    max_ape_pct=float(np.abs(np.expm1(-tgt)) * 100)))
        records.append(dict(experiment=f'{prefix}_validation', series=series, horizon=0, mape=float(val_gain),
                            max_ape_pct=float('nan')))
    return pd.DataFrame(records)


def summarise(results, decisions, frame):
    valid = frame[frame.experiment.str.endswith('validation')].set_index('series').mape
    outer_gain = {}
    for series in results:
        d = results[series][1]
        outer_gain[series] = float(d.d.mean())
    joined = pd.DataFrame({'val_gain': valid, 'outer_gain': pd.Series(outer_gain)}).dropna()
    corr = None
    if len(joined) > 3 and joined.val_gain.nunique() > 1 and joined.outer_gain.nunique() > 1:
        corr = float(joined.rank().corr().iloc[0, 1])
    certified = int(sum(eprocess_value(list(weekly_differences(results[s][1]))) >= 1 / ALPHA
                        for s in results))
    return dict(n_series=len(results), val_outer_rank_correlation=corr,
                series_certified_at_alpha=certified,
                mean_val_gain=float(joined.val_gain.mean()) if len(joined) else None,
                mean_outer_gain=float(joined.outer_gain.mean()) if len(joined) else None,
                deploy_rate=float(np.mean([len([v for v in decisions[s].values() if v]) / max(len(decisions[s]), 1)
                                           for s in decisions])) if decisions else None)


def run(direction, directory=DATA, db=None, limit=None):
    store = panels(directory)
    if limit:
        store = dict(list(store.items())[:limit])
    results = {name: walk_forward(price) for name, price in store.items()}
    results = {k: v for k, v in results.items() if not v[1].empty}
    decisions, trace = replay(results, direction)
    frame = evaluate(results, decisions, f'E021_{direction}')
    summary = summarise(results, decisions, frame)
    summary['direction'] = direction
    out = BASE / 'logs'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{direction}_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    trace.to_csv(out / f'{direction}_cross_section.csv', index=False)
    (BASE / 'results').mkdir(parents=True, exist_ok=True)
    frame.to_parquet(BASE / 'results' / f'{direction}_scores.parquet', index=False)
    with closing(sqlite3.connect(db or BASE.parents[1] / 'leaderboard.sqlite', timeout=60)) as con, con:
        con.execute('CREATE TABLE IF NOT EXISTS leaderboard(experiment TEXT, commodity TEXT, horizon TEXT, mape REAL, rmse REAL, n INTEGER, target_series TEXT, created_at TEXT, max_ape_pct REAL, PRIMARY KEY(experiment,commodity,horizon))')
        if 'max_ape_pct' not in [row[1] for row in con.execute('PRAGMA table_info(leaderboard)')]:
            con.execute('ALTER TABLE leaderboard ADD COLUMN max_ape_pct REAL')
        scored = frame[~frame.experiment.str.endswith('_validation')]
        agg = scored.groupby(['experiment', 'series']).agg(
            mape=('mape', 'mean'), max_ape_pct=('max_ape_pct', 'max'))
        rows = [(exp, series, 'all', float(row.mape), float('nan'), 0, series, '', float(row.max_ape_pct))
                for (exp, series), row in agg.iterrows()]
        con.executemany('INSERT OR REPLACE INTO leaderboard(experiment, commodity, horizon, mape, rmse, n, target_series, created_at, max_ape_pct) VALUES (?,?,?,?,?,?,?,?,?)', rows)
    log.info('%s summary: %s', direction, json.dumps(summary))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', *DIRECTIONS])
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.command == 'prepare':
        index = prepare()
        print(json.dumps(index[:5], indent=2), f'... total {len(index)}')
    else:
        print(json.dumps(run(args.command, limit=args.limit), indent=2))


if __name__ == '__main__':
    main()
