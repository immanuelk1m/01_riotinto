"""E019: cross-sectional deployment of the immutable E015 candidate with online FDR control.

E018 showed that per-commodity anytime-valid gates refuse to deploy anything on
this panel: a 0.0002 capped-APE edge against 0.015 weekly noise is not certifiable
from 236-384 weeks. E019 does not change the evidence, it changes the **family**:
the same bounded weekly differences are treated as six simultaneous hypotheses, so
a rule can buy power by allowing a controlled fraction of deployed-but-worse
commodities instead of demanding per-commodity proof.

Directions (one Colab pane each):
  ebh   e-BH over the six per-commodity e-values every week at level ``ALPHA_FDR``
        (Wang & Ramdas e-BH; online generalisation of Fischer, Xu & Ramdas 2024).
  cap   calibration-after-adaptive-pick surrogate: pick commodities with a positive
        recent mean gain, then keep only picks above a cutoff calibrated on the
        historical selected set with an adaptive level
        ``alpha_{t+1} = alpha_t + gamma (alpha - err_t)``
        (CAP / adaptive conformal inference, Bao et al. 2024-2025).
  pool  hierarchical evidence: one pooled e-process over all commodities must pass
        ``1/alpha`` before any single commodity may deploy, and a commodity then
        needs its own e-process at the Bonferroni split ``m/alpha``.

Every statistic reads only labels whose ``target_date`` is already realised at the
origin, forecasts already issued are never revised, and the underlying candidate
forecast is untouched.
"""

import argparse
import hashlib
import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT = BASE / 'inputs'
DIRECTIONS = ('ebh', 'cap', 'pool')
TARGETS = {
    'gold': 'Com_LBMA_Gold_Price_PM_USD_V_0',
    'copper': 'Com_LME_Cu_Cash_Bid',
    'corn': 'Com_Barchart_Cash_National_Corn_Index',
    'coffee': 'Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag',
    'wti_crude_oil': 'Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel',
    'wheat': 'Com_Barchart_Cash_Soft_Red_Wheat_Index',
}
KEYS = ['origin', 'horizon']
ALPHA_FDR = 0.10      # target fraction of deployed-but-worse commodities
MIN_LABELS = 13       # E016/E018 parity: no decision before 13 mature weeks
CAP = 1.0             # bounded capped absolute percentage error
LAMBDA_MAX = 0.5      # keeps 1 + lambda * d > 0 for d in [-1, 1]
RECENT = 26           # E016's gate window, kept as the "recent" evidence window
ACI_GAMMA = 0.10      # step size of the adaptive level in `cap`


def validate(df):
    required = set(KEYS + ['target_date', 'y_true', 'last_price', 'y_raw', 'y_cand', 'n_train'])
    if required - set(df) or df.empty or df.duplicated(KEYS).any():
        raise ValueError('Missing fields, empty frame, or duplicate forecasts')
    if set(df.horizon) != set(range(1, 9)) or not df.groupby('origin').size().eq(8).all():
        raise ValueError('Incomplete horizons')
    dates = pd.DatetimeIndex(sorted(df.origin.unique()))
    if not dates.equals(pd.date_range(dates[0], dates[-1], freq='W-SUN')):
        raise ValueError('Incomplete weekly origins')
    if not (df.target_date == df.origin + pd.to_timedelta(df.horizon * 7, unit='D')).all():
        raise ValueError('Wrong target dates')
    for col in ['last_price', 'y_raw', 'y_cand']:
        if not np.isfinite(df[col]).all() or not df[col].gt(0).all():
            raise ValueError(f'Invalid {col}')
    observed = df.y_true.dropna()
    if not np.isfinite(observed).all() or not observed.gt(0).all():
        raise ValueError('Invalid observed targets')


def load(commodity, input_dir=INPUT):
    path = Path(input_dir) / f'predictions_{commodity}.parquet'
    if Path(input_dir) == INPUT and not path.exists():
        path = BASE.parent / 'E015' / 'results' / 'frozen' / path.name
    frame = pd.read_parquet(path)
    for col in ('origin', 'target_date'):
        frame[col] = pd.to_datetime(frame[col])
    validate(frame)
    return frame.sort_values(KEYS).reset_index(drop=True)


def ape(forecast, target_logret):
    return np.minimum(CAP, np.abs(np.expm1(forecast - target_logret)))


def hedge_bet(history):
    if not history:
        return 0.0
    d = np.asarray(history, dtype=float)
    mean, var = float(d.mean()), float(d.var())
    if var <= 0:
        return LAMBDA_MAX
    return float(np.clip(2 * mean / (var + mean ** 2), 0.0, LAMBDA_MAX))


def eprocess_value(history):
    """e-process for H: mean difference <= 0, with predictable bets."""
    log_e, seen = 0.0, []
    for d in history:
        log_e += float(np.log1p(hedge_bet(seen) * d))
        seen.append(d)
    return float(np.exp(log_e))


def weekly_panel(frame):
    """One bounded observation per target week: the eight horizons averaged."""
    f = frame.copy()
    target_logret = np.log(f.y_true / f.last_price)
    cand = np.log(f.y_cand / f.last_price)
    f['d'] = ape(np.zeros(len(f)), target_logret) - ape(cand, target_logret)
    mature = f.dropna(subset=['y_true'])
    return mature.groupby('target_date').d.mean()


def ebh_reject(e_values, alpha=ALPHA_FDR):
    """e-BH: reject the k largest e-values with k = max{i : E_(i) >= m/(alpha i)}."""
    m = len(e_values)
    order = np.argsort(-np.asarray(e_values))
    best = 0
    for i, idx in enumerate(order, start=1):
        if e_values[idx] >= m / (alpha * i):
            best = i
    return {int(idx) for idx in order[:best]}


def replay(panels, origins, direction, alpha=ALPHA_FDR):
    """Cross-sectional replay. Decisions are made at each origin from matured weeks only."""
    if direction not in DIRECTIONS:
        raise ValueError(f'Unknown direction {direction}')
    names = list(panels)
    trace, decisions = [], {c: {} for c in names}
    alpha_t = alpha
    err_history = []
    for t in origins:
        histories = {c: panels[c][panels[c].index <= t].to_numpy() for c in names}
        e_vals = [eprocess_value(list(histories[c])) for c in names]
        deploy = {c: False for c in names}
        detail = ''
        if all(len(histories[c]) >= MIN_LABELS for c in names):
            if direction == 'ebh':
                rejected = ebh_reject(e_vals, alpha)
                deploy = {c: (i in rejected) for i, c in enumerate(names)}
                detail = f'rejected={sorted(names[i] for i in rejected)}'
            elif direction == 'pool':
                pooled = float(np.mean([np.log(max(v, 1e-300)) for v in e_vals]))
                gate = pooled >= np.log(1.0 / alpha)
                if gate:
                    deploy = {c: e_vals[i] >= len(names) / alpha for i, c in enumerate(names)}
                detail = f'pooled_log_e={pooled:.3g} gate={gate}'
            else:
                picks = {}
                for c in names:
                    hist = histories[c]
                    recent = hist[-RECENT:]
                    picks[c] = bool(len(recent) and float(np.mean(recent)) > 0)
                selected = np.array([histories[c][-RECENT:].mean() for c in names if picks[c]])
                if len(selected):
                    cutoff = float(np.quantile(selected, 1.0 - alpha_t))
                    deploy = {c: picks[c] and float(histories[c][-RECENT:].mean()) >= cutoff
                              for c in names}
                detail = f'alpha_t={alpha_t:.3f} picks={sum(picks.values())}'
        for c in names:
            decisions[c][t] = deploy[c]
        deployed_now = [c for c in names if deploy[c]]
        err = float(np.mean([histories[c][-1] < 0 for c in deployed_now])) if deployed_now else 0.0
        err_history.append(err)
        if direction == 'cap':
            alpha_t = float(np.clip(alpha_t + ACI_GAMMA * (alpha - err), 0.0, 0.5))
        trace.append(dict(origin=t, deployed=len(deployed_now), deployed_names=','.join(deployed_now),
                          alpha_t=alpha_t, err=err, detail=detail,
                          max_e=float(max(e_vals)) if e_vals else np.nan))
    return decisions, pd.DataFrame(trace)


def run(commodity_frames, direction, input_dir=INPUT):
    panels = {c: weekly_panel(f) for c, f in commodity_frames.items()}
    origins = sorted(set().union(*[set(f.origin) for f in commodity_frames.values()]))
    decisions, trace = replay(panels, origins, direction)
    results, logs = BASE / 'results' / direction, BASE / 'logs'
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    for c, source in commodity_frames.items():
        out = source.copy()
        deploy_map = decisions[c]
        out['deploy'] = out.origin.map(lambda o: deploy_map.get(o, False)).astype(bool)
        out['y_pred'] = np.where(out.deploy, out.y_cand, out.last_price).astype(float)
        validate(out)
        out.to_parquet(results / f'predictions_{c}.parquet', index=False)
        source_hash = hashlib.sha256(pd.util.hash_pandas_object(source, index=True).values.tobytes()).hexdigest()
        (logs / f'{c}_{direction}_protocol.json').write_text(json.dumps(dict(
            source='E015 frozen immutable candidate archive', source_frame_sha256=source_hash,
            direction=direction, alpha_fdr=ALPHA_FDR, min_labels=MIN_LABELS, error_cap=CAP,
            recent=RECENT, aci_gamma=ACI_GAMMA, target_available='target_date <= origin',
            publication_lag_weeks=0, untouched_holdout=False), indent=2))
        diag = diagnostics(out)
        diag.to_csv(logs / f'{c}_{direction}_transfer.csv', index=False)
        summary = dict(direction=direction, commodity=c, n_blocks=len(diag),
                       deploy_fraction=float(diag.deploy_fraction.mean()) if len(diag) else None,
                       block_win_rate_positive=float((diag.deployed_gain > 0).mean()) if len(diag) else None,
                       unit='26-week block averaged over all eight horizons', untouched_holdout=False)
        (logs / f'{c}_{direction}_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    trace.to_csv(logs / f'cross_section_{direction}.csv', index=False)
    logging.info('%s complete commodities=%d', direction, len(commodity_frames))


def diagnostics(frame):
    origins = sorted(frame.origin.unique())
    rows = []
    for block, start in enumerate(range(0, len(origins), 26)):
        date = origins[start]
        test = frame[(frame.origin >= date) & (frame.origin < date + pd.Timedelta(weeks=26))].dropna(subset=['y_true'])
        if test.empty:
            continue
        gain = lambda x, col: float((np.abs(x.last_price / x.y_true - 1) - np.abs(x[col] / x.y_true - 1)).mean())
        rows.append(dict(block=block, origin=date, n_test=len(test),
                         candidate_gain=gain(test, 'y_cand'), deployed_gain=gain(test, 'y_pred'),
                         deploy_fraction=float(test.deploy.mean())))
    return pd.DataFrame(rows)


def leaderboard(direction, db=None, input_dir=INPUT):
    results = BASE / 'results' / direction
    if set(results.glob('predictions_*.parquet')) != {results / f'predictions_{c}.parquet' for c in TARGETS}:
        raise ValueError('Expected all six results')
    records = []
    for c in TARGETS:
        df = pd.read_parquet(results / f'predictions_{c}.parquet')
        source = load(c, input_dir)
        validate(df)
        if not df[KEYS + ['target_date']].equals(source[KEYS + ['target_date']]):
            raise ValueError('Source grid mismatch')
        for suffix, col in (('', 'y_pred'), ('_candidate', 'y_cand')):
            for h, group in [(str(h), g) for h, g in df.groupby('horizon')] + [('all', df)]:
                g = group.dropna(subset=['y_true'])
                err = g[col] - g.y_true
                records.append((f'E019_{direction}{suffix}', c, h,
                                float((err.abs() / g.y_true).mean() * 100), float(np.sqrt((err ** 2).mean())),
                                len(g), TARGETS[c], datetime.now().isoformat(timespec='seconds')))
    with closing(sqlite3.connect(db or BASE.parents[1] / 'leaderboard.sqlite', timeout=60)) as con, con:
        con.execute('CREATE TABLE IF NOT EXISTS leaderboard(experiment TEXT, commodity TEXT, horizon TEXT, mape REAL, rmse REAL, n INTEGER, target_series TEXT, created_at TEXT, PRIMARY KEY(experiment,commodity,horizon))')
        con.executemany('INSERT OR REPLACE INTO leaderboard VALUES (?,?,?,?,?,?,?,?)', records)


def diagnostics_fdr(direction):
    """Retrospective share of deployed-and-worse commodity-weeks (never used to decide)."""
    trace = pd.read_csv(BASE / 'logs' / f'cross_section_{direction}.csv')
    trace['deployed_names'] = trace['deployed_names'].map(lambda v: '' if pd.isna(v) else str(v))
    deployed, worse = 0, 0
    for _, row in trace.iterrows():
        if not row.deployed_names:
            continue
        for c in row.deployed_names.split(','):
            deployed += 1
            frame = pd.read_parquet(BASE / 'results' / direction / f'predictions_{c}.parquet')
            sub = frame[(frame.origin == pd.Timestamp(row.origin))].dropna(subset=['y_true'])
            if sub.empty:
                continue
            err = np.abs(sub.y_cand / sub.y_true - 1) - np.abs(sub.last_price / sub.y_true - 1)
            worse += int(err.mean() > 0)
    return dict(deployed=deployed, worse=worse,
                deployed_worse_rate=(worse / deployed) if deployed else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=[*DIRECTIONS, 'leaderboard', 'fdr'])
    parser.add_argument('args', nargs='*')
    parser.add_argument('--input-dir', type=Path, default=INPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.command == 'leaderboard':
        if len(args.args) != 1 or args.args[0] not in DIRECTIONS:
            parser.error('leaderboard requires one direction')
        leaderboard(args.args[0], input_dir=args.input_dir)
    elif args.command == 'fdr':
        if len(args.args) != 1 or args.args[0] not in DIRECTIONS:
            parser.error('fdr requires one direction')
        print(json.dumps(diagnostics_fdr(args.args[0]), indent=2))
    else:
        wanted = args.args or list(TARGETS)
        for c in wanted:
            if c not in TARGETS:
                parser.error(f'Unknown commodity {c}')
        frames = {c: load(c, args.input_dir) for c in wanted}
        run(frames, args.command, args.input_dir)


if __name__ == '__main__':
    main()
