"""Delayed affine calibration of immutable E014 ERM forecasts.

Research-inspired ablation, not a reproduction. Targets are available at
target_date; E014's source-vintage and forced-lag limitations remain.
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
DIRECTIONS = ('static52', 'recent26', 'drift26')
TARGETS = {
    'gold': 'Com_LBMA_Gold_Price_PM_USD_V_0',
    'copper': 'Com_LME_Cu_Cash_Bid',
    'corn': 'Com_Barchart_Cash_National_Corn_Index',
    'coffee': 'Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag',
    'wti_crude_oil': 'Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel',
    'wheat': 'Com_Barchart_Cash_Soft_Red_Wheat_Index',
}
KEYS = ['origin', 'horizon']
MIN_FIT, MIN_GATE, RIDGE = 13, 13, 0.01
DRIFT_RATIO, CORRECTION_CLIP, BIAS_CLIP = 1.5, 0.10, 0.05


def validate(df):
    required = set(KEYS + ['target_date', 'y_true', 'last_price', 'y_raw', 'n_train'])
    if required - set(df) or df.empty or df.duplicated(KEYS).any():
        raise ValueError('Missing fields, empty frame, or duplicate forecasts')
    if set(df.horizon) != set(range(1, 9)) or not df.groupby('origin').size().eq(8).all():
        raise ValueError('Incomplete horizons')
    dates = pd.DatetimeIndex(sorted(df.origin.unique()))
    if not dates.equals(pd.date_range(dates[0], dates[-1], freq='W-SUN')):
        raise ValueError('Incomplete weekly origins')
    if not (df.target_date == df.origin + pd.to_timedelta(df.horizon * 7, unit='D')).all():
        raise ValueError('Wrong target dates')
    for col in ['last_price', 'y_raw', *[c for c in ('y_pred', 'y_cand') if c in df]]:
        if not np.isfinite(df[col]).all() or not df[col].gt(0).all():
            raise ValueError(f'Invalid {col}')
    observed = df.y_true.dropna()
    if not np.isfinite(observed).all() or not observed.gt(0).all():
        raise ValueError('Invalid observed targets')


def load(commodity, input_dir=INPUT):
    path = Path(input_dir) / f'predictions_{commodity}.parquet'
    if Path(input_dir) == INPUT and not path.exists():
        path = BASE.parent / 'E014' / 'results' / 'erm' / path.name
    frame = pd.read_parquet(path)
    for col in ('origin', 'target_date'):
        frame[col] = pd.to_datetime(frame[col])
    validate(frame)
    return frame.sort_values(KEYS).reset_index(drop=True)


def affine(x, y):
    """Ridge toward identity, with a bounded intercept and slope."""
    design = np.column_stack((x, np.ones(len(x))))
    penalty = np.diag([RIDGE, RIDGE])
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y + penalty @ np.array([1., 0.]))
    return float(np.clip(beta[0], 0., 1.)), float(np.clip(beta[1], -BIAS_CLIP, BIAS_CLIP))


def replay(frame, direction):
    if direction not in DIRECTIONS:
        raise ValueError(direction)
    validate(frame)
    outputs, trace = [], []
    for h, group in frame.groupby('horizon', sort=True):
        g = group.sort_values('origin').reset_index(drop=True)
        raw = np.log(g.y_raw.to_numpy() / g.last_price.to_numpy())
        issued = np.zeros(len(g))
        mature, cursor = [], 0
        for i, row in g.iterrows():
            while cursor < i and g.at[cursor, 'target_date'] <= row.origin:
                if pd.notna(g.at[cursor, 'y_true']):
                    mature.append(cursor)
                cursor += 1
            window, drift, ratio = 52, False, None
            # Compare two non-overlapping sets of 26 matured raw errors.
            if len(mature) >= 52:
                ids = np.asarray(mature[-52:])
                target = np.log(g.loc[ids, 'y_true'].to_numpy() / g.loc[ids, 'last_price'].to_numpy())
                err = np.minimum(np.abs(np.expm1(raw[ids] - target)), 1.)
                ratio = float(err[26:].mean() / max(err[:26].mean(), 1e-6))
                drift = ratio > DRIFT_RATIO
            if direction == 'recent26' or (direction == 'drift26' and drift):
                window = 26
            fit = np.asarray(mature[-window:], dtype=int)
            scale, bias = 0., 0.
            if len(fit) >= MIN_FIT:
                target = np.log(g.loc[fit, 'y_true'].to_numpy() / g.loc[fit, 'last_price'].to_numpy())
                scale, bias = affine(raw[fit], target)
                corrected = scale * raw[i] + bias
                issued[i] = float(np.clip(corrected, raw[i] - CORRECTION_CLIP, raw[i] + CORRECTION_CLIP))
            # Gate uses previously issued candidates, never in-sample fitted loss.
            gate = np.asarray(mature[-26:], dtype=int)
            gain, deploy = None, False
            if len(gate) >= MIN_GATE and len(fit) >= MIN_FIT:
                target = np.log(g.loc[gate, 'y_true'].to_numpy() / g.loc[gate, 'last_price'].to_numpy())
                gain = float(np.abs(np.expm1(-target)).mean() - np.abs(np.expm1(issued[gate] - target)).mean())
                deploy = gain > 0
            rec = row.to_dict()
            rec.update(y_cand=float(row.last_price * np.exp(issued[i])),
                       y_pred=float(row.last_price * np.exp(issued[i] if deploy else 0.)),
                       deploy=deploy, kappa=scale, bias=bias, n_updates=len(mature))
            outputs.append(rec)
            trace.append(dict(origin=row.origin, horizon=h, n_updates=len(mature),
                              last_update_target=g.at[mature[-1], 'target_date'] if mature else pd.NaT,
                              fit_n=len(fit), window=window, drift=drift, drift_ratio=ratio,
                              scale=scale, bias=bias, validation_gain=gain, deploy=deploy))
    return (pd.DataFrame(outputs).sort_values(KEYS).reset_index(drop=True),
            pd.DataFrame(trace).sort_values(KEYS).reset_index(drop=True))


def diagnostics(frame):
    """Retrospective block diagnostics; never called from replay decisions."""
    origins = sorted(frame.origin.unique())
    rows = []
    for block, start in enumerate(range(0, len(origins), 26)):
        date = origins[start]
        val = frame[(frame.target_date <= date)].dropna(subset=['y_true']).groupby('horizon').tail(26)
        test = frame[(frame.origin >= date) & (frame.origin < date + pd.Timedelta(weeks=26))].dropna(subset=['y_true'])
        if test.empty:
            continue
        gain = lambda x, col: float((np.abs(x.last_price / x.y_true - 1) - np.abs(x[col] / x.y_true - 1)).mean())
        counts = val.groupby('horizon').size()
        valid = len(counts) == 8 and bool(counts.ge(MIN_GATE).all())
        rows.append(dict(block=block, origin=date, n_val=len(val), n_test=len(test),
                         val_gain=gain(val, 'y_cand') if valid else None,
                         test_gain=gain(test, 'y_cand'), deploy_test_gain=gain(test, 'y_pred'),
                         deploy_fraction=float(test.deploy.mean()), diagnostic_only=True))
    return pd.DataFrame(rows)


def run(commodity, direction, input_dir=INPUT):
    source = load(commodity, input_dir)
    out, trace = replay(source, direction)
    validate(out)
    assert out[KEYS].equals(source[KEYS])
    results, logs = BASE / 'results' / direction, BASE / 'logs'
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(pd.util.hash_pandas_object(source, index=True).values.tobytes()).hexdigest()
    (logs / f'{commodity}_{direction}_protocol.json').write_text(json.dumps(dict(
        source='E014 ERM immutable raw forecast archive', source_frame_sha256=source_hash,
        direction=direction, min_fit=MIN_FIT, min_gate=MIN_GATE, ridge=RIDGE,
        drift_ratio=DRIFT_RATIO, correction_clip=CORRECTION_CLIP, bias_clip=BIAS_CLIP,
        target_available='target_date <= origin', publication_lag_weeks=0,
        untouched_holdout=False), indent=2))
    out.to_parquet(results / f'predictions_{commodity}.parquet', index=False)
    trace.to_csv(logs / f'{commodity}_{direction}_calibration.csv', index=False)
    tr = diagnostics(out)
    tr.to_csv(logs / f'{commodity}_{direction}_transfer.csv', index=False)
    wins = tr[tr.val_gain > 0]
    summary = dict(direction=direction, n_blocks=len(tr), n_val_wins=len(wins),
                   candidate_transfer_rate=float((wins.test_gain > 0).mean()) if len(wins) else None,
                   unit='26-week block averaged over all eight horizons', untouched_holdout=False)
    (logs / f'{commodity}_{direction}_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    logging.info('%s %s complete rows=%d', direction, commodity, len(out))


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
        for col in ('y_true', 'last_price', 'y_raw'):
            if not np.allclose(df[col], source[col], equal_nan=True):
                raise ValueError('Source value mismatch')
        for suffix, col in (('', 'y_pred'), ('_candidate', 'y_cand')):
            for h, group in [(str(h), g) for h, g in df.groupby('horizon')] + [('all', df)]:
                g = group.dropna(subset=['y_true'])
                err = g[col] - g.y_true
                records.append((f'E016_{direction}{suffix}', c, h,
                                float((err.abs() / g.y_true).mean() * 100), float(np.sqrt((err**2).mean())),
                                len(g), TARGETS[c], datetime.now().isoformat(timespec='seconds')))
    with closing(sqlite3.connect(db or BASE.parents[1] / 'leaderboard.sqlite', timeout=60)) as con, con:
        con.execute('CREATE TABLE IF NOT EXISTS leaderboard(experiment TEXT, commodity TEXT, horizon TEXT, mape REAL, rmse REAL, n INTEGER, target_series TEXT, created_at TEXT, PRIMARY KEY(experiment,commodity,horizon))')
        con.executemany('INSERT OR REPLACE INTO leaderboard VALUES (?,?,?,?,?,?,?,?)', records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=[*DIRECTIONS, 'leaderboard'])
    parser.add_argument('args', nargs='*')
    parser.add_argument('--input-dir', type=Path, default=INPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.command == 'leaderboard':
        if len(args.args) != 1 or args.args[0] not in DIRECTIONS:
            parser.error('leaderboard requires one direction')
        leaderboard(args.args[0], input_dir=args.input_dir)
    else:
        for c in args.args or TARGETS:
            if c not in TARGETS:
                parser.error(f'Unknown commodity {c}')
            run(c, args.command, args.input_dir)


if __name__ == '__main__':
    main()
