"""E018: anytime-valid deployment gates on the immutable E015 frozen candidate.

Research-inspired ablation, not a reproduction. E016 deployed the calibrated
candidate only when the last 26 mature candidate errors beat persistence on a
point estimate (``gain > 0``). That rule carries no uncertainty control, so a
26-week window of noise can flip the decision; in the completed
regime-transfer round the gate reversed a candidate that had already passed the
accuracy, block-transfer and commodity-safety checks. This experiment changes
only the decision rule: the candidate forecast stays the immutable E015 frozen
``y_cand`` (coffee: 7.87 MAPE versus 7.89 for persistence, but 8.07 once the
E015 gate deploys it).

Directions (one Colab pane each):
  eproc  betting e-process on the mature loss difference, HEDGE-style
         predictable bet (Waudby-Smith & Ramdas 2023/2024). Deploy while
         E_t >= 1/alpha.
  cs_weight  time-uniform sub-Gaussian confidence sequence on the mean
         difference (Howard et al. 2020/2021), converted into a continuous
         deployment weight w = clip(lower_bound / EFFECT_SCALE, 0, 1) instead
         of a binary switch, so a tiny but positive bound still buys a small
         move toward the candidate rather than nothing.
  smcs   sequential model confidence set over {persistence, raw GRU, shrunk GRU}
         built from pairwise e-processes with running intersection
         (Arnold, Gavrilopoulos, Schulz & Ziegel 2024, JRSS-B). Deploy the best
         surviving model only after persistence is excluded.

Every decision reads only labels whose ``target_date`` is already realised at the
origin, and no issued forecast is ever revised. Loss differences are capped
absolute percentage errors so the bounded-loss assumptions behind the e-process
and the confidence sequence hold.
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
DIRECTIONS = ('eproc', 'cs_weight', 'smcs')
TARGETS = {
    'gold': 'Com_LBMA_Gold_Price_PM_USD_V_0',
    'copper': 'Com_LME_Cu_Cash_Bid',
    'corn': 'Com_Barchart_Cash_National_Corn_Index',
    'coffee': 'Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag',
    'wti_crude_oil': 'Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel',
    'wheat': 'Com_Barchart_Cash_Soft_Red_Wheat_Index',
}
KEYS = ['origin', 'horizon']
ALPHA = 0.05          # per-decision error budget for all three directions
MIN_LABELS = 13       # E016's minimum for a deploy decision, kept for parity
CAP = 1.0             # cap on absolute percentage error -> differences in [-1, 1]
LAMBDA_MAX = 0.5      # keeps 1 + lambda * d > 0 for d in [-1, 1]
SHRINK_MIN = 13       # minimum mature labels before estimating kappa
SHRINK_FRACTION = 0.5  # middle model of the sequential confidence set
EFFECT_SCALE = 0.01    # 1 percentage point: weight=1 once the bound proves that much


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
    """Capped absolute percentage error; persistence is forecast == 0."""
    return np.minimum(CAP, np.abs(np.expm1(forecast - target_logret)))


def hedge_bet(history):
    """Predictable betting fraction for the next observation (past data only)."""
    if not history:
        return 0.0
    d = np.asarray(history, dtype=float)
    mean = float(d.mean())
    var = float(d.var())
    if var <= 0:
        return LAMBDA_MAX
    lam = 2 * mean / (var + mean ** 2)
    return float(np.clip(lam, 0.0, LAMBDA_MAX))


def eprocess_value(history):
    """e-process for H: mean loss difference <= 0, using predictable bets."""
    log_e, seen = 0.0, []
    for d in history:
        lam = hedge_bet(seen)
        log_e += float(np.log1p(lam * d))
        seen.append(d)
    return float(np.exp(log_e))


def cs_lower(history, alpha=ALPHA):
    """Sub-Gaussian time-uniform lower bound for d bounded in [-1, 1]."""
    n = len(history)
    if n < 2:
        return -np.inf
    mean = float(np.mean(history))
    t = float(n)
    rad = np.sqrt(2.0 * np.log(np.log2(2.0 * t) / alpha) / t)
    return mean - rad


def pairwise_e(better, worse, alpha=ALPHA):
    """e-process that model `better` has not a worse mean loss than `worse`."""
    diff = np.asarray(better, dtype=float) - np.asarray(worse, dtype=float)
    return eprocess_value(list(diff))


def smcs_members(losses, alpha=ALPHA):
    """Sequential model confidence set: models not yet excluded, running intersection."""
    names = list(losses)
    min_e = {}
    for i in names:
        vals = []
        for j in names:
            if i == j:
                continue
            vals.append(pairwise_e(losses[j], losses[i], alpha))
        min_e[i] = min(vals) if vals else 1.0
    return names, min_e


def weekly_panel(frame):
    """One observation per target week: the eight horizons averaged together.

    Horizons overlap, so this trades a little independence for the power a
    per-horizon test lacks; both losses stay capped, so differences stay in
    [-1, 1] and the e-process assumptions hold.
    """
    f = frame.copy()
    target_logret = np.log(f.y_true / f.last_price)
    raw = np.log(f.y_cand / f.last_price)
    f['loss_p'] = ape(np.zeros(len(f)), target_logret)
    f['loss_c'] = ape(raw, target_logret)
    f['loss_s'] = ape(SHRINK_FRACTION * raw, target_logret)
    f['d'] = f.loss_p - f.loss_c
    mature = f.dropna(subset=['y_true'])
    return mature.groupby('target_date').agg(d=('d', 'mean'), loss_p=('loss_p', 'mean'),
                                             loss_c=('loss_c', 'mean'), loss_s=('loss_s', 'mean'))


def replay(frame, direction, alpha=ALPHA):
    """Causal commodity-level replay. Returns (predictions, decision trace)."""
    if direction not in DIRECTIONS:
        raise ValueError(f'Unknown direction {direction}')
    panel = weekly_panel(frame)
    origins = sorted(frame.origin.unique())
    trace, decisions = [], {}
    excluded = set()
    for origin in origins:
        history = panel[panel.index <= origin]
        d = history.d.to_numpy()
        deploy, detail, e_val, lb, weight = False, '', np.nan, np.nan, 0.0
        if len(d) >= MIN_LABELS:
            if direction == 'eproc':
                e_val = eprocess_value(list(d))
                deploy = e_val >= 1.0 / alpha
                weight = 1.0 if deploy else 0.0
                detail = f'e={e_val:.3g}'
            elif direction == 'cs_weight':
                lb = cs_lower(list(d), alpha)
                weight = float(np.clip(lb / EFFECT_SCALE, 0.0, 1.0))
                deploy = weight > 0
                detail = f'lb={lb:.4f} w={weight:.3f}'
            else:
                losses = {'persistence': history.loss_p.to_numpy(),
                          'candidate': history.loss_c.to_numpy(),
                          'shrunk': history.loss_s.to_numpy()}
                names, min_e = smcs_members(losses, alpha)
                excluded |= {n for n in names if min_e[n] < 1.0 / alpha}
                alive = [n for n in names if n not in excluded]
                deploy = 'persistence' not in alive and bool(alive)
                weight = 1.0 if deploy else 0.0
                e_val = min(min_e.values()) if min_e else np.nan
                detail = f'set={sorted(alive)}'
        decisions[origin] = (deploy, weight)
        trace.append(dict(origin=origin, n_labels=len(d), e_value=e_val, cs_lower=lb,
                          deploy=deploy, detail=detail,
                          mean_diff=float(d.mean()) if len(d) else np.nan))
    out = frame.copy()
    cand_logret = np.log(out.y_cand / out.last_price)
    out['deploy'] = out.origin.map(lambda o: decisions[o][0]).astype(bool)
    out['weight'] = out.origin.map(lambda o: decisions[o][1]).astype(float)
    out['y_pred'] = (out.last_price * np.exp(out.weight * np.log(out.y_cand / out.last_price))).astype(float)
    out['n_labels'] = out.origin.map({t['origin']: t['n_labels'] for t in trace}).astype(int)
    out = out.sort_values(KEYS).reset_index(drop=True)
    trace = pd.DataFrame(trace).sort_values('origin').reset_index(drop=True)
    return out, trace


def diagnostics(frame, min_gate=MIN_LABELS):
    """Retrospective block diagnostics; never used by replay decisions."""
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
        source='E015 frozen immutable candidate archive', source_frame_sha256=source_hash,
        direction=direction, alpha=ALPHA, min_labels=MIN_LABELS, error_cap=CAP,
        lambda_max=LAMBDA_MAX, target_available='target_date <= origin',
        publication_lag_weeks=0, untouched_holdout=False), indent=2))
    out.to_parquet(results / f'predictions_{commodity}.parquet', index=False)
    trace.to_csv(logs / f'{commodity}_{direction}_decisions.csv', index=False)
    diag = diagnostics(out)
    diag.to_csv(logs / f'{commodity}_{direction}_transfer.csv', index=False)
    summary = dict(direction=direction, n_blocks=len(diag),
                   deploy_fraction=float(diag.deploy_fraction.mean()) if len(diag) else None,
                   block_win_rate_positive=float((diag.deployed_gain > 0).mean()) if len(diag) else None,
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
        for suffix, col in (('', 'y_pred'), ('_candidate', 'y_cand')):
            for h, group in [(str(h), g) for h, g in df.groupby('horizon')] + [('all', df)]:
                g = group.dropna(subset=['y_true'])
                err = g[col] - g.y_true
                peak = float((err.abs() / g.y_true.abs() * 100).max())
                records.append((f'E018_{direction}{suffix}', c, h,
                                float((err.abs() / g.y_true).mean() * 100), float(np.sqrt((err ** 2).mean())),
                                len(g), TARGETS[c], datetime.now().isoformat(timespec='seconds'), peak))
    with closing(sqlite3.connect(db or BASE.parents[1] / 'leaderboard.sqlite', timeout=60)) as con, con:
        con.execute('CREATE TABLE IF NOT EXISTS leaderboard(experiment TEXT, commodity TEXT, horizon TEXT, mape REAL, rmse REAL, n INTEGER, target_series TEXT, created_at TEXT, PRIMARY KEY(experiment,commodity,horizon))')
        if 'max_ape_pct' not in [row[1] for row in con.execute('PRAGMA table_info(leaderboard)')]:
            con.execute('ALTER TABLE leaderboard ADD COLUMN max_ape_pct REAL')
        con.executemany('INSERT OR REPLACE INTO leaderboard(experiment, commodity, horizon, mape, rmse, n, target_series, created_at, max_ape_pct) VALUES (?,?,?,?,?,?,?,?,?)', records)


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
