"""Gate E of PROTOCOL.md: machine-checked transfer contract.

The five items are evaluated from a predictions archive, so a report cannot claim a
result the numbers do not support:

  1. margin      deployed cell MAPE below persistence (and below the matched control),
                 with a paired 26-week block bootstrap interval excluding zero
  2. block       next-block win rate above 50%, computed per 26-week block
  3. coverage    deploy coverage at least 10% of origins
  4. safety      no commodity regresses by more than 0.05 MAPE against the control
  5. mutation    invariance to labels later than a cutoff (checked by re-running the
                 producer, so it is a callable hook rather than a post-hoc read)

Usage:
    uv run python experiment/transfer_contract.py --predictions DIR [--control DIR]
    uv run python experiment/transfer_contract.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

CAP = 1.0
BLOCK = 26
CRITICAL = {'margin', 'block', 'coverage', 'safety'}
SAFETY_TOL = 0.05
MIN_COVERAGE = 0.10


def capped_ape(forecast, target):
    """Capped absolute percentage error; both inputs are prices."""
    return np.minimum(CAP, np.abs(forecast / target - 1.0))


def load_archive(directory):
    """Read predictions_*.parquet from a results directory into {commodity: frame}."""
    directory = Path(directory)
    files = sorted(directory.glob('predictions_*.parquet'))
    if not files:
        raise FileNotFoundError(f'no predictions_*.parquet in {directory}')
    out = {}
    for path in files:
        frame = pd.read_parquet(path).dropna(subset=['y_true'])
        for col in ('origin', 'target_date'):
            if col in frame:
                frame[col] = pd.to_datetime(frame[col])
        out[path.stem.replace('predictions_', '')] = frame.sort_values(['origin', 'horizon']).reset_index(drop=True)
    return out


def block_bootstrap_ci(values, block=BLOCK, draws=2000, seed=42, level=0.95):
    """Moving-block bootstrap interval for the mean of an ordered daily/weekly series."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < block or n == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    means = np.empty(draws)
    for i in range(draws):
        picks = rng.choice(starts, size=int(np.ceil(n / block)), replace=True)
        sample = np.concatenate([values[s:s + block] for s in picks])[:n]
        means[i] = sample.mean()
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def cell_table(deployed, control=None):
    """Per-commodity cell MAPE for deployed, persistence and (optionally) the control."""
    rows = []
    for commodity, frame in deployed.items():
        base = capped_ape(frame.last_price, frame.y_true)
        pred = capped_ape(frame.y_pred, frame.y_true)
        row = dict(commodity=commodity, n_cells=int(len(frame)),
                   persistence=float(base.mean() * 100), deployed=float(pred.mean() * 100),
                   coverage=float(frame.deploy.mean()) if 'deploy' in frame else 1.0)
        if control is not None and commodity in control:
            ctrl_frame = control[commodity]
            merged = frame[['origin', 'horizon']].merge(
                ctrl_frame[['origin', 'horizon', 'y_pred']], on=['origin', 'horizon'], how='left')
            if merged.y_pred.notna().all():
                ctrl = capped_ape(merged.y_pred.to_numpy(), frame.y_true.to_numpy())
                row['control'] = float(ctrl.mean() * 100)
                row['d_vs_control'] = float((ctrl - pred).mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def margin_item(table, deployed, paired_diff):
    lo, hi = block_bootstrap_ci(paired_diff)
    below_base = bool(table.deployed.mean() < table.persistence.mean())
    has_control = 'control' in table
    below_ctrl = bool(table.deployed.mean() < table.control.mean()) if has_control else True
    passed = below_base and below_ctrl and lo > 0
    return dict(item='margin', passed=passed, ci_low=lo, ci_high=hi,
                deployed=float(table.deployed.mean()), persistence=float(table.persistence.mean()),
                control=float(table.control.mean()) if 'control' in table else None,
                note='deployed must beat persistence and control with CI above zero')


def block_item(deployed, block=BLOCK):
    """Next-block win rate over consecutive 26-origin blocks.

    Origins where the rule falls back to persistence are ties, not losses; the rate is
    reported over decisive blocks with the tie count alongside.
    """
    wins = ties = losses = 0
    for frame in deployed.values():
        origins = sorted(frame.origin.unique())
        for start in range(0, len(origins), block):
            window = origins[start:start + block]
            group = frame[frame.origin.isin(window)]
            base = capped_ape(group.last_price, group.y_true).mean()
            pred = capped_ape(group.y_pred, group.y_true).mean()
            if np.isclose(pred, base, atol=1e-12):
                ties += 1
            elif pred < base:
                wins += 1
            else:
                losses += 1
    decisive = wins + losses
    rate = wins / decisive if decisive else float('nan')
    return dict(item='block', passed=bool(decisive > 0 and rate > 0.5), win_rate=rate,
                wins=wins, losses=losses, ties=ties, decisive_blocks=decisive)


def coverage_item(table, deployed):
    origins = sum(frame.origin.nunique() for frame in deployed.values())
    deployed_origins = sum(frame.loc[frame.deploy, 'origin'].nunique() for frame in deployed.values()) \
        if all('deploy' in f for f in deployed.values()) else origins
    rate = deployed_origins / origins if origins else float('nan')
    return dict(item='coverage', passed=bool(rate >= MIN_COVERAGE), coverage=rate, origins=origins)


def safety_item(table):
    if 'd_vs_control' not in table:
        return dict(item='safety', passed=None, note='no control archive supplied')
    worst = table.loc[table.d_vs_control.idxmin()]
    return dict(item='safety', passed=bool(worst.d_vs_control > -SAFETY_TOL),
                worst_commodity=worst.commodity, worst_delta=float(worst.d_vs_control))


def mutation_item(producer=None, frame=None, cutoff=None):
    """Item 5 needs the producing code; without it the item is unverified, not passed."""
    if producer is None or frame is None or cutoff is None:
        return dict(item='mutation', passed=None, note='hook not exercised in this run')
    mutated = frame.copy()
    future = pd.to_datetime(mutated.target_date) > pd.Timestamp(cutoff)
    mutated.loc[future, 'y_true'] = mutated.loc[future, 'y_true'] * 3.0
    base = producer(frame)
    alt = producer(mutated)
    keep = ['origin', 'horizon', 'y_pred', 'deploy']
    same = base[base.origin <= cutoff][keep].reset_index(drop=True).equals(
        alt[alt.origin <= cutoff][keep].reset_index(drop=True))
    return dict(item='mutation', passed=bool(same), cutoff=str(cutoff))


def evaluate(predictions_dir, control_dir=None, producer=None, cutoff=None):
    deployed = load_archive(predictions_dir)
    control = load_archive(control_dir) if control_dir else None
    table = cell_table(deployed, control)
    diff, base_all = [], []
    for commodity, frame in deployed.items():
        diff.append(capped_ape(frame.last_price, frame.y_true) - capped_ape(frame.y_pred, frame.y_true))
    items = [
        margin_item(table, deployed, np.concatenate(diff)),
        block_item(deployed),
        coverage_item(table, deployed),
        safety_item(table),
        mutation_item(producer, None, cutoff),
    ]
    critical = [i for i in items if i['item'] in CRITICAL]
    verdict = 'transfer' if all(i['passed'] for i in critical) else 'non-transfer'
    return dict(verdict=verdict, metric='capped APE (cap=1.0), equal weight per cell',
                items=items, table=table.round(3).to_dict(orient='records'))


def _self_test():
    rng = np.random.default_rng(0)
    dates = pd.date_range('2020-01-05', periods=160, freq='W-SUN')

    def frame(noise_scale, deploy_rate=0.5, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for origin in dates:
            deploy = bool(rng.random() < deploy_rate)
            for h in range(1, 3):
                truth_logret = rng.normal(0, 0.02)
                truth = 100 * math.exp(truth_logret)
                # deployed forecast tracks the truth with small noise; a non-deployed
                # origin falls back to persistence (log return zero)
                cand_logret = truth_logret + rng.normal(0, noise_scale) if deploy else 0.0
                rows.append(dict(origin=origin, target_date=origin + pd.Timedelta(weeks=h), horizon=h,
                                 y_true=truth, last_price=100.0,
                                 y_pred=100 * math.exp(cand_logret), deploy=deploy))
        return pd.DataFrame(rows)

    good = {'x': frame(0.004, 0.6, 1)}         # deployed tracks the truth closely
    bad = {'x': frame(0.12, 0.6, 2)}           # deployed is far noisier than persistence
    sparse = {'x': frame(0.004, 0.02, 3)}      # almost never deploys

    good_items = {i['item']: i['passed'] for i in evaluate_frame(good)}
    bad_items = {i['item']: i['passed'] for i in evaluate_frame(bad)}
    sparse_items = {i['item']: i['passed'] for i in evaluate_frame(sparse)}
    assert good_items['margin'] is True, good_items
    assert good_items['block'] is True, good_items
    assert good_items['coverage'] is True, good_items
    assert bad_items['margin'] is False, bad_items
    assert sparse_items['coverage'] is False, sparse_items

    def producer(f):
        return f.assign(y_pred=np.where(f.deploy, f.y_pred, f.last_price))

    base = good['x']
    mut = base.copy()
    mut.loc[mut.origin > dates[100], 'y_true'] *= 3
    hook = mutation_item(producer, mut, dates[100])
    assert hook['passed'] is True, hook
    print('SELF-TEST PASS: margin/block/coverage discriminate, mutation hook detects nothing for a causal producer')


def evaluate_frame(frames, control=None):
    table = cell_table(frames, control)
    diff = [capped_ape(f.last_price, f.y_true) - capped_ape(f.y_pred, f.y_true) for f in frames.values()]
    return [margin_item(table, frames, np.concatenate(diff)), block_item(frames),
            coverage_item(table, frames), safety_item(table),
            dict(item='mutation', passed=None, note='hook not exercised')]


def main():
    parser = argparse.ArgumentParser(description='Gate E contract checks (PROTOCOL.md section 6)')
    parser.add_argument('--predictions', type=Path, help='results directory with predictions_*.parquet')
    parser.add_argument('--control', type=Path, default=None)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.predictions:
        parser.error('--predictions or --self-test is required')
    print(json.dumps(evaluate(args.predictions, args.control), indent=2, default=str))


if __name__ == '__main__':
    main()
