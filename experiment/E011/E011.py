"""E011: delayed-label EWLS correction of E010 raw GRU forecasts.

Fixed recipe, inspired by delayed adaptation and memory-horizon aggregation;
not a reproduction of MELO. Each horizon has separate state. The weekly close
is assumed available at its target_date (source publication lags are unknown).
Only issued expert forecasts are scored, and no issued forecast is revised.
These already-inspected historical test periods are retrospective evaluation,
not a new untouched holdout. No hyperparameters are selected on these results.
"""
import argparse
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
EXP = "E011_delay"
H = 8
HALF_LIVES = (13, 52, 104)
EXPERTS = ("raw",) + tuple(f"{kind}_{half}" for kind in ("intercept", "affine") for half in HALF_LIVES)
WARMUP, RIDGE, SHARE, SCORE_HALF_LIFE = 26, 1.0, 0.02, 26
TARGETS = {
    "gold": "Com_LBMA_Gold_Price_PM_USD_V_0",
    "copper": "Com_LME_Cu_Cash_Bid",
    "corn": "Com_Barchart_Cash_National_Corn_Index",
    "coffee": "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag",
    "wti_crude_oil": "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel",
}
log = logging.getLogger(EXP)


def validate(frame, source=None):
    required = {"origin", "target_date", "horizon", "y_true", "last_price", "y_raw", "n_train"}
    if required - set(frame):
        raise ValueError(f"Missing columns: {required - set(frame)}")
    if frame.empty or frame.duplicated(["origin", "horizon"]).any():
        raise ValueError("Empty or duplicate origin/horizon")
    if set(frame.horizon) != set(range(1, H + 1)) or not frame.groupby("origin").size().eq(H).all():
        raise ValueError("Incomplete horizons")
    dates = pd.to_datetime(frame.origin)
    if not pd.to_datetime(frame.target_date).eq(dates + pd.to_timedelta(frame.horizon * 7, unit="D")).all():
        raise ValueError("Target dates mismatch")
    unique = pd.DatetimeIndex(sorted(dates.unique()))
    if not unique.equals(pd.date_range(unique[0], unique[-1], freq="7D")):
        raise ValueError("Missing weekly origins")
    for col in ("last_price", "y_raw", "y_pred", "y_equal"):
        if col in frame and (not np.isfinite(frame[col]).all() or not frame[col].gt(0).all()):
            raise ValueError(f"Invalid {col}")
    observed = frame.y_true.dropna()
    if not np.isfinite(observed).all() or not observed.gt(0).all():
        raise ValueError("Invalid observed targets")
    if source is not None:
        left = frame.sort_values(["origin", "horizon"]).reset_index(drop=True)
        right = source.sort_values(["origin", "horizon"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(left[list(required)], right[list(required)], check_dtype=False)


class Bank:
    def __init__(self):
        self.gram = [np.zeros((2, 2)) for _ in EXPERTS[1:]]
        self.rhs = [np.zeros(2) for _ in EXPERTS[1:]]
        self.mass = np.zeros(6)
        self.loss = np.zeros(len(EXPERTS))
        self.target_square = 0.0
        self.score_mass = 0.0
        self.n = 0
        self.last_target = None

    def update(self, issued, target, target_date):
        raw, issued_experts = issued
        decay = 2 ** (-1 / SCORE_HALF_LIFE)
        self.loss = decay * self.loss + (issued_experts - target) ** 2
        self.target_square = decay * self.target_square + target ** 2
        self.score_mass = decay * self.score_mass + 1
        for j, name in enumerate(EXPERTS[1:]):
            kind, half = name.split("_")
            decay = 2 ** (-1 / int(half))
            x = np.array([1.0, raw if kind == "affine" else 0.0])
            self.gram[j] = decay * self.gram[j] + np.outer(x, x)
            self.rhs[j] = decay * self.rhs[j] + x * (target - raw)
            self.mass[j] = decay * self.mass[j] + 1
        self.n += 1
        self.last_target = target_date

    def forecast(self, raw):
        forecasts = [raw]
        for j, name in enumerate(EXPERTS[1:]):
            kind, _ = name.split("_")
            # Scale the slope ridge to the observed predictor second moment.
            diagonal = np.array([RIDGE, max(self.gram[j][1, 1] / max(self.mass[j], 1), 1e-4) * RIDGE])
            coef = np.linalg.solve(self.gram[j] + np.diag(diagonal), self.rhs[j])
            correction = coef @ np.array([1.0, raw if kind == "affine" else 0.0])
            forecasts.append(raw + float(np.clip(correction, -0.5, 0.5)))
        forecasts = np.clip(np.array(forecasts), -2.0, 2.0)
        forecasts[0] = raw
        prior = np.array([0.5] + [0.5 / 6] * 6)
        if self.n < WARMUP:
            weights = np.array([1.0] + [0.0] * 6)
        else:
            normalized = self.loss / max(self.target_square, self.score_mass * 1e-6)
            logits = np.log(prior) - normalized
            weights = np.exp(logits - logits.max())
            weights = (1 - SHARE) * weights / weights.sum() + SHARE * prior
        return forecasts, weights


def process(source):
    source = source.copy()
    for col in ("origin", "target_date"):
        source[col] = pd.to_datetime(source[col])
    source = source.sort_values(["origin", "horizon"]).reset_index(drop=True)
    validate(source)
    banks = {h: Bank() for h in range(1, H + 1)}
    pending, predictions, diagnostics = [], [], []
    for origin, group in source.groupby("origin", sort=True):
        still_pending = []
        for record in pending:
            date, h, raw, forecasts, truth, price = record
            if date <= origin:
                if np.isfinite(truth):
                    banks[h].update((raw, forecasts), np.log(truth / price), date)
            else:
                still_pending.append(record)
        pending = still_pending
        for row in group.to_dict("records"):
            h, price = int(row["horizon"]), row["last_price"]
            bank = banks[h]
            raw = float(np.log(row["y_raw"] / price))
            forecasts, weights = bank.forecast(raw)
            output = dict(row)
            output.update(y_pred=price * np.exp(weights @ forecasts),
                          y_equal=price * np.exp(forecasts.mean()), n_updates=bank.n,
                          deploy=bool(bank.n >= WARMUP))
            predictions.append(output)
            diagnostics.append(dict(origin=origin, horizon=h, n_updates=bank.n,
                                    latest_target=bank.last_target,
                                    **{f"weight_{name}": float(w) for name, w in zip(EXPERTS, weights)},
                                    **{f"return_{name}": float(p) for name, p in zip(EXPERTS, forecasts)}))
            pending.append((row["target_date"], h, raw, forecasts.copy(), row["y_true"], price))
    output, diagnostic = pd.DataFrame(predictions), pd.DataFrame(diagnostics)
    validate(output, source)
    assert (diagnostic.latest_target.dropna() <= diagnostic.loc[diagnostic.latest_target.notna(), "origin"]).all()
    assert np.allclose(diagnostic[[f"weight_{n}" for n in EXPERTS]].sum(axis=1), 1)
    return output, diagnostic


def metrics(frame, col):
    observed = frame.dropna(subset=["y_true"])
    error = observed[col] - observed.y_true
    return float((error.abs() / observed.y_true.abs()).mean() * 100), float(np.sqrt((error ** 2).mean())), len(observed)


def run(commodity, input_dir, output_dir):
    source = pd.read_parquet(input_dir / f"predictions_{commodity}.parquet")
    output, diagnostic = process(source)
    results, logs = output_dir / "results" / "delay", output_dir / "logs"
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    output.to_parquet(results / f"predictions_{commodity}.parquet", index=False)
    diagnostic.to_csv(logs / f"{commodity}_delay_weights.csv", index=False)
    summaries = []
    for col in ("y_pred", "y_equal", "y_raw", "last_price"):
        mape, rmse, n = metrics(output, col)
        summaries.append(dict(commodity=commodity, prediction=col, mape=mape, rmse=rmse, n=n))
        log.info("[%s] %s MAPE=%.4f RMSE=%.4f n=%d", commodity, col, mape, rmse, n)
    pd.DataFrame(summaries).to_csv(logs / f"{commodity}_metrics.csv", index=False)


def write_leaderboard(input_dir, output_dir, database):
    results = output_dir / "results" / "delay"
    expected = {results / f"predictions_{c}.parquet" for c in TARGETS}
    if set(results.glob("predictions_*.parquet")) != expected:
        raise ValueError("Expected complete five-commodity result set")
    records, now = [], datetime.now().isoformat(timespec="seconds")
    for commodity in TARGETS:
        frame = pd.read_parquet(results / f"predictions_{commodity}.parquet")
        validate(frame, pd.read_parquet(input_dir / f"predictions_{commodity}.parquet"))
        for exp, col in ((EXP, "y_pred"), (EXP + "_equal", "y_equal"), (EXP + "_raw", "y_raw")):
            for horizon, group in [(str(h), g) for h, g in frame.groupby("horizon")] + [("all", frame)]:
                mape, rmse, n = metrics(group, col)
                scored = group.dropna(subset=["y_true"])
                peak = float(((scored[col] - scored.y_true).abs() / scored.y_true.abs() * 100).max())
                if not n or not np.isfinite([mape, rmse]).all():
                    raise ValueError("Invalid metrics")
                records.append((exp, commodity, horizon, mape, rmse, n, TARGETS[commodity], now, peak))
    with sqlite3.connect(database, timeout=60) as con:
        con.execute("CREATE TABLE IF NOT EXISTS leaderboard(experiment TEXT, commodity TEXT, horizon TEXT, mape REAL, rmse REAL, n INTEGER, target_series TEXT, created_at TEXT, PRIMARY KEY(experiment, commodity, horizon))")
        if "max_ape_pct" not in [row[1] for row in con.execute("PRAGMA table_info(leaderboard)")]:
            con.execute("ALTER TABLE leaderboard ADD COLUMN max_ape_pct REAL")
        con.executemany("INSERT OR REPLACE INTO leaderboard(experiment, commodity, horizon, mape, rmse, n, target_series, created_at, max_ape_pct) VALUES (?,?,?,?,?,?,?,?,?)", records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("delay", "leaderboard"))
    parser.add_argument("commodities", nargs="*")
    parser.add_argument("--input-dir", type=Path, default=BASE / "inputs")
    parser.add_argument("--output-dir", type=Path, default=BASE)
    parser.add_argument("--database", type=Path, default=BASE.parents[1] / "leaderboard.sqlite")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "leaderboard":
        write_leaderboard(args.input_dir, args.output_dir, args.database)
    else:
        for commodity in args.commodities or TARGETS:
            if commodity not in TARGETS:
                raise ValueError(f"Unknown commodity {commodity}")
            run(commodity, args.input_dir, args.output_dir)
        (args.output_dir / "logs" / "recipe.json").write_text(json.dumps(dict(
            half_lives=HALF_LIVES, warmup=WARMUP, ridge=RIDGE, share=SHARE,
            score_half_life=SCORE_HALF_LIFE, experts=EXPERTS,
            protocol="target_date <= origin; immutable issued forecasts; retrospective"), indent=2))


if __name__ == "__main__":
    main()
