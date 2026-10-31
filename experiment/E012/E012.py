"""E012: delayed fixed-share aggregation of immutable GRU forecasts.

Research-inspired ablation, not a reproduction of MELO/ELF. Input predictions
come from E009/E010 retrospective walk-forward runs, so this is a retrospective
prequential experiment, not a new untouched holdout. All hyperparameters below
are fixed before replay. Target-week close is assumed available at that week's
origin; no additional source publication lag is represented by these archives.
"""
import argparse
import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT = BASE / "inputs"
RESULTS = BASE / "results" / "hedge"
LOGS = BASE / "logs"
EXP = "E012_gru_hedge"
TARGETS = {
    "gold": "Com_LBMA_Gold_Price_PM_USD_V_0",
    "copper": "Com_LME_Cu_Cash_Bid",
    "corn": "Com_Barchart_Cash_National_Corn_Index",
    "coffee": "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag",
    "wti_crude_oil": "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel",
}
EXPERTS = ["naive", "e009_raw", "e009_candidate", "e009_gated",
           "e010_raw", "e010_candidate", "e010_gated"]
GRID = [(half, share) for half in (13, 52, 104) for share in (0.01, 0.05)]
ETA, LOSS_SCALE, META_HALF, META_SHARE = 2.0, 0.01, 52, 0.01
KEYS = ["origin", "horizon"]


def validate_frame(df):
    required = {*KEYS, "target_date", "y_true", "last_price", "n_train"}
    if required - set(df):
        raise ValueError(f"Missing columns: {required - set(df)}")
    if df.empty or df.duplicated(KEYS).any():
        raise ValueError("Empty or duplicate origin/horizon")
    if df.origin.isna().any() or df.target_date.isna().any():
        raise ValueError("Missing dates")
    if set(df.horizon.unique()) != set(range(1, 9)) or not df.groupby("origin").size().eq(8).all():
        raise ValueError("Incomplete horizons")
    origins = pd.DatetimeIndex(sorted(df.origin.unique()))
    if not origins.equals(pd.date_range(origins[0], origins[-1], freq="7D")):
        raise ValueError("Missing weekly origins")
    if not (df.target_date == df.origin + pd.to_timedelta(df.horizon * 7, unit="D")).all():
        raise ValueError("Target date mismatch")
    for col in ["last_price", *[c for c in df if c.startswith("y_") and c != "y_true"]]:
        if not np.isfinite(df[col]).all() or not df[col].gt(0).all():
            raise ValueError(f"Invalid price: {col}")
    observed = df.y_true.dropna()
    if not np.isfinite(observed).all() or not observed.gt(0).all():
        raise ValueError("Invalid observed target")


def load(commodity, input_dir=INPUT):
    frames = []
    for exp in ("E009", "E010"):
        path = Path(input_dir) / exp / f"predictions_{commodity}.parquet"
        if Path(input_dir) == INPUT and not path.exists():
            direction = "gru" if exp == "E009" else "gru_reg"
            path = BASE.parent / exp / "results" / direction / path.name
        df = pd.read_parquet(path)
        for col in ("origin", "target_date"):
            df[col] = pd.to_datetime(df[col])
        validate_frame(df)
        if {"y_raw", "y_cand", "y_pred"} - set(df):
            raise ValueError(f"{exp}: missing GRU variants")
        frames.append(df.sort_values(KEYS).reset_index(drop=True))
    a, b = frames
    if not a[KEYS + ["target_date"]].equals(b[KEYS + ["target_date"]]):
        raise ValueError("E009/E010 origin, horizon or target mismatch")
    for col in ("y_true", "last_price"):
        if not np.allclose(a[col], b[col], rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError(f"E009/E010 {col} mismatch")
    out = b[KEYS + ["target_date", "y_true", "last_price", "n_train"]].copy()
    out["naive"] = out.last_price
    for name, frame in (("e009", a), ("e010", b)):
        for variant, col in (("raw", "y_raw"), ("candidate", "y_cand"), ("gated", "y_pred")):
            out[f"{name}_{variant}"] = frame[col]
    return out


def update(weights, loss, half_life, share):
    # Discount accumulated evidence, then mix with the uniform prior.
    z = np.log(np.maximum(weights, 1e-300)) * (0.5 ** (1 / half_life)) - ETA * loss
    z = np.exp(z - z.max())
    return (1 - share) * z / z.sum() + share / len(z)


def bounded_loss(pred, truth):
    return np.minimum((pred - truth) ** 2 / LOSS_SCALE, 1.0)


def aggregate(frame):
    """Return forecasts and issuance-time state; never read an immature target."""
    validate_frame(frame)
    outputs, traces = [], []
    for h, group in frame.groupby("horizon", sort=True):
        g = group.sort_values("origin").reset_index(drop=True)
        p = np.log(g[EXPERTS].to_numpy() / g.last_price.to_numpy()[:, None])
        bank = np.full((len(GRID), len(EXPERTS)), 1 / len(EXPERTS))
        meta = np.full(len(GRID), 1 / len(GRID))
        issued = np.zeros((len(g), len(GRID)))
        cursor, n_updates, last_update = 0, 0, pd.NaT
        for i, row in g.iterrows():
            while cursor < i and g.at[cursor, "target_date"] <= row.origin:
                truth = g.at[cursor, "y_true"]
                if pd.notna(truth):
                    target = np.log(truth / g.at[cursor, "last_price"])
                    loss = bounded_loss(p[cursor], target)
                    for k, (half, share) in enumerate(GRID):
                        bank[k] = update(bank[k], loss, half, share)
                    meta = update(meta, bounded_loss(issued[cursor], target), META_HALF, META_SHARE)
                    n_updates += 1
                    last_update = g.at[cursor, "target_date"]
                cursor += 1
            issued[i] = bank @ p[i]
            effective = meta @ bank
            pred = float(meta @ issued[i])
            rec = row.to_dict()
            rec.update(y_pred=row.last_price * np.exp(pred),
                       y_equal=row.last_price * np.exp(p[i].mean()),
                       y_e010=row.e010_gated, n_updates=n_updates)
            outputs.append(rec)
            tr = dict(origin=row.origin, horizon=h, n_updates=n_updates,
                      last_update_target=last_update,
                      max_weight=float(effective.max()),
                      entropy=float(-(effective * np.log(effective)).sum()))
            tr.update({f"weight_{name}": float(w) for name, w in zip(EXPERTS, effective)})
            tr.update({f"meta_h{half}_s{share}": float(w) for (half, share), w in zip(GRID, meta)})
            traces.append(tr)
    return (pd.DataFrame(outputs).sort_values(KEYS).reset_index(drop=True),
            pd.DataFrame(traces).sort_values(KEYS).reset_index(drop=True))


def transfer_diagnostics(df):
    """Read future labels only after replay, for descriptive transfer measurement."""
    records = []
    origins = sorted(df.origin.unique())
    for cut in range(52, len(origins), 26):
        date = origins[cut]
        end = date + pd.Timedelta(weeks=26)
        for h, g in df.groupby("horizon"):
            val = g[(g.target_date <= date) & (g.origin >= date - pd.Timedelta(weeks=60))].dropna(subset=["y_true"]).tail(52)
            test = g[(g.origin >= date) & (g.origin < end)].dropna(subset=["y_true"])
            if len(val) < 24 or test.empty:
                continue
            vr, tr = [], []
            for name in EXPERTS:
                vl = float(np.square(np.log(val[name] / val.y_true)).mean())
                tl = float(np.square(np.log(test[name] / test.y_true)).mean())
                vr.append(vl)
                tr.append(tl)
            corr = float(pd.Series(vr).rank().corr(pd.Series(tr).rank())) if np.std(vr) and np.std(tr) else np.nan
            for k, name in enumerate(EXPERTS):
                records.append(dict(cutoff=date, horizon=h, expert=name, n_val=len(val), n_test=len(test),
                                    val_loss=vr[k], future_loss=tr[k],
                                    val_gain=vr[0] - vr[k], future_gain=tr[0] - tr[k],
                                    ranking_transfer=corr, val_winner=bool(k == int(np.argmin(vr))),
                                    diagnostic_only=True))
    return pd.DataFrame(records)


def run(commodity, input_dir=INPUT):
    source = load(commodity, input_dir)
    out, weights = aggregate(source)
    validate_frame(out)
    if not out[KEYS].equals(source[KEYS]):
        raise ValueError("Output origins differ from source")
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(RESULTS / f"predictions_{commodity}.parquet", index=False)
    weights.to_csv(LOGS / f"{commodity}_hedge_weights.csv", index=False)
    transfer_diagnostics(out).to_csv(LOGS / f"{commodity}_hedge_transfer.csv", index=False)
    logging.info("%s complete rows=%d", commodity, len(out))


def leaderboard(db_path=None, input_dir=INPUT):
    expected = {RESULTS / f"predictions_{c}.parquet" for c in TARGETS}
    if set(RESULTS.glob("predictions_*.parquet")) != expected:
        raise ValueError("Expected exactly five complete commodity outputs")
    records = []
    now = datetime.now().isoformat(timespec="seconds")
    for commodity in TARGETS:
        df = pd.read_parquet(RESULTS / f"predictions_{commodity}.parquet")
        validate_frame(df)
        source = load(commodity, input_dir)
        if not df[KEYS].equals(source[KEYS]):
            raise ValueError("Output/input origin mismatch")
        for col in ("y_true", "last_price"):
            if not np.allclose(df[col], source[col], rtol=1e-12, atol=1e-12, equal_nan=True):
                raise ValueError(f"Output/input {col} mismatch")
        for exp, col in ((EXP, "y_pred"), (EXP + "_equal", "y_equal")):
            for h, g in [(str(h), g) for h, g in df.groupby("horizon")] + [("all", df)]:
                d = g.dropna(subset=["y_true"])
                err = d[col] - d.y_true
                if d.empty:
                    raise ValueError("No scored observations")
                mape = float((err.abs() / d.y_true.abs()).mean() * 100)
                rmse = float(np.sqrt(np.square(err).mean()))
                records.append((exp, commodity, h, mape, rmse, len(d), TARGETS[commodity], now))
    db_path = db_path or BASE.parents[1] / "leaderboard.sqlite"
    with closing(sqlite3.connect(db_path, timeout=60)) as con, con:
        con.execute("CREATE TABLE IF NOT EXISTS leaderboard(experiment TEXT, commodity TEXT, horizon TEXT, mape REAL, rmse REAL, n INTEGER, target_series TEXT, created_at TEXT, PRIMARY KEY(experiment, commodity, horizon))")
        con.executemany("INSERT OR REPLACE INTO leaderboard VALUES (?,?,?,?,?,?,?,?)", records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["hedge", "leaderboard"])
    parser.add_argument("commodities", nargs="*")
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--db", type=Path)
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "leaderboard":
        leaderboard(args.db, args.input_dir)
    else:
        (LOGS / "protocol.json").write_text(json.dumps(dict(experts=EXPERTS, grid=GRID, eta=ETA,
            loss_scale=LOSS_SCALE, meta_half=META_HALF, meta_share=META_SHARE,
            target_available="target_date <= origin", source="retrospective E009/E010 GRU archive",
            publication_lag_weeks=0, untouched_holdout=False), indent=2))
        for commodity in args.commodities or TARGETS:
            if commodity not in TARGETS:
                raise ValueError(f"Unknown commodity {commodity}")
            run(commodity, args.input_dir)


if __name__ == "__main__":
    main()
