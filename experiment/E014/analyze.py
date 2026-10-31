"""Validate E014 artifacts and write compact metric/curve audit summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import E014 as experiment


EXP_DIR = Path(__file__).resolve().parent
DIRECTIONS = tuple(experiment.core.DIRECTIONS)
COMMODITIES = tuple(experiment.TARGETS)
SHARED_FIVE = tuple(name for name in COMMODITIES if name != "wheat")


def metrics(frame: pd.DataFrame, column: str) -> dict:
    valid = frame.dropna(subset=["y_true"])
    error = valid[column] - valid.y_true
    return {
        "mape": float((error.abs() / valid.y_true.abs()).mean() * 100),
        "rmse": float(np.sqrt((error**2).mean())),
        "n": int(len(valid)),
    }


def audit_curves(direction: str, commodity: str) -> dict:
    path = EXP_DIR / "logs" / f"{commodity}_{direction}_curves.json"
    curves = json.loads(path.read_text())
    if not curves:
        raise ValueError(f"Empty curves: {path}")
    epoch_zero = early_stopped = full_epochs = 0
    gaps = []
    for block in curves:
        history = block["history"]
        if block["best_epoch"] == 0:
            epoch_zero += 1
        if block["stop_epoch"] < experiment.core.EPOCHS:
            early_stopped += 1
        else:
            full_epochs += 1
        for row in history:
            numeric = [row["train_loss"], row["objective"], row["environment_variance"], *row["environment_losses"]]
            if row["early_val_loss"] is not None:
                numeric.append(row["early_val_loss"])
            if not np.isfinite(numeric).all() or len(row["environment_losses"]) != 3:
                raise ValueError(f"Invalid curve row: {path} block={block['block']} epoch={row['epoch']}")
        selected = history[block["best_epoch"]]
        if selected["early_val_loss"] is not None:
            gaps.append(float(selected["early_val_loss"] - selected["train_loss"]))
    return {
        "blocks": len(curves),
        "early_stopped_blocks": early_stopped,
        "full_epoch_blocks": full_epochs,
        "epoch_zero_blocks": epoch_zero,
        "deploy_blocks": int(sum(bool(block["deploy"]) for block in curves)),
        "mean_selected_val_train_gap": float(np.mean(gaps)),
        "max_selected_val_train_gap": float(np.max(gaps)),
    }


def main() -> None:
    records = []
    curve_audit = {}
    frames = {}
    for direction in DIRECTIONS:
        experiment.configure(direction)
        frames[direction] = {}
        curve_audit[direction] = {}
        for commodity in COMMODITIES:
            frame = pd.read_parquet(EXP_DIR / "results" / direction / f"predictions_{commodity}.parquet")
            experiment.core.validate_results(commodity, frame)
            frames[direction][commodity] = frame
            curve_audit[direction][commodity] = audit_curves(direction, commodity)
            for model, column in (("deploy", "y_pred"), ("candidate", "y_cand"), ("raw", "y_raw"), ("naive", "last_price")):
                records.append({"direction": direction, "commodity": commodity, "model": model, **metrics(frame, column)})
    table = pd.DataFrame(records)
    table.to_csv(EXP_DIR / "results" / "metrics_by_commodity.csv", index=False)

    aggregates = []
    for direction in DIRECTIONS:
        for group, commodities in (("shared_five", SHARED_FIVE), ("all_six", COMMODITIES)):
            for model in ("deploy", "candidate", "raw", "naive"):
                selected = table[(table.direction == direction) & (table.model == model) & table.commodity.isin(commodities)]
                aggregates.append({
                    "direction": direction,
                    "group": group,
                    "model": model,
                    "mean_commodity_mape": float(selected.mape.mean()),
                    "mean_commodity_rmse": float(selected.rmse.mean()),
                    "n": int(selected.n.sum()),
                })
    aggregate_table = pd.DataFrame(aggregates)
    aggregate_table.to_csv(EXP_DIR / "results" / "metrics_aggregate.csv", index=False)

    prior = {}
    for direction in DIRECTIONS:
        values = []
        for commodity in SHARED_FIVE:
            path = EXP_DIR.parent / "E013" / "results" / direction / f"predictions_{commodity}.parquet"
            if path.exists():
                values.append(metrics(pd.read_parquet(path), "y_pred")["mape"])
        prior[direction] = float(np.mean(values)) if len(values) == len(SHARED_FIVE) else None

    transfer = {}
    for direction in DIRECTIONS:
        transfer[direction] = {}
        for commodity in COMMODITIES:
            summary = json.loads((EXP_DIR / "logs" / f"{commodity}_{direction}_summary.json").read_text())
            rate = summary["candidate_transfer_rate"]
            if rate is not None and not math.isfinite(rate):
                raise ValueError(f"Invalid transfer rate {direction}/{commodity}")
            transfer[direction][commodity] = summary

    summary = {
        "experiment": "E014_sheet_pool",
        "directions": list(DIRECTIONS),
        "commodities": list(COMMODITIES),
        "metrics": json.loads(table.to_json(orient="records")),
        "aggregates": json.loads(aggregate_table.to_json(orient="records")),
        "prior_e013_shared_five_mean_commodity_mape": prior,
        "curve_audit": curve_audit,
        "transfer": transfer,
        "warning": "Forced dt-lag0 variables make E014 non-causal diagnostic evidence.",
    }
    (EXP_DIR / "results" / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(aggregate_table.to_string(index=False))
    print("PASS 18 prediction files, all curve rows, transfer summaries, and E010 shared-five grids")


if __name__ == "__main__":
    main()
