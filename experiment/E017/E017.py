"""E017 sheet-pool wrapper around the E013 invariant-risk GRU core."""

from __future__ import annotations

import importlib.util
import logging
import json
import math
import time

import numpy as np
import sys
from pathlib import Path

import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
CORE_PATH = EXP_DIR / "e013_core.py"
if not CORE_PATH.exists():
    CORE_PATH = EXP_DIR.parent / "E013" / "E013.py"
spec = importlib.util.spec_from_file_location("e017_e013_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load E013 core from {CORE_PATH}")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

DIRECTIONS = {"identity": 0, "qmap52": 52, "qmap104": 104}
core.DIRECTIONS = {name: 0.0 for name in DIRECTIONS}
for _name in ("load", "split_rows", "scaler", "sequences", "fit_model", "predict", "decision",
              "environment_ids", "validate_results", "metrics", "H", "SEQ", "REFIT", "EPOCHS"):
    globals()[_name] = getattr(core, _name)
log = core.log

TARGETS = {
    "gold": "Com_LBMA_Gold_Price_PM_USD_V_0",
    "copper": "Com_LME_Cu_Cash_Bid",
    "corn": "Com_Barchart_Cash_National_Corn_Index",
    "coffee": "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag",
    "wti_crude_oil": "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel",
    "wheat": "Com_Barchart_Cash_Soft_Red_Wheat_Index",
}


def configure(direction: str) -> None:
    global DIRECTION, PENALTY, LOG_DIR, RES_DIR, WINDOW
    if direction not in core.DIRECTIONS:
        raise ValueError(f"Unknown direction {direction}")
    core.EXP_DIR = EXP_DIR
    core.ROOT = EXP_DIR.parents[1]
    snapshot = EXP_DIR if (EXP_DIR / "data").exists() else EXP_DIR.parent / "E014"
    core.DATA_DIR = snapshot / "data"
    core.FEAT_DIR = snapshot / "features"
    core.LOG_DIR = EXP_DIR / "logs"
    core.LEADERBOARD = EXP_DIR / "leaderboard.sqlite"
    core.TARGETS = TARGETS
    core.DIRECTION = direction
    core.PENALTY = core.DIRECTIONS[direction]
    core.EXP = f"E017_sheet_pool_{direction}"
    core.RES_DIR = EXP_DIR / "results" / direction
    DIRECTION, PENALTY, WINDOW = direction, 0.0, DIRECTIONS[direction]
    LOG_DIR, RES_DIR = core.LOG_DIR, core.RES_DIR



def fit_mapping(F, source_end, fit_end, window):
    """Fit recent->source marginal maps; no row after fit_end is read.

    Only sheet features are mapped. Target log-price and return (last two
    columns) remain unchanged. Constant/sparse marginals use identity.
    """
    if not 0 <= source_end <= fit_end < len(F):
        raise ValueError("Invalid mapping chronology")
    if window == 0:
        return [None] * F.shape[1]
    if window not in (52, 104):
        raise ValueError("Unsupported window")
    source = F[:source_end + 1]
    recent = F[max(0, fit_end - window + 1):fit_end + 1]
    maps = []
    for col in range(F.shape[1]):
        a = source[:, col]; a = np.sort(a[np.isfinite(a)])
        b = recent[:, col]; b = np.sort(b[np.isfinite(b)])
        if col >= F.shape[1] - 2 or min(len(a), len(b)) < 16 or np.unique(a).size < 2 or np.unique(b).size < 2:
            maps.append(None)
            continue
        knots, starts, counts = np.unique(b, return_index=True, return_counts=True)
        probabilities = (starts + (counts - 1) / 2) / (len(b) - 1)
        mapped = np.quantile(a, probabilities)
        maps.append((knots, mapped))
    return maps


def apply_mapping(F, mapping):
    if len(mapping) != F.shape[1]:
        raise ValueError("Mapping width mismatch")
    output = F.copy()
    for col, spec in enumerate(mapping):
        if spec is not None:
            mask = np.isfinite(F[:, col])
            output[mask, col] = np.interp(F[mask, col], spec[0], spec[1])
    return output


def run(commodity):
    import torch

    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = load(commodity)
    first, N = int(0.8 * S["N"]), S["N"]
    rows, trace, curves = [], [], []
    start = time.time()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.info("[%s] device=%s torch=%s N=%d first=%s features=%d", commodity, device,
             torch.__version__, N, S["weeks"][first].date(), len(S["names"]))
    for block, cut in enumerate(range(first, N, REFIT)):
        available, train, early, gate = split_rows(S, cut)
        stats = scaler(S["F"], train.max())
        early_map = fit_mapping(S["F"], train.max(), early.min(), WINDOW)
        gate_map = fit_mapping(S["F"], train.max(), gate.min(), WINDOW)
        model, epochs, history = fit_model(torch, sequences(S["F"], train, stats), S["Z"][train], device,
                                          EPOCHS, (sequences(apply_mapping(S["F"], early_map), early, stats), S["Z"][early]))
        if epochs == 0:
            kappa, t, rel, deploy = 0.0, 0.0, np.zeros(H), False
            gate_mape_rel = np.zeros(H)
        else:
            P = predict(torch, model, sequences(apply_mapping(S["F"], gate_map), gate, stats), device)
            kappa, t, rel, deploy = decision(P, S["Z"][gate])
            gate_target = S["Z"][gate] * S["scale"][gate]
            gate_candidate = kappa * P * S["scale"][gate]
            gate_mape_rel = np.abs(np.expm1(gate_candidate - gate_target)).mean(0) / np.maximum(
                np.abs(np.expm1(-gate_target)).mean(0), 1e-12) - 1
        stats = scaler(S["F"], available.max())
        outer_map = fit_mapping(S["F"], available.max(), cut, WINDOW)
        refit_epochs = max(epochs, 1)
        model, _, final_history = fit_model(torch, sequences(S["F"], available, stats), S["Z"][available],
                                            device, refit_epochs)
        curves.append(dict(block=block, origin=str(S["weeks"][cut].date()), best_epoch=epochs,
                           stop_epoch=history[-1]["epoch"], refit_epochs=refit_epochs,
                           kappa=kappa, t=t, deploy=deploy,
                           n_train=len(train), n_early=len(early), n_gate=len(gate), n_refit=len(available),
                           direction=DIRECTION, penalty=PENALTY,
                           mapping_window=WINDOW, mapping_fit_ends={
                               "early": str(S["weeks"][early.min()].date()),
                               "gate": str(S["weeks"][gate.min()].date()),
                               "outer": str(S["weeks"][cut].date())},
                           environments=[dict(start=str(S["weeks"][train[environment_ids(len(train)) == g].min()].date()),
                                              end=str(S["weeks"][train[environment_ids(len(train)) == g].max()].date()),
                                              n=int((environment_ids(len(train)) == g).sum())) for g in range(3)],
                           train_end=str(S["weeks"][train.max()].date()),
                           early_start=str(S["weeks"][early.min()].date()),
                           early_end=str(S["weeks"][early.max()].date()),
                           gate_start=str(S["weeks"][gate.min()].date()),
                           gate_end=str(S["weeks"][gate.max()].date()),
                           history=history, refit_history=final_history))
        (LOG_DIR / f"{commodity}_{DIRECTION}_curves.json").write_text(json.dumps(curves, indent=2, allow_nan=False))
        q = np.arange(cut, min(cut + REFIT, N))
        raw = predict(torch, model, sequences(apply_mapping(S["F"], outer_map), q, stats), device) * S["scale"][q]
        if not np.isfinite(raw).all():
            raise ValueError("Non-finite forecast")
        for j, i in enumerate(q):
            T, price = S["weeks"][i], float(S["pff"].iloc[i])
            for h in range(1, H + 1):
                cand = kappa * float(raw[j, h - 1])
                target = T + pd.Timedelta(weeks=h)
                rows.append(dict(origin=T, target_date=target, horizon=h, retune=block,
                                 y_true=S["price"].get(target, np.nan), last_price=price,
                                 y_raw=price * math.exp(float(raw[j, h - 1])), y_cand=price * math.exp(cand),
                                 y_pred=price * math.exp(cand if deploy else 0), deploy=deploy,
                                 n_train=len(available), kappa=kappa))
        trace.extend(dict(retune=block, origin=str(S["weeks"][cut].date()), h=h, kappa=kappa,
                          t=t, deploy=deploy, val_rel=float(rel[h - 1]), epochs=epochs,
                          val_mape_rel=float(gate_mape_rel[h - 1]),
                          n_train=len(available), n_early=len(early), n_gate=len(gate)) for h in range(1, H + 1))
        log.info("[%s] block=%d epochs=%d stop=%d kappa=%.3f t=%.3f deploy=%s elapsed=%.0fs", commodity,
                 block, epochs, history[-1]["epoch"], kappa, t, deploy, time.time() - start)
    df, tr = pd.DataFrame(rows), pd.DataFrame(trace)
    d = df.dropna(subset=["y_true"])
    base = (np.log(d.y_true / d.last_price) ** 2).groupby([d.retune, d.horizon]).mean()
    for name, col in (("test_rel", "y_pred"), ("cand_test_rel", "y_cand"), ("raw_test_rel", "y_raw")):
        gain = (np.log(d[col] / d.y_true) ** 2).groupby([d.retune, d.horizon]).mean() / base - 1
        tr[name] = [gain.get((r, h), np.nan) for r, h in zip(tr.retune, tr.h)]
    base_mape = (np.abs(d.last_price / d.y_true - 1)).groupby([d.retune, d.horizon]).mean()
    for name, col in (("test_mape_rel", "y_pred"), ("cand_test_mape_rel", "y_cand"), ("raw_test_mape_rel", "y_raw")):
        gain = np.abs(d[col] / d.y_true - 1).groupby([d.retune, d.horizon]).mean() / base_mape - 1
        tr[name] = [gain.get((r, h), np.nan) for r, h in zip(tr.retune, tr.h)]
    paired = tr.dropna(subset=["val_mape_rel", "cand_test_mape_rel"])
    eligible = paired.val_mape_rel < 0
    summary = dict(direction=DIRECTION, penalty=PENALTY,
                   n_blocks=int(tr.retune.nunique()), n_val_wins=int(eligible.sum()),
                   metric="MAPE relative to persistence",
                   candidate_transfer_rate=float((paired.loc[eligible, "cand_test_mape_rel"] < 0).mean()) if eligible.any() else None,
                   val_test_rank_correlation=float(paired[["val_mape_rel", "cand_test_mape_rel"]].rank().corr().iloc[0, 1])
                   if paired.val_mape_rel.nunique() > 1 and paired.cand_test_mape_rel.nunique() > 1 else None,
                   note="Retrospective diagnostics only; no outer-result direction selection.")
    (LOG_DIR / f"{commodity}_{DIRECTION}_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    assert len(df) == (N - first) * H
    validate_results(commodity, df)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns="retune").to_parquet(RES_DIR / f"predictions_{commodity}.parquet", index=False)
    tr.to_csv(LOG_DIR / f"{commodity}_{DIRECTION}_transfer.csv", index=False)
    for col in ("y_pred", "y_cand", "y_raw", "last_price"):
        log.info("[%s] %s MAPE=%.4f RMSE=%.4f n=%d", commodity, col, *metrics(df, col))


def main() -> None:
    args = sys.argv[1:]
    cmd = args.pop(0) if args else "identity"
    direction = args.pop(0) if cmd == "leaderboard" and args else cmd
    configure(direction)
    core.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(core.LOG_DIR / f"{core.EXP}.log")],
    )
    if cmd == "leaderboard":
        files = set(core.RES_DIR.glob("predictions_*.parquet"))
        expected = {core.RES_DIR / f"predictions_{commodity}.parquet" for commodity in TARGETS}
        if files != expected or args:
            raise ValueError(f"Incomplete results missing={expected-files} extra={files-expected}")
        core.write_leaderboard({
            commodity: pd.read_parquet(core.RES_DIR / f"predictions_{commodity}.parquet")
            for commodity in TARGETS
        })
    else:
        for commodity in args or list(TARGETS):
            if commodity not in TARGETS:
                raise ValueError(f"Unknown commodity {commodity}")
            run(commodity)
        core.log.info("%s done", core.EXP)


if __name__ == "__main__":
    main()

