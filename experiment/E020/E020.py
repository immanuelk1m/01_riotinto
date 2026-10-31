"""E020 model-side regime adaptation on the frozen E014 sheet snapshot.

Round 1 and 2 showed the binding constraint is the evidence base, not the
decision rule: every anytime-valid gate refused to deploy, and the one rule that
beat persistence (E019 cap) failed its deployed-worse contract. E020 asks the
complementary question with a lever that can move MAPE by far more than 0.02
percentage points: does changing what the deployed model is trained on help?

Directions (one Colab GPU pane each), all ERM (penalty zero), all reading only
labels realised at or before the origin:
  base    E015-fresh control: refit on every mature row with a refit scaler
  recent  TFPS/regime-expert inspired: refit on the last RECENT_WEEKS mature rows
          only, with a scaler fitted on that window
  tta     TAIP/test-time-adaptation inspired: refit on all mature rows, then
          continue the same weights for TTA_EPOCHS on the last TTA_WEEKS rows
          with the gate scaler retained
"""
import copy
import hashlib
import json
import logging
import math
import random
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DIRECTIONS = {"base": 0.0, "recent": 0.0, "tta": 0.0}
DIRECTION, PENALTY = "base", 0.0
EXP = "E020_base"
EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
SNAPSHOT = EXP_DIR if (EXP_DIR / "data").exists() else EXP_DIR.parent / "E014"
DATA_DIR, FEAT_DIR = SNAPSHOT / "data", SNAPSHOT / "features"
LOG_DIR, RES_DIR = EXP_DIR / "logs", EXP_DIR / "results" / DIRECTION
LEADERBOARD = ROOT / "leaderboard.sqlite"
TARGETS = {
    "gold": "Com_LBMA_Gold_Price_PM_USD_V_0",
    "copper": "Com_LME_Cu_Cash_Bid",
    "corn": "Com_Barchart_Cash_National_Corn_Index",
    "coffee": "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag",
    "wti_crude_oil": "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel",
    "wheat": "Com_Barchart_Cash_Soft_Red_Wheat_Index",
}
H, SEQ, REFIT, EARLY_WEEKS, GATE_WEEKS = 8, 52, 26, 52, 52
HIDDEN, EPOCHS, PATIENCE, BATCH, SEED = 8, 80, 10, 64, 42
MIN_DELTA, DROPOUT, LR, WEIGHT_DECAY, BETA = 1e-5, 0.3, 3e-4, 1e-2, 0.5
RECENT_WEEKS, TTA_WEEKS, TTA_EPOCHS = 156, 52, 5
log = logging.getLogger(EXP)


def load(commodity):
    X = pd.read_parquet(DATA_DIR / f"{commodity}_X.parquet")
    X = X.set_index(pd.to_datetime(X.pop("week_end"))).astype(float)
    px = pd.read_parquet(DATA_DIR / f"{commodity}_price.parquet")
    price = px.set_index(pd.to_datetime(px["dt"]))["value"].astype(float).resample("W-SUN").last()
    weeks = pd.date_range(max(price.first_valid_index(), X.index[0]), price.index[-1], freq="W-SUN")
    price, X = price.reindex(weeks), X.reindex(weeks)
    pff = price.ffill()
    lp = np.log(pff.where(pff > 0))
    names = [f["name"] for f in json.loads((FEAT_DIR / f"{commodity}_features.json").read_text())["features"]]
    names = list(dict.fromkeys([*names, "tgt_logp", "tgt_r1"]))
    F = X.assign(tgt_logp=lp, tgt_r1=lp.diff())[names].replace([np.inf, -np.inf], np.nan).ffill()
    sigma = lp.diff().rolling(52, min_periods=26).std().to_numpy()
    scale = sigma[:, None] * np.sqrt(np.arange(1, H + 1))
    Y = np.column_stack([(np.log(price.where(price > 0)).shift(-h) - lp).values for h in range(1, H + 1)])
    Z = Y / np.where(scale > 1e-8, scale, np.nan)
    last = min(X.dropna(how="all").index[-1], price.last_valid_index())
    return dict(weeks=weeks, F=F.to_numpy(), Z=Z, scale=scale, price=price,
                pff=pff, N=int((weeks <= last).sum()), names=names)


def split_rows(S, cut):
    rows = np.arange(SEQ - 1, cut - H + 1)
    rows = rows[np.isfinite(S["Z"][rows]).all(axis=1)]
    gate_start = cut - H - GATE_WEEKS + 1
    early_end = gate_start - H - 1
    early_start = early_end - EARLY_WEEKS + 1
    train = rows[rows + H < early_start]
    early = rows[(rows >= early_start) & (rows <= early_end)]
    gate = rows[rows >= gate_start]
    if len(train) < 100 or min(len(early), len(gate)) < 24:
        raise ValueError(f"Insufficient history train={len(train)} early={len(early)} gate={len(gate)}")
    assert train.max() + H < early.min() and early.max() + H < gate.min()
    assert gate.max() + H <= cut
    return rows, train, early, gate


def scaler(F, end):
    A = F[:end + 1]
    med = pd.DataFrame(A).median().fillna(0).to_numpy()
    A = np.where(np.isfinite(A), A, med)
    mean, std = A.mean(0), A.std(0)
    return med, mean, np.where(std > 1e-8, std, 1)


def sequences(F, rows, stats):
    med, mean, std = stats
    F = np.clip((np.where(np.isfinite(F), F, med) - mean) / std, -10, 10)
    return F[rows[:, None] + np.arange(1 - SEQ, 1)].astype(np.float32)


def seed_torch(torch):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment_ids(n):
    """Three contiguous equal-count environments, within supplied training rows."""
    if n < 3:
        raise ValueError("At least three training rows required")
    return np.minimum(np.arange(n) * 3 // n, 2)


def environment_objective(torch, row_losses, env, penalty, population_weights=None):
    risks = torch.stack([row_losses[env == g].mean() for g in range(3)])
    pooled = row_losses.mean() if population_weights is None else (risks * population_weights).sum()
    variance = risks.var(unbiased=False)
    return pooled + penalty * variance, pooled, variance, risks


def fit_model(torch, X, Y, device, epochs, validation=None, initial_model=None):
    seed_torch(torch)

    class GRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(X.shape[-1], HIDDEN, batch_first=True)
            self.dropout = torch.nn.Dropout(DROPOUT)
            self.head = torch.nn.Linear(HIDDEN, H)

        def forward(self, x):
            _, hidden = self.gru(x)
            return self.head(self.dropout(hidden[-1]))

    model = GRU().to(device) if initial_model is None else initial_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.SmoothL1Loss(beta=BETA)
    per_item = torch.nn.SmoothL1Loss(beta=BETA, reduction="none")
    tx, ty = torch.as_tensor(X, device=device), torch.as_tensor(Y.astype(np.float32), device=device)
    env = torch.as_tensor(environment_ids(len(X)), device=device)
    groups = [torch.where(env == g)[0] for g in range(3)]
    proportions = torch.as_tensor([len(g) / len(X) for g in groups], device=device)
    if validation is not None:
        vx, vy = [torch.as_tensor(a.astype(np.float32), device=device) for a in validation]

    def measure(epoch):
        model.eval()
        with torch.no_grad():
            row_losses = per_item(model(tx), ty).mean(1)
            objective, pooled, variance, risks = environment_objective(torch, row_losses, env, PENALTY)
            train_loss = float(pooled.item())
            val_loss = float(criterion(model(vx), vy).item()) if validation is not None else None
        if not np.isfinite(train_loss) or (val_loss is not None and not np.isfinite(val_loss)):
            raise ValueError("Non-finite evaluation loss")
        return dict(epoch=epoch, train_loss=train_loss, early_val_loss=val_loss,
                    objective=float(objective.item()), environment_variance=float(variance.item()),
                    environment_losses=risks.cpu().tolist(), penalty=PENALTY)

    history = [measure(0)]
    best_loss = history[0]["early_val_loss"] if validation is not None else math.inf
    best_epoch, stale = 0, 0
    best_state = copy.deepcopy(model.state_dict()) if validation is not None else None
    for epoch in range(1, epochs + 1):
        model.train()
        # Every mini-batch contains all environments; population weights retain
        # pooled ERM weighting even when environments differ slightly in size.
        for _ in range(math.ceil(len(tx) / BATCH)):
            ids = torch.cat([g[torch.randint(len(g), (BATCH // 3,), device=device)] for g in groups])
            optimizer.zero_grad(set_to_none=True)
            item_loss = per_item(model(tx[ids]), ty[ids]).mean(1)
            loss, _, _, _ = environment_objective(torch, item_loss, env[ids], PENALTY, proportions)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        history.append(measure(epoch))
        if validation is not None:
            val_loss = history[-1]["early_val_loss"]
            if val_loss < best_loss - MIN_DELTA:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= PATIENCE:
                break
    if validation is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch if validation is not None else epochs, history


def predict(torch, model, X, device):
    with torch.no_grad():
        return model(torch.as_tensor(X, device=device)).cpu().numpy()


def decision(P, Z):
    kappa = float(np.clip((P * Z).sum() / max((P ** 2).sum(), 1e-12), 0, 1))
    D = Z ** 2 - (Z - kappa * P) ** 2
    d = D.mean(1)
    dc, n = d - d.mean(), len(d)
    variance = dc @ dc / n + 2 * sum((1 - lag / H) * (dc[lag:] @ dc[:-lag]) / n for lag in range(1, H))
    t = float(d.mean() / math.sqrt(max(variance, 1e-12) / n))
    rel = ((Z - kappa * P) ** 2).sum(0) / np.maximum((Z ** 2).sum(0), 1e-12) - 1
    return kappa, t, rel, bool(kappa > 0 and t >= 1)


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def scaler_hash(stats):
    return hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in stats)).hexdigest()


def window_rows(available, weeks):
    """Last `weeks` mature rows, inclusive; never empty for weeks >= 1."""
    return available[available >= available.max() - weeks + 1]


def deployment_model(torch, S, available, gate_model, gate_stats, epochs, device):
    """The sole ablation boundary; no future rows or labels are used."""
    if DIRECTION not in DIRECTIONS:
        raise ValueError(f"Unknown direction {DIRECTION}")
    refit_epochs = max(epochs, 1)
    if DIRECTION == "base":
        stats = scaler(S["F"], available.max())
        model, _, history = fit_model(torch, sequences(S["F"], available, stats),
                                      S["Z"][available], device, refit_epochs)
        return model, stats, refit_epochs, history, len(available)
    if DIRECTION == "recent":
        rows = window_rows(available, RECENT_WEEKS)
        stats = scaler(S["F"], rows.max())
        model, _, history = fit_model(torch, sequences(S["F"], rows, stats),
                                      S["Z"][rows], device, refit_epochs)
        return model, stats, refit_epochs, history, len(rows)
    if DIRECTION == "tta":
        stats = scaler(S["F"], available.max())
        model, _, history = fit_model(torch, sequences(S["F"], available, stats),
                                      S["Z"][available], device, refit_epochs)
        recent = window_rows(available, TTA_WEEKS)
        model, _, tta_history = fit_model(torch, sequences(S["F"], recent, stats),
                                          S["Z"][recent], device, TTA_EPOCHS,
                                          initial_model=model)
        return model, stats, refit_epochs + TTA_EPOCHS, history + tta_history, len(available)
    raise ValueError(f"Unknown direction {DIRECTION}")


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
        model, epochs, history = fit_model(torch, sequences(S["F"], train, stats), S["Z"][train], device,
                                          EPOCHS, (sequences(S["F"], early, stats), S["Z"][early]))
        if epochs == 0:
            kappa, t, rel, deploy = 0.0, 0.0, np.zeros(H), False
            gate_mape_rel = np.zeros(H)
        else:
            P = predict(torch, model, sequences(S["F"], gate, stats), device)
            kappa, t, rel, deploy = decision(P, S["Z"][gate])
            gate_target = S["Z"][gate] * S["scale"][gate]
            gate_candidate = kappa * P * S["scale"][gate]
            gate_mape_rel = np.abs(np.expm1(gate_candidate - gate_target)).mean(0) / np.maximum(
                np.abs(np.expm1(-gate_target)).mean(0), 1e-12) - 1
        gate_model_hash, gate_scaler_hash = model_hash(model), scaler_hash(stats)
        model, stats, refit_epochs, final_history, deployed_n_train = deployment_model(
            torch, S, available, model, stats, epochs, device)
        deployed_model_hash, deployed_scaler_hash = model_hash(model), scaler_hash(stats)
        curves.append(dict(block=block, origin=str(S["weeks"][cut].date()), best_epoch=epochs,
                           stop_epoch=history[-1]["epoch"], refit_epochs=refit_epochs,
                           kappa=kappa, t=t, deploy=deploy,
                           n_train=len(train), n_early=len(early), n_gate=len(gate),
                           n_refit=len(available),
                           deployed_n_train=deployed_n_train,
                           direction=DIRECTION, penalty=PENALTY,
                           gate_model_hash=gate_model_hash, deployed_model_hash=deployed_model_hash,
                           gate_scaler_hash=gate_scaler_hash, deployed_scaler_hash=deployed_scaler_hash,
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
        raw = predict(torch, model, sequences(S["F"], q, stats), device) * S["scale"][q]
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
                                 n_train=deployed_n_train, kappa=kappa))
        trace.extend(dict(retune=block, origin=str(S["weeks"][cut].date()), h=h, kappa=kappa,
                          t=t, deploy=deploy, val_rel=float(rel[h - 1]), epochs=epochs,
                          val_mape_rel=float(gate_mape_rel[h - 1]),
                          n_train=deployed_n_train, n_early=len(early), n_gate=len(gate)) for h in range(1, H + 1))
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


def metrics(df, col):
    d = df.dropna(subset=["y_true"])
    err = d[col] - d.y_true
    return float((err.abs() / d.y_true.abs()).mean() * 100), float(np.sqrt((err ** 2).mean())), len(d)


def validate_results(commodity, df):
    required = {"origin", "target_date", "horizon", "y_true", "y_pred", "y_cand", "y_raw", "last_price", "deploy", "n_train"}
    if required - set(df.columns):
        raise ValueError(f"{commodity}: missing columns {sorted(required - set(df.columns))}")
    if df.empty or df.duplicated(["origin", "horizon"]).any():
        raise ValueError(f"{commodity}: empty or duplicate origin/horizon")
    if set(df.horizon.unique()) != set(range(1, H + 1)) or not (df.groupby("origin").horizon.nunique() == H).all():
        raise ValueError(f"{commodity}: incomplete horizons")
    origins = pd.DatetimeIndex(pd.to_datetime(df.origin).unique()).sort_values()
    if not origins.equals(pd.date_range(origins.min(), origins.max(), freq="W-SUN")):
        raise ValueError(f"{commodity}: missing weekly origins")
    if not (pd.to_datetime(df.target_date) == pd.to_datetime(df.origin) + pd.to_timedelta(df.horizon * 7, unit="D")).all():
        raise ValueError(f"{commodity}: target dates mismatch")
    for col in ("y_pred", "y_cand", "y_raw", "last_price"):
        if not np.isfinite(df[col]).all() or not (df[col] > 0).all():
            raise ValueError(f"{commodity}: invalid {col}")
    reference = EXP_DIR.parent / "E014" / "results" / "erm" / f"predictions_{commodity}.parquet"
    if reference.exists():
        expected = pd.read_parquet(reference, columns=["origin", "horizon"])
        actual = df[["origin", "horizon"]]
        expected = expected.assign(origin=pd.to_datetime(expected.origin)).sort_values(["origin", "horizon"]).reset_index(drop=True)
        actual = actual.assign(origin=pd.to_datetime(actual.origin)).sort_values(["origin", "horizon"]).reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError(f"{commodity}: origin/horizon grid differs from E014 reference")


def write_leaderboard(frames):
    if set(frames) != set(TARGETS):
        raise ValueError("Expected all six commodities")
    records = []
    now = datetime.now().isoformat(timespec="seconds")
    for commodity, df in frames.items():
        validate_results(commodity, df)
        for exp, col in ((EXP, "y_pred"), (EXP + "_candidate", "y_cand"), (EXP + "_raw", "y_raw"), ("E015_naive", "last_price")):
            for h, g in [(str(h), g) for h, g in df.groupby("horizon")] + [("all", df)]:
                mape, rmse, n = metrics(g, col)
                if n == 0 or not np.isfinite([mape, rmse]).all():
                    raise ValueError(f"{commodity}: invalid metrics")
                records.append((exp, commodity, h, mape, rmse, n, TARGETS[commodity], now))
                if h == "all":
                    log.info("[%s] %s MAPE=%.4f RMSE=%.4f n=%d", commodity, exp, mape, rmse, n)
    with sqlite3.connect(LEADERBOARD, timeout=60) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS leaderboard(
            experiment TEXT, commodity TEXT, horizon TEXT, mape REAL, rmse REAL, n INTEGER,
            target_series TEXT, created_at TEXT, PRIMARY KEY(experiment, commodity, horizon))""")
        con.executemany("INSERT OR REPLACE INTO leaderboard VALUES (?,?,?,?,?,?,?,?)", records)


def configure(direction):
    global DIRECTION, PENALTY, EXP, RES_DIR
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction {direction}")
    DIRECTION, PENALTY = direction, DIRECTIONS[direction]
    EXP = f"E020_{direction}"
    RES_DIR = EXP_DIR / "results" / direction


def main():
    args = sys.argv[1:]
    cmd = args.pop(0) if args else "base"
    direction = args.pop(0) if cmd == "leaderboard" and args else cmd
    configure(direction)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / f"{EXP}.log")])
    if cmd == "leaderboard":
        files = set(RES_DIR.glob("predictions_*.parquet"))
        expected = {RES_DIR / f"predictions_{c}.parquet" for c in TARGETS}
        if files != expected or args:
            raise ValueError(f"Incomplete results missing={expected-files} extra={files-expected}")
        write_leaderboard({c: pd.read_parquet(RES_DIR / f"predictions_{c}.parquet") for c in TARGETS})
    else:
        for commodity in args or list(TARGETS):
            if commodity not in TARGETS:
                raise ValueError(f"Unknown commodity {commodity}")
            run(commodity)
        log.info("%s done", EXP)


if __name__ == "__main__":
    main()


