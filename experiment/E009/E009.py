"""E009: 52-week GRU, eight direct outputs, expanding 26-week refits.

Uses the E008 snapshot/features for the original five commodities. Validation is
the last 104 realised origin weeks, with an eight-week purge before validation.
Epoch selection and shrink/gate share validation (not an independent significance
test). Report raw, shrunk and gated predictions separately to expose that limit.
Run on Colab: python E009.py gru [commodity ...]
Import results locally: uv run experiment/E009/E009.py leaderboard [gru]
"""

import copy
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

EXP = "E009_gru"
EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
DATA_DIR, FEAT_DIR = EXP_DIR / "data", EXP_DIR / "features"
LOG_DIR, RES_DIR = EXP_DIR / "logs", EXP_DIR / "results" / "gru"
LEADERBOARD = ROOT / "leaderboard.sqlite"
TARGETS = {
    "gold": "Com_LBMA_Gold_Price_PM_USD_V_0",
    "copper": "Com_LME_Cu_Cash_Bid",
    "corn": "Com_Barchart_Cash_National_Corn_Index",
    "coffee": "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag",
    "wti_crude_oil": "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel",
}
H, SEQ, REFIT, VAL_WEEKS = 8, 52, 26, 104
HIDDEN, EPOCHS, PATIENCE, BATCH, SEED = 32, 80, 8, 64, 42
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
    boundary = cut - H - VAL_WEEKS + 1
    train, val = rows[rows + H < boundary], rows[rows >= boundary]
    if len(train) < 100 or len(val) < 30:
        raise ValueError(f"Insufficient realised history: train={len(train)} val={len(val)}")
    assert train.max() + H < val.min() and rows.max() + H <= cut
    return rows, train, val


def scaler(F, end):
    # Every statistic is fitted only on rows available to the training partition.
    A = F[:end + 1]
    med = pd.DataFrame(A).median().fillna(0).to_numpy()
    A = np.where(np.isfinite(A), A, med)
    mean, std = A.mean(0), A.std(0)
    return med, mean, np.where(std > 1e-8, std, 1)


def sequences(F, rows, stats):
    med, mean, std = stats
    F = np.clip((np.where(np.isfinite(F), F, med) - mean) / std, -10, 10)
    indices = rows[:, None] + np.arange(1 - SEQ, 1)
    return F[indices].astype(np.float32)


def seed_torch(torch):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_model(torch, X, Y, device, epochs, validation=None):
    seed_torch(torch)

    class GRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(X.shape[-1], HIDDEN, batch_first=True)
            self.head = torch.nn.Linear(HIDDEN, H)

        def forward(self, x):
            _, hidden = self.gru(x)
            return self.head(hidden[-1])

    model = GRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    tx, ty = torch.as_tensor(X, device=device), torch.as_tensor(Y.astype(np.float32), device=device)
    if validation is not None:
        vx, vy = [torch.as_tensor(a.astype(np.float32), device=device) for a in validation]
    best_loss, best_epoch, best_state, stale = math.inf, 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(tx), device=device)
        for ids in order.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(tx[ids]) - ty[ids]) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(((model(vx) - vy) ** 2).mean().item())
            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= PATIENCE:
                break
    if validation is not None:
        if best_state is None:
            raise ValueError("No finite validation checkpoint")
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch if validation is not None else epochs


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


def run(commodity):
    import torch  # Training dependency only; local result import needs no torch.

    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = load(commodity)
    first, N = int(0.8 * S["N"]), S["N"]
    rows, trace = [], []
    start = time.time()
    log.info("[%s] device=%s torch=%s N=%d first_origin=%s features=%d", commodity, device,
             torch.__version__, N, S["weeks"][first].date(), len(S["names"]))
    for block, cut in enumerate(range(first, N, REFIT)):
        available, train, val = split_rows(S, cut)
        stats = scaler(S["F"], train.max())
        model, epochs = fit_model(torch, sequences(S["F"], train, stats), S["Z"][train], device, EPOCHS,
                                  (sequences(S["F"], val, stats), S["Z"][val]))
        P = predict(torch, model, sequences(S["F"], val, stats), device)
        kappa, t, rel, deploy = decision(P, S["Z"][val])
        # Refit from scratch for selected epochs with all labels realised by cut.
        stats = scaler(S["F"], available.max())
        model, _ = fit_model(torch, sequences(S["F"], available, stats), S["Z"][available], device, epochs)
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
                                 n_train=len(available), kappa=kappa))
        trace.extend(dict(retune=block, origin=str(S["weeks"][cut].date()), h=h, kappa=kappa,
                          t=t, deploy=deploy, val_rel=float(rel[h - 1]), epochs=epochs,
                          n_train=len(available), n_val=len(val)) for h in range(1, H + 1))
        log.info("[%s] block=%d origins=%d/%d epochs=%d kappa=%.3f t=%.3f deploy=%s elapsed=%.0fs",
                 commodity, block, min(cut + REFIT, N) - first, N - first, epochs, kappa, t, deploy, time.time() - start)
    df, tr = pd.DataFrame(rows), pd.DataFrame(trace)
    d = df.dropna(subset=["y_true"])
    base = (np.log(d.y_true / d.last_price) ** 2).groupby([d.retune, d.horizon]).mean()
    for name, col in (("test_rel", "y_pred"), ("cand_test_rel", "y_cand"), ("raw_test_rel", "y_raw")):
        gain = (np.log(d[col] / d.y_true) ** 2).groupby([d.retune, d.horizon]).mean() / base - 1
        tr[name] = [gain.get((r, h), np.nan) for r, h in zip(tr.retune, tr.h)]
    assert len(df) == (N - first) * H and not df.duplicated(["origin", "horizon"]).any()
    for col in ("y_pred", "y_cand", "y_raw", "last_price"):
        assert np.isfinite(df[col]).all() and (df[col] > 0).all()
    RES_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns="retune").to_parquet(RES_DIR / f"predictions_{commodity}.parquet", index=False)
    tr.to_csv(LOG_DIR / f"{commodity}_gru_transfer.csv", index=False)
    log.info("[%s] saved %d predictions (%.0fs)", commodity, len(df), time.time() - start)
    for col in ("y_pred", "y_cand", "y_raw", "last_price"):
        log.info("[%s] %s MAPE=%.4f RMSE=%.4f n=%d", commodity, col, *metrics(df, col))


def metrics(df, col):
    d = df.dropna(subset=["y_true"])
    err = d[col] - d.y_true
    return float((err.abs() / d.y_true.abs()).mean() * 100), float(np.sqrt((err ** 2).mean())), len(d)


def write_leaderboard(commodity, df):
    required = {"origin", "target_date", "horizon", "y_true", "y_pred", "y_cand", "y_raw", "last_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{commodity}: missing result columns {sorted(missing)}")
    if df.duplicated(["origin", "horizon"]).any():
        raise ValueError(f"{commodity}: duplicate origin/horizon rows")
    if set(df["horizon"].unique()) != set(range(1, H + 1)):
        raise ValueError(f"{commodity}: incomplete horizons")
    if not (df.groupby("origin")["horizon"].nunique() == H).all():
        raise ValueError(f"{commodity}: incomplete origin forecasts")
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(LEADERBOARD, timeout=60) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS leaderboard(
            experiment TEXT, commodity TEXT, horizon TEXT, mape REAL, rmse REAL, n INTEGER,
            target_series TEXT, created_at TEXT, PRIMARY KEY(experiment, commodity, horizon))""")
        for exp, col in ((EXP, "y_pred"), (EXP + "_candidate", "y_cand"), (EXP + "_raw", "y_raw"), ("naive", "last_price")):
            for h, g in [(str(h), g) for h, g in df.groupby("horizon")] + [("all", df)]:
                mape, rmse, n = metrics(g, col)
                assert n > 0 and np.isfinite(mape) and np.isfinite(rmse)
                con.execute("INSERT OR REPLACE INTO leaderboard VALUES (?,?,?,?,?,?,?,?)",
                            (exp, commodity, h, mape, rmse, n, TARGETS[commodity], now))
                if h == "all":
                    log.info("[%s] %s h=all MAPE=%.4f RMSE=%.4f n=%d", commodity, exp, mape, rmse, n)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "E009_gru.log")])
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gru"
    if cmd == "leaderboard":
        if sys.argv[2:] not in ([], ["gru"]):
            raise ValueError("Only gru results are supported")
        files = sorted(RES_DIR.glob("predictions_*.parquet"))
        expected = {RES_DIR / f"predictions_{commodity}.parquet" for commodity in TARGETS}
        if set(files) != expected:
            missing = sorted(path.name for path in expected - set(files))
            extra = sorted(path.name for path in set(files) - expected)
            raise FileNotFoundError(f"Incomplete E009 results: missing={missing} extra={extra}")
        for f in files:
            write_leaderboard(f.stem.removeprefix("predictions_"), pd.read_parquet(f))
    elif cmd == "gru":
        commodities = sys.argv[2:] or list(TARGETS)
        for commodity in commodities:
            if commodity not in TARGETS:
                raise ValueError(f"Unknown commodity: {commodity}")
            run(commodity)
        log.info("E009_gru done")
    else:
        raise ValueError("Usage: E009.py gru [commodity ...] | leaderboard [gru]")


if __name__ == "__main__":
    main()
