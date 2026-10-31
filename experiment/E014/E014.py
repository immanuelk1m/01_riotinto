"""E014 sheet-pool wrapper around the E013 invariant-risk GRU core."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
CORE_PATH = EXP_DIR / "e013_core.py"
if not CORE_PATH.exists():
    CORE_PATH = EXP_DIR.parent / "E013" / "E013.py"
spec = importlib.util.spec_from_file_location("e014_e013_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load E013 core from {CORE_PATH}")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

TARGETS = {
    "gold": "Com_LBMA_Gold_Price_PM_USD_V_0",
    "copper": "Com_LME_Cu_Cash_Bid",
    "corn": "Com_Barchart_Cash_National_Corn_Index",
    "coffee": "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag",
    "wti_crude_oil": "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel",
    "wheat": "Com_Barchart_Cash_Soft_Red_Wheat_Index",
}


def configure(direction: str) -> None:
    if direction not in core.DIRECTIONS:
        raise ValueError(f"Unknown direction {direction}")
    core.EXP_DIR = EXP_DIR
    core.ROOT = EXP_DIR.parents[1]
    core.DATA_DIR = EXP_DIR / "data"
    core.FEAT_DIR = EXP_DIR / "features"
    core.LOG_DIR = EXP_DIR / "logs"
    core.LEADERBOARD = core.ROOT / "leaderboard.sqlite"
    core.TARGETS = TARGETS
    core.DIRECTION = direction
    core.PENALTY = core.DIRECTIONS[direction]
    core.EXP = f"E014_sheet_pool_{direction}"
    core.RES_DIR = EXP_DIR / "results" / direction


def main() -> None:
    args = sys.argv[1:]
    cmd = args.pop(0) if args else "inv01"
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
            core.run(commodity)
        core.log.info("%s done", core.EXP)


if __name__ == "__main__":
    main()
