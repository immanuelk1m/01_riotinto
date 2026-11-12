"""고정 스냅샷 연구 실행: python tools/run_study.py [--phase anchor|features|adaptation]."""
import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.study_runner import execute_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "conf/studies/p0_p1_v1.yml")
    parser.add_argument("--phase", choices=["anchor", "features", "adaptation"])
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()
    logging.getLogger("kedro").setLevel(logging.WARNING)
    os.environ.setdefault("WANDB_SILENT", "true")
    execute_study(args.study, ROOT, args.phase, args.wandb_mode)


if __name__ == "__main__":
    main()
