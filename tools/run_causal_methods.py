"""후속 인과발견 방법(`analysis/causal_methods.py`)을 `causal_study` 와 같은 조건으로 돌리고 리더보드에 올린다.

    uv run python tools/run_causal_methods.py [--methods cdnod cedar ...] [--wandb-mode online|offline|disabled]

- 입력: 대조군 창과 같은 학습 패널(늦게 시작한 변수의 시작 전 = 0, 결측 없는 연속 구간), 같은 선택·확인 구간.
- 방법은 가격과 이어진 링크만 낸다. 부모 → 회귀 기여 → top-3·F1 채점은 `analysis.causal_study.worker` 를 그대로 쓴다.
- 캐시: 결과 폴더는 공통 코드·입력·잠금 파일의 지문, 적합 하나는 그 방법의 `spec.json`(어댑터 소스·패키지 버전)으로 재사용한다.
  어댑터 하나를 고치면 그 방법만 다시 돈다. `--methods` 로 일부만 돌려도 이미 끝난 다른 방법은 결과표에 남는다.
- 비교 대상은 `matched_pcmci`(같은 창의 PCMCI+). 이 run 은 `causal_study` 에 이미 있으므로 다시 올리지 않는다.
"""
import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)  # statsmodels adfuller 반환형 예고 — 값은 같다

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import causal_common as cc  # noqa: E402
from analysis import causal_methods as cm  # noqa: E402
from analysis.causal_study import Spec, cached, contiguous, paired_ci, worker as score_worker, zero_before_start  # noqa: E402

OUT = ROOT / "data/07_reporting/causal_methods"
PHASES = {"selection": "2023-01-01", "confirmation": "2025-01-01"}  # 학습 끝(포함 안 함) — causal_study.job 과 같다
CONTROL = "matched_pcmci"
INPUTS = [cc.PANEL, cc.KEY, cc.RELEASE_FILE, cc.ROOT / "data/01_raw/exogenous.parquet"]


def training_panel(series, phase):
    """causal_study.job 의 대조군 창과 같은 학습 패널."""
    end = PHASES[phase]
    X, info = cc.make_panel(series, fit_end=end)
    T = contiguous(zero_before_start(X.loc[X.index < end]))
    if len(T) < max(52, 2 * X.shape[1] * (cc.TAU + 1)):
        raise ValueError("Insufficient contiguous complete training weeks")
    return X, info, T


def fingerprint():
    h = hashlib.sha256()
    for p in [Path(__file__), ROOT / "analysis/causal_study.py", ROOT / "analysis/causal_common.py", ROOT / "uv.lock", *INPUTS]:
        h.update(str(p.name).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:20]


def run_job(path, series, phase, name):
    """작업 프로세스: 방법으로 링크를 찾고, 부모 그래프를 만들어 causal_study 의 채점 worker 에 넘긴다."""
    warnings.filterwarnings("ignore")
    try:
        began = time.monotonic()
        X, info, T = training_panel(series, phase)
        sink = cc.sinks(X, info)
        logp = np.log(cc.prices[cc.prices.series == series].set_index("week").value).reindex(T.index).dropna().values
        out = cm.METHODS[name][0](T, sink, cc.TAU, logp)
        parents = cm.parents_from_edges(out["edges"], set(X.columns[sink]))
        g = np.full((X.shape[1], X.shape[1], cc.TAU + 1), "", dtype="<U3")
        for v, t in parents:
            g[X.columns.get_loc(v), 0, t] = "-->"
        (path / "method.json").write_text(json.dumps(dict(
            edges=[[v, int(t), m] for v, t, m in out["edges"]], parents=sorted([v, t] for v, t in parents),
            extra=out.get("extra", {}), discover_seconds=time.monotonic() - began, train_rows=len(T)), indent=1, default=str))
    except Exception:
        (path / "error.txt").write_text(traceback.format_exc())
        return
    score_worker(path, series, phase, Spec(matched=True), {"graph": g})


def run(methods=None, wandb_mode="online"):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A causal methods run is already active")
        root = OUT / fingerprint()
        root.mkdir(exist_ok=True)
        (OUT / "latest.json").write_text(json.dumps({"path": str(root.resolve())}))
        (root / "inputs.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}, indent=2))
        specs, records, results = {}, [], {}
        for name in cm.METHODS:
            specs[name] = cm.method_spec(name, ROOT)
            text = json.dumps(specs[name], indent=2, sort_keys=True)
            for phase in PHASES:
                for s in cc.SERIES:
                    path = root / phase / name / s
                    if (path / "spec.json").exists() and (path / "spec.json").read_text() != text:
                        shutil.rmtree(path)  # 어댑터·패키지가 바뀌었다
                    path.mkdir(parents=True, exist_ok=True)
                    result = cached(path) if (path / "spec.json").exists() else None
                    status = "cached" if result else "pending"
                    if result is None and (methods is None or name in methods):
                        (path / "error.txt").unlink(missing_ok=True)
                        (path / "spec.json").write_text(text)
                        (root / "status.json").write_text(json.dumps(dict(status="running", current_job=dict(name=name, phase=phase, series=s))))
                        proc = mp.get_context("spawn").Process(target=run_job, args=(path, s, phase, name))
                        proc.start()
                        proc.join()
                        result = cached(path)
                        status = "complete" if result else "failed"
                    elif result is None and (path / "error.txt").exists():
                        status = "failed"
                    if status == "pending":
                        continue
                    meta = json.loads((path / "method.json").read_text()) if (path / "method.json").exists() else {}
                    row = dict(name=name, phase=phase, series=s, parent="" if name == CONTROL else CONTROL,
                               axis="regime" if name in cm.REGIME else "method", status=status,
                               seconds=meta.get("discover_seconds", 0.0) + (result or {}).get("compute_seconds", 0.0))
                    if result:
                        results[phase, name, s] = result
                        row |= result["scores"] | dict(candidate_count=result["candidate_count"])
                    records.append(row)
                    pd.DataFrame(records).to_csv(root / "summary.csv", index=False)
                    print(f"{phase} {name} {s}: {status} ({row['seconds']:.0f}s)", flush=True)
        done = lambda n, p: all((p, n, s) in results for s in cc.SERIES)  # noqa: E731
        comparisons = []
        for phase in PHASES:
            for name in dict.fromkeys(r["name"] for r in records if r["phase"] == phase):
                if name != CONTROL and done(name, phase) and done(CONTROL, phase):
                    d, lo, hi = paired_ci({s: results[phase, name, s]["hits"] for s in cc.SERIES},
                                          {s: results[phase, CONTROL, s]["hits"] for s in cc.SERIES})
                    comparisons.append(dict(phase=phase, name=name, parent=CONTROL, delta=d, low=lo, high=hi,
                                            improved=phase == "confirmation" and lo > 0))
        pd.DataFrame(comparisons, columns=["phase", "name", "parent", "delta", "low", "high", "improved"]).to_csv(root / "comparisons.csv", index=False)
        ran = {r["name"] for r in records}
        (root / "selection.json").write_text(json.dumps(dict(study="causal_methods", winner=None, selected=None,
                                                             specs={k: v for k, v in specs.items() if k in ran}), indent=2))
        state = "complete" if records and all(r["status"] in ("complete", "cached") for r in records) else "incomplete"
        (root / "status.json").write_text(json.dumps(dict(status=state)))
    if wandb_mode != "disabled" and records:
        from tools.log_causal_leaderboard import upload
        print(upload(root, wandb_mode, exclude={CONTROL}))
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", choices=list(cm.METHODS))
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()
    methods = None if args.methods is None else {CONTROL, *args.methods}  # 비교 대상은 항상 함께 돈다(캐시되면 바로 끝난다)
    print(run(methods, args.wandb_mode))
