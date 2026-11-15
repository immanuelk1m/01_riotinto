"""인과 연구 결과 폴더 → W&B 리더보드(`sonet_krr/causal`). 폴더의 CSV·JSON 만 읽고 모델을 다시 적합하지 않는다.

    uv run python tools/log_causal_leaderboard.py [--path <결과 폴더>] [--wandb-mode online|offline|disabled]

방법 변형 하나 = run 하나(id = fingerprint + 이름의 해시). 다시 올리면 같은 run 을 덮어쓴다.
`analysis/causal_study.py` 에 넣지 않는다 — 그 파일은 fingerprint 에 들어가 캐시된 적합을 무효로 만든다.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import causal_common as cc  # noqa: E402 — import 만 한다(수정하면 fingerprint 가 바뀐다)
from analysis.causal_study import paired_ci  # noqa: E402

ENTITY, PROJECT = "sonet_krr", "causal"
# 연구 코드의 이름 → 리더보드 이름. 연구 코드에서 바꾸면 fingerprint 가 바뀌어 모든 적합을 다시 돌리므로 여기서 바꾼다.
# run id 는 연구 코드의 이름으로 계산한다(이름만 바꿔도 같은 run 을 덮어쓴다).
NAMES = {"baseline": "PCMCI"}
ORIGINAL = {v: k for k, v in NAMES.items()}
ROOT_NAME = NAMES["baseline"]  # 연구가 기준(root)으로 쓰는 변형
STUDIES = {"causal_study": "data/07_reporting/causal_study", "causal_methods": "data/07_reporting/causal_methods"}
PROTOCOL = "split_v1"  # 선택 2023-07~2024-12(학습 <2023-01), 확인 2025-01~(학습 <2025-01)
DONE = ["complete", "cached"]
COLUMNS = {"답지 주 수": "weeks_scored", "전체 주 수": "weeks_all", "입력 밖 인용": "cited_outside_panel",
           "top1 우연": "top1_chance", "top1 (전체 주)": "top1_all_weeks", "top3 우연": "top3_chance",
           "top3 (전체 주)": "top3_all_weeks", "방향 일치 (top3 적중 주)": "direction_match"}
F1 = ["f1_3", "precision_3", "recall_3", "f1_3_chance", "set_f1", "set_precision", "set_recall"]
MEANS = ["top3", "top1", "top3_chance", "top3_lift", "top3_all_weeks", "direction_match", *F1]
K = 3


def f1_scores(contrib, cited):
    """주 × 변수 기여와 주별 인용 변수 집합(패널 안) → (F1 값 dict, 주별 F1@3 Series).

    주별: 후보 T = |기여| 상위 3(0 제외, `cc._top`), F1 = 2|T∩C| / (|T|+|C|). 분모 주 = 인용이 있는 주(top-3 적중률과 같다).
    집합: S = 기여가 0 이 아닌 적이 있는 변수, G = 구간의 인용 변수 합집합.
    """
    cited = cited[cited.map(bool)]
    top = cc._top(contrib, cited.index, K)
    picks = [set(top.columns[t]) for t in top.values]
    tp = pd.Series([len(c & t) for c, t in zip(cited, picks)], index=cited.index, dtype=float)
    n_pick = pd.Series([len(t) for t in picks], index=cited.index, dtype=float)
    n_cited = cited.map(len).astype(float)
    f1 = (2 * tp / (n_pick + n_cited)).rename("f1_3")
    S, G = set(contrib.columns[(contrib != 0).any()]), set().union(*cited)
    p = len(S & G) / len(S) if S else 0.0
    r = len(S & G) / len(G) if G else float("nan")
    return dict(top3=float((tp > 0).mean()), f1_3=float(f1.mean()),
                precision_3=float((tp / n_pick.where(n_pick > 0)).fillna(0).mean()), recall_3=float((tp / n_cited).mean()),
                f1_3_chance=float((2 * K * n_cited / contrib.shape[1] / (K + n_cited)).mean()),  # 무작위 3개의 기대 F1(F1 이 TP 에 선형)
                set_precision=p, set_recall=r, set_f1=2 * p * r / (p + r) if p + r else 0.0), f1


def add_f1(directory, cells, comparisons, keys=None):
    """완료된 적합마다 저장된 `contributions.parquet` 로 F1 을 붙이고, F1@3 짝 비교 행을 만든다."""
    if keys is None:
        recorded = {Path(k).name: v for k, v in json.loads((directory / "inputs.json").read_text()).items()}
        if hashlib.sha256(cc.KEY.read_bytes()).hexdigest() != recorded.get(cc.KEY.name):
            raise ValueError(f"{cc.KEY.name} changed after this run; F1 would use a different answer key than top-3")
        keys = {s: cc.load_key(s) for s in cells.series.unique()}
    scores, weekly = [], {}
    for r in cells[cells.status.isin(DONE)].itertuples():
        contrib = pd.read_parquet(directory / r.phase / r.name / r.series / "contributions.parquet")
        cols = set(contrib.columns)
        values, weekly[r.phase, r.name, r.series] = f1_scores(contrib, keys[r.series].cited.loc[contrib.index].map(lambda v: v & cols))
        scores.append(dict(phase=r.phase, name=r.name, series=r.series, top3_check=values.pop("top3"), **values))
    cells = cells.merge(pd.DataFrame(scores, columns=["phase", "name", "series", "top3_check", *F1]),
                        on=["phase", "name", "series"], how="left")
    off = cells[cells.top3_check.notna() & ((cells.top3_check - cells.top3).abs() > 1e-12)]
    if len(off):  # 순위 규칙이 연구와 같다는 증거 — 다르면 F1 도 믿을 수 없다
        raise ValueError("Recomputed top-3 differs from summary.csv:\n" + off[["phase", "name", "series", "top3", "top3_check"]].to_string())
    extra = []
    for c in comparisons.itertuples():
        series = cells[(cells.phase == c.phase) & (cells.name == c.name)].series
        d, lo, hi = paired_ci({s: weekly[c.phase, c.name, s] for s in series}, {s: weekly[c.phase, c.parent, s] for s in series})
        extra.append(dict(phase=c.phase, name=c.name, parent=c.parent, delta=d, low=lo, high=hi, improved=c.phase == "confirmation" and lo > 0))
    comparisons = pd.concat([comparisons.assign(metric="top3"), pd.DataFrame(extra, columns=comparisons.columns).assign(metric="f1_3")],
                            ignore_index=True)
    return cells.drop(columns="top3_check"), comparisons


def rows(directory, keys=None):
    """결과 폴더 → (runs {이름: id·group·tags·config·summary}, 리더보드 표, 원자재별 표, 비교표). keys = {원자재: 답지}(테스트용)."""
    directory = Path(directory)
    cells = pd.read_csv(directory / "summary.csv").rename(columns=COLUMNS)
    cells[["parent", "axis"]] = cells[["parent", "axis"]].astype(object).fillna("")
    cells["top3_lift"] = cells.top3 - cells.top3_chance
    cells, comparisons = add_f1(directory, cells, pd.read_csv(directory / "comparisons.csv"), keys)
    for frame in (cells, comparisons):
        frame[["name", "parent"]] = frame[["name", "parent"]].replace(NAMES)
    selection = json.loads((directory / "selection.json").read_text())
    study = selection.get("study", "causal_study")
    selection = dict(winner=NAMES.get(selection["winner"], selection["winner"]), selected=NAMES.get(selection["selected"], selection["selected"]),
                     specs={NAMES.get(k, k): v for k, v in selection["specs"].items()})
    inputs = {Path(k).name: v for k, v in json.loads((directory / "inputs.json").read_text()).items()}  # 로컬 경로는 올리지 않는다
    series, fingerprint = list(cells.series.unique()), directory.name
    runs = {}
    for name, group in cells.groupby("name", sort=False):
        spec, summary = selection["specs"].get(name, {}), {}
        for phase, sub in group.groupby("phase", sort=False):
            done = sub.status.isin(DONE)
            full = len(sub) == len(series) and bool(done.all())
            parent = sub.parent.iloc[0]
            summary |= {f"{phase}/complete": full, f"{phase}/parent": parent, f"{phase}/axis": sub.axis.iloc[0],
                        f"{phase}/seconds": float(sub.seconds.sum())}
            for r in sub.itertuples():
                summary[f"{phase}/status/{r.series}"] = r.status
                if r.status in DONE:
                    summary |= {f"{phase}/{m}/{r.series}": float(getattr(r, m)) for m in ("top3", "top1", "f1_3", "set_f1")}
            if full:  # 끝나지 않은 변형은 평균을 적지 않는다 — 0점으로 읽히면 안 된다
                summary |= {f"{phase}/{k}": float(v) for k, v in sub[MEANS].mean().dropna().items()}
            for c in comparisons[(comparisons.name == name) & (comparisons.phase == phase)].itertuples():
                prefix = "" if c.metric == "top3" else c.metric + "_"
                for role in ["parent"] * (c.parent == (parent or ROOT_NAME)) + ["baseline"] * (c.parent == ROOT_NAME):
                    summary |= {f"{phase}/{prefix}delta_vs_{role}": float(c.delta), f"{phase}/{prefix}ci_low_vs_{role}": float(c.low),
                                f"{phase}/{prefix}ci_high_vs_{role}": float(c.high)}
                    if phase == "confirmation":  # 선택 구간의 차이는 "개선 확인"으로 읽지 않는다
                        summary[f"{phase}/{prefix}improved_vs_{role}"] = bool(c.improved)
        tags = ["current"] + ["incomplete"] * (not group.status.isin(DONE).all()) + ["selected"] * (name == selection["selected"]) \
            + ["winner"] * (name == selection["winner"]) + ["matched_window"] * bool(spec.get("matched"))
        runs[name] = dict(
            id=hashlib.sha1(f"{fingerprint}:{ORIGINAL.get(name, name)}".encode()).hexdigest()[:12],
            group=f"{study}.{group.axis.iloc[0]}", tags=tags, summary=summary,
            config=dict(spec, name=name, study=study, parent=next((p for p in group.parent if p), ""),
                        axis=group.axis.iloc[0], fingerprint=fingerprint, inputs=inputs, protocol=PROTOCOL,
                        uses_answer_key=False))
    board = pd.DataFrame([
        dict(phase=phase, name=name, tags=",".join(t for t in run["tags"] if t != "current"),
             **{k.split("/", 1)[1].replace("/", "_"): v for k, v in run["summary"].items() if k.startswith(phase + "/")})
        for name, run in runs.items() for phase in ("confirmation", "selection") if f"{phase}/complete" in run["summary"]])
    board.insert(0, "study", study)
    board.insert(3, "window", ["matched" if "matched_window" in t.split(",") else "full" for t in board.tags])
    return runs, ranked(board), cells.assign(study=study), comparisons.assign(study=study)


def ranked(board):
    """순위는 같은 구간·같은 학습 창(full: 결측 주를 검정에서 뺀 전체 창, matched: 시작 전 = 0 으로 채운 연속 창) 안에서만 매긴다."""
    board = board.drop(columns=["rank", "rank_f1"], errors="ignore")
    for m in ("top3", "f1_3"):
        if m not in board:
            board[m] = float("nan")
    by = board.groupby(["phase", "window"])
    board.insert(4, "rank", by.top3.rank(ascending=False, method="min"))
    board.insert(5, "rank_f1", by.f1_3.rank(ascending=False, method="min"))  # 정렬 기준은 top-3 그대로
    return board.sort_values(["phase", "window", "rank", "name"], na_position="last", ignore_index=True)


def combined(exclude_by_study=None):
    """모든 연구의 최신 결과를 한 표로. causal_methods 의 대조군(matched_pcmci)은 causal_study 에 이미 있어 뺀다."""
    exclude_by_study = exclude_by_study or {"causal_methods": {"matched_pcmci"}}
    boards, cells, comps = [], [], []
    for study, base in STUDIES.items():
        latest = ROOT / base / "latest.json"
        if not latest.exists():
            continue
        _, b, c, k = rows(Path(json.loads(latest.read_text())["path"]))
        drop = exclude_by_study.get(study, set())
        boards.append(b[~b.name.isin(drop)]), cells.append(c[~c.name.isin(drop)]), comps.append(k)
    return ranked(pd.concat(boards, ignore_index=True)), pd.concat(cells, ignore_index=True), pd.concat(comps, ignore_index=True)


def upload(directory, mode="online", project=PROJECT, exclude=()):
    """결과 폴더의 변형마다 run 하나(`exclude` 는 다른 연구에 이미 있는 run), 그리고 모든 연구를 합친 `leaderboard` run."""
    directory = Path(directory)
    runs, board, cells, comparisons = rows(directory)
    runs = {k: v for k, v in runs.items() if k not in exclude}
    print(board.reindex(columns=["phase", "window", "rank", "name", "top3", "top3_chance", "delta_vs_parent", "ci_low_vs_parent", "ci_high_vs_parent",
                                 "rank_f1", "f1_3", "f1_3_chance", "f1_3_delta_vs_parent", "set_f1", "set_precision", "set_recall", "tags"]
                        ).to_string(index=False, float_format="%.3f"))
    if mode == "disabled":
        return None
    import wandb

    os.environ.setdefault("WANDB_SILENT", "true")
    fingerprint, study = directory.name, next(iter(runs.values()))["config"]["study"]
    common = dict(entity=ENTITY, project=project, resume="allow", reinit="create_new", mode=mode,
                  dir=str(directory), save_code=False)
    for name, run in runs.items():
        with wandb.init(id=run["id"], name=name, group=run["group"], job_type="causal", config=run["config"],
                        tags=run["tags"], **common) as r:
            r.tags = run["tags"]
            r.summary.update(run["summary"])
    table = lambda d: wandb.Table(dataframe=d.astype(object).where(d.notna(), None))  # noqa: E731 — NaN 은 빈칸으로
    board, cells, comparisons = combined()
    with wandb.init(id="leaderboard", name="leaderboard", job_type="leaderboard",
                    config=dict(fingerprint=fingerprint, protocol=PROTOCOL), **common) as r:
        r.log({"leaderboard": table(board), "cells": table(cells), "comparisons": table(comparisons)})
        url = r.url
    if mode == "online":  # 옛 fingerprint 의 run 은 남기되 `current` 만 뗀다
        for old in wandb.Api(timeout=30).runs(f"{ENTITY}/{project}", filters={"tags": "current", "config.study": study, "config.fingerprint": {"$ne": fingerprint}}):
            old.tags = [t for t in old.tags if t != "current"]
            old.update()
    return url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, help="결과 폴더(생략하면 causal_study/latest.json)")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--project", default=PROJECT)
    args = parser.parse_args()
    path = args.path or Path(json.loads((ROOT / "data/07_reporting/causal_study/latest.json").read_text())["path"])
    print(upload(path, args.wandb_mode, args.project) or "")
