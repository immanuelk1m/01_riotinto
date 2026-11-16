"""노트북의 주별 귀속 결과를 리더보드 결과 폴더 형식(`causal_study` 와 같다)으로 내보낸다.

노트북 마지막 셀에서 `export(...)` 를 부르면 `data/07_reporting/causal_notebooks/<tag>/<지문>/` 에
`summary.csv`·`comparisons.csv`·`selection.json`·`inputs.json`·`<구간>/<이름>/<원자재>/contributions.parquet` 를 쓴다.
올리기: `uv run python tools/log_causal_leaderboard.py --path <그 폴더>`. 채점은 노트북과 같은 `score_weeks`·`week_hits` 다.

`protocol` 은 노트북의 적합·채점 구간이다. 다른 `protocol` 의 run 은 같은 순위에 넣지 않는다.
- `in_sample_full`: 전체 기간으로 그래프·계수를 적합하고 같은 기간의 답지 주를 채점한다(PCMCI-LPCMCI, RCBNB-MB).
- `fit_lt_2025`: 2025-01 이전으로 적합하고, 선택 구간(적합 구간 안)·확인 구간(2025-01 이후)을 채점한다(ATTRIBUTION-ABLATION).
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import causal_common as cc  # noqa: E402
from analysis.causal_study import paired_ci  # noqa: E402

OUT = ROOT / "data/07_reporting/causal_notebooks"
INPUTS = [cc.PANEL, cc.KEY, cc.RELEASE_FILE, cc.WEATHER_FILE, ROOT / "data/01_raw/exogenous.parquet"]


def _fingerprint(notebook):
    nb = json.loads(Path(notebook).read_text())
    h = hashlib.sha256("".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code").encode())
    for p in [Path(cc.__file__), Path(__file__), ROOT / "uv.lock", *INPUTS]:
        if p.exists():
            h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()[:20]


def export(tag, notebook, rows, protocol, specs=None):
    """rows: dict(name, phase, series, contrib, X, key, parent='', axis='', uses_answer_key=False, override={}, status=None) 의 목록.
    contrib = 주 × 변수 귀속 점수(노트북이 채점한 그대로), X = 그 원자재의 패널, key = 그 구간의 답지.
    override = 노트북이 바꿔 적은 점수(예: 부호가 방향이 아닌 기준선의 방향 일치 = NaN). status = 계산하지 않은 행(예: "skipped")."""
    root = OUT / tag / _fingerprint(notebook)
    root.mkdir(parents=True, exist_ok=True)
    name = lambda n: f"{tag}.{n}"  # noqa: E731 — 노트북 사이·연구 사이에 이름이 겹치지 않게
    records, hits, spec_out = [], {}, {}
    for r in rows:
        n = name(r["name"])
        spec_out[n] = dict(notebook=Path(notebook).name, uses_answer_key=bool(r.get("uses_answer_key")), **(specs or {}).get(r["name"], {}))
        if r.get("status"):  # 노트북이 돌리지 않은 조합: 0점이 아니라 미완료로 남긴다
            records.append(dict(name=n, phase=r["phase"], series=r["series"], parent=name(r["parent"]) if r.get("parent") else "",
                                axis=r.get("axis", "") or "notebook", status=r["status"], seconds=0.0))
            continue
        weeks = r["key"].index[r["key"].index.isin(r["X"].index)]
        path = root / r["phase"] / n / r["series"]
        path.mkdir(parents=True, exist_ok=True)
        r["contrib"].reindex(weeks).fillna(0).to_parquet(path / "contributions.parquet")
        hits[r["phase"], n, r["series"]] = cc.week_hits(r["contrib"], r["X"], r["key"], 3)
        records.append(dict(name=n, phase=r["phase"], series=r["series"], parent=name(r["parent"]) if r.get("parent") else "",
                            axis=r.get("axis", "") or "notebook", status="complete", seconds=0.0,
                            **(cc.score_weeks(r["contrib"], r["X"], r["key"]) | r.get("override", {})),
                            candidate_count=int((r["contrib"].reindex(weeks).fillna(0) != 0).any().sum())))
    summary = pd.DataFrame(records)
    summary.to_csv(root / "summary.csv", index=False)
    comparisons = []
    every = set(summary.series)
    for (phase, n), g in summary.groupby(["phase", "name"], sort=False):
        parent = g.parent.iloc[0]
        series = list(g.series)
        if parent and set(series) == every and all((phase, k, s) in hits for k in (n, parent) for s in series):
            d, lo, hi = paired_ci({s: hits[phase, n, s] for s in series}, {s: hits[phase, parent, s] for s in series})
            comparisons.append(dict(phase=phase, name=n, parent=parent, delta=d, low=lo, high=hi, improved=phase == "confirmation" and lo > 0))
    pd.DataFrame(comparisons, columns=["phase", "name", "parent", "delta", "low", "high", "improved"]).to_csv(root / "comparisons.csv", index=False)
    (root / "selection.json").write_text(json.dumps(dict(study=f"notebook.{tag}", protocol=protocol, winner=None, selected=None,
                                                         specs=spec_out), indent=2, ensure_ascii=False, default=str))
    (root / "inputs.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS if p.exists()}, indent=2))
    (root / "status.json").write_text(json.dumps(dict(status="complete")))
    (OUT / tag / "latest.json").write_text(json.dumps({"path": str(root.resolve())}))
    print(f"exported {len(summary)} rows, {len(comparisons)} comparisons → {root}")
    return root
