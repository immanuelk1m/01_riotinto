import hashlib
import json

import numpy as np
import pandas as pd
from tools.log_causal_leaderboard import f1_scores, rows

WEEKS = pd.date_range("2025-01-05", periods=10, freq="W-SUN")


def test_f1_scores():
    weeks = WEEKS[:5]
    contrib = pd.DataFrame(0.0, index=weeks, columns=list("abcd"))
    contrib.loc[weeks[0], "a"] = 1                       # 후보 1개, 적중 → 1.0
    contrib.loc[weeks[1], ["a", "b", "c"]] = [3, 2, -1]  # 후보 3개 중 적중 → 0.5
    contrib.loc[weeks[2], "b"] = 1                       # 빗나감 → 0
    contrib.loc[weeks[3], "d"] = 1                       # 인용 없는 주 → 분모에서 빠진다
    cited = pd.Series([{"a"}, {"a"}, {"a"}, set(), {"a"}], index=weeks)  # 마지막 주: 후보 없음 → 0
    values, weekly = f1_scores(contrib, cited)
    assert list(weekly) == [1.0, 0.5, 0.0, 0.0]
    assert values["top3"] == 0.5 and values["recall_3"] == 0.5
    assert np.isclose(values["f1_3"], 0.375) and np.isclose(values["precision_3"], (1 + 1 / 3) / 4)
    assert np.isclose(values["f1_3_chance"], 2 * 3 / 4 / 4)
    # 집합: S = {a, b, c, d}(기여가 0 이 아닌 적이 있는 변수), G = {a}
    assert values["set_precision"] == 0.25 and values["set_recall"] == 1.0 and np.isclose(values["set_f1"], 0.4)


def test_rows(tmp_path):
    directory = tmp_path / "abc123"
    keys = {s: pd.DataFrame({"cited": [{"a"}] * len(WEEKS)}, index=WEEKS) for s in ("coffee", "gold")}
    cells = []

    def add(name, series, status="complete", top3=None, picks=1, parent=""):
        cells.append({"name": name, "phase": "confirmation", "series": series, "parent": parent, "axis": "model", "status": status,
                      "seconds": 1.0, "top1": top3, "top3": top3, "top3 우연": top3 and 0.1, "top3 (전체 주)": top3,
                      "방향 일치 (top3 적중 주)": None})
        path = directory / "confirmation" / name / series
        path.mkdir(parents=True)
        if status != "complete":
            return
        c = pd.DataFrame(0.0, index=WEEKS, columns=list("abcd"))
        hits = round(top3 * len(WEEKS))
        c.iloc[:hits, :picks] = 1.0                    # 적중 주: a 포함 picks 개
        c.iloc[hits:, 1:1 + picks] = 1.0               # 나머지 주: a 없이 picks 개
        c.to_parquet(path / "contributions.parquet")

    add("baseline", "coffee", top3=0.2), add("baseline", "gold", top3=0.4)
    add("robust", "coffee", top3=0.4, picks=3, parent="baseline"), add("robust", "gold", top3=0.6, picks=3, parent="baseline")
    add("lpcmci", "coffee", top3=0.9), add("lpcmci", "gold", status="timeout")
    pd.DataFrame(cells).to_csv(directory / "summary.csv", index=False)
    pd.DataFrame([dict(phase="confirmation", name="robust", parent="baseline", delta=0.2, low=0.1, high=0.3, improved=True)]
                 ).to_csv(directory / "comparisons.csv", index=False)
    (directory / "selection.json").write_text(json.dumps(dict(winner="robust", selected="robust", specs={"robust": {"model": "robust"}})))
    (directory / "inputs.json").write_text(json.dumps({"/home/someone/data/panel.parquet": "sha"}))

    runs, board, cells, comparisons = rows(directory, keys)
    robust, baseline, lpcmci = (runs[n]["summary"] for n in ("robust", "PCMCI", "lpcmci"))
    assert np.isclose(robust["confirmation/top3"], 0.5) and np.isclose(robust["confirmation/top3_lift"], 0.4)
    assert robust["confirmation/delta_vs_parent"] == robust["confirmation/delta_vs_baseline"] == 0.2
    assert robust["confirmation/improved_vs_parent"] is True
    # F1@3: 후보 3개면 적중 주가 0.5, 1개면 1.0 → robust (0.2 + 0.3)/2, baseline (0.2 + 0.4)/2
    assert np.isclose(robust["confirmation/f1_3"], 0.25) and np.isclose(baseline["confirmation/f1_3"], 0.3)
    assert np.isclose(robust["confirmation/recall_3"], robust["confirmation/top3"])
    assert np.isclose(robust["confirmation/f1_3_delta_vs_parent"], -0.05) and robust["confirmation/f1_3_improved_vs_parent"] is False
    assert np.isclose(robust["confirmation/set_f1/coffee"], 0.4)  # S = {a, b, c, d}, G = {a}
    assert set(comparisons.metric) == {"top3", "f1_3"}
    tags = runs["robust"]["tags"]
    assert {"selected", "winner", "current"} <= set(tags) and "incomplete" not in tags
    assert runs["robust"]["config"]["inputs"] == {"panel.parquet": "sha"} and runs["robust"]["config"]["model"] == "robust"
    # 끝나지 않은 변형: 평균은 없고 원자재별 값·상태는 남는다
    assert "confirmation/top3" not in lpcmci and "confirmation/f1_3" not in lpcmci
    assert lpcmci["confirmation/top3/coffee"] == 0.9 and lpcmci["confirmation/f1_3/coffee"] == 0.9
    assert lpcmci["confirmation/status/gold"] == "timeout" and "incomplete" in runs["lpcmci"]["tags"]
    assert list(board.name) == ["robust", "PCMCI", "lpcmci"] and "baseline" not in set(cells.name) | set(cells.parent) and list(board["rank"][:2]) == [1, 2]
    assert list(board.rank_f1[:2]) == [2, 1] and pd.isna(board["rank"][2]) and "top3_chance" in cells
    # id 는 fingerprint(폴더 이름) + 이름으로 정해진다
    assert runs["robust"]["id"] == rows(directory, keys)[0]["robust"]["id"] != runs["PCMCI"]["id"]
    assert runs["PCMCI"]["id"] == hashlib.sha1(b"abc123:baseline").hexdigest()[:12]  # 이름을 바꿔도 id 는 그대로
    assert set(comparisons.parent) == {"PCMCI"}
    other = tmp_path / "def456"
    directory.rename(other)
    assert rows(other, keys)[0]["robust"]["id"] != runs["robust"]["id"]


def test_recomputed_top3_must_match(tmp_path):
    directory = tmp_path / "x"
    path = directory / "confirmation" / "baseline" / "coffee"
    path.mkdir(parents=True)
    pd.DataFrame(0.0, index=WEEKS, columns=["a"]).to_parquet(path / "contributions.parquet")  # 적중 0
    pd.DataFrame([{"name": "baseline", "phase": "confirmation", "series": "coffee", "parent": "", "axis": "model", "status": "complete",
                   "seconds": 1.0, "top1": 0.5, "top3": 0.5, "top3 우연": 0.1}]).to_csv(directory / "summary.csv", index=False)
    pd.DataFrame(columns=["phase", "name", "parent", "delta", "low", "high", "improved"]).to_csv(directory / "comparisons.csv", index=False)
    keys = {"coffee": pd.DataFrame({"cited": [{"a"}] * len(WEEKS)}, index=WEEKS)}
    try:
        rows(directory, keys)
    except ValueError as e:
        assert "Recomputed top-3" in str(e)
    else:
        raise AssertionError("mismatch not caught")
