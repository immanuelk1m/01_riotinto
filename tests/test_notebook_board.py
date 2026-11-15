import json

import pandas as pd

from analysis import notebook_board as nb
from tools.log_causal_leaderboard import rows

WEEKS = pd.date_range("2024-01-07", periods=8, freq="W-SUN")


def test_export_matches_leaderboard(tmp_path, monkeypatch):
    monkeypatch.setattr(nb, "OUT", tmp_path)
    notebook = tmp_path / "x.ipynb"
    notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["1"]}]}))
    X = pd.DataFrame({"price": 0.1, "a": 1.0, "b": 1.0}, index=WEEKS)
    key = pd.DataFrame({"cited": [{"a"}] * 8, "type": "t", "ret": 1.0}, index=WEEKS)
    hit = pd.DataFrame({"a": 1.0, "b": 0.0}, index=WEEKS)        # 모든 주 적중
    miss = pd.DataFrame({"a": 0.0, "b": 1.0}, index=WEEKS)       # 모든 주 빗나감
    row = lambda name, s, c, **kw: dict(name=name, phase="all", series=s, contrib=c, X=X, key=key, **kw)  # noqa: E731
    root = nb.export("t", notebook, [row("base", "coffee", miss), row("base", "gold", miss),
                                     row("prior", "coffee", hit, parent="base", uses_answer_key=True),
                                     row("prior", "gold", hit, parent="base", uses_answer_key=True),
                                     row("half", "coffee", hit, parent="base"), dict(name="half", phase="all", series="gold", X=X, key=key,
                                                                                     parent="base", status="skipped")], "in_sample_full")
    summary = pd.read_csv(root / "summary.csv")
    assert set(summary.name) == {"t.base", "t.prior", "t.half"} and summary.loc[summary.name == "t.prior", "top3"].eq(1).all()
    comparisons = pd.read_csv(root / "comparisons.csv")
    assert list(comparisons.name) == ["t.prior"] and comparisons.delta.iloc[0] == 1.0  # 한 원자재가 빠진 변형은 비교하지 않는다
    keys = {s: key for s in ("coffee", "gold")}
    runs, board, _, _ = rows(root, keys)
    assert runs["t.prior"]["config"]["protocol"] == "in_sample_full" and "answer_key" in runs["t.prior"]["tags"]
    assert "incomplete" in runs["t.half"]["tags"] and "all/top3" not in runs["t.half"]["summary"]
    # 답지를 쓰는 변형은 따로 순위를 매긴다
    assert board.set_index("name").loc["t.prior", "rank"] == 1 and board.set_index("name").loc["t.base", "rank"] == 1
