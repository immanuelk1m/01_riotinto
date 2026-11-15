import ast
from pathlib import Path

from analysis.causal_methods import METHODS, parents_from_edges

ROOT = Path(__file__).resolve().parents[1]


def test_parents_from_edges_applies_time_order_and_same_week_assumption():
    edges = [("a", 2, "-->"), ("b", 0, "o-o"), ("c", 0, "o-o"), ("d", 0, "<--"), ("e", 0, "-->"), ("price", 1, "-->"), ("price", 0, "")]
    # 시차 링크는 방향과 무관하게 부모, 같은 주는 `-->` 이거나 sink 변수일 때만, 가격 자기 시차는 통제 항으로 남는다
    assert parents_from_edges(edges, sink_vars={"b", "d"}) == {("a", 2), ("b", 0), ("d", 0), ("e", 0), ("price", 1)}


def test_no_neural_networks():
    """신경망 모델은 쓰지 않는다: 어댑터·옮긴 코드는 torch·lightning 을 import 하지 않고, Causal-TS 는 CD-NOTS·CDNOTS+·CEDAR 만 부른다."""
    for f in ("analysis/causal_methods.py", "analysis/castor_linear.py", "analysis/rcbnb.py"):
        tree = ast.parse((ROOT / f).read_text())
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not {m for m in names if m.split(".")[0] in {"torch", "lightning", "pytorch_lightning"}}, f
        called = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "causalts" for a in n.names}
        assert called <= {"run_cdnots", "run_cdnots_plus", "run_cedar"}, f
    assert "grace" not in " ".join(METHODS).lower()


def test_castor_recovers_two_regimes():
    from analysis.castor_linear import demo
    err, out = demo()
    assert len(out["graphs"]) == 2 and err < 0.1
