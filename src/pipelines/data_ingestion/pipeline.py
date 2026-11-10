from kedro.pipeline import Node, Pipeline

from . import nodes as n


def create_pipeline(**kwargs) -> Pipeline:
    """Postgres 스냅샷. 실험과 분리해 두어 ablation 은 고정된 스냅샷 위에서 돈다."""
    return Pipeline([
        Node(n.load_prices, "params:targets", "raw_prices", name="load_prices"),
        Node(n.load_exogenous, ["params:targets", "params:exogenous"], "raw_exogenous", name="load_exogenous"),
    ])
