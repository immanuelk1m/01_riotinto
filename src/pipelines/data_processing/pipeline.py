from kedro.pipeline import Node, Pipeline

from . import nodes as n


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(n.make_targets, ["raw_prices", "params:horizon", "params:origin_start"], "targets",
             name="make_targets"),
    ])
