from kedro.pipeline import Node, Pipeline

from . import nodes as n


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(n.to_weekly, "raw_prices", "weekly_prices", name="to_weekly"),
        Node(n.make_targets, ["weekly_prices", "params:horizon", "params:origin_start"], "targets",
             name="make_targets"),
    ])
