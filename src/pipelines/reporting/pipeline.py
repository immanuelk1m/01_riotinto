from functools import partial

from kedro.pipeline import Node, Pipeline

from src.experiments import experiments, namespace

from . import nodes as n


def create_pipeline(**kwargs) -> Pipeline:
    """evaluation: 실험마다 평가표 + W&B run 하나(modeling 과 같은 namespace), 마지막에 전체 요약 한 장."""
    spaces = {name: namespace(name, spec) for name, spec in experiments().items()}
    per_experiment = [
        Pipeline(
            [
                Node(n.evaluate, ["forecasts", "params:ape_cap"], "metrics", name="evaluate"),
                Node(partial(n.log_run, name),
                     ["params:spec", "parameters", "targets", "model_input", "forecasts", "metrics"],
                     None, name="log_run"),
            ],
            namespace=space,
            inputs={"targets"},
            parameters={"params:spec": f"params:experiments.{name}", "params:ape_cap": "params:ape_cap"},
        )
        for name, space in spaces.items()
    ]
    summary = Pipeline([
        Node(partial(n.summarize, list(spaces)), [f"{space}.metrics" for space in spaces.values()], "summary",
             name="summarize"),
    ])
    return sum(per_experiment, Pipeline([])) + summary
