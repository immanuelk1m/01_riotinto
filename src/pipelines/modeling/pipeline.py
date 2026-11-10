from kedro.pipeline import Node, Pipeline

from src.experiments import experiments, namespace

from . import nodes as n


def create_pipeline(**kwargs) -> Pipeline:
    """model → learning → inference, 실험마다 namespace `<입력 형식>.<모델 카테고리>.<실험>` 으로 한 벌씩."""
    return sum(
        (
            Pipeline(
                [
                    Node(n.select_input, ["tabular", "feature_columns", "params:spec", "params:horizon"],
                         "model_input", name="select_input"),
                    Node(n.make_model, "params:spec", "model", name="make_model"),
                    Node(n.fit, ["model", "model_input", "params:horizon"], "fitted", name="fit"),
                    Node(n.predict, ["fitted", "model_input", "params:horizon"], "forecasts", name="predict"),
                ],
                namespace=namespace(name, spec),
                inputs={"tabular", "feature_columns"},
                parameters={"params:spec": f"params:experiments.{name}", "params:horizon": "params:horizon"},
            )
            for name, spec in experiments().items()
        ),
        Pipeline([]),
    )
