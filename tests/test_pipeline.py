from pathlib import Path

from kedro.framework.startup import bootstrap_project

from src.experiments import experiments, namespace


def test_registry_splits_data_and_runs_every_experiment_end_to_end():
    bootstrap_project(Path.cwd())
    from src.pipeline_registry import register_pipelines

    pipelines = register_pipelines()
    assert {node.name for node in pipelines["data"].nodes} == {"load_prices", "load_exogenous"}
    default = pipelines["__default__"]
    assert not {"load_prices", "load_exogenous"} & {node.name for node in default.nodes}
    for name, spec in experiments().items():
        nodes = default.only_nodes_with_namespaces([namespace(name, spec)]).nodes
        assert {node.name.rsplit(".", 1)[1] for node in nodes} == \
            {"select_input", "make_model", "fit", "predict", "evaluate", "log_run"}
