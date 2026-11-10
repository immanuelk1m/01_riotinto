from kedro.pipeline import Node, Pipeline

from . import nodes as n


def create_pipeline(**kwargs) -> Pipeline:
    """피처와 모델 입력: 모든 실험이 공유하는 두 파일(tabular·timeseries)."""
    return Pipeline([
        Node(n.build_features,
             ["targets", "raw_exogenous", "params:experiments", "params:exogenous", "params:horizon"],
             ["features", "feature_columns"], name="build_features"),
        Node(n.make_tabular, ["features", "targets"], "tabular", name="make_tabular"),
        Node(n.make_timeseries, ["features", "targets"], "timeseries", name="make_timeseries"),
    ])
