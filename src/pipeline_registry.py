from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """`pipelines/<이름>/` 을 모두 찾아 등록한다.

    `data`(= data_ingestion, Postgres → data/01_raw)는 스냅샷 갱신 때만 돌리므로 `__default__` 에서 뺀다.
    """
    pipelines = find_pipelines(raise_errors=True)
    pipelines.pop("__default__")
    pipelines["data"] = pipelines.pop("data_ingestion")
    pipelines["__default__"] = sum((p for name, p in pipelines.items() if name != "data"), Pipeline([]))
    return pipelines
