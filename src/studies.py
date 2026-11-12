"""고정 연구 명세의 계보와 변경 축을 검증하고 실행 대기열을 고른다."""
from copy import deepcopy
from pathlib import Path

import yaml

from src.experiments import check

LINEAGE_KEYS = {"study_id", "root", "parent", "axis", "changed_paths", "primary_series", "comparators", "kind"}
SERIES = {"coffee", "copper", "gold"}
AXIS_PATHS = {
    "target": ("target",),
    "representation": ("features", "feature_params.price_lags", "feature_params.lags"),
    "lag_schedule": ("feature_params.lags.offsets",),
    "depth": ("model_params.max_depth",),
    "learning_rate": ("model_params.learning_rate",),
    "min_child_weight": ("model_params.min_child_weight",),
    "train_start": ("train_start",),
    "exog_columns": ("features", "feature_params.exog"),
    "exog_diff": ("feature_params.exog.diff",),
    "drivers_basket": ("features", "feature_params.exog"),
    "window": ("train_window_weeks",),
    "shrink": ("prediction",),
    "ensemble": ("prediction",),
    "missing": ("preprocessing.missing", "preprocessing.scaling"),
    "scaling": ("preprocessing.scaling",),
    "model_family": ("model", "model_params"),
    "forecast_strategy": ("forecast_strategy", "model_params.multi_strategy"),
}


def _flatten(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            result.update(_flatten(item, path))
        else:
            result[path] = item
    return result


def spec_diff(parent: dict, child: dict) -> list[str]:
    """메타데이터를 제외한 실제 설정 차이. 목록은 하나의 값으로 비교한다."""
    left, right = (_flatten({k: v for k, v in s.items() if k not in {"study", "lineage"}})
                   for s in (parent, child))
    missing = object()
    return sorted(k for k in left.keys() | right.keys() if left.get(k, missing) != right.get(k, missing))


def _dependencies(spec: dict) -> list[str]:
    lineage = spec["lineage"]
    dependencies = [lineage["parent"]] if lineage["parent"] is not None else []
    prediction = spec.get("prediction", {})
    if prediction.get("method") == "shrink":
        dependencies.append(prediction["source"])
    elif prediction.get("method") == "mean":
        dependencies.extend(prediction["sources"])
    dependencies.extend(lineage["comparators"])
    return list(dict.fromkeys(dependencies))


def _order(specs: dict[str, dict]) -> list[str]:
    ordered, visiting, visited = [], set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError(f"연구 계보/예측 의존성 순환: {name}")
        if name in visited:
            return
        if name not in specs:
            raise ValueError(f"연구에 없는 부모/의존 실험: {name}")
        visiting.add(name)
        for dependency in _dependencies(specs[name]):
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        ordered.append(name)

    # 참고 기준선은 계보상의 학습 부모가 아니지만 비교 전에 실행한다.
    for name, spec in specs.items():
        if spec["lineage"]["kind"] == "benchmark":
            visit(name)
    for name in specs:
        visit(name)
    return ordered


def _validate_prediction(name: str, spec: dict) -> None:
    lineage, prediction = spec["lineage"], spec.get("prediction")
    if lineage["kind"] != "derived":
        if prediction is not None:
            raise ValueError(f"실험 {name}: 학습/benchmark 실험에는 prediction 을 지정할 수 없다")
        return
    if not isinstance(prediction, dict):
        raise ValueError(f"실험 {name}: derived 실험에 prediction 이 필요하다")
    method = prediction.get("method")
    if method == "shrink":
        alpha = prediction.get("alpha")
        if (set(prediction) != {"method", "source", "alpha"}
                or not isinstance(prediction.get("source"), str)
                or isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1
                or prediction["source"] != lineage["parent"] or lineage["axis"] != "shrink"):
            raise ValueError(f"실험 {name}: 잘못된 shrink source/alpha/axis")
    elif method == "mean":
        sources = prediction.get("sources")
        if (set(prediction) != {"method", "sources"} or not isinstance(sources, list)
                or len(sources) < 2 or not all(isinstance(s, str) for s in sources)
                or len(sources) != len(set(sources)) or lineage["parent"] not in sources
                or lineage["axis"] != "ensemble"):
            raise ValueError(f"실험 {name}: 잘못된 mean sources/axis")
    else:
        raise ValueError(f"실험 {name}: 모르는 prediction method {method!r}")


def validate_study(study: dict) -> None:
    """계보, 한 축의 실제 차이, 모델 등록과 파생 예측 의존성을 실행 전에 검사한다."""
    if not isinstance(study, dict) or not isinstance(study.get("study_id"), str) or not study["study_id"]:
        raise ValueError("study_id 가 있는 연구 명세가 필요하다")
    specs, root = study.get("experiments"), study.get("root")
    if not isinstance(specs, dict) or not specs or root not in specs:
        raise ValueError("연구 experiments 에 root 실험이 있어야 한다")
    for name, spec in specs.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            raise ValueError("실험 이름과 spec 이 올바르지 않다")
        lineage = spec.get("lineage")
        if not isinstance(lineage, dict) or set(lineage) != LINEAGE_KEYS:
            raise ValueError(f"실험 {name}: lineage 필드가 올바르지 않다 (필수: {sorted(LINEAGE_KEYS)})")
        check(name, spec)
        if lineage["study_id"] != study["study_id"] or lineage["root"] != root:
            raise ValueError(f"실험 {name}: study_id/root 불일치")
        if lineage["kind"] not in {"fit", "derived", "benchmark"}:
            raise ValueError(f"실험 {name}: 모르는 kind")
        primary, comparisons, changes = (lineage[k] for k in ("primary_series", "comparators", "changed_paths"))
        if (not isinstance(primary, list) or not primary or not all(isinstance(s, str) for s in primary)
                or len(primary) != len(set(primary)) or not set(primary) <= SERIES):
            raise ValueError(f"실험 {name}: primary_series 가 올바르지 않다")
        if (not isinstance(comparisons, list) or not all(isinstance(c, str) for c in comparisons)
                or len(comparisons) != len(set(comparisons)) or name in comparisons):
            raise ValueError(f"실험 {name}: comparators 가 올바르지 않다")
        if (not isinstance(changes, list) or not all(isinstance(c, str) for c in changes)
                or len(changes) != len(set(changes))):
            raise ValueError(f"실험 {name}: changed_paths 가 올바르지 않다")
        if lineage["parent"] is not None and not isinstance(lineage["parent"], str):
            raise ValueError(f"실험 {name}: parent 는 실험 이름 또는 null 이어야 한다")
        if name == root:
            if lineage["parent"] is not None or lineage["kind"] != "fit" or lineage["axis"] != "root" or changes:
                raise ValueError("root 는 부모·변경 경로가 없는 fit 실험이어야 한다")
        elif lineage["kind"] == "benchmark":
            if lineage["parent"] is not None or lineage["axis"] != "benchmark" or changes:
                raise ValueError(f"실험 {name}: benchmark 는 ablation 부모·변경 경로를 갖지 않는다")
        elif lineage["parent"] is None:
            raise ValueError(f"실험 {name}: ablation 부모가 없다")
        _validate_prediction(name, spec)

    ordered = _order(specs)
    for name in ordered:
        spec, lineage = specs[name], specs[name]["lineage"]
        parent_name = lineage["parent"]
        if parent_name is None:
            continue
        parent = specs[parent_name]
        if parent["lineage"]["kind"] == "benchmark":
            raise ValueError(f"실험 {name}: benchmark 를 XGB ablation 의 부모로 사용할 수 없다")
        axis = lineage["axis"]
        if axis not in AXIS_PATHS:
            raise ValueError(f"실험 {name}: 모르는 변경 축 {axis!r}")
        actual = spec_diff(parent, spec)
        if not actual or actual != sorted(lineage["changed_paths"]):
            raise ValueError(f"실험 {name}: changed_paths 불일치 (실제: {actual})")
        if any(not any(path == prefix or path.startswith(prefix + ".") for prefix in AXIS_PATHS[axis])
               for path in actual):
            raise ValueError(f"실험 {name}: 선언한 {axis} 축 외의 변경이 있다: {actual}")
        if axis == "missing" and (parent.get("preprocessing", {}).get("scaling", "none")
                                  != spec.get("preprocessing", {}).get("scaling", "none")):
            raise ValueError(f"실험 {name}: missing 축에서 scaling 을 변경할 수 없다")
        if axis == "forecast_strategy":
            if (parent.get("forecast_strategy", "mimo") != "mimo"
                    or spec.get("forecast_strategy", "mimo") != "direct"
                    or parent["model"] != spec["model"]):
                raise ValueError(f"실험 {name}: forecast_strategy 축은 동일 모델의 mimo → direct 만 허용한다")
            if ("model_params.multi_strategy" in actual
                    and "multi_strategy" in spec.get("model_params", {})):
                raise ValueError(f"실험 {name}: forecast_strategy 축은 multi_strategy 제거만 허용한다")
        ancestor = parent_name
        while specs[ancestor]["lineage"]["parent"] is not None:
            ancestor = specs[ancestor]["lineage"]["parent"]
        if ancestor != root:
            raise ValueError(f"실험 {name}: root 에 연결되지 않았다")

    phases = study.get("phases", {})
    if not isinstance(phases, dict):
        raise ValueError("phases 는 단계 이름과 실험 목록의 mapping 이어야 한다")
    for phase, names in phases.items():
        if (not isinstance(phase, str) or not isinstance(names, list) or not names
                or not all(isinstance(n, str) for n in names) or len(names) != len(set(names))
                or not set(names) <= specs.keys()):
            raise ValueError(f"잘못된 연구 단계: {phase!r}")


def load_study(path: str | Path) -> dict:
    study = yaml.safe_load(Path(path).read_text())
    if isinstance(study, dict) and isinstance(study.get("experiments"), dict):
        # YAML anchors share objects; mutations of one selected spec must not change another.
        study["experiments"] = {name: deepcopy(spec) for name, spec in study["experiments"].items()}
    validate_study(study)
    return study


def ordered_experiments(study: dict) -> dict[str, dict]:
    validate_study(study)
    return {name: deepcopy(study["experiments"][name]) for name in _order(study["experiments"])}


def select_experiments(study: dict, phase: str | None = None) -> dict[str, dict]:
    """선택 단계와 부모·component·보조 비교 실험 및 naive 기준선을 함께 돌려준다."""
    ordered = ordered_experiments(study)
    if phase is None:
        return ordered
    if phase not in study.get("phases", {}):
        raise ValueError(f"모르는 연구 단계: {phase!r}")
    wanted = set()

    def include(name):
        if name in wanted:
            return
        wanted.add(name)
        for dependency in _dependencies(ordered[name]):
            include(dependency)

    for name in study["phases"][phase]:
        include(name)
    include(study["root"])
    for name, spec in ordered.items():
        if spec["lineage"]["kind"] == "benchmark" and spec["model"] == "naive":
            include(name)
    return {name: spec for name, spec in ordered.items() if name in wanted}
