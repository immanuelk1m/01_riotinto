"""실험 목록(`parameters_experiments.yml`)과 실험별 namespace. modeling·reporting 파이프라인이 함께 쓴다."""
from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

from src.features import FEATURES
from src.models import MODELS
from src.utils import DEFAULT_TARGET, TARGETS

SPEC_KEYS = {"study", "model", "features", "model_params", "feature_params", "target", "train_start",
             "lineage", "train_window_weeks", "prediction", "preprocessing", "forecast_strategy"}


def experiments() -> dict[str, dict]:
    # ponytail: 파이프라인 생성 시점엔 실행 env 를 모른다 — base + local 만 읽고 `--env` 는 반영되지 않는다.
    conf = OmegaConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE), base_env="base", default_run_env="local")
    specs = conf["parameters"]["experiments"]
    for name, spec in specs.items():
        check(name, spec)
        if "prediction" in spec:
            raise ValueError(f"파생 연구 실험 {name}: tools/run_study.py 로 실행한다 (일반 파이프라인 재학습 금지)")
    return specs


def check(name: str, spec: dict) -> None:
    if unknown := set(spec) - SPEC_KEYS:
        raise ValueError(f"실험 {name}: 모르는 키 {sorted(unknown)} (허용: {sorted(SPEC_KEYS)})")
    if spec["model"] not in MODELS:
        raise ValueError(f"실험 {name}: 미등록 모델 {spec['model']!r} (등록: {sorted(MODELS)})")
    strategy = spec.get("forecast_strategy", "mimo")
    if strategy not in ("mimo", "direct"):
        raise ValueError(f"실험 {name}: forecast_strategy 는 mimo 또는 direct 여야 한다")
    if spec["model"] == "lightgbm" and strategy != "direct":
        raise ValueError(f"실험 {name}: lightgbm 은 forecast_strategy direct 를 요구한다")
    if (spec["model"] == "xgb" and strategy == "direct"
            and "multi_strategy" in spec.get("model_params", {})):
        raise ValueError(f"실험 {name}: direct XGB 는 multi_strategy 를 생략해야 한다")
    if unknown := set(spec.get("features", [])) - set(FEATURES):
        raise ValueError(f"실험 {name}: 미등록 피처 {sorted(unknown)} (등록: {sorted(FEATURES)})")
    target = spec.get("target", DEFAULT_TARGET)
    if target not in TARGETS:
        raise ValueError(f"실험 {name}: 모르는 target {target!r} (허용: {sorted(TARGETS)})")
    if (fixed := getattr(MODELS[spec["model"]], "target", target)) != target:
        raise ValueError(f"실험 {name}: 모델 {spec['model']!r} 은 target {fixed!r} 로만 예측한다")
    if "train_window_weeks" in spec:
        window = spec["train_window_weeks"]
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError(f"실험 {name}: train_window_weeks 는 양의 정수여야 한다")
    if "prediction" in spec and spec.get("lineage", {}).get("kind") != "derived":
        raise ValueError(f"실험 {name}: prediction 은 계보가 있는 derived 실험에만 허용된다")
    if "preprocessing" in spec:
        preprocessing = spec["preprocessing"]
        if not isinstance(preprocessing, dict) or set(preprocessing) - {"missing", "scaling"}:
            raise ValueError(f"실험 {name}: preprocessing 은 missing/scaling 키만 허용한다")
        missing = preprocessing.get("missing", "native")
        scaling = preprocessing.get("scaling", "none")
        if missing not in ("native", "ffill") or scaling not in ("none", "standard"):
            raise ValueError(f"실험 {name}: 잘못된 preprocessing missing/scaling 값")
        if scaling == "standard" and missing != "ffill":
            raise ValueError(f"실험 {name}: standard scaling 은 ffill 을 요구한다")


def namespace(name: str, spec: dict) -> str:
    """`<입력 형식>.<모델 카테고리>.<실험>` — 앞 두 마디가 05·06 산출물의 폴더다.

    카테고리 = 모델이 들어 있는 `models/` 하위 폴더, 입력 형식 = 모델 클래스의 `input`(없으면 tabular).
    """
    model = MODELS[spec["model"]]
    return f"{getattr(model, 'input', 'tabular')}.{model.__module__.split('.')[2]}.{name}"
