"""실험 목록(`parameters_experiments.yml`)과 실험별 namespace. modeling·reporting 파이프라인이 함께 쓴다."""
from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

from src.features import FEATURES
from src.models import MODELS

SPEC_KEYS = {"study", "model", "features", "model_params", "feature_params"}


def experiments() -> dict[str, dict]:
    # ponytail: 파이프라인 생성 시점엔 실행 env 를 모른다 — base + local 만 읽고 `--env` 는 반영되지 않는다.
    conf = OmegaConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE), base_env="base", default_run_env="local")
    specs = conf["parameters"]["experiments"]
    for name, spec in specs.items():
        if unknown := set(spec) - SPEC_KEYS:
            raise ValueError(f"실험 {name}: 모르는 키 {sorted(unknown)} (허용: {sorted(SPEC_KEYS)})")
        if spec["model"] not in MODELS:
            raise ValueError(f"실험 {name}: 미등록 모델 {spec['model']!r} (등록: {sorted(MODELS)})")
        if unknown := set(spec.get("features", [])) - set(FEATURES):
            raise ValueError(f"실험 {name}: 미등록 피처 {sorted(unknown)} (등록: {sorted(FEATURES)})")
    return specs


def namespace(name: str, spec: dict) -> str:
    """`<입력 형식>.<모델 카테고리>.<실험>` — 앞 두 마디가 05·06 산출물의 폴더다.

    카테고리 = 모델이 들어 있는 `models/` 하위 폴더, 입력 형식 = 모델 클래스의 `input`(없으면 tabular).
    """
    model = MODELS[spec["model"]]
    return f"{getattr(model, 'input', 'tabular')}.{model.__module__.split('.')[2]}.{name}"
