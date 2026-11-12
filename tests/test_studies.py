from collections import Counter
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from src.experiments import check, experiments
from src.studies import load_study, ordered_experiments, select_experiments, spec_diff, validate_study

CATALOGUE = Path(__file__).resolve().parents[1] / "conf/studies/p0_p1_v1.yml"


@pytest.fixture
def study():
    return load_study(CATALOGUE)


def test_catalogue_has_the_planned_counts_and_every_ablation_reaches_base_xgb(study):
    specs = ordered_experiments(study)
    assert Counter(s["lineage"]["kind"] for s in specs.values()) == {"fit": 10, "derived": 2, "benchmark": 2}
    positions = {name: i for i, name in enumerate(specs)}
    for name, spec in specs.items():
        if spec["lineage"]["kind"] == "benchmark":
            assert spec["lineage"]["parent"] is None
            continue
        ancestor = name
        while (parent := specs[ancestor]["lineage"]["parent"]) is not None:
            assert positions[parent] < positions[ancestor]
            ancestor = parent
        assert ancestor == "base_xgb"
    for name, spec in specs.items():
        for source in spec.get("prediction", {}).get("sources", []):
            assert positions[source] < positions[name]


def test_root_and_regularization_steps_are_explicit_and_separate(study):
    specs = study["experiments"]
    root = specs["base_xgb"]
    assert root["target"] == "level" and root["features"] == ["price_lags"]
    assert root["feature_params"]["price_lags"]["lags"] == [1, 2, 4, 12, 52]
    assert {k: root["model_params"][k] for k in ("max_depth", "learning_rate", "min_child_weight", "n_estimators")} == {
        "max_depth": 6, "learning_rate": 0.3, "min_child_weight": 1, "n_estimators": 100,
    }
    for child, parameter in [("xgb_ret8_depth2", "max_depth"), ("xgb_ret8_depth2_lr003", "learning_rate"),
                             ("xgb_ret8_reg", "min_child_weight")]:
        assert spec_diff(specs[specs[child]["lineage"]["parent"]], specs[child]) == [f"model_params.{parameter}"]


def test_phase_selection_includes_ancestors_components_and_comparators(study):
    subset = deepcopy(study)
    subset["phases"] = {"one": ["xgb_ret8_reg_2015_shrink025"]}
    selected = select_experiments(subset, "one")
    assert {"base_xgb", "naive", "xgb_ret8_reg", "xgb_ret8_reg_2015", "xgb_ret8_reg_2015_shrink025"} <= selected.keys()
    assert "drivers_all" not in selected and "drift" not in selected
    assert len(select_experiments(study, "anchor")) == 10
    with pytest.raises(ValueError, match="모르는 연구 단계"):
        select_experiments(study, "missing")


def test_unknown_parent_and_component_are_rejected(study):
    study["experiments"]["drivers_all"]["lineage"]["parent"] = "missing"
    with pytest.raises(ValueError, match="없는 부모/의존"):
        validate_study(study)
    study = load_study(CATALOGUE)
    study["experiments"]["xgb_ret8_reg_2015_roll156"]["lineage"]["comparators"] = ["missing"]
    with pytest.raises(ValueError, match="없는 부모/의존"):
        validate_study(study)


def test_parent_and_prediction_dependency_cycles_are_rejected(study):
    study["experiments"]["xgb_ret8_reg_2015"]["lineage"]["parent"] = "xgb_ret8_reg_2015_roll156"
    with pytest.raises(ValueError, match="순환"):
        validate_study(study)
    study = load_study(CATALOGUE)
    study["experiments"]["xgb_ret8_reg_2015"]["lineage"]["comparators"] = ["xgb_ret8_reg_2015_roll156"]
    with pytest.raises(ValueError, match="순환"):
        validate_study(study)


def test_unrecorded_changes_and_wrong_axis_are_rejected(study):
    spec = study["experiments"]["xgb_ret8_depth2"]
    spec["model_params"]["learning_rate"] = 0.05
    with pytest.raises(ValueError, match="changed_paths 불일치"):
        validate_study(study)
    spec["lineage"]["changed_paths"] = spec_diff(study["experiments"]["xgb_ret8"], spec)
    with pytest.raises(ValueError, match="축 외의 변경"):
        validate_study(study)


def test_wrong_root_and_nonroot_without_parent_are_rejected(study):
    study["experiments"]["drivers_all"]["lineage"]["root"] = "other"
    with pytest.raises(ValueError, match="root 불일치"):
        validate_study(study)
    study = load_study(CATALOGUE)
    study["experiments"]["drivers_all"]["lineage"]["parent"] = None
    with pytest.raises(ValueError, match="부모가 없다"):
        validate_study(study)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, True, float("nan"), "0.5"])
def test_invalid_shrink_alpha_is_rejected(study, alpha):
    study["experiments"]["xgb_ret8_reg_2015_shrink025"]["prediction"]["alpha"] = alpha
    with pytest.raises(ValueError, match="shrink source/alpha/axis"):
        validate_study(study)


def test_derived_cannot_disguise_a_fit_and_benchmarks_cannot_be_ablation_parents(study):
    study["experiments"]["xgb_ret8_reg_2015_shrink025"]["lineage"]["kind"] = "fit"
    with pytest.raises(ValueError, match="prediction"):
        validate_study(study)
    study = load_study(CATALOGUE)
    study["experiments"]["drivers_all"]["lineage"]["parent"] = "naive"
    with pytest.raises(ValueError, match="benchmark.*부모"):
        validate_study(study)


def test_yaml_anchors_and_selected_specs_do_not_share_mutable_model_settings(study):
    study["experiments"]["base_xgb"]["model_params"]["max_depth"] = 99
    assert study["experiments"]["xgb_logdiff"]["model_params"]["max_depth"] == 6
    fresh = load_study(CATALOGUE)
    selected = select_experiments(fresh)
    selected["base_xgb"]["model_params"]["max_depth"] = 99
    assert fresh["experiments"]["base_xgb"]["model_params"]["max_depth"] == 6


def test_spec_diff_handles_added_empty_dicts_lists_and_metadata():
    left = {"features": ["lags"], "feature_params": {}, "study": "a", "lineage": {"parent": None}}
    right = {"features": ["lags", "exog"], "feature_params": {"exog": {}}, "study": "b", "lineage": {}}
    assert spec_diff(left, right) == ["feature_params", "feature_params.exog", "features"]
    assert spec_diff(left, dict(left, study="other", lineage={"changed": True})) == []


def test_prepare_queue_is_separate_and_legacy_configuration_remains_valid(tmp_path):
    existing = Path("conf/base/parameters_experiments.yml")
    original = existing.read_bytes()
    output = tmp_path / "queue.yml"
    result = subprocess.run([sys.executable, "tools/prepare_study.py", "--phase", "anchor", "--output", str(output)],
                            check=True, capture_output=True, text=True)
    assert "10 experiments" in result.stdout
    assert len(yaml.safe_load(output.read_text())["experiments"]) == 10
    assert existing.read_bytes() == original
    for name, spec in experiments().items():
        check(name, spec)


def test_prepare_rejects_overwriting_current_queue():
    existing = Path("conf/base/parameters_experiments.yml")
    original = existing.read_bytes()
    result = subprocess.run([sys.executable, "tools/prepare_study.py", "--output", str(existing)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "덮어쓰지 않는다" in result.stderr
    assert existing.read_bytes() == original
