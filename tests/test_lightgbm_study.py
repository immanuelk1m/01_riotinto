from copy import deepcopy
from pathlib import Path

import pytest

from src.experiments import check
from src.studies import load_study, select_experiments, spec_diff, validate_study

CATALOGUES = Path(__file__).resolve().parents[1] / "conf/studies"
NEW = ["xgb_ret8_reg_2015_ffill", "xgb_direct", "base_lightgbm_direct"]


@pytest.fixture
def study():
    return load_study(CATALOGUES / "lightgbm_direct_v1.yml")


def test_catalogue_preserves_anchors_and_complete_lineage(study):
    original = load_study(CATALOGUES / "p0_p1_v1.yml")
    gp = load_study(CATALOGUES / "gp_baseline_v1.yml")
    assert len(original["experiments"]) == 14
    assert len(gp["experiments"]) == 6
    specs = study["experiments"]
    assert len(specs) == 13
    assert study["phases"]["anchor"] == original["phases"]["anchor"]
    for name in study["phases"]["anchor"]:
        assert spec_diff(original["experiments"][name], specs[name]) == []
    assert set(select_experiments(study, "features")) == set(specs)
    assert len(select_experiments(study, "anchor")) == 10
    for name in NEW:
        lineage = specs[name]["lineage"]
        assert lineage["primary_series"] == ["coffee", "copper", "gold"]
        assert lineage["comparators"] == ["drift"]
        assert lineage["changed_paths"] == spec_diff(specs[lineage["parent"]], specs[name])
        ancestor = name
        while specs[ancestor]["lineage"]["parent"] is not None:
            ancestor = specs[ancestor]["lineage"]["parent"]
        assert ancestor == "base_xgb"
        assert specs[name]["target"] == "logdiff"
        assert specs[name]["train_start"] == "2015-01-04"
        assert specs[name]["preprocessing"] == {"missing": "ffill", "scaling": "none"}
    assert specs["xgb_direct"]["lineage"]["changed_paths"] == [
        "forecast_strategy", "model_params.multi_strategy"]
    assert specs["base_lightgbm_direct"]["model_params"]["n_jobs"] == 1


@pytest.mark.parametrize("strategy", [None, "recursive", "direct-h-mature", True])
def test_invalid_strategy_rejected(study, strategy):
    spec = deepcopy(study["experiments"]["xgb_direct"])
    spec["forecast_strategy"] = strategy
    with pytest.raises(ValueError, match="forecast_strategy"):
        check("invalid", spec)


def test_lightgbm_requires_direct_and_xgb_omits_multi_strategy(study):
    spec = deepcopy(study["experiments"]["base_lightgbm_direct"])
    del spec["forecast_strategy"]
    with pytest.raises(ValueError, match="lightgbm.*direct"):
        check("invalid", spec)
    spec = deepcopy(study["experiments"]["xgb_direct"])
    spec["model_params"]["multi_strategy"] = "one_output_per_tree"
    with pytest.raises(ValueError, match="multi_strategy"):
        check("invalid", spec)


@pytest.mark.parametrize("field,value", [
    ("train_window_weeks", 156), ("train_start", "2020-01-05"),
    ("label_maturity", "per_horizon"), ("forecast_strategy", "mimo"),
])
def test_strategy_axis_cannot_hide_other_changes(study, field, value):
    spec = study["experiments"]["xgb_direct"]
    spec[field] = value
    spec["lineage"]["changed_paths"] = spec_diff(
        study["experiments"]["xgb_ret8_reg_2015_ffill"], spec)
    with pytest.raises(ValueError):
        validate_study(study)


def test_strategy_axis_cannot_change_model_parameters(study):
    spec = study["experiments"]["xgb_direct"]
    spec["model_params"]["max_depth"] = 4
    spec["lineage"]["changed_paths"] = spec_diff(
        study["experiments"]["xgb_ret8_reg_2015_ffill"], spec)
    with pytest.raises(ValueError, match="축 외의 변경"):
        validate_study(study)


def test_changed_paths_must_match_actual_spec(study):
    study["experiments"]["xgb_direct"]["lineage"]["changed_paths"] = ["forecast_strategy"]
    with pytest.raises(ValueError, match="changed_paths"):
        validate_study(study)


def test_strategy_axis_cannot_change_model_family(study):
    spec = study["experiments"]["xgb_direct"]
    spec["model"] = "lightgbm"
    spec["lineage"]["changed_paths"] = spec_diff(
        study["experiments"]["xgb_ret8_reg_2015_ffill"], spec)
    with pytest.raises(ValueError, match="축 외의 변경"):
        validate_study(study)
