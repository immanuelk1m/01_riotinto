from copy import deepcopy
from pathlib import Path

import pytest

from src.experiments import check
from src.studies import load_study, ordered_experiments, select_experiments, spec_diff, validate_study

CATALOGUES = Path(__file__).resolve().parents[1] / "conf/studies"
CHAIN = ["base_xgb", "base_xgb_ffill", "base_xgb_ffill_scaled", "base_gaussian_process"]


@pytest.fixture
def study():
    return load_study(CATALOGUES / "gp_baseline_v1.yml")


def test_gp_catalogue_preserves_root_and_benchmarks_and_traces_each_axis(study):
    original = load_study(CATALOGUES / "p0_p1_v1.yml")
    specs = ordered_experiments(study)
    assert set(specs) == {*CHAIN, "naive", "drift"}
    assert len(original["experiments"]) == 14
    for name in ["base_xgb", "naive", "drift"]:
        assert spec_diff(original["experiments"][name], specs[name]) == []
    assert [name for name in specs if name in CHAIN] == CHAIN
    for name in CHAIN:
        spec = specs[name]
        assert spec["target"] == "level"
        assert spec["features"] == ["price_lags"]
        assert spec["feature_params"] == {"price_lags": {"lags": [1, 2, 4, 12, 52]}}
        assert spec["train_start"] is None
        assert "train_window_weeks" not in spec
        assert spec["lineage"]["primary_series"] == ["coffee", "copper", "gold"]
        assert spec["lineage"]["study_id"] == "gp_baseline_v1"
    for parent, child, axis in zip(CHAIN, CHAIN[1:], ["missing", "scaling", "model_family"]):
        lineage = specs[child]["lineage"]
        assert lineage["parent"] == parent
        assert lineage["axis"] == axis
        assert lineage["changed_paths"] == spec_diff(specs[parent], specs[child])
    assert specs[CHAIN[-1]]["model"] == "gaussian_process"
    assert specs[CHAIN[-1]]["model_params"]["normalize_y"] is True
    assert set(select_experiments(study, "anchor")) == {"base_xgb", "naive", "drift"}
    assert set(select_experiments(study, "features")) == set(specs)


@pytest.mark.parametrize("preprocessing", [None, [], {"unknown": "ffill"}, {"missing": "bfill"},
                                            {"scaling": "robust"}, {"scaling": "standard"},
                                            {"missing": "native", "scaling": "standard"}])
def test_invalid_preprocessing_is_rejected(preprocessing):
    with pytest.raises(ValueError, match="preprocessing|ffill"):
        check("invalid", {"model": "xgb", "preprocessing": preprocessing})


@pytest.mark.parametrize("preprocessing", [{}, {"missing": "native", "scaling": "none"},
                                            {"missing": "ffill"},
                                            {"missing": "ffill", "scaling": "standard"}])
def test_supported_preprocessing_and_legacy_defaults(preprocessing):
    check("legacy", {"model": "xgb"})
    check("supported", {"model": "xgb", "preprocessing": preprocessing})


@pytest.mark.parametrize("name,field,value", [
    ("base_xgb_ffill", "target", "logdiff"),
    ("base_xgb_ffill_scaled", "model_params", {}),
    ("base_gaussian_process", "target", "logdiff"),
])
def test_axis_rejects_unrelated_changes_even_when_declared(study, name, field, value):
    spec = study["experiments"][name]
    spec[field] = value
    parent = study["experiments"][spec["lineage"]["parent"]]
    spec["lineage"]["changed_paths"] = spec_diff(parent, spec)
    with pytest.raises(ValueError, match="축 외"):
        validate_study(study)


def test_missing_axis_cannot_also_change_effective_scaling(study):
    spec = study["experiments"]["base_xgb_ffill"]
    spec["preprocessing"]["scaling"] = "standard"
    with pytest.raises(ValueError, match="missing 축에서 scaling"):
        validate_study(study)


def test_declared_paths_cannot_hide_model_parameter_changes(study):
    spec = study["experiments"]["base_gaussian_process"]
    spec["lineage"]["changed_paths"] = ["model"]
    with pytest.raises(ValueError, match="changed_paths 불일치"):
        validate_study(study)


def test_model_family_step_keeps_preprocessing_identical(study):
    specs = study["experiments"]
    assert specs[CHAIN[-1]]["preprocessing"] == specs[CHAIN[-2]]["preprocessing"]
    changed = deepcopy(study)
    changed["experiments"][CHAIN[-1]]["preprocessing"]["scaling"] = "none"
    spec = changed["experiments"][CHAIN[-1]]
    spec["lineage"]["changed_paths"] = spec_diff(changed["experiments"][CHAIN[-2]], spec)
    with pytest.raises(ValueError, match="축 외"):
        validate_study(changed)
