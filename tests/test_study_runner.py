from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.pipelines.modeling import nodes as modeling
from src.studies import spec_diff
from src.study_runner import derive_forecasts, execute_study, validate_expected_cells


def tiny_study(tmp_path):
    (tmp_path / "conf/base").mkdir(parents=True)
    (tmp_path / "data/01_raw").mkdir(parents=True)
    (tmp_path / "uv.lock").write_text("test environment\n")
    parameters = {"targets": {"coffee": "test_price"}, "horizon": 2,
                  "origin_start": "2024-01-01", "ape_cap": 1.0}
    (tmp_path / "conf/base/parameters.yml").write_text(yaml.safe_dump(parameters))
    (tmp_path / "conf/base/parameters_exogenous.yml").write_text(yaml.safe_dump({"exogenous": {}}))
    weeks = pd.date_range("2020-01-05", periods=240, freq="W-SUN")
    prices = pd.DataFrame({"series": "coffee", "week": weeks,
                           "value": np.exp(np.cumsum(0.001 + 0.01 * np.sin(np.arange(len(weeks)))))})
    prices.to_parquet(tmp_path / "data/01_raw/prices.parquet", index=False)
    pd.DataFrame(columns=["series", "col_name", "week", "value", "code", "drivers", "ml"]).to_parquet(
        tmp_path / "data/01_raw/exogenous.parquet", index=False)
    lineage = {"study_id": "tiny", "root": "base_xgb", "parent": None, "axis": "root",
               "changed_paths": [], "primary_series": ["coffee"], "comparators": [], "kind": "fit"}
    base = {"model": "xgb", "features": ["price_lags"], "target": "level",
            "feature_params": {"price_lags": {"lags": [1, 2]}},
            "model_params": {"n_estimators": 2, "max_depth": 1, "tree_method": "hist",
                             "multi_strategy": "multi_output_tree", "random_state": 0},
            "study": "tiny.root", "lineage": lineage}
    specs = {"base_xgb": base}
    for name, prediction in [("rolling", None), ("shrink", {"method": "shrink", "source": "base_xgb", "alpha": 0.5}),
                             ("average", {"method": "mean", "sources": ["base_xgb", "rolling"]})]:
        spec = deepcopy(base)
        spec["lineage"].update(parent="base_xgb", axis="window" if prediction is None else
                               ("shrink" if name == "shrink" else "ensemble"),
                               kind="fit" if prediction is None else "derived")
        if prediction is None:
            spec["train_window_weeks"] = 52
        else:
            spec["prediction"] = prediction
        spec["lineage"]["changed_paths"] = spec_diff(base, spec)
        specs[name] = spec
    for model in ["naive", "drift"]:
        spec = {"model": model, "target": "logdiff", "study": "tiny.benchmark", "lineage": deepcopy(lineage)}
        spec["lineage"].update(kind="benchmark", axis="benchmark")
        specs[model] = spec
    study = {"study_id": "tiny", "root": "base_xgb", "experiments": specs,
             "phases": {"anchor": ["base_xgb", "naive", "drift"], "features": ["rolling"],
                        "adaptation": ["shrink", "average"]}}
    path = tmp_path / "study.yml"
    path.write_text(yaml.safe_dump(study, sort_keys=False))
    return path


def test_study_runs_real_kedro_nodes_preserves_artifacts_and_resumes_without_fit(tmp_path, monkeypatch):
    study = tiny_study(tmp_path)
    fitted = []
    original = modeling.fit

    def record_fit(model, *args):
        fitted.append(type(model).__name__)
        return original(model, *args)

    monkeypatch.setattr(modeling, "fit", record_fit)
    directory = execute_study(study, tmp_path, wandb_mode="disabled", bootstrap_samples=25)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["runs"]) == 6
    assert len(fitted) == 4  # two XGB, naive, drift; neither derived run fits
    assert all(r["status"] == "complete" for r in manifest["runs"].values())
    assert all((directory / "runs" / r["run_id"] / "forecasts.parquet").exists() for r in manifest["runs"].values())
    assert not (tmp_path / "data/07_reporting/summary.parquet").exists()
    assert not (tmp_path / "conf/base/parameters_experiments.yml").exists()
    comparisons = pd.read_parquet(directory / "comparisons.parquet")
    assert {"root", "naive", "parent"} <= set(comparisons.reference_role)
    assert comparisons.loc[comparisons.experiment == comparisons.reference, "delta_loss_pct"].eq(0).all()
    lineage = json.loads((directory / "lineage.json").read_text())
    assert all(row["root_run_id"] == manifest["runs"]["base_xgb"]["run_id"] for row in lineage)

    def no_fit(*args):
        raise AssertionError("completed run must not fit again")

    monkeypatch.setattr(modeling, "fit", no_fit)
    assert execute_study(study, tmp_path, wandb_mode="disabled", bootstrap_samples=25) == directory
    assert execute_study(study, tmp_path, phase="anchor", wandb_mode="disabled", bootstrap_samples=25) == directory
    assert pd.read_parquet(directory / "summary.parquet").experiment.nunique() == 6
    assert len(json.loads((directory / "lineage.json").read_text())) == 6
    shared = directory / "shared/tabular.parquet"
    original_shared = shared.read_bytes()
    altered = pd.read_parquet(shared)
    altered.loc[0, "y1"] = 123.0
    altered.to_parquet(shared)
    with pytest.raises(ValueError, match="공유 입력 산출물 손상"):
        execute_study(study, tmp_path, wandb_mode="disabled", bootstrap_samples=25)
    shared.write_bytes(original_shared)
    first_record = next(iter(manifest["runs"].values()))
    forecasts = directory / "runs" / first_record["run_id"] / "forecasts.parquet"
    forecasts.write_bytes(b"broken output")
    with pytest.raises(ValueError, match="산출물 손상"):
        execute_study(study, tmp_path, wandb_mode="disabled", bootstrap_samples=25)


def test_derived_endpoints_and_component_pairing_are_exact():
    frame = pd.DataFrame({"series": ["coffee", "coffee"], "origin": pd.to_datetime(["2024-01-07"] * 2),
                          "horizon": [1, 2], "fc": [110.0, 121.0], "actual": [108.0, 120.0], "base": [100.0] * 2})
    naive = derive_forecasts({"method": "shrink", "alpha": 0}, frame)
    unchanged = derive_forecasts({"method": "shrink", "alpha": 1}, frame)
    np.testing.assert_array_equal(naive.fc, frame.base)
    pd.testing.assert_frame_equal(unchanged, frame)
    mean = derive_forecasts({"method": "mean"}, frame, naive.iloc[::-1])
    np.testing.assert_allclose(mean.fc, [105, 110.5])
    with pytest.raises(ValueError, match="different actual"):
        derive_forecasts({"method": "mean"}, frame, naive.assign(actual=1))


def test_identical_experiment_in_another_study_reuses_the_finished_wandb_run(monkeypatch):
    from types import SimpleNamespace
    from src import study_runner

    spec = {"model": "naive", "target": "logdiff", "study": "a.benchmark", "lineage": {"study_id": "a"}}
    record = {"run_id": "local", "identity": {"data_sha": "d"}, "outputs": {"forecasts.parquet": "f"}}
    key = study_runner.dedupe_key("naive", spec, record)
    other_study = {**spec, "study": "b.benchmark", "lineage": {"study_id": "b"}}
    assert study_runner.dedupe_key("naive", other_study, record) == key
    assert study_runner.dedupe_key("naive", spec, {**record, "outputs": {"forecasts.parquet": "g"}}) != key

    seen = []
    monkeypatch.setattr(study_runner, "find_logged_run",
                        lambda k, own: seen.append((k, own)) or SimpleNamespace(id="kept", url="https://w/kept"))
    logged = study_runner.log_study_run("naive", spec, record, {}, None, None, None, None, "online", None)
    assert seen == [(key, "local")]
    assert logged == {"mode": "online", "logged": True, "url": "https://w/kept", "run_id": "kept", "reused": True}
    assert study_runner.wandb_run_id({"run_id": "local", "wandb": logged}) == "kept"
