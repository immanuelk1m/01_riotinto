"""고정 명세를 Kedro 노드로 실행하고 run별 산출물·비교·W&B 계보를 보존한다."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import Node, Pipeline
from kedro.runner import SequentialRunner
from kedro_datasets.json import JSONDataset
from kedro_datasets.pandas import ParquetDataset
from kedro_datasets.pickle import PickleDataset
from threadpoolctl import threadpool_limits

from src.experiments import namespace
from src.pipelines.data_processing.nodes import make_targets
from src.pipelines.feature_engineering.nodes import build_features, make_tabular, make_timeseries
from src.pipelines.modeling import nodes as modeling
from src.pipelines.reporting.nodes import _features, _sha, evaluate, run_summary
from src.studies import load_study, ordered_experiments, select_experiments, spec_diff
from src.study_reporting import aligned_forecasts, annual_comparison, compare_forecasts, holm_adjust

LOG = logging.getLogger(__name__)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def source_files(root: Path) -> list[Path]:
    return sorted([*root.glob("src/**/*.py"), *root.glob("tools/*.py")])


def source_fingerprint(root: Path) -> str:
    return digest({str(p.relative_to(root)): file_sha(p) for p in source_files(root)})


def environment() -> dict:
    packages = ["kedro", "kedro-datasets", "numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "lightgbm", "wandb"]
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in packages}}


def fit_with_audit(model, model_input, spec, horizon, exogenous, checkpoint_dir=None):
    if spec.get("lineage", {}).get("study_id") == "lightgbm_direct_v1":
        with threadpool_limits(limits=1):
            fitted, audit = _fit_with_audit(model, model_input, spec, horizon, exogenous, checkpoint_dir)
        for row in audit:
            row["computational_threads"] = 1
        return fitted, audit
    return _fit_with_audit(model, model_input, spec, horizon, exogenous, checkpoint_dir)


def _fit_with_audit(model, model_input, spec, horizon, exogenous, checkpoint_dir=None):
    if checkpoint_dir is not None and spec["model"] == "gaussian_process":
        probe = model_input.copy()
        ready = probe.is_origin & probe[[f"y{h}" for h in range(1, horizon + 1)]].notna().all(axis=1)
        first = probe.loc[ready].groupby("series").week.min()
        probe["is_origin"] = ready & probe.week.eq(probe.series.map(first))
        calibration = modeling.fit(model, probe, spec, horizon, exogenous, checkpoint_dir=checkpoint_dir)
        print("GP_CALIBRATION " + json.dumps({s: fitted.study_model_info for (s, _), fitted in calibration.items()}), flush=True)
        del calibration, probe
        fitted = modeling.fit(model, model_input, spec, horizon, exogenous, checkpoint_dir=checkpoint_dir)
    else:
        fitted = modeling.fit(model, model_input, spec, horizon, exogenous)
    audit = []
    for (series, origin), trained in fitted.items():
        info = dict(trained.study_training_info)
        info.update(series=series, origin=str(origin.date()),
                    feature_columns=trained.study_feature_columns)
        if hasattr(trained, "get_booster"):
            config = json.loads(trained.get_booster().save_config())["learner"]
            info["effective_xgb_parameters"] = {
                **config["gradient_booster"]["tree_train_param"],
                **config["learner_train_param"],
                "num_trees": config["gradient_booster"]["gbtree_model_param"]["num_trees"],
                "device": config["generic_param"]["device"],
                "seed": config["generic_param"]["seed"],
            }
        audit.append(info)
    return fitted, audit


def derive_forecasts(prediction: dict, *frames: pd.DataFrame) -> pd.DataFrame:
    if not frames:
        raise ValueError("파생 예측의 source 가 없다")
    first, _ = aligned_forecasts(frames[0], frames[0])
    method = prediction["method"]
    if method == "shrink":
        if len(frames) != 1:
            raise ValueError("shrink 에는 source 하나만 허용한다")
        alpha = prediction["alpha"]
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha 는 0..1 이어야 한다")
        if alpha == 0:
            first["fc"] = first.base
        elif alpha != 1:
            first["fc"] = first.base * np.exp(alpha * np.log(first.fc / first.base))
    elif method == "mean":
        aligned = [aligned_forecasts(first, frame)[1] for frame in frames]
        first["fc"] = np.mean([frame.fc.to_numpy() for frame in aligned], axis=0)
    else:
        raise ValueError(f"모르는 파생 예측: {method}")
    aligned_forecasts(first, first)
    return first


def _memory(value):
    return MemoryDataset(data=value, copy_mode="assign")


def run_shared_inputs(directory: Path, specs: dict, parameters: dict, exogenous_spec: dict) -> None:
    """일반 실행의 data/03·04를 건드리지 않는 동일한 공유 입력 노드."""
    datasets = {
        "raw_prices": ParquetDataset(filepath=str(directory / "inputs/prices.parquet")),
        "raw_exogenous": ParquetDataset(filepath=str(directory / "inputs/exogenous.parquet")),
        "horizon": _memory(parameters["horizon"]), "origin_start": _memory(parameters["origin_start"]),
        "specs": _memory(specs), "exogenous_spec": _memory(exogenous_spec),
        "targets": ParquetDataset(filepath=str(directory / "shared/targets.parquet")),
        "feature_columns": JSONDataset(filepath=str(directory / "shared/feature_columns.json")),
        "tabular": ParquetDataset(filepath=str(directory / "shared/tabular.parquet")),
        "timeseries": ParquetDataset(filepath=str(directory / "shared/timeseries.parquet")),
    }
    pipeline = Pipeline([
        Node(make_targets, ["raw_prices", "horizon", "origin_start"], "targets", name="make_targets"),
        Node(build_features, ["targets", "raw_exogenous", "specs", "exogenous_spec", "horizon"],
             ["features", "feature_columns"], name="build_features"),
        Node(make_tabular, ["features", "targets"], "tabular", name="make_tabular"),
        Node(make_timeseries, ["features", "targets"], "timeseries", name="make_timeseries"),
    ])
    SequentialRunner().run(pipeline, DataCatalog(datasets))


def run_experiment(name: str, spec: dict, directory: Path, shared: Path, parameters: dict,
                   exogenous_spec: dict, source_paths: dict[str, Path]) -> None:
    datasets = {
        "tabular": ParquetDataset(filepath=str(shared / "tabular.parquet")),
        "feature_columns": JSONDataset(filepath=str(shared / "feature_columns.json")),
        "spec": _memory(spec), "horizon": _memory(parameters["horizon"]),
        "exogenous_spec": _memory(exogenous_spec), "ape_cap": _memory(parameters["ape_cap"]),
        "forecasts": ParquetDataset(filepath=str(directory / "forecasts.parquet")),
        "metrics": ParquetDataset(filepath=str(directory / "metrics.parquet")),
    }
    if "prediction" in spec:
        sources = []
        for i, path in enumerate(source_paths.values()):
            key = f"source_{i}"
            datasets[key] = ParquetDataset(filepath=str(path))
            sources.append(key)
        datasets["prediction"] = _memory(spec["prediction"])
        nodes = [Node(derive_forecasts, ["prediction", *sources], "forecasts", name="derive")]
    else:
        datasets.update({
            "fitted": PickleDataset(filepath=str(directory / "fitted.pkl")),
            "training": JSONDataset(filepath=str(directory / "training.json")),
        })
        nodes = [
            Node(modeling.select_input, ["tabular", "feature_columns", "spec", "horizon"],
                 "model_input", name="select_input"),
            Node(modeling.make_model, "spec", "model", name="make_model"),
            Node(partial(fit_with_audit, checkpoint_dir=directory / "checkpoints"),
                 ["model", "model_input", "spec", "horizon", "exogenous_spec"],
                 ["fitted", "training"], name="fit"),
            Node(modeling.predict, ["fitted", "model_input", "spec", "horizon"], "forecasts", name="predict"),
        ]
    nodes.append(Node(evaluate, ["forecasts", "ape_cap"], "metrics", name="evaluate"))
    # Namespace remains the same input/model-category/experiment contract as ordinary Kedro runs.
    unscoped = Pipeline(nodes)
    pipeline = Pipeline(unscoped, namespace=namespace(name, spec),
                        inputs=unscoped.inputs(),
                        outputs={"forecasts", "metrics", *({"fitted", "training"} if "prediction" not in spec else set())})
    # Modular outputs above are deliberately unprefixed disk datasets; transient model/input nodes are namespaced.
    SequentialRunner().run(pipeline, DataCatalog(datasets))


def validate_expected_cells(forecasts: pd.DataFrame, targets: pd.DataFrame, horizon: int) -> None:
    ycols = [f"y{h}" for h in range(1, horizon + 1)]
    origins = targets[targets.is_origin & targets[ycols].notna().all(axis=1)]
    expected = pd.MultiIndex.from_frame(
        origins.rename(columns={"week": "origin"})[["series", "origin"]]
        .merge(pd.DataFrame({"horizon": range(1, horizon + 1)}), how="cross")
    ).sort_values()
    checked, _ = aligned_forecasts(forecasts, forecasts)
    actual = pd.MultiIndex.from_frame(checked[["series", "origin", "horizon"]]).sort_values()
    if not actual.equals(expected):
        raise ValueError("예상 평가 셀과 예측 셀이 다르다")


def _references(name: str, spec: dict, root: str, naive: str) -> dict[str, str]:
    references = {"naive": naive, "root": root}
    if spec["lineage"]["parent"]:
        references["parent"] = spec["lineage"]["parent"]
    for other in spec["lineage"]["comparators"]:
        references[f"additional:{other}"] = other
    return references


def dedupe_key(name: str, spec: dict, record: dict) -> str:
    """연구가 달라도 같은 실험·설정·데이터·예측이면 같은 W&B run 으로 본다."""
    return digest({"experiment": name, "spec": {k: v for k, v in spec.items() if k not in {"study", "lineage"}},
                   "data_sha": record["identity"]["data_sha"], "forecasts": record["outputs"]["forecasts.parquet"]})


def wandb_run_id(record: dict) -> str:
    """재사용한 run 이면 그 W&B id, 아니면 로컬 run id."""
    return record.get("wandb", {}).get("run_id") or record["run_id"]


def find_logged_run(key: str, own_id: str):
    """같은 dedupe_key 로 이미 끝난 W&B run(자기 자신 제외)."""
    import wandb

    runs = wandb.Api(timeout=30).runs("sonet_krr/riotinto", filters={"config.dedupe_key": key, "state": "finished"})
    return next((run for run in runs if run.id != own_id), None)


def log_study_run(name: str, spec: dict, record: dict, records: dict, metrics: pd.DataFrame,
                  forecasts: pd.DataFrame, comparisons: pd.DataFrame, annual: pd.DataFrame,
                  wandb_mode: str, study_directory: Path) -> dict:
    if wandb_mode == "disabled":
        return {"mode": "disabled", "logged": False}
    import wandb

    lineage = spec["lineage"]
    key = dedupe_key(name, spec, record)
    if wandb_mode == "online" and (existing := find_logged_run(key, record["run_id"])) is not None:
        return {"mode": wandb_mode, "logged": True, "url": existing.url, "run_id": existing.id, "reused": True}
    references = {"root_run_id": wandb_run_id(records[lineage["root"]]),
                  "parent_run_id": wandb_run_id(records[lineage["parent"]]) if lineage["parent"] else None,
                  "component_run_ids": {s: wandb_run_id(records[s]) for s in record["source_experiments"]}}
    config = {**record["identity"], **references, "experiment": name, "dedupe_key": key,
              "data_last_week": record["data_last_week"], "evaluation_class": "retrospective_exploration"}
    with wandb.init(project="riotinto", entity="sonet_krr", id=record["run_id"], name=name,
                    group=spec["study"], job_type=lineage["kind"],
                    tags=[lineage["study_id"], lineage["axis"], "base_xgb_lineage"],
                    config=config, resume="allow", reinit="create_new", mode=wandb_mode,
                    dir=str(study_directory), save_code=False, settings=wandb.Settings(init_timeout=30)) as run:
        run.log({"metrics": wandb.Table(dataframe=metrics),
                 "cells": wandb.Table(dataframe=forecasts.assign(origin=forecasts.origin.dt.strftime("%Y-%m-%d"))),
                 "comparisons": wandb.Table(dataframe=comparisons), "annual": wandb.Table(dataframe=annual)})
        summary = run_summary(metrics)
        training_path = study_directory / "runs" / record["run_id"] / "training.json"
        if training_path.exists():
            training = json.loads(training_path.read_text())
            counts = [row["n_train"] for row in training]
            summary.update(n_train_min=min(counts), n_train_max=max(counts))
            heads = []
            for row in training:
                for head in row.get("model_info", {}).get("heads", []):
                    flattened = {key: (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)
                                 for key, value in head.items()}
                    heads.append({"series": row["series"], "origin": row["origin"],
                                  "training_start": row["training_start"], "training_end": row["training_end"],
                                  "cutoff": row["cutoff"], **flattened})
            if heads:
                run.log({"direct_heads": wandb.Table(dataframe=pd.DataFrame(heads))})
                summary["direct_head_count"] = len(heads)
        for _, row in comparisons[(comparisons.block_length == 3) & (comparisons.scope == "overall")].iterrows():
            summary[f"gain_vs_{row.reference_role}"] = float(row.gain_pct)
        summary.update(root_run_id=references["root_run_id"], parent_run_id=references["parent_run_id"])
        run.summary.update(summary)
        url = run.url if wandb_mode == "online" else None
    return {"mode": wandb_mode, "logged": wandb_mode == "online", "url": url, "run_id": record["run_id"]}


def execute_study(study_path: str | Path, root: str | Path = ".", phase: str | None = None,
                  wandb_mode: str = "online", bootstrap_samples: int | None = None) -> Path:
    root, study_path = Path(root).resolve(), Path(study_path).resolve()
    study = load_study(study_path)
    all_specs = ordered_experiments(study)
    selected = select_experiments(study, phase)
    naive = next(name for name, spec in all_specs.items()
                 if spec["lineage"]["kind"] == "benchmark" and spec["model"] == "naive")
    parameters = yaml.safe_load((root / "conf/base/parameters.yml").read_text())
    exogenous_spec = yaml.safe_load((root / "conf/base/parameters_exogenous.yml").read_text())["exogenous"]
    source_sha = source_fingerprint(root)
    context = {"study_sha": digest(study), "source_sha": source_sha,
               "protocol_sha": digest(parameters), "exogenous_config_sha": digest(exogenous_spec),
               "uv_lock_sha": file_sha(root / "uv.lock"), "environment": environment(),
               "raw_sha": {name: file_sha(root / f"data/01_raw/{name}.parquet")
                           for name in ("prices", "exogenous")}}
    # The analysis settings affect p-values and are also part of the immutable execution identity.
    evaluation = study.get("evaluation", {})
    if evaluation.get("confidence", 0.95) != 0.95:
        raise ValueError("이 평가 구현은 95% 신뢰구간만 지원한다")
    repetitions = bootstrap_samples if bootstrap_samples is not None else evaluation.get("bootstrap_repetitions", 2000)
    blocks = tuple([evaluation.get("block_origins", 3), *evaluation.get("sensitivity_blocks", [2, 4])])
    bootstrap_seed = evaluation.get("bootstrap_seed", 0)
    context["bootstrap_samples"] = repetitions
    directory = root / "data/07_reporting/studies" / study["study_id"] / digest(context)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["context"] != context:
            raise ValueError("실행 지문 경로와 manifest 내용이 다르다")
    else:
        manifest = {"study_id": study["study_id"], "execution_fingerprint": directory.name,
                    "context": context, "status": "running", "runs": {}}
        write_json(manifest_path, manifest)
    write_json(directory / "study.json", study)
    write_json(directory / "parameters.json", {**parameters, "exogenous": exogenous_spec})
    for name, fingerprint in context["raw_sha"].items():
        target = directory / "inputs" / f"{name}.parquet"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / f"data/01_raw/{name}.parquet", target)
        if file_sha(target) != fingerprint:
            raise ValueError(f"고정 스냅샷 지문 불일치: {name}")
    for path in source_files(root):
        target = directory / "source" / path.relative_to(root)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        if file_sha(path) != file_sha(target):
            raise ValueError("저장 소스와 실행 소스가 다르다")
    shared_files = ["targets.parquet", "tabular.parquet", "timeseries.parquet", "feature_columns.json"]
    if "shared_outputs" not in manifest:
        if any(record.get("status") == "complete" for record in manifest["runs"].values()):
            raise ValueError("완료 모델에 대응하는 공유 입력 무결성 기록이 없다")
        run_shared_inputs(directory, all_specs, parameters, exogenous_spec)
        manifest["shared_outputs"] = {name: file_sha(directory / "shared" / name) for name in shared_files}
        write_json(manifest_path, manifest)
    for name, fingerprint in manifest["shared_outputs"].items():
        path = directory / "shared" / name
        if not path.exists() or file_sha(path) != fingerprint:
            raise ValueError(f"공유 입력 산출물 손상: {name}")
    shared = directory / "shared"
    targets, tabular = (pd.read_parquet(shared / f"{name}.parquet") for name in ("targets", "tabular"))
    feature_columns = json.loads((shared / "feature_columns.json").read_text())
    data_sha = _sha(targets)
    features = {}
    for name, spec in all_specs.items():
        frame = modeling.select_input(tabular, feature_columns, spec, parameters["horizon"])
        features[name] = _sha(_features(frame, parameters["horizon"]))
    # All records are immutable identities; completion is tracked separately for safe resume.
    records = manifest["runs"]
    for i, (name, spec) in enumerate(selected.items(), 1):
        parent = spec["lineage"]["parent"]
        prediction = spec.get("prediction", {})
        sources = ([prediction["source"]] if prediction.get("method") == "shrink"
                   else prediction.get("sources", []))
        identity = {"study_id": study["study_id"], "experiment": name, "spec": spec,
                    "data_sha": data_sha, "exogenous_sha": context["raw_sha"]["exogenous"],
                    "features_sha": features[name], "execution": context,
                    "parent_run_id": records[parent]["run_id"] if parent else None,
                    "component_run_ids": {s: records[s]["run_id"] for s in sources},
                    "component_forecasts_sha": {s: records[s]["outputs"]["forecasts.parquet"] for s in sources}}
        rid = digest(identity)[:20]
        output = directory / "runs" / rid
        previous = records.get(name)
        if previous and previous["run_id"] != rid:
            raise ValueError(f"동일 실행 안에서 run identity 가 바뀌었다: {name}")
        if previous and previous.get("status") == "complete":
            if any(not (output / f).exists() or file_sha(output / f) != h for f, h in previous["outputs"].items()):
                raise ValueError(f"완료 기록의 산출물 손상: {name}")
            print(f"REUSE {i}/{len(selected)} {name} {rid}", flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        records[name] = {"run_id": rid, "identity": identity, "status": "running",
                         "source_experiments": sources, "data_last_week": str(targets.week.max().date())}
        write_json(manifest_path, manifest)
        print(f"TRAIN {i}/{len(selected)} {name} [{spec['lineage']['kind']}]", flush=True)
        source_paths = {s: directory / "runs" / records[s]["run_id"] / "forecasts.parquet" for s in sources}
        try:
            run_experiment(name, spec, output, shared, parameters, exogenous_spec, source_paths)
            forecasts = pd.read_parquet(output / "forecasts.parquet")
            validate_expected_cells(forecasts, targets, parameters["horizon"])
            files = ["forecasts.parquet", "metrics.parquet"]
            if "prediction" not in spec:
                files += ["fitted.pkl", "training.json"]
            records[name].update(status="complete", outputs={f: file_sha(output / f) for f in files})
            print(f"DONE {i}/{len(selected)} {name} gain_naive={run_summary(pd.read_parquet(output / 'metrics.parquet'))['gain_pct']:.3f}%", flush=True)
        except Exception as error:
            records[name].update(status="failed", error=str(error))
            manifest["status"] = "failed"
            write_json(manifest_path, manifest)
            raise
        write_json(manifest_path, manifest)
    if source_fingerprint(root) != source_sha:
        raise ValueError("실행 중 소스가 변경됐다. 해당 실행을 완료 처리하지 않는다")
    manifest["status"] = "evaluating"
    write_json(manifest_path, manifest)
    # Phase execution accumulates evidence: later phases must not drop completed earlier hypotheses.
    completed = {name: spec for name, spec in all_specs.items()
                 if records.get(name, {}).get("status") == "complete"}
    for name in completed:
        output = directory / "runs" / records[name]["run_id"]
        if any(not (output / f).exists() or file_sha(output / f) != h
               for f, h in records[name]["outputs"].items()):
            raise ValueError(f"완료 기록의 산출물 손상: {name}")
    frames = {name: pd.read_parquet(directory / "runs" / records[name]["run_id"] / "forecasts.parquet") for name in completed}
    comparisons, annual, losses, metrics_list = [], [], [], []
    for name, spec in completed.items():
        frame = frames[name]
        reference_map = _references(name, spec, study["root"], naive)
        for role, other in reference_map.items():
            comparison = compare_forecasts(frame, frames[other], parameters["ape_cap"],
                                          bootstrap_samples=repetitions, block_lengths=blocks, seed=bootstrap_seed)
            comparison = comparison.assign(experiment=name, reference=other, reference_role=role,
                                           axis=spec["lineage"]["axis"])
            comparison["primary"] = ((comparison.scope == "series") & comparison.series.isin(spec["lineage"]["primary_series"]))
            if len(spec["lineage"]["primary_series"]) == len(parameters["targets"]):
                comparison.loc[comparison.scope == "overall", "primary"] = True
            comparisons.append(comparison)
            annual.append(annual_comparison(frame, frames[other], parameters["ape_cap"])
                          .assign(experiment=name, reference=other, reference_role=role))
        ape = np.abs(frame.fc / frame.actual - 1)
        losses.append(frame.assign(experiment=name, ape_pct=ape * 100,
                                   capped_ape_pct=ape.clip(upper=parameters["ape_cap"]) * 100))
        metrics_list.append(pd.read_parquet(directory / "runs" / records[name]["run_id"] / "metrics.parquet")
                            .assign(experiment=name, run_id=records[name]["run_id"], parent=spec["lineage"]["parent"],
                                    root=study["root"], axis=spec["lineage"]["axis"], kind=spec["lineage"]["kind"]))
    comparisons = pd.concat(comparisons, ignore_index=True)
    comparisons["holm_pvalue"] = np.nan
    comparisons["family_size"] = 0
    comparisons["family_evaluated"] = 0
    # Claim families: predefined axis, reference role, declared primary scope and block choice.
    eligible = comparisons.primary & (comparisons.experiment != comparisons.reference)
    for (axis, role, scope, series, _), family in comparisons[eligible].groupby(
            ["axis", "reference_role", "scope", "series", "block_length"]):
        family_size = 0
        for other_name, other_spec in all_specs.items():
            primary = other_spec["lineage"]["primary_series"]
            other_references = _references(other_name, other_spec, study["root"], naive)
            if (other_spec["lineage"]["axis"] == axis and role in other_references
                    and other_references[role] != other_name
                    and ((scope == "series" and series in primary)
                         or (scope == "overall" and len(primary) == len(parameters["targets"])))):
                family_size += 1
        pvalues = np.r_[family.bootstrap_pvalue.to_numpy(), np.full(family_size - len(family), np.nan)]
        comparisons.loc[family.index, "holm_pvalue"] = holm_adjust(pvalues)[:len(family)]
        comparisons.loc[family.index, "family_size"] = family_size
        comparisons.loc[family.index, "family_evaluated"] = len(family)
    annual = pd.concat(annual, ignore_index=True)
    pd.concat(losses, ignore_index=True).to_parquet(directory / "origin_losses.parquet", index=False)
    pd.concat(metrics_list, ignore_index=True).to_parquet(directory / "summary.parquet", index=False)
    comparisons.to_parquet(directory / "comparisons.parquet", index=False)
    annual.to_parquet(directory / "annual.parquet", index=False)
    lineage = [{"experiment": name, **spec["lineage"], "run_id": records[name]["run_id"],
                "root_run_id": records[study["root"]]["run_id"],
                "parent_run_id": records[spec["lineage"]["parent"]]["run_id"] if spec["lineage"]["parent"] else None,
                "actual_changes": spec_diff(all_specs[spec["lineage"]["parent"]], spec) if spec["lineage"]["parent"] else []}
               for name, spec in completed.items()]
    write_json(directory / "lineage.json", lineage)
    manifest["status"] = "logging"
    write_json(manifest_path, manifest)
    for i, (name, spec) in enumerate(completed.items(), 1):
        record = records[name]
        run_comparisons = comparisons[comparisons.experiment == name].reset_index(drop=True)
        run_annual = annual[annual.experiment == name].reset_index(drop=True)
        analysis_sha = digest({"comparisons": _sha(run_comparisons), "annual": _sha(run_annual)})
        if (record.get("wandb", {}).get("logged") and wandb_mode == "online"
                and record["wandb"].get("analysis_sha") == analysis_sha):
            continue
        print(f"LOG {i}/{len(completed)} {name}", flush=True)
        try:
            record["wandb"] = log_study_run(
                name, spec, record, records, pd.read_parquet(directory / "runs" / record["run_id"] / "metrics.parquet"),
                frames[name], run_comparisons, run_annual, wandb_mode, directory)
            record["wandb"]["analysis_sha"] = analysis_sha
        except Exception as error:
            manifest.update(status="logging_failed", error=str(error))
            write_json(manifest_path, manifest)
            raise
        write_json(manifest_path, manifest)
    manifest.update(status="complete", selected=list(selected), completed=list(completed), wandb_mode=wandb_mode,
                    online_logged=sum(bool(records[n].get("wandb", {}).get("logged")) for n in completed))
    write_json(manifest_path, manifest)
    print(f"STUDY_COMPLETE {directory} records={len(completed)} online_logged={manifest['online_logged']}", flush=True)
    return directory
