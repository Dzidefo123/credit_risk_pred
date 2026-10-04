"""Export frozen evidence to optional local MLflow without fitting or scoring."""

import json
import math
import os
import time
from pathlib import Path

from credit_risk.tracking.settings import TrackingConfig
from credit_risk.utils.config import load_config
from credit_risk.validation.runner import digest

MANIFESTS = {"origination": "experiment.json", "validation": "validation.json"}


def flatten_values(payload, prefix=""):
    result = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_values(value, name))
        else:
            result[name] = value
    return result


def prepare_export(run_dir, kind, include_models=True):
    if kind not in MANIFESTS:
        raise ValueError("Tracking supports origination and validation evidence")
    directory = Path(run_dir).resolve()
    manifest_path = directory / MANIFESTS[kind]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = [manifest_path]
    for name, expected in manifest["artifacts_sha256"].items():
        path = (directory / name).resolve()
        if path.parent != directory or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Tracking artifact must stay within experiment directory")
        if digest(path) != expected:
            raise ValueError(f"Tracking artifact checksum mismatch: {name}")
        if path.suffix in (".json", ".png") or (include_models and path.suffix == ".joblib"):
            artifacts.append(path)
    params = {
        key: json.dumps(value, sort_keys=True)
        for key, value in flatten_values(
            manifest["config"] if kind == "origination" else manifest["selection"]["config"]
        ).items()
    }
    params.update(
        stage=kind,
        source_sha256=manifest["source_sha256"],
        source_manifest_sha256=digest(manifest_path),
        source_package_version=manifest["package_version"],
    )
    if kind == "origination":
        metrics_source = {"development": manifest["development_metrics"]}
    else:
        params.update(
            {
                f"calibration.{key}": value
                for key, value in manifest["selection"]["selected_methods"].items()
            }
        )
        params["preferred_candidate"] = manifest["selection"]["preferred_candidate"]
        metrics_source = {
            "development": manifest["selection"]["development_metrics"],
            "historical_final_holdout": manifest["final_metrics"],
        }
    metrics = {
        key: float(value)
        for key, value in flatten_values(metrics_source).items()
        if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
    }
    return params, metrics, artifacts


def export_to_mlflow(run_dir, kind, config_path, run_name=None):
    config = load_config(config_path, TrackingConfig)
    params, metrics, artifacts = prepare_export(run_dir, kind, config.include_models)
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "true"
    # Lazy import: ordinary training, analytics and API operation do not require MLflow.
    try:
        from mlflow.entities import Metric, Param
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise ValueError(
            "Install optional tracking dependencies: uv sync --extra tracking"
        ) from exc
    base = Path(config_path).resolve().parent
    database = (base / config.database_path).resolve()
    root = (base / config.artifact_root).resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=f"sqlite:///{database.as_posix()}")
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            config.experiment_name, artifact_location=root.as_uri()
        )
    else:
        if experiment.lifecycle_stage != "active" or experiment.artifact_location != root.as_uri():
            raise ValueError(
                "Existing tracking experiment differs from configured local artifact root"
            )
        experiment_id = experiment.experiment_id
    run = client.create_run(
        experiment_id,
        tags={
            "mlflow.runName": run_name or f"{kind}-evidence-export",
            "credit_risk.stage": kind,
            "credit_risk.export_only": "true",
            "credit_risk.model_promoted": "false",
        },
    )
    run_id = run.info.run_id
    try:
        now = int(time.time() * 1000)
        # Batches stay below server limits and preserve every flattened aggregate metric.
        entities = [Metric(key, value, now, 0) for key, value in metrics.items()]
        for start in range(0, len(entities), 500):
            client.log_batch(run_id, metrics=entities[start : start + 500])
        parameters = [Param(key, str(value)) for key, value in params.items()]
        for start in range(0, len(parameters), 100):
            client.log_batch(run_id, params=parameters[start : start + 100])
        for artifact in artifacts:
            client.log_artifact(run_id, str(artifact), artifact_path="evidence")
        client.set_terminated(run_id, status="FINISHED")
    except Exception as exc:
        client.set_terminated(run_id, status="FAILED")
        raise ValueError("Local MLflow export failed; inspect the failed tracking run") from exc
    return dict(
        status="recorded",
        stage=kind,
        run_id=run_id,
        experiment_id=experiment_id,
        tracking_uri=f"sqlite:///{database.as_posix()}",
        artifact_uri=run.info.artifact_uri,
        metric_count=len(metrics),
        parameter_count=len(params),
        artifacts=[p.name for p in artifacts],
        source_manifest_sha256=params["source_manifest_sha256"],
        fitting_performed=False,
        final_test_scored=False,
        model_promoted=False,
    )
