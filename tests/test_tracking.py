"""Optional local tracking exports existing evidence with checksums and no row data."""

import json
from pathlib import Path

import pytest

from credit_risk.tracking.mlflow import export_to_mlflow, prepare_export
from credit_risk.validation.runner import digest


def evidence(tmp_path):
    run = tmp_path / "source"
    run.mkdir()
    (run / "model.joblib").write_bytes(b"inert model artifact; never deserialized")
    (run / "development_predictions.csv").write_text("private,row,labels", encoding="utf-8")
    m = {
        "config": {"threshold": 0.1, "nested": {"seed": 42}},
        "source_sha256": "a" * 64,
        "package_version": "test",
        "development_metrics": {"model": {"brier": 0.12, "roc_auc": 0.8, "unavailable": None}},
        "artifacts_sha256": {p.name: digest(p) for p in run.iterdir()},
    }
    (run / "experiment.json").write_text(json.dumps(m), encoding="utf-8")
    return run, m


def test_export_preparation_checks_all_artifacts_but_omits_row_data(tmp_path):
    run, m = evidence(tmp_path)
    params, metrics, artifacts = prepare_export(run, "origination")
    assert metrics == {"development.model.brier": 0.12, "development.model.roc_auc": 0.8}
    assert params["source_manifest_sha256"] == digest(run / "experiment.json")
    assert {p.name for p in artifacts} == {"experiment.json", "model.joblib"}
    assert {p.name for p in prepare_export(run, "origination", False)[2]} == {"experiment.json"}
    (run / "development_predictions.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        prepare_export(run, "origination")


def test_artifact_paths_cannot_escape(tmp_path):
    run, m = evidence(tmp_path)
    m["artifacts_sha256"] = {"../escaped.joblib": "wrong"}
    (run / "experiment.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        prepare_export(run, "origination")
    with pytest.raises(ValueError):
        prepare_export(run, "unsupported")


def test_optional_mlflow_sqlite_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "true")
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    run, m = evidence(tmp_path)
    config = tmp_path / "tracking.yaml"
    config.write_text(
        "database_path: tracking.db\nartifact_root: artifacts\n"
        "experiment_name: unit-test\ninclude_models: true\n",
        encoding="utf-8",
    )
    original = digest(run / "experiment.json")
    first = export_to_mlflow(run, "origination", config, "unit-export")
    second = export_to_mlflow(run, "origination", config, "unit-export-2")
    assert first["experiment_id"] == second["experiment_id"]
    assert first["run_id"] != second["run_id"]
    client = MlflowClient(tracking_uri=first["tracking_uri"])
    record = client.get_run(first["run_id"])
    assert record.info.status == "FINISHED"
    assert record.data.metrics["development.model.brier"] == 0.12
    assert record.data.params["nested.seed"] == "42"
    assert record.data.tags["credit_risk.export_only"] == "true"
    names = {Path(p.path).name for p in client.list_artifacts(first["run_id"], "evidence")}
    assert names == {"experiment.json", "model.joblib"}
    assert digest(run / "experiment.json") == original
    assert not first["final_test_scored"] and not first["fitting_performed"]


def test_missing_optional_dependency_gives_actionable_error_before_writes(tmp_path, monkeypatch):
    import builtins

    run, _ = evidence(tmp_path)
    config = tmp_path / "tracking.yaml"
    config.write_text(
        "database_path: optional.db\nartifact_root: optional-files\n", encoding="utf-8"
    )
    original_import = builtins.__import__

    def block(name, *args, **kwargs):
        if name.startswith("mlflow"):
            raise ImportError("optional dependency absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(ValueError, match="extra tracking"):
        export_to_mlflow(run, "origination", config)
    assert not (tmp_path / "optional.db").exists()
    assert not (tmp_path / "optional-files").exists()


def test_failed_export_is_marked_failed_in_local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    run, _ = evidence(tmp_path)
    config = tmp_path / "tracking.yaml"
    config.write_text(
        "database_path: failed.db\nartifact_root: files\nexperiment_name: failure-test\n",
        encoding="utf-8",
    )

    def broken(*args, **kwargs):
        raise OSError("forced artifact write failure")

    monkeypatch.setattr(MlflowClient, "log_artifact", broken)
    with pytest.raises(ValueError, match="failed tracking run"):
        export_to_mlflow(run, "origination", config)
    client = MlflowClient(tracking_uri=f"sqlite:///{(tmp_path / 'failed.db').as_posix()}")
    experiment = client.get_experiment_by_name("failure-test")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1 and runs[0].info.status == "FAILED"
