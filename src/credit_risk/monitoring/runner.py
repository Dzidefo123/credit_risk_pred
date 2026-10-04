"""Freeze development reference and score label-free comparison populations."""

import inspect
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk import __version__
from credit_risk.data.loaders import load_origination_csv
from credit_risk.data.validation import ORIGINATION_FEATURES
from credit_risk.decisioning.runner import load_selected_model
from credit_risk.monitoring.drift import compare_reference, fit_reference, pd_to_score
from credit_risk.monitoring.settings import MonitoringConfig
from credit_risk.validation.runner import digest, verify_experiment, write_json


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new directory")
    return path


def model_identity(selection, validation):
    return dict(
        model_sha256=validation["artifacts_sha256"][
            f"{selection['preferred_candidate']}_selected.joblib"
        ],
        selection_sha256=validation["artifacts_sha256"]["selection.json"],
        candidate=selection["preferred_candidate"],
        calibration_method=selection["selected_methods"][selection["preferred_candidate"]],
    )


def freeze_monitor_reference(csv_path, run_dir, validation_dir, output_dir, config=None):
    config = config or MonitoringConfig()
    output = fresh_directory(output_dir)
    experiment, _, frame, splits, _ = verify_experiment(csv_path, run_dir)
    model, selection, validation = load_selected_model(validation_dir, run_dir, experiment)
    if validation["source_sha256"] != experiment["source_sha256"]:
        raise ValueError("Selected model source differs")
    rows = splits["development"]
    features = frame.iloc[rows].loc[:, ORIGINATION_FEATURES]
    probabilities = model.predict_proba(features)[:, 1]
    reference = fit_reference(features, probabilities, config)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "reference.json", reference)
    pd.DataFrame({"row_position": rows}).to_csv(output / "reference_positions.csv", index=False)
    manifest = dict(
        package_version=__version__,
        reference_partition="development",
        temporal_reference=False,
        final_test_scored=False,
        source_sha256=experiment["source_sha256"],
        model_identity=model_identity(selection, validation),
        config=config.model_dump(),
        measurement_code_sha256=digest(inspect.getfile(fit_reference)),
        artifacts_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()},
    )
    write_json(output / "reference_manifest.json", manifest)
    return manifest


def load_reference(reference_dir):
    directory = Path(reference_dir)
    manifest = json.loads((directory / "reference_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts_sha256"].items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Reference artifact must stay within its directory")
        if digest(directory / name) != expected:
            raise ValueError(f"Reference checksum mismatch: {name}")
    if manifest["measurement_code_sha256"] != digest(inspect.getfile(fit_reference)):
        raise ValueError("Reference measurement code changed; freeze new reference")
    reference = json.loads((directory / "reference.json").read_text(encoding="utf-8"))
    return reference, manifest


def guard_reserved_population(current, source, splits):
    """Protect the already-consumed original holdout from reuse as a monitor cohort."""
    reserved = source.iloc[splits["test"]]
    if "source_row_id" in current and "source_row_id" in reserved:
        if current.source_row_id.isin(reserved.source_row_id).any():
            raise ValueError("Comparison includes reserved final-test source row identifiers")
    original_groups = pd.util.hash_pandas_object(
        reserved.loc[:, ORIGINATION_FEATURES].astype(float), index=False
    )
    current_groups = pd.util.hash_pandas_object(
        current.loc[:, ORIGINATION_FEATURES].astype(float), index=False
    )
    if current_groups.isin(original_groups).any():
        raise ValueError("Comparison includes reserved final-test predictor groups")


def run_monitoring(
    source_csv,
    run_dir,
    validation_dir,
    reference_dir,
    current_csv,
    output_dir,
    config=None,
    current_manifest=None,
):
    config = config or MonitoringConfig()
    output = fresh_directory(output_dir)
    reference, reference_manifest = load_reference(reference_dir)
    experiment, _, source_frame, splits, _ = verify_experiment(source_csv, run_dir)
    model, selection, validation = load_selected_model(validation_dir, run_dir, experiment)
    identity = model_identity(selection, validation)
    if (
        identity != reference_manifest["model_identity"]
        or experiment["source_sha256"] != reference_manifest["source_sha256"]
        or validation["source_sha256"] != experiment["source_sha256"]
    ):
        raise ValueError("Model/source differs from frozen monitoring reference")
    provenance = {
        "data_kind": "unverified_external_comparison",
        "observation_dates_available": False,
    }
    if current_manifest is not None:
        provenance = json.loads(Path(current_manifest).read_text(encoding="utf-8"))
        if provenance["artifacts_sha256"][Path(current_csv).name] != digest(current_csv):
            raise ValueError("Comparison population checksum mismatch")
    data = load_origination_csv(current_csv, require_target=False)
    guard_reserved_population(data.frame, source_frame, splits)
    features = data.frame.loc[:, ORIGINATION_FEATURES]
    probabilities = model.predict_proba(features)[:, 1]
    comparison = compare_reference(reference, features, probabilities, config)
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "row_position": range(len(features)),
            "pd": probabilities,
            "score": pd_to_score(probabilities, config),
        }
    )
    if "source_row_id" in data.frame:
        predictions.insert(1, "source_row_id", data.frame.source_row_id.to_numpy())
    predictions.to_csv(output / "monitor_predictions.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    names = list(ORIGINATION_FEATURES)
    axes[0].barh(names, [comparison["metrics"][n]["psi"] for n in names], color="#286b8c")
    axes[0].axvline(config.psi.warning, color="#cf942b", linestyle="--", label="Warning policy")
    axes[0].axvline(config.psi.critical, color="#bb4747", linestyle="--", label="Critical policy")
    axes[0].set_xscale("symlog", linthresh=0.1)
    axes[0].set_xlabel("Feature stability (PSI / CSI equivalent; symlog axis)")
    axes[0].legend()
    axes[1].hist(
        reference["profiles"]["pd"]["numeric_values"],
        bins=np.linspace(0, 1, 31),
        density=True,
        alpha=0.5,
        label="Reference",
    )
    axes[1].hist(
        probabilities, bins=np.linspace(0, 1, 31), density=True, alpha=0.5, label="Comparison"
    )
    axes[1].set_xlabel("Calibrated two-year delinquency PD")
    axes[1].legend()
    fig.suptitle(f"Population comparison — {comparison['status']} (dates unverified)")
    fig.savefig(output / "monitoring.png", dpi=160)
    plt.close(fig)
    result = dict(
        package_version=__version__,
        **comparison,
        config=config.model_dump(),
        provenance=provenance,
        current_source_sha256=digest(current_csv),
        reference_manifest_sha256=digest(Path(reference_dir) / "reference_manifest.json"),
        model_identity=identity,
        final_test_scored=False,
        target_semantics=validation["target_semantics"],
        current_quality=asdict(data.quality),
        source_code_sha256={
            Path(p).name: digest(p)
            for p in [__file__, inspect.getfile(fit_reference), inspect.getfile(MonitoringConfig)]
        },
        limitations=[
            "No temporal drift claim without dated monitoring populations",
            "Feature/score/PD drift does not prove deteriorated calibration or discrimination",
            "Score is a deterministic PD transform, not independent scorecard evidence",
            "Thresholds are illustrative governance settings; alerts require investigation",
            "No automatic recalibration, retraining, model promotion or policy change",
        ],
        artifacts_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()},
    )
    write_json(output / "monitoring.json", result)
    return result
