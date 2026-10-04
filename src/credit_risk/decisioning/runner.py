"""Compare policies on frozen development rows without fitting or test scoring."""

import inspect
import json
from hashlib import sha256
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from credit_risk import __version__
from credit_risk.data.validation import ORIGINATION_FEATURES, ORIGINATION_TARGET
from credit_risk.decisioning.policy import decide, policy_summary
from credit_risk.decisioning.settings import PolicyComparisonConfig
from credit_risk.models.calibration import CalibratedPDModel, ProbabilityCalibrator
from credit_risk.validation.runner import digest, verify_experiment, write_json


def verify_calibration_lock(run_dir, experiment, validation, selection):
    marker = json.loads((Path(run_dir) / "test_consumption.json").read_text(encoding="utf-8"))
    lock = marker["selection"]
    expected = sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
    if marker["selection_sha256"] != expected or validation["selection_sha256"] != expected:
        raise ValueError("Calibration selection lock differs")
    if (
        lock["source_sha256"] != experiment["source_sha256"]
        or lock["choices"] != selection["selected_methods"]
        or lock["preferred_candidate"] != selection["preferred_candidate"]
        or lock["calibration_code_sha256"] != digest(inspect.getfile(ProbabilityCalibrator))
    ):
        raise ValueError("Frozen calibration provenance changed")
    for candidate, expected_hash in lock["model_hashes"].items():
        if expected_hash != experiment["artifacts_sha256"][f"{candidate}.joblib"]:
            raise ValueError("Calibration base model differs")


def load_selected_model(validation_dir, run_dir=None, experiment=None):
    """Only load trusted local pickles after checking selection and model integrity."""
    directory = Path(validation_dir)
    manifest = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    selection_path = directory / "selection.json"
    if digest(selection_path) != manifest["artifacts_sha256"]["selection.json"]:
        raise ValueError("Selection checksum mismatch")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection != manifest["selection"] or selection["test_used_for_selection"]:
        raise ValueError("Selection differs or used final test")
    candidate = selection["preferred_candidate"]
    if candidate not in ("xgboost", "logistic_regression"):
        raise ValueError("Unknown selected candidate")
    model_path = directory / f"{candidate}_selected.joblib"
    if digest(model_path) != manifest["artifacts_sha256"][model_path.name]:
        raise ValueError("Selected model checksum mismatch")
    if run_dir is not None:
        verify_calibration_lock(run_dir, experiment, manifest, selection)
    model = joblib.load(model_path)
    if not isinstance(model, CalibratedPDModel):
        raise ValueError("Expected calibrated PD wrapper")
    if model.calibrator.method != selection["selected_methods"][candidate]:
        raise ValueError("Selected calibration method differs")
    return model, selection, manifest


def run_policy_comparison(csv_path, run_dir, validation_dir, output_dir, config=None):
    config = config or PolicyComparisonConfig()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Policy output is not empty; choose a new directory")
    experiment, _, frame, splits, _ = verify_experiment(csv_path, run_dir)
    model, selection, validation = load_selected_model(validation_dir, run_dir, experiment)
    if validation["source_sha256"] != experiment["source_sha256"]:
        raise ValueError("Calibrated model source differs from experiment")
    rows = splits["development"]
    applicants = frame.iloc[rows].copy()
    probabilities = model.predict_proba(applicants.loc[:, ORIGINATION_FEATURES])[:, 1]
    results = []
    output.mkdir(parents=True, exist_ok=True)
    for policy in config.policies:
        decisions = decide(applicants, probabilities, policy)
        decisions.insert(0, "row_position", rows)
        if "source_row_id" in applicants:
            decisions.insert(1, "source_row_id", applicants.source_row_id.to_numpy())
        decisions.to_csv(output / f"{policy.name}_decisions.csv", index=False)
        results.append(
            dict(policy=policy.name, **policy_summary(decisions, applicants[ORIGINATION_TARGET]))
        )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, key, title in zip(
        axes,
        ["approval_rate", "expected_bad_rate", "expected_loss_proxy"],
        ["Auto approval rate", "Approved mean PD", "Loss proxy (income units)"],
        strict=True,
    ):
        ax.bar(
            [r["policy"] for r in results],
            [r[key] if r[key] is not None else float("nan") for r in results],
            color="#286b8c",
        )
        ax.set_title(title)
    fig.suptitle("Development policy comparison — illustrative assumptions")
    fig.savefig(output / "policy_comparison.png", dpi=160)
    plt.close(fig)
    manifest = dict(
        package_version=__version__,
        partition="development",
        final_test_scored=False,
        target_semantics=validation["target_semantics"],
        source_sha256=experiment["source_sha256"],
        selection_sha256=digest(Path(validation_dir) / "selection.json"),
        model_sha256=validation["artifacts_sha256"][
            f"{selection['preferred_candidate']}_selected.joblib"
        ],
        source_code_sha256={
            Path(path).name: digest(path)
            for path in [inspect.getfile(decide), inspect.getfile(PolicyComparisonConfig), __file__]
        },
        preferred_candidate=selection["preferred_candidate"],
        config=config.model_dump(),
        policies=results,
        limitations=[
            "Development comparison is not independent policy validation",
            "Historical selected outcomes are not future funded outcomes or reject inference",
            "Two-year delinquency PD times assumed LGD/EAD is a loss proxy",
            "Source income units and DebtRatio basis are unverified; limits are hypothetical",
            "Manual reviews receive no automatic offered exposure; no policy champion is selected",
        ],
        artifacts_sha256={f.name: digest(f) for f in sorted(output.iterdir()) if f.is_file()},
    )
    write_json(output / "policy_comparison.json", manifest)
    return manifest
