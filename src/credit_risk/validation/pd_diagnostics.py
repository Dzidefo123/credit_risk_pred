"""Training-only cross-fitted diagnostics, isolated from historical model bundles."""

import warnings
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

from credit_risk.data.validation import (
    ORIGINATION_FEATURES,
    ORIGINATION_TARGET,
    validate_origination,
)
from credit_risk.models.pd import build_pd_model, predict_probability
from credit_risk.utils.config import ModelConfig
from credit_risk.validation.calibration import probability_diagnostics
from credit_risk.validation.diagnostics import group_bootstrap_comparison
from credit_risk.validation.holdout_registry import (
    HISTORICAL_SOURCE_SHA256,
    HoldoutRegistry,
    sample_fingerprints,
    verify_repository_registry,
)
from credit_risk.validation.metrics import binary_metrics
from credit_risk.validation.thresholds import threshold_analysis, threshold_diagnostics

METRICS = ("roc_auc", "gini", "pr_auc", "average_precision", "brier", "log_loss")
ASSIGNMENTS_SHA256 = "efbdc36ee1286f0c06f302197174efbf9884ecb0f8e8c4b9d010569265db04e2"
TRAIN_ROWS, TRAIN_EVENTS = 67562, 4514
TRAIN_POSITIONS_SHA256 = "892d79c1fed9be83efbae3370195cba20bfe387e8284fad5377df0ff5d8e81c2"


def file_digest(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_only(source, experiment, registry_path):
    """Parse only anchored training positions; excluded CSV records are skipped.

    Hashing the raw file for identity reads bytes only, never holdout outcomes.
    Fixed source/split/position anchors prevent a substituted partition being
    presented as training. No model bundle or prediction artifact is opened.
    """
    experiment, source = Path(experiment), Path(source)
    registry = HoldoutRegistry(registry_path)
    verify_repository_registry(registry_path)
    assignments_path = experiment / "split_assignments.csv"
    if file_digest(assignments_path) != ASSIGNMENTS_SHA256:
        raise ValueError("Saved split assignment identity changed")
    assignments = pd.read_csv(
        assignments_path, usecols=["row_position", "partition", "feature_group"]
    )
    train = assignments.loc[assignments.partition == "train"].sort_values("row_position")
    positions = train.row_position.to_numpy(dtype="<i8")
    positions_digest = sha256(positions.tobytes()).hexdigest()
    if positions_digest != TRAIN_POSITIONS_SHA256:
        raise ValueError("Saved training positions changed")
    registry.check(
        HISTORICAL_SOURCE_SHA256, [f"legacy-pandas-v1:{int(g):016x}" for g in train.feature_group]
    )
    source_digest = file_digest(source)
    if source_digest != HISTORICAL_SOURCE_SHA256:
        raise ValueError("Source identity changed")
    included = set(positions.tolist())
    frame = pd.read_csv(source, skiprows=lambda row: row > 0 and row - 1 not in included)
    frame = frame.rename(
        columns={
            "NumberOfTime30-59DaysPastDueNotWorse": "NumberOfTime30_59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse": "NumberOfTime60_89DaysPastDueNotWorse",
            "Unnamed: 0": "source_row_id",
        }
    )
    validate_origination(frame)
    if len(frame) != TRAIN_ROWS or int(frame[ORIGINATION_TARGET].sum()) != TRAIN_EVENTS:
        raise ValueError("Training rows/outcome count changed")
    actual_groups = pd.util.hash_pandas_object(frame.loc[:, ORIGINATION_FEATURES], index=False)
    if not np.array_equal(actual_groups.to_numpy(), train.feature_group.to_numpy()):
        raise ValueError("Parsed training profiles disagree with saved identities")
    registry.check(source_digest, sample_fingerprints(frame))
    return frame, {
        "source_sha256": source_digest,
        "assignments_sha256": ASSIGNMENTS_SHA256,
        "training_positions_sha256": positions_digest,
        "registry_sha256": file_digest(registry_path),
        "partition": "original train only",
        "rows": len(frame),
        "events": TRAIN_EVENTS,
        "original_holdout_access": "excluded at CSV parsing; no frozen model/prediction loading",
    }


def cross_validate_candidates(
    frame,
    config: ModelConfig,
    *,
    folds=5,
    seed=42,
    thresholds=(0.03, 0.05, 0.1, 0.2, 0.5),
    bootstrap_samples=200,
):
    """Generic research helper; repository entry point supplies verified train only.

    All preprocessing is fitted separately inside each training fold. Fixed
    threshold grids are descriptive, never optimized on OOF outcomes. A separate
    prevalence threshold comes only from each fold's training labels.
    """
    validate_origination(frame)
    if type(folds) is not int or folds < 2 or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Invalid folds/seed")
    thresholds = tuple(thresholds)
    y = frame[ORIGINATION_TARGET].to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("CV requires both outcomes")
    groups = pd.util.hash_pandas_object(frame.loc[:, ORIGINATION_FEATURES], index=False).to_numpy()
    if any(len(np.unique(groups[y == label])) < folds for label in (0, 1)):
        raise ValueError("Insufficient outcome groups for requested folds")
    threshold_analysis(y, np.full(len(y), 0.5), thresholds=thresholds)
    x = frame.loc[:, ORIGINATION_FEATURES]
    names = ("logistic_regression", "xgboost")
    predictions = {name: np.full(len(y), np.nan) for name in names}
    records = {name: [] for name in names}
    design = []
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for number, (train, evaluation) in enumerate(splitter.split(x, y, groups), 1):
        if any(len(np.unique(y[indices])) != 2 for indices in (train, evaluation)):
            raise ValueError("A CV fold lacks both outcomes")
        overlap = np.intersect1d(groups[train], groups[evaluation])
        if len(overlap):
            raise ValueError("Predictor group leakage across folds")
        prevalence = float(y[train].mean())
        design.append(
            {
                "fold": number,
                "training_rows": len(train),
                "evaluation_rows": len(evaluation),
                "evaluation_events": int(y[evaluation].sum()),
                "group_overlap": 0,
                "evaluation_positions_sha256": sha256(
                    evaluation.astype("<i8").tobytes()
                ).hexdigest(),
                "training_prevalence_threshold": prevalence,
            }
        )
        for name in names:
            model = build_pd_model(name, config, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(x.iloc[train], y[train])
            p = predict_probability(model, x.iloc[evaluation])
            predictions[name][evaluation] = p
            records[name].append(
                {
                    "fold": number,
                    "metrics": binary_metrics(y[evaluation], p),
                    "training_prevalence_threshold": threshold_diagnostics(
                        y[evaluation], p, threshold=prevalence
                    ),
                }
            )
    candidates = {}
    for name in names:
        p = predictions[name]
        if not np.isfinite(p).all():
            raise ValueError("OOF coverage incomplete")
        candidates[name] = {
            "folds": records[name],
            "cv_summary": {
                metric: {
                    "mean": float(np.mean([f["metrics"][metric] for f in records[name]])),
                    "std": float(np.std([f["metrics"][metric] for f in records[name]], ddof=1)),
                    "fold_values": [f["metrics"][metric] for f in records[name]],
                }
                for metric in METRICS
            },
            "oof": probability_diagnostics(y, p),
            "threshold_analysis": threshold_analysis(y, p, thresholds=thresholds),
        }
    uncertainty = group_bootstrap_comparison(
        y, predictions, groups, samples=bootstrap_samples, confidence=0.95, seed=seed + 1
    )
    uncertainty.update(
        seed=seed + 1,
        interval_type="percentile, conditional on fixed cross-fitted predictions",
        limitations=(
            "Shared training folds; no refitting uncertainty or unidentified borrower dependence"
        ),
    )
    return {
        "schema_version": 1,
        "evaluation": "training-only cross-sectional OOF; not out-of-time",
        "probability_status": "fresh raw candidates; no recalibration of evaluated predictions",
        "target": "two-year serious-delinquency outcome",
        "rows": len(y),
        "events": int(y.sum()),
        "observed_event_rate": float(y.mean()),
        "fold_count": folds,
        "seed": seed,
        "config": config.model_dump(),
        "fold_design": design,
        "candidates": candidates,
        "uncertainty": uncertainty,
        "threshold_protocol": (
            "Predeclared grid plus fold-training prevalence; no optimized OOF threshold"
        ),
        "historical_locked_holdout_result": {
            "status": "retained historical; not reevaluated or comparable to CV",
            "roc_auc": 0.868152,
            "brier": 0.048545,
            "log_loss": 0.176030,
        },
    }


def run_training_diagnostics(root, source):
    root = Path(root)
    frame, provenance = load_training_only(
        source, root / "artifacts/phase4-origination-001", root / "reports/holdout_registry.json"
    )
    from credit_risk.utils.config import load_config

    result = cross_validate_candidates(frame, load_config(root / "configs/model.yaml", ModelConfig))
    result["provenance"] = provenance
    result["versions"] = {
        name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "scipy")
    }
    result["diagnostic_source_sha256"] = {
        name: file_digest(root / "src/credit_risk/validation" / name)
        for name in ("calibration.py", "thresholds.py", "pd_diagnostics.py", "diagnostics.py")
    }
    result["reused_source_sha256"] = {
        name: file_digest(root / name)
        for name in (
            "src/credit_risk/models/pd.py",
            "src/credit_risk/features/origination.py",
            "src/credit_risk/utils/config.py",
            "src/credit_risk/validation/metrics.py",
            "scripts/pd_diagnostics.py",
        )
    }
    if file_digest(root / "reports/holdout_registry.json") != provenance["registry_sha256"]:
        raise ValueError("Registry changed during diagnostics")
    return result
