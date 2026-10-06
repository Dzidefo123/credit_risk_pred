"""Nested, training-only probability calibration; no historical model loading."""

import warnings
from dataclasses import asdict, dataclass
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
from credit_risk.models.calibration import ProbabilityCalibrator
from credit_risk.models.pd import build_pd_model, predict_probability
from credit_risk.utils.config import ModelConfig, load_config
from credit_risk.validation.calibration import probability_diagnostics
from credit_risk.validation.diagnostics import group_bootstrap_comparison, reliability_diagnostics
from credit_risk.validation.metrics import binary_metrics
from credit_risk.validation.pd_diagnostics import METRICS, file_digest, load_training_only

METHODS = ("raw", "sigmoid", "isotonic")
MODELS = ("logistic_regression", "xgboost")


@dataclass(frozen=True)
class CalibrationStudyConfig:
    outer_folds: int = 5
    inner_folds: int = 5
    seed: int = 42
    epsilon: float = 1e-6
    minimum_calibration_events: int = 20
    minimum_brier_gain: float = 0.0001
    bootstrap_samples: int = 200
    confidence_level: float = 0.95
    material_auc_change: float = 0.001

    def __post_init__(self):
        if (
            type(self.outer_folds) is not int
            or self.outer_folds < 2
            or type(self.inner_folds) is not int
            or self.inner_folds < 3
            or type(self.seed) is not int
            or not 0 <= self.seed <= 2**32 - 100
            or type(self.minimum_calibration_events) is not int
            or self.minimum_calibration_events < 1
            or type(self.bootstrap_samples) is not int
            or self.bootstrap_samples < 20
        ):
            raise ValueError("Invalid calibration study folds, seed, events or bootstrap samples")
        if (
            not np.finfo(float).eps <= self.epsilon < 0.5
            or not np.isfinite(self.minimum_brier_gain)
            or self.minimum_brier_gain <= 0
            or not 0 < self.confidence_level < 1
            or not np.isfinite(self.material_auc_change)
            or self.material_auc_change <= 0
        ):
            raise ValueError("Invalid calibration study numerical settings")


def predictor_groups(frame):
    return pd.util.hash_pandas_object(frame.loc[:, ORIGINATION_FEATURES], index=False).to_numpy()


def positions_digest(positions):
    return sha256(np.asarray(positions, dtype="<i8").tobytes()).hexdigest()


def inner_roles(training_frame, study, seed):
    """Only outer-training rows are available to this split function."""
    validate_origination(training_frame)
    labels = training_frame[ORIGINATION_TARGET].to_numpy()
    groups = predictor_groups(training_frame)
    if any(len(np.unique(groups[labels == label])) < study.inner_folds for label in (0, 1)):
        raise ValueError("Insufficient outcome groups for inner partitions")
    splitter = StratifiedGroupKFold(n_splits=study.inner_folds, shuffle=True, random_state=seed)
    held_out = [evaluation for _, evaluation in splitter.split(training_frame, labels, groups)]
    roles = {
        "calibration": held_out[0],
        "selection": held_out[1],
        "base_fit": np.sort(np.concatenate(held_out[2:])),
    }
    for role, positions in roles.items():
        if len(np.unique(labels[positions])) != 2:
            raise ValueError(f"{role} requires both outcomes")
        if (
            role != "base_fit"
            and min(np.bincount(labels[positions].astype(int), minlength=2))
            < study.minimum_calibration_events
        ):
            raise ValueError(f"Insufficient events/non-events in {role}")
    for a, b in (
        ("base_fit", "calibration"),
        ("base_fit", "selection"),
        ("calibration", "selection"),
    ):
        if len(np.intersect1d(groups[roles[a]], groups[roles[b]])):
            raise ValueError("Inner predictor-group overlap")
    return roles


def choose_on_inner_selection(scores, minimum_brier_gain):
    """Predeclared Brier objective; raw wins without a gain and no log-loss harm.

    Only scores from the separate inner-selection partition may be supplied by
    the nested workflow. No outer outcomes enter this function.
    """
    raw = scores["raw"]
    eligible = [
        method
        for method in METHODS[1:]
        if raw["brier"] - scores[method]["brier"] >= minimum_brier_gain
        and scores[method]["log_loss"] <= raw["log_loss"]
    ]
    return (
        min(eligible, key=lambda m: (scores[m]["brier"], METHODS.index(m))) if eligible else "raw"
    )


def fit_outer_training(training_frame, config, study, seed):
    """Fit/select with only outer-training data; no evaluation labels argument.

    Inner role 0 fits calibrators; role 1 chooses methods; the remaining roles
    fit base pipelines. Preprocessing is fitted only within base_fit. Base models
    are not subsequently refitted on calibrator data (which would change their
    probability distribution). All three methods share the exact same base fit.
    """
    roles = inner_roles(training_frame, study, seed)
    labels = training_frame[ORIGINATION_TARGET].to_numpy()
    predictors = training_frame.loc[:, ORIGINATION_FEATURES]
    fitted, metadata = {}, {}
    for name in MODELS:
        model = build_pd_model(name, config, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(predictors.iloc[roles["base_fit"]], labels[roles["base_fit"]])
        calibration_p = predict_probability(model, predictors.iloc[roles["calibration"]])
        calibrators = {
            method: ProbabilityCalibrator(method, epsilon=study.epsilon).fit(
                calibration_p, labels[roles["calibration"]]
            )
            for method in METHODS[1:]
        }
        selection_raw = predict_probability(model, predictors.iloc[roles["selection"]])
        # Raw bypasses ProbabilityCalibrator.transform, preserving exact endpoints.
        selection_predictions = {
            "raw": selection_raw,
            **{method: cal.transform(selection_raw) for method, cal in calibrators.items()},
        }
        scores = {
            method: binary_metrics(labels[roles["selection"]], p)
            for method, p in selection_predictions.items()
        }
        selected = choose_on_inner_selection(scores, study.minimum_brier_gain)
        fitted[name] = (model, calibrators, selected)
        metadata[name] = {
            "selected_method": selected,
            "selection_metrics": scores,
            "calibrator_parameters": {
                method: cal.parameters() for method, cal in calibrators.items()
            },
        }
    design = {
        role: {
            "rows": len(indices),
            "events": int(labels[indices].sum()),
            "relative_positions_sha256": positions_digest(indices),
        }
        for role, indices in roles.items()
    }
    return fitted, {
        "inner_seed": seed,
        "roles": design,
        "inner_group_overlap": 0,
        "models": metadata,
    }


def predict_outer(fitted, predictors):
    """Accept predictors only; outcomes are unavailable to fitted calibrators."""
    output = {}
    for name, (model, calibrators, selected) in fitted.items():
        raw = predict_probability(model, predictors.loc[:, ORIGINATION_FEATURES])
        probabilities = {
            "raw": raw.copy(),
            **{method: cal.transform(raw) for method, cal in calibrators.items()},
        }
        for p in probabilities.values():
            if p.shape != raw.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError("Invalid outer probabilities")
        probabilities["inner_selected"] = probabilities[selected].copy()
        output[name] = probabilities
    return output


def ranking_diagnostics(labels, raw, transformed, method, tolerance):
    order = np.argsort(raw, kind="stable")
    inversions = int(np.count_nonzero(np.diff(transformed[order]) < -1e-14))
    if inversions:
        raise ValueError("Non-monotonic calibration mapping")
    delta = binary_metrics(labels, transformed)["roc_auc"] - binary_metrics(labels, raw)["roc_auc"]
    return {
        "auc_delta": delta,
        "material_auc_change": abs(delta) > tolerance,
        "rank_inversions": inversions,
        "raw_unique_probabilities": len(np.unique(raw)),
        "calibrated_unique_probabilities": len(np.unique(transformed)),
        "interpretation": "Isotonic ties may alter AUC"
        if method == "isotonic"
        else "Positive-slope sigmoid preserves ranking except numerical clipping/ties",
    }


def summarize(labels, probabilities, records):
    diagnostics = probability_diagnostics(labels, probabilities)
    diagnostics["reliability"] = reliability_diagnostics(
        labels, probabilities, np.linspace(0, 1, 11)
    )
    for row in diagnostics["reliability"]["bins"]:
        row["sparse"] = row["rows"] < 100
    return {
        "pooled_oof": diagnostics,
        "folds": records,
        "cv_summary": {
            metric: {
                "mean": float(np.mean([f["metrics"][metric] for f in records])),
                "std": float(np.std([f["metrics"][metric] for f in records], ddof=1)),
                "fold_values": [f["metrics"][metric] for f in records],
            }
            for metric in METRICS
        },
    }


def evidence_recommendation(model_results, comparisons, minimum_gain):
    """Post-study research assessment, never used to fit/select an outer predictor.

    Require a conditional Brier interval wholly beyond the prespecified minimum
    gain and no log-loss deterioration in its conditional interval. Otherwise
    raw's lower complexity wins. This is not a fresh-test significance claim.
    """
    raw = model_results["raw"]["pooled_oof"]["metrics"]
    qualifying = []
    for method in METHODS[1:]:
        bounds = comparisons[method]["paired_differences"]["intervals"]
        if (
            bounds["brier"]["upper"] is not None
            and bounds["log_loss"]["upper"] is not None
            and bounds["brier"]["upper"] < -minimum_gain
            and bounds["log_loss"]["upper"] <= 0
        ):
            qualifying.append(method)
    if qualifying:
        choice = min(
            qualifying,
            key=lambda m: (model_results[m]["pooled_oof"]["metrics"]["brier"], METHODS.index(m)),
        )
        reason = "Conditional paired evidence meets prespecified Brier gain and log-loss guard"
    else:
        choice = "raw"
        reason = (
            "No calibrated method meets both prespecified conditional evidence requirements; "
            "prefer simplicity"
        )
    best = min(METHODS, key=lambda m: model_results[m]["pooled_oof"]["metrics"]["brier"])
    return {
        "recommendation": choice.upper(),
        "reason": reason,
        "lowest_point_brier_method": best,
        "point_brier_gain_of_best": raw["brier"]
        - model_results[best]["pooled_oof"]["metrics"]["brier"],
        "scope": "post-study research recommendation; no artifact/policy promotion",
    }


def nested_calibration_study(frame, config: ModelConfig, study=None, *, progress=None):
    """Generic study helper; repository entry point provides anchored train only."""
    study = study or CalibrationStudyConfig()
    validate_origination(frame)
    labels = frame[ORIGINATION_TARGET].to_numpy()
    groups = predictor_groups(frame)
    if any(len(np.unique(groups[labels == label])) < study.outer_folds for label in (0, 1)):
        raise ValueError("Insufficient outcome groups for outer folds")
    methods = (*METHODS, "inner_selected")
    oof = {name: {method: np.full(len(frame), np.nan) for method in methods} for name in MODELS}
    records = {name: {method: [] for method in methods} for name in MODELS}
    design = []
    splitter = StratifiedGroupKFold(
        n_splits=study.outer_folds, shuffle=True, random_state=study.seed
    )
    for number, (training, evaluation) in enumerate(splitter.split(frame, labels, groups), 1):
        if len(np.unique(labels[evaluation])) != 2 or len(np.unique(labels[training])) != 2:
            raise ValueError("Outer folds require both outcomes")
        if len(np.intersect1d(groups[training], groups[evaluation])):
            raise ValueError("Outer predictor-group overlap")
        if progress:
            progress(f"Fitting outer fold {number}/{study.outer_folds}")
        fitted, inner = fit_outer_training(frame.iloc[training], config, study, study.seed + number)
        predicted = predict_outer(fitted, frame.iloc[evaluation].loc[:, ORIGINATION_FEATURES])
        # Evaluation labels are first used for scoring after all fit/selection has finished.
        evaluation_labels = labels[evaluation]
        design.append(
            {
                "outer_fold": number,
                "training_rows": len(training),
                "evaluation_rows": len(evaluation),
                "evaluation_events": int(evaluation_labels.sum()),
                "outer_group_overlap": 0,
                "training_positions_sha256": positions_digest(training),
                "evaluation_positions_sha256": positions_digest(evaluation),
                "inner": inner,
            }
        )
        for name in MODELS:
            for method in methods:
                p = predicted[name][method]
                oof[name][method][evaluation] = p
                record = {"fold": number, "metrics": binary_metrics(evaluation_labels, p)}
                if method in METHODS[1:]:
                    record["ranking"] = ranking_diagnostics(
                        evaluation_labels,
                        predicted[name]["raw"],
                        p,
                        method,
                        study.material_auc_change,
                    )
                records[name][method].append(record)
    results, comparisons, recommendations = {}, {}, {}
    for name in MODELS:
        if any(not np.isfinite(p).all() for p in oof[name].values()):
            raise ValueError("Incomplete outer-fold prediction coverage")
        results[name] = {
            method: summarize(labels, oof[name][method], records[name][method])
            for method in methods
        }
        comparisons[name] = {}
        for method in METHODS[1:]:
            if progress:
                progress(f"Paired bootstrap: {name} raw vs {method}")
            paired = group_bootstrap_comparison(
                labels,
                {"raw": oof[name]["raw"], method: oof[name][method]},
                groups,
                samples=study.bootstrap_samples,
                confidence=study.confidence_level,
                seed=study.seed + 50,
            )
            paired.update(
                seed=study.seed + 50,
                interval_type="percentile, conditional on fixed nested OOF predictions",
                point_differences={
                    metric: results[name][method]["pooled_oof"]["metrics"][metric]
                    - results[name]["raw"]["pooled_oof"]["metrics"][metric]
                    for metric in ("brier", "log_loss", "roc_auc")
                },
            )
            comparisons[name][method] = paired
        recommendations[name] = evidence_recommendation(
            results[name], comparisons[name], study.minimum_brier_gain
        )
    return {
        "schema_version": 1,
        "target": "two-year serious-delinquency outcome",
        "rows": len(frame),
        "events": int(labels.sum()),
        "study_config": asdict(study),
        "model_config": config.model_dump(),
        "design": (
            "Outer grouped CV; inner roles 0 calibration, 1 selection, remaining roles base-fit"
        ),
        "selection_protocol": {
            "primary": "Brier",
            "minimum_absolute_gain": study.minimum_brier_gain,
            "guard": "log loss must not worsen on inner selection",
            "ties": "raw unless minimum gain; sigmoid before isotonic on equal eligible Brier",
            "outer_role": "scoring only; retrospective recommendations do not change predictions",
        },
        "fold_design": design,
        "models": results,
        "paired_comparisons": comparisons,
        "recommendations": recommendations,
        "historical_metrics": {
            "status": "retained historical, not reevaluated; separate from Tasks 4 and 5",
            "roc_auc": 0.868152,
            "brier": 0.048545,
            "log_loss": 0.176030,
        },
    }


def run_calibration_study(root, source, *, progress=None):
    root = Path(root)
    frame, provenance = load_training_only(
        source, root / "artifacts/phase4-origination-001", root / "reports/holdout_registry.json"
    )
    protected = {
        "reports/holdout_registry.json",
        "reports/model_validation/PD_DIAGNOSTICS.md",
        "reports/model_validation/pd_diagnostics.json",
        "reports/model_validation/pd_diagnostics.png",
    }
    before = {name: file_digest(root / name) for name in sorted(protected)}
    result = nested_calibration_study(
        frame, load_config(root / "configs/model.yaml", ModelConfig), progress=progress
    )
    result["provenance"] = provenance
    result["preserved_task4_and_registry_sha256"] = before
    result["versions"] = {
        name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "scipy")
    }
    result["source_code_sha256"] = {
        name: file_digest(root / name)
        for name in (
            "src/credit_risk/validation/calibration_study.py",
            "src/credit_risk/validation/pd_diagnostics.py",
            "src/credit_risk/validation/calibration.py",
            "src/credit_risk/validation/diagnostics.py",
            "src/credit_risk/models/calibration.py",
            "src/credit_risk/models/pd.py",
            "src/credit_risk/features/origination.py",
            "src/credit_risk/utils/config.py",
            "src/credit_risk/validation/metrics.py",
            "scripts/calibration_study.py",
        )
    }
    if any(file_digest(root / name) != digest for name, digest in before.items()):
        raise ValueError("Task 4 evidence or registry changed during calibration study")
    return result
