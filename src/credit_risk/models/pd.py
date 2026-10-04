"""Reproducible origination candidates; calibration and final test stay reserved."""

import inspect
import json
import logging
import platform
import warnings
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from credit_risk import __version__
from credit_risk.data.loaders import load_origination_csv
from credit_risk.data.validation import (
    ORIGINATION_FEATURES,
    ORIGINATION_TARGET,
    validate_origination,
)
from credit_risk.features.origination import OriginationFeatures
from credit_risk.utils.config import ModelConfig
from credit_risk.validation.metrics import binary_metrics

LOGGER = logging.getLogger("credit_risk")


def split_origination(
    frame: pd.DataFrame, config: ModelConfig, seed: int = 42
) -> dict[str, np.ndarray]:
    """Stratify exact-predictor groups, retaining duplicates and all valid rows.

    Fractions apply to groups, so row proportions can differ slightly. Stratify
    by any positive label in a group; label is never part of the group key.
    Source indices are not assumed to be borrower identities.
    """
    validate_origination(frame)
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a valid 32-bit nonnegative integer")
    groups = pd.util.hash_pandas_object(frame.loc[:, ORIGINATION_FEATURES], index=False)
    summary = pd.DataFrame(
        {"group": groups.to_numpy(), "label": frame[ORIGINATION_TARGET].to_numpy()}
    )
    group_labels = summary.groupby("group", sort=True).label.max()
    keys = group_labels.index.to_numpy()

    def separate(keys, fraction):
        try:
            return train_test_split(
                keys,
                test_size=fraction,
                random_state=seed,
                stratify=group_labels.loc[keys].to_numpy(),
            )
        except ValueError as exc:
            raise ValueError(
                f"Insufficient independent groups for stratified partitions: {exc}"
            ) from exc

    pool, test = separate(keys, config.test_fraction)
    pool, calibration = separate(pool, config.calibration_fraction / (1 - config.test_fraction))
    train, development = separate(
        pool, config.validation_fraction / (1 - config.test_fraction - config.calibration_fraction)
    )
    partitions = {
        name: np.flatnonzero(groups.isin(selected).to_numpy())
        for name, selected in {
            "train": train,
            "development": development,
            "calibration": calibration,
            "test": test,
        }.items()
    }
    for name, positions in partitions.items():
        if set(frame.iloc[positions][ORIGINATION_TARGET].unique()) != {0, 1}:
            raise ValueError(
                f"{name} must contain both outcomes; supply a larger representative sample"
            )
    return partitions


def build_pd_model(kind: str, config: ModelConfig, seed: int = 42) -> Pipeline:
    if kind not in ("logistic_regression", "xgboost"):
        raise ValueError(f"Unknown PD model: {kind}")
    steps = [
        (
            "features",
            OriginationFeatures(
                config.clip_lower_quantile,
                config.clip_upper_quantile,
                log_transform=kind == "logistic_regression",
            ),
        ),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
    ]
    if kind == "logistic_regression":
        steps.extend(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=config.logistic.c,
                        max_iter=config.logistic.max_iter,
                        tol=config.logistic.tolerance,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    else:
        steps.append(
            (
                "model",
                XGBClassifier(
                    **config.xgboost.model_dump(),
                    random_state=seed,
                    tree_method="hist",
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=1,
                ),
            )
        )
    return Pipeline(steps)


def predict_probability(model: Pipeline, predictors: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(predictors)[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model returned invalid probabilities")
    return probabilities


def run_origination_experiment(
    csv_path: str | Path, output_dir: str | Path, config: ModelConfig | None = None, seed: int = 42
) -> dict:
    """Fit only train, evaluate only development, and persist self-contained pipelines."""
    config = config or ModelConfig()
    csv_path, directory = Path(csv_path), Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Experiment output is not empty; choose a new run directory")
    source_hash = sha256(csv_path.read_bytes()).hexdigest()
    data = load_origination_csv(csv_path)
    frame = data.frame.reset_index(drop=True)
    partitions = split_origination(frame, config, seed)
    X = frame.loc[:, ORIGINATION_FEATURES]
    y = frame[ORIGINATION_TARGET].astype(int)
    train, development = partitions["train"], partitions["development"]
    directory.mkdir(parents=True, exist_ok=True)
    assignments = pd.DataFrame({"row_position": np.arange(len(frame)), "partition": ""})
    assignments["feature_group"] = pd.util.hash_pandas_object(X, index=False).to_numpy()
    if "source_row_id" in frame:
        assignments["source_row_id"] = frame.source_row_id
    for name, positions in partitions.items():
        assignments.loc[positions, "partition"] = name
    assignments.to_csv(directory / "split_assignments.csv", index=False)
    metadata = {
        "package_version": __version__,
        "target_semantics": data.target_semantics,
        "probability_status": "raw_uncalibrated",
        "source_sha256": source_hash,
        "feature_names": list(ORIGINATION_FEATURES),
        "seed": seed,
        "config": config.model_dump(),
        "data_quality": asdict(data.quality),
        "validation_type": "grouped cross-sectional development holdout; not out-of-time",
        "final_test_scored": False,
        "calibration_scored": False,
        "partitions": {
            name: {
                "rows": len(rows),
                "bad_count": int(y.iloc[rows].sum()),
                "bad_rate": float(y.iloc[rows].mean()),
                "row_positions_sha256": sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest(),
            }
            for name, rows in partitions.items()
        },
        "versions": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")
        },
        "python": platform.python_version(),
        "source_code_sha256": {
            Path(path).name: sha256(Path(path).read_bytes()).hexdigest()
            for path in [
                __file__,
                inspect.getfile(OriginationFeatures),
                inspect.getfile(ModelConfig),
                inspect.getfile(binary_metrics),
            ]
        },
        "development_metrics": {},
        "artifacts_sha256": {},
    }
    predictions = pd.DataFrame(
        {"row_position": development, "label": y.iloc[development].to_numpy()}
    )
    prevalence = float(y.iloc[train].mean())
    metadata["development_metrics"]["constant_train_prevalence"] = binary_metrics(
        y.iloc[development], np.full(len(development), prevalence), config.classification_threshold
    )
    for kind in (config.baseline, config.challenger):
        LOGGER.info("Training candidate", extra={"details": {"model": kind, "rows": len(train)}})
        model = build_pd_model(kind, config, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X.iloc[train], y.iloc[train])
        probability = predict_probability(model, X.iloc[development])
        metadata["development_metrics"][kind] = binary_metrics(
            y.iloc[development], probability, config.classification_threshold
        )
        predictions[kind] = probability
        path = directory / f"{kind}.joblib"
        joblib.dump(
            {
                "pipeline": model,
                "metadata": {
                    key: metadata[key]
                    for key in (
                        "package_version",
                        "target_semantics",
                        "source_sha256",
                        "probability_status",
                        "feature_names",
                        "seed",
                        "versions",
                    )
                },
                "model_kind": kind,
            },
            path,
        )
        feature_names = model.named_steps["imputer"].get_feature_names_out(ORIGINATION_FEATURES)
        if kind == "logistic_regression":
            pd.DataFrame(
                {
                    "transformed_feature": feature_names,
                    "standardized_coefficient": model.named_steps["model"].coef_[0],
                }
            ).to_csv(directory / "logistic_coefficients.csv", index=False)
        else:
            pd.DataFrame(
                {
                    "transformed_feature": feature_names,
                    "gain_importance": model.named_steps["model"].feature_importances_,
                }
            ).to_csv(directory / "xgboost_importance.csv", index=False)
        LOGGER.info("Candidate evaluated on development only", extra={"details": {"model": kind}})
    predictions.to_csv(directory / "development_predictions.csv", index=False)
    for path in sorted(directory.iterdir()):
        if path.is_file():
            metadata["artifacts_sha256"][path.name] = sha256(path.read_bytes()).hexdigest()
    (directory / "experiment.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metadata
