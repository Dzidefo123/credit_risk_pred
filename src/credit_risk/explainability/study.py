"""Fold-isolated explanation research on the anchored original training partition."""

import warnings
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

from credit_risk.data.validation import (
    ORIGINATION_FEATURES,
    ORIGINATION_TARGET,
    validate_origination,
)
from credit_risk.explainability.logistic import coefficient_table, local_logistic
from credit_risk.explainability.stability import (
    coefficient_stability,
    descriptive,
    feature_ranks,
    rank_agreement,
)
from credit_risk.explainability.xgboost import tree_shap
from credit_risk.models.pd import build_pd_model
from credit_risk.utils.config import ModelConfig, load_config
from credit_risk.validation.pd_diagnostics import file_digest, load_training_only


def score_stratified_sample(probabilities, size, seed):
    p = np.asarray(probabilities, dtype=float)
    if (
        p.ndim != 1
        or not len(p)
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
        or type(size) is not int
        or size < 1
    ):
        raise ValueError("Invalid explanation sample inputs")
    size = min(size, len(p))
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    membership = np.searchsorted(edges[1:-1], p, side="right")
    counts = np.bincount(membership)
    desired = counts * size / len(p)
    quotas = np.floor(desired).astype(int)
    for index in sorted(range(len(counts)), key=lambda i: (-(desired[i] - quotas[i]), i))[
        : size - quotas.sum()
    ]:
        quotas[index] += 1
    rng = np.random.default_rng(seed)
    positions = np.sort(
        np.concatenate(
            [
                rng.choice(np.flatnonzero(membership == i), int(q), replace=False)
                for i, q in enumerate(quotas)
                if q
            ]
        )
    )
    return positions, {
        "strategy": "proportional predicted-score deciles; no labels used",
        "population_counts": counts.tolist(),
        "sample_counts": quotas.tolist(),
        "size": len(positions),
        "seed": seed,
        "positions_sha256": sha256(positions.astype("<i8").tobytes()).hexdigest(),
    }


def dependence_summary(values, contributions, bins=12):
    x, s = np.asarray(values, dtype=float), np.asarray(contributions, dtype=float)
    if (
        x.ndim != 1
        or x.shape != s.shape
        or not len(x)
        or not np.isfinite(s).all()
        or np.isinf(x).any()
    ):
        raise ValueError("Invalid dependence vectors")
    valid = np.isfinite(x)
    x, s = x[valid], s[valid]
    if not len(x):
        return {"bins": [], "missing_rows": int((~valid).sum()), "spearman_rho": None}
    unique = np.unique(x)
    rows = []
    if len(unique) <= 24:
        masks = [(f"value={value:g}", x == value) for value in unique]
        definition = "exact observed input values (<=24 unique)"
    else:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        membership = np.searchsorted(edges[1:-1], x, side="right")
        masks = [(f"quantile bin {i + 1}", membership == i) for i in range(max(1, len(edges) - 1))]
        definition = "12 quantile bins of observed nonmissing raw input; unique edges"
    for label, mask in masks:
        if not mask.any():
            continue
        rows.append(
            {
                "bin": label,
                "rows": int(mask.sum()),
                "median_input": float(np.median(x[mask])),
                "mean_shap": float(s[mask].mean()),
                "shap_q25": float(np.quantile(s[mask], 0.25)),
                "shap_q75": float(np.quantile(s[mask], 0.75)),
                "sparse": int(mask.sum()) < 100,
            }
        )
    rho = None if np.ptp(x) == 0 or np.ptp(s) == 0 else float(spearmanr(x, s).statistic)
    return {
        "bins": rows,
        "binning": definition,
        "missing_rows": int((~valid).sum()),
        "spearman_rho": rho,
        "interpretation": "observed input vs model contribution; not causal or partial dependence",
    }


def explain_fold(training_frame, evaluation_predictors, config, seed, interaction_rows):
    """Evaluation outcomes cannot enter: only predictors are accepted here."""
    if set(evaluation_predictors.columns) != set(ORIGINATION_FEATURES):
        raise ValueError("Explanation evaluation accepts only the ten predictors")
    labels = training_frame[ORIGINATION_TARGET].to_numpy()
    predictors = training_frame.loc[:, ORIGINATION_FEATURES]
    outputs = {}
    for kind in ("logistic_regression", "xgboost"):
        model = build_pd_model(kind, config, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(predictors, labels)
        if kind == "logistic_regression":
            outputs["logistic"] = {
                "coefficients": coefficient_table(model),
                "local": local_logistic(model, evaluation_predictors),
            }
        else:
            shap = tree_shap(model, evaluation_predictors)
            positions, sampling = score_stratified_sample(
                shap["probability"], interaction_rows, seed + 100
            )
            pairs = tree_shap(model, evaluation_predictors.iloc[positions], interactions=True)
            names, interactions = pairs["features"], pairs["interactions"]
            pair_rows = [
                {
                    "feature_a": a,
                    "feature_b": names[j],
                    "mean_abs_pair_contribution": float(2 * np.abs(interactions[:, i, j]).mean()),
                    "mean_signed_pair_contribution": float(2 * interactions[:, i, j].mean()),
                }
                for i, a in enumerate(names)
                for j in range(i + 1, len(names))
            ]
            outputs["xgboost"] = {
                "shap": shap,
                "interactions": pair_rows,
                "interaction_sampling": sampling,
                "max_interaction_error": pairs["max_interaction_error"],
            }
    return outputs


def illustrative_locals(outputs):
    tree, logistic = outputs["xgboost"]["shap"], outputs["logistic"]["local"]
    order = np.argsort(tree["probability"], kind="stable")
    rows = []
    for q in (0.1, 0.5, 0.9):
        index = int(order[int(np.rint(q * (len(order) - 1)))])
        example = {
            "example": f"fold1-p{int(q * 100)}",
            "selection": "first evaluation fold; XGB predicted-score percentile",
            "quantile": q,
            "models": {},
        }
        for name, values in (("logistic", logistic), ("xgboost", tree)):
            baseline = float(
                values["baseline"] if name == "logistic" else values["baseline"][index]
            )
            contributions = dict(
                zip(values["features"], values["contributions"][index].tolist(), strict=True)
            )
            example["models"][name] = {
                "baseline_log_odds": baseline,
                "contributions_log_odds": contributions,
                "margin": float(values["margin"][index]),
                "raw_probability": float(values["probability"][index]),
                "additivity_error": float(
                    abs(baseline + sum(contributions.values()) - values["margin"][index])
                ),
            }
        rows.append(example)
    return rows


def explanation_study(
    frame, config: ModelConfig, *, folds=5, seed=42, interaction_rows=200, progress=None
):
    validate_origination(frame)
    if (
        type(folds) is not int
        or folds < 2
        or type(seed) is not int
        or not 0 <= seed <= 2**32 - 101
        or type(interaction_rows) is not int
        or interaction_rows < 1
    ):
        raise ValueError("Invalid explanation folds, seed or interaction size")
    labels = frame[ORIGINATION_TARGET].to_numpy()
    groups = pd.util.hash_pandas_object(frame.loc[:, ORIGINATION_FEATURES], index=False).to_numpy()
    if any(len(np.unique(groups[labels == label])) < folds for label in (0, 1)):
        raise ValueError("Insufficient outcome groups for explanation folds")
    logistic_folds, tree_folds, design, local_examples, pairs_by_fold = [], [], [], [], []
    oof_shap = {}
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for number, (training, evaluation) in enumerate(splitter.split(frame, labels, groups), 1):
        if any(len(np.unique(labels[indices])) != 2 for indices in (training, evaluation)):
            raise ValueError("Explanation folds require both outcomes")
        if len(np.intersect1d(groups[training], groups[evaluation])):
            raise ValueError("Explanation predictor-group leakage")
        if progress:
            progress(f"Fit and explain fold {number}/{folds}")
        outputs = explain_fold(
            frame.iloc[training],
            frame.iloc[evaluation].loc[:, ORIGINATION_FEATURES],
            config,
            seed,
            interaction_rows,
        )
        logistic_folds.append({"fold": number, "coefficients": outputs["logistic"]["coefficients"]})
        shap = outputs["xgboost"]["shap"]
        magnitudes = {
            name: float(np.abs(shap["contributions"][:, i]).mean())
            for i, name in enumerate(shap["features"])
        }
        ranks = feature_ranks(magnitudes)
        features = []
        for i, name in enumerate(shap["features"]):
            values = shap["contributions"][:, i]
            oof_shap.setdefault(name, np.zeros(len(frame)))[evaluation] = values
            features.append(
                {
                    "feature": name,
                    "mean_abs_shap": magnitudes[name],
                    "mean_signed_shap": float(values.mean()),
                    "rank": ranks[name],
                }
            )
        tree_folds.append(
            {
                "fold": number,
                "evaluation_rows": len(evaluation),
                "features": features,
                "max_additivity_error": shap["max_additivity_error"],
                "max_interaction_error": outputs["xgboost"]["max_interaction_error"],
                "interaction_sampling": outputs["xgboost"]["interaction_sampling"],
            }
        )
        pairs_by_fold.append(outputs["xgboost"]["interactions"])
        design.append(
            {
                "fold": number,
                "training_rows": len(training),
                "evaluation_rows": len(evaluation),
                "evaluation_events": int(labels[evaluation].sum()),
                "group_overlap": 0,
                "training_positions_sha256": sha256(training.astype("<i8").tobytes()).hexdigest(),
                "evaluation_positions_sha256": sha256(
                    evaluation.astype("<i8").tobytes()
                ).hexdigest(),
            }
        )
        if number == 1:
            local_examples = illustrative_locals(outputs)
    coefficient_rows = coefficient_stability(logistic_folds)
    fold_importances = [
        {row["feature"]: row["mean_abs_shap"] for row in fold["features"]} for fold in tree_folds
    ]
    pooled = {name: float(np.abs(values).mean()) for name, values in oof_shap.items()}
    pooled_ranks = feature_ranks(pooled)
    tree_rows = []
    for name in sorted(oof_shap):
        values = oof_shap[name]
        importance = [fold.get(name, 0.0) for fold in fold_importances]
        ranks = [
            feature_ranks({key: fold.get(key, 0.0) for key in oof_shap})[name]
            for fold in fold_importances
        ]
        tree_rows.append(
            {
                "feature": name,
                "pooled_mean_abs_shap": pooled[name],
                "global_rank": pooled_ranks[name],
                "fold_mean_abs_shap": importance,
                "fold_ranks": ranks,
                "fold_importance": descriptive(importance),
                "rank": descriptive(ranks),
                "present_folds": sum(name in fold for fold in fold_importances),
                "distribution": dict(
                    zip(
                        ("p05", "p25", "median", "p75", "p95"),
                        np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                        strict=True,
                    )
                ),
            }
        )
    logistic_magnitudes = {
        name: float(
            np.mean(
                [
                    abs(
                        next(
                            row["standardized_coefficient"]
                            for row in fold["coefficients"]
                            if row["feature"] == name
                        )
                    )
                    for fold in logistic_folds
                ]
            )
        )
        for name in ORIGINATION_FEATURES
    }
    main_logistic_ranks = feature_ranks(logistic_magnitudes)
    main_tree_ranks = feature_ranks({name: pooled[name] for name in ORIGINATION_FEATURES})
    stability_lookup = {row["feature"]: row for row in coefficient_rows}
    comparison = [
        {
            "feature": name,
            "logistic_standardized_rank": main_logistic_ranks[name],
            "logistic_mean_abs_standardized_coefficient": logistic_magnitudes[name],
            "logistic_direction": stability_lookup[name]["dominant_sign"],
            "logistic_sign_flip": stability_lookup[name]["sign_flip"],
            "xgboost_shap_rank": main_tree_ranks[name],
        }
        for name in ORIGINATION_FEATURES
    ]
    top = sorted(ORIGINATION_FEATURES, key=lambda name: (-pooled[name], name))[:5]
    dependence = {
        name: dependence_summary(frame[name].to_numpy(dtype=float), oof_shap[name]) for name in top
    }
    pairs = {}
    for fold_pairs in pairs_by_fold:
        for pair in fold_pairs:
            key = (pair["feature_a"], pair["feature_b"])
            pairs.setdefault(key, []).append(pair["mean_abs_pair_contribution"])
    interactions = [
        {
            "feature_a": key[0],
            "feature_b": key[1],
            "mean_abs_pair_contribution": float(np.mean(values)),
            "fold_values": values,
            "estimated_folds": len(values),
        }
        for key, values in pairs.items()
    ]
    interactions.sort(
        key=lambda row: (-row["mean_abs_pair_contribution"], row["feature_a"], row["feature_b"])
    )
    return {
        "schema_version": 1,
        "target": "two-year serious-delinquency outcome",
        "rows": len(frame),
        "seed": seed,
        "model_config": config.model_dump(),
        "fold_design": design,
        "logistic_folds": logistic_folds,
        "logistic_stability": coefficient_rows,
        "xgboost_folds": tree_folds,
        "xgboost_stability": tree_rows,
        "xgboost_rank_agreement": rank_agreement(fold_importances, ORIGINATION_FEATURES),
        "cross_model_comparison": comparison,
        "dependence": dependence,
        "leading_interactions": interactions[:3],
        "local_examples": local_examples,
        "methodology": {
            "folds": folds,
            "global_shap_rows": len(frame),
            "interaction_rows_per_fold": interaction_rows,
            "shap_implementation": "XGBoost native exact TreeSHAP (approx_contribs=False)",
            "shap_scale": "raw margin / log-odds",
            "background": "tree-path-dependent training cover; no external background",
            "global_importance": (
                "row-weighted pooled mean absolute SHAP; fold means/SD descriptive"
            ),
            "missing_indicator_policy": (
                "separate transformed columns; absent tree columns "
                "contribute zero, absent coefficients are not estimated"
            ),
            "interaction_pair_scale": "2 * off-diagonal SHAP interaction allocation; raw margin",
            "local_selection": (
                "fold 1 XGB score p10/p50/p90, deterministic ties by evaluation order"
            ),
        },
        "historical_metrics": {
            "status": "retained historical, not reevaluated",
            "roc_auc": 0.868152,
            "brier": 0.048545,
            "log_loss": 0.176030,
        },
    }


def run_explainability_study(root, source, *, progress=None):
    root = Path(root)
    frame, provenance = load_training_only(
        source, root / "artifacts/phase4-origination-001", root / "reports/holdout_registry.json"
    )
    protected = [
        root / "reports/holdout_registry.json",
        *sorted((root / "reports/model_validation").glob("*")),
    ]
    before = {
        str(path.relative_to(root)).replace("\\", "/"): file_digest(path)
        for path in protected
        if path.is_file()
        and path.name
        in {
            "holdout_registry.json",
            "PD_DIAGNOSTICS.md",
            "pd_diagnostics.json",
            "pd_diagnostics.png",
            "CALIBRATION_STUDY.md",
            "calibration_study.json",
            "calibration_comparison.png",
        }
    }
    result = explanation_study(
        frame, load_config(root / "configs/model.yaml", ModelConfig), progress=progress
    )
    result["provenance"] = provenance
    result["preserved_evidence_sha256"] = before
    result["versions"] = {
        name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "scipy")
    }
    result["source_code_sha256"] = {
        str(path.relative_to(root)).replace("\\", "/"): file_digest(path)
        for path in sorted((root / "src/credit_risk/explainability").glob("*.py"))
    }
    for name in (
        "models/pd.py",
        "features/origination.py",
        "utils/config.py",
        "validation/metrics.py",
        "validation/pd_diagnostics.py",
    ):
        relative = "src/credit_risk/" + name
        result["source_code_sha256"][relative] = file_digest(root / relative)
    result["source_code_sha256"]["scripts/explainability_study.py"] = file_digest(
        root / "scripts/explainability_study.py"
    )
    if any(file_digest(root / name) != digest for name, digest in before.items()):
        raise ValueError("Existing evidence or registry changed during explanation study")
    return result
