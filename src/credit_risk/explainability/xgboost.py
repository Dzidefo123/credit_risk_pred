"""Native exact tree-path-dependent TreeSHAP on raw-margin/log-odds scale."""

import numpy as np
from scipy.special import expit
from xgboost import DMatrix

from credit_risk.explainability.logistic import transformed_names
from credit_risk.models.pd import predict_probability


def tree_shap(pipeline, predictors, *, interactions=False):
    names = transformed_names(pipeline)
    matrix = np.asarray(pipeline[:-1].transform(predictors), dtype=float)
    booster = pipeline.named_steps["model"].get_booster()
    if (
        matrix.ndim != 2
        or not len(matrix)
        or matrix.shape[1] != len(names)
        or booster.num_features() != len(names)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("TreeSHAP transformed feature alignment is invalid")
    data = DMatrix(matrix, feature_names=names)
    values = np.asarray(
        booster.predict(data, pred_contribs=True, approx_contribs=False), dtype=float
    )
    margin = np.asarray(booster.predict(data, output_margin=True), dtype=float)
    if values.shape != (len(matrix), len(names) + 1) or not np.isfinite(values).all():
        raise ValueError("Invalid TreeSHAP dimensions/values")
    error = float(np.max(np.abs(values.sum(axis=1) - margin)))
    if not np.allclose(values.sum(axis=1), margin, atol=1e-5, rtol=1e-5):
        raise ValueError("TreeSHAP local additivity failed")
    if not np.allclose(
        expit(margin), predict_probability(pipeline, predictors), atol=1e-6, rtol=1e-6
    ):
        raise ValueError("TreeSHAP margin disagrees with pipeline probabilities")
    result = {
        "features": names,
        "contributions": values[:, :-1],
        "baseline": values[:, -1],
        "margin": margin,
        "probability": expit(margin),
        "max_additivity_error": error,
        "scale": "raw margin / log-odds",
    }
    if interactions:
        pair_values = np.asarray(
            booster.predict(data, pred_interactions=True, approx_contribs=False), dtype=float
        )
        if (
            pair_values.shape != (len(matrix), len(names) + 1, len(names) + 1)
            or not np.isfinite(pair_values).all()
        ):
            raise ValueError("Invalid SHAP interaction dimensions/values")
        if not np.allclose(
            pair_values.sum(axis=2), values, atol=1e-5, rtol=1e-5
        ) or not np.allclose(pair_values, pair_values.transpose(0, 2, 1), atol=1e-5, rtol=1e-5):
            raise ValueError("SHAP interaction additivity/symmetry failed")
        result["interactions"] = pair_values[:, :-1, :-1]
        result["max_interaction_error"] = float(np.max(np.abs(pair_values.sum(axis=2) - values)))
    return result
