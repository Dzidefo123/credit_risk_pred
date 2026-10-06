"""Logistic associations in explicitly identified preprocessing coordinates."""

import numpy as np
from scipy.special import expit


def transformed_names(pipeline):
    names = list(
        pipeline.named_steps["imputer"].get_feature_names_out(
            pipeline.named_steps["features"].get_feature_names_out()
        )
    )
    if len(set(names)) != len(names):
        raise ValueError("Duplicate transformed feature names")
    return names


def coefficient_table(pipeline):
    names = transformed_names(pipeline)
    scaler, model = pipeline.named_steps["scaler"], pipeline.named_steps["model"]
    beta = np.asarray(model.coef_, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)
    if (
        beta.shape != (1, len(names))
        or scales.shape != (len(names),)
        or not np.isfinite(beta).all()
        or not np.isfinite(scales).all()
        or (scales <= 0).any()
    ):
        raise ValueError("Invalid binary logistic coefficient/scaler alignment")
    rows = []
    for name, coefficient, scale, variance in zip(names, beta[0], scales, scaler.var_, strict=True):
        transformed_beta = coefficient / scale
        if max(abs(coefficient), abs(transformed_beta)) > 700:
            raise ValueError("Odds ratio would overflow")
        rows.append(
            {
                "feature": name,
                "standardized_coefficient": float(coefficient),
                "transformed_coefficient": float(transformed_beta),
                "training_scale": float(scale),
                "constant_training_feature": bool(variance == 0),
                "odds_ratio_per_training_sd": float(np.exp(coefficient)),
                "odds_ratio_per_transformed_unit": float(np.exp(transformed_beta)),
                "direction": "positive"
                if coefficient > 1e-8
                else "negative"
                if coefficient < -1e-8
                else "near_zero",
            }
        )
    return rows


def local_logistic(pipeline, predictors):
    names = transformed_names(pipeline)
    z = np.asarray(pipeline[:-1].transform(predictors), dtype=float)
    model = pipeline.named_steps["model"]
    if z.ndim != 2 or z.shape[1] != len(names) or not len(z) or not np.isfinite(z).all():
        raise ValueError("Invalid transformed logistic predictors")
    contributions = z * model.coef_[0]
    baseline = float(model.intercept_[0])
    margin = np.asarray(pipeline.decision_function(predictors), dtype=float)
    if not np.allclose(baseline + contributions.sum(axis=1), margin, atol=1e-10, rtol=1e-10):
        raise ValueError("Logistic contribution additivity failed")
    return {
        "features": names,
        "contributions": contributions,
        "baseline": baseline,
        "margin": margin,
        "probability": expit(margin),
        "scale": "raw log-odds",
    }
