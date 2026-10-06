"""Diagnostic logistic recalibration fits; these never change predictions."""

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit

from credit_risk.validation.diagnostics import calibration_edges, reliability_diagnostics
from credit_risk.validation.metrics import binary_metrics


def calibration_diagnostics(labels, probabilities, *, epsilon=1e-12):
    """CITL uses a fixed slope of one; the separate slope fit estimates both terms.

    Clip only diagnostic logits to [epsilon, 1-epsilon], avoiding infinite logits
    at exact endpoints. No probability used for Brier/log loss is changed.
    CIs are deliberately omitted: cross-fitted predictions share training data,
    and borrower clusters are unknown. An iid Wald interval is not justified here.
    """
    binary_metrics(labels, probabilities)
    if not np.isfinite(epsilon) or not np.finfo(float).eps <= epsilon < 0.5:
        raise ValueError("epsilon must be finite, at least machine epsilon, and below 0.5")
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    safe = np.clip(p, epsilon, 1 - epsilon)
    x = np.log(safe) - np.log1p(-safe)
    result = {
        "epsilon": epsilon,
        "clipped_rows": int(np.count_nonzero(safe != p)),
        "calibration_intercept": None,
        "joint_intercept": None,
        "calibration_slope": None,
        "slope_status": "undefined",
        "confidence_intervals": None,
        "interval_reason": "No iid assumption for shared-fit OOF predictions or unknown borrowers",
    }
    if len(np.unique(y)) != 2:
        result["slope_status"] = "single-class outcome: no finite logistic MLE"
        return result
    result["calibration_intercept"] = float(brentq(lambda a: np.sum(expit(a + x) - y), -100, 100))
    if np.ptp(x) < 1e-10:
        result["slope_status"] = "constant logits: joint slope unidentifiable"
        return result
    # Complete/quasi separation has no finite unpenalized logistic MLE.
    if x[y == 0].max() <= x[y == 1].min() or x[y == 1].max() <= x[y == 0].min():
        result["slope_status"] = "separated outcomes: no finite joint MLE"
        return result
    design = np.column_stack([np.ones(len(x)), x])

    def objective(beta):
        linear = design @ beta
        return float(np.mean(np.logaddexp(0, linear) - y * linear))

    def gradient(beta):
        return design.T @ (expit(design @ beta) - y) / len(y)

    fit = minimize(objective, [0.0, 1.0], jac=gradient, method="BFGS", options={"gtol": 1e-9})
    fitted = expit(design @ fit.x)
    information = design.T @ ((fitted * (1 - fitted))[:, None] * design)
    if (
        not np.isfinite(fit.x).all()
        or np.max(np.abs(gradient(fit.x))) > 1e-7
        or np.linalg.cond(information) > 1e12
    ):
        result["slope_status"] = "joint MLE failed convergence/information check"
        return result
    result.update(
        joint_intercept=float(fit.x[0]), calibration_slope=float(fit.x[1]), slope_status="estimated"
    )
    return result


def probability_diagnostics(labels, probabilities):
    """Existing reliability bins with counts, sparse flags and logistic diagnostics."""
    metrics = binary_metrics(labels, probabilities)
    reliability = reliability_diagnostics(labels, probabilities, calibration_edges(probabilities))
    for row in reliability["bins"]:
        row["sparse"] = row["rows"] < 100
    return {
        "metrics": metrics,
        "reliability": reliability,
        "calibration": calibration_diagnostics(labels, probabilities),
    }
