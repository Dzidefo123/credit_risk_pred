"""Initial discrimination and raw-probability metrics, with explicit definitions."""

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true, probabilities, threshold: float = 0.1) -> dict[str, float | None]:
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Labels and probabilities must be equal-length nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Labels must be binary and probabilities finite in [0,1]")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be in (0,1)")
    prediction = p >= threshold
    result = {
        "roc_auc": None,
        "gini": None,
        "ks": None,
        "pr_auc": None,
        "average_precision": None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "observed_bad_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "classification_threshold": float(threshold),
    }
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        result.update(
            roc_auc=float(roc_auc_score(y, p)),
            gini=float(2 * roc_auc_score(y, p) - 1),
            ks=float(np.max(np.abs(tpr - fpr))),
            pr_auc=float(auc(recall, precision)),
            average_precision=float(average_precision_score(y, p)),
        )
    return result
