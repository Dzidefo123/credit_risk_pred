"""Explicit research thresholds; no underwriting optimum is implied."""

import numpy as np

from credit_risk.validation.metrics import binary_metrics


def threshold_diagnostics(labels, probabilities, *, threshold):
    binary_metrics(labels, probabilities)  # Reuse the existing input contract.
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    y, predicted = np.asarray(labels), np.asarray(probabilities) >= threshold
    tp = int(np.sum((y == 1) & predicted))
    tn = int(np.sum((y == 0) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "threshold": float(threshold),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "balanced_accuracy": (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None,
        "predicted_positive_rate": float(predicted.mean()),
    }


def threshold_analysis(labels, probabilities, *, thresholds):
    values = list(thresholds)
    if not values:
        raise ValueError("Supply at least one explicit threshold")
    return [threshold_diagnostics(labels, probabilities, threshold=t) for t in values]


def select_threshold(labels, probabilities, *, thresholds, objective, partition):
    """Select on explicitly identified training/development data, never final test.

    Caller must truthfully identify provenance; this generic helper cannot infer
    it from arrays. The repository workflow separately verifies saved train rows.
    On ties choose the larger supplied threshold (fewer flagged observations).
    """
    if partition not in {"train", "development"}:
        raise ValueError("Threshold selection requires train/development data")
    if objective not in {"f1", "youden_j"}:
        raise ValueError("objective must be f1 or youden_j")
    rows = threshold_analysis(labels, probabilities, thresholds=thresholds)
    if len(np.unique(labels)) != 2:
        raise ValueError("Threshold selection requires both outcomes")

    def value(row):
        return row["f1"] if objective == "f1" else row["recall"] + row["specificity"] - 1

    chosen = max(rows, key=lambda row: (value(row), row["threshold"]))
    return {
        "threshold": chosen["threshold"],
        "objective": objective,
        "objective_value": value(chosen),
        "selection_partition": partition,
        "status": "research objective, not a lending optimum",
    }
