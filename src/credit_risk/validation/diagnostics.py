"""Reliability bins and paired group-bootstrap uncertainty for frozen predictions."""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from credit_risk.validation.metrics import binary_metrics


def calibration_edges(probabilities, n_bins: int = 10) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid calibration probabilities")
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("At least two requested calibration bins required")
    return np.unique(np.r_[0.0, np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 1.0])


def reliability_diagnostics(labels, probabilities, edges, confidence: float = 0.95) -> dict:
    binary_metrics(labels, probabilities)
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or edges[0] != 0
        or edges[-1] != 1
        or not (np.diff(edges) > 0).all()
    ):
        raise ValueError("Edges must increase strictly from zero to one")
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be in (0,1)")
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    z = float(norm.ppf(0.5 + confidence / 2))
    rows, ece = [], 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = bins == index
        count = int(mask.sum())
        row = {
            "lower": float(lower),
            "upper": float(upper),
            "rows": count,
            "mean_probability": None,
            "observed_bad_rate": None,
            "wilson_lower": None,
            "wilson_upper": None,
        }
        if count:
            observed, predicted = float(y[mask].mean()), float(p[mask].mean())
            denominator = 1 + z * z / count
            center = (observed + z * z / (2 * count)) / denominator
            half = (
                z
                * np.sqrt(observed * (1 - observed) / count + z * z / (4 * count * count))
                / denominator
            )
            row.update(
                mean_probability=predicted,
                observed_bad_rate=observed,
                wilson_lower=float(max(0, center - half)),
                wilson_upper=float(min(1, center + half)),
            )
            ece += count / len(y) * abs(observed - predicted)
        rows.append(row)
    return {
        "bins": rows,
        "ece": float(ece),
        "observed_expected_ratio": float(y.sum() / p.sum()) if p.sum() > 0 else None,
        "bin_interval_type": "approximate row-binomial Wilson; not borrower-cluster robust",
    }


def group_bootstrap_comparison(
    labels,
    predictions: dict[str, np.ndarray],
    groups,
    samples: int = 200,
    confidence: float = 0.95,
    seed: int = 43,
) -> dict:
    y = np.asarray(labels)
    groups = np.asarray(groups)
    if groups.shape != y.shape or not len(predictions) or samples < 20 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap inputs")
    for probabilities in predictions.values():
        binary_metrics(y, probabilities)
    _, membership = np.unique(groups, return_inverse=True)
    n_groups = int(membership.max()) + 1
    rng = np.random.default_rng(seed)
    values = {
        name: {
            metric: [] for metric in ("roc_auc", "gini", "brier", "log_loss", "average_precision")
        }
        for name in predictions
    }
    skipped = 0
    for _ in range(samples):
        group_weights = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)
        weights = group_weights[membership]
        if len(np.unique(y[weights > 0])) < 2:
            skipped += 1
            continue
        for name, probability in predictions.items():
            p = np.asarray(probability, dtype=float)
            safe = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
            auc = float(roc_auc_score(y, p, sample_weight=weights))
            result = {
                "roc_auc": auc,
                "gini": 2 * auc - 1,
                "brier": float(np.average((p - y) ** 2, weights=weights)),
                "log_loss": float(
                    np.average(-y * np.log(safe) - (1 - y) * np.log1p(-safe), weights=weights)
                ),
                "average_precision": float(average_precision_score(y, p, sample_weight=weights)),
            }
            for metric, value in result.items():
                values[name][metric].append(value)
    tail = (1 - confidence) / 2

    def interval(observations):
        if not observations:
            return {"lower": None, "upper": None}
        bounds = np.quantile(observations, [tail, 1 - tail])
        return {"lower": float(bounds[0]), "upper": float(bounds[1])}

    result = {
        "confidence_level": confidence,
        "requested_samples": samples,
        "valid_samples": samples - skipped,
        "skipped_single_class": skipped,
        "resampling_unit": "exact-predictor group",
        "interval_type": "percentile, frozen models",
        "intervals": {
            name: {metric: interval(data) for metric, data in metrics.items()}
            for name, metrics in values.items()
        },
        "paired_differences": {},
    }
    names = list(predictions)
    if len(names) == 2:
        result["paired_differences"] = {
            "direction": f"{names[1]} minus {names[0]}",
            "intervals": {
                metric: interval(
                    (
                        np.asarray(values[names[1]][metric]) - np.asarray(values[names[0]][metric])
                    ).tolist()
                )
                for metric in values[names[0]]
            },
        }
    return result
