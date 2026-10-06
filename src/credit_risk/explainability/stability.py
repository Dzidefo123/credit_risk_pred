"""Descriptive fold stability; no independent-fold or temporal inference."""

import numpy as np
from scipy.stats import rankdata, spearmanr


def feature_ranks(values):
    if not values or any(not isinstance(k, str) for k in values):
        raise ValueError("Nonempty named feature magnitudes required")
    names = sorted(values)
    x = np.asarray([values[name] for name in names], dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Feature magnitudes must be finite and nonnegative")
    return dict(zip(names, rankdata(-x, method="average").tolist(), strict=True))


def sign_stability(coefficients, *, tolerance=1e-8):
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or tolerance < 0:
        raise ValueError("Invalid coefficients/sign tolerance")
    counts = {
        "positive": int(np.sum(values > tolerance)),
        "negative": int(np.sum(values < -tolerance)),
        "near_zero": int(np.sum(np.abs(values) <= tolerance)),
    }
    return {
        "sign_counts": counts,
        "sign_consistency": max(counts.values()) / len(values),
        "sign_flip": counts["positive"] > 0 and counts["negative"] > 0,
        "dominant_sign": max(counts, key=counts.get),
        "sign_tolerance": tolerance,
    }


def descriptive(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite descriptive observations required")
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else None,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def coefficient_stability(folds):
    if len(folds) < 2:
        raise ValueError("At least two coefficient folds required")
    names = sorted({row["feature"] for fold in folds for row in fold["coefficients"]})
    lookups = [{row["feature"]: row for row in fold["coefficients"]} for fold in folds]
    rankings = [
        feature_ranks({name: abs(row["standardized_coefficient"]) for name, row in lookup.items()})
        for lookup in lookups
    ]
    rows = []
    for name in names:
        values = [lookup[name]["standardized_coefficient"] for lookup in lookups if name in lookup]
        ranks = [rank[name] for rank in rankings if name in rank]
        rows.append(
            {
                "feature": name,
                "estimated_folds": len(values),
                "fold_coefficients": [
                    lookup[name]["standardized_coefficient"] if name in lookup else None
                    for lookup in lookups
                ],
                "fold_ranks": [rank.get(name) for rank in rankings],
                "coefficient": descriptive(values),
                "rank": descriptive(ranks),
                **sign_stability(values),
            }
        )
    return rows


def rank_agreement(fold_importances, features):
    features = list(features)
    if len(features) < 2 or len(fold_importances) < 2:
        raise ValueError("At least two aligned features/folds required")
    vectors = [
        np.array([feature_ranks(fold)[name] for name in features]) for fold in fold_importances
    ]
    pairs = []
    for i, left in enumerate(vectors):
        for j in range(i + 1, len(vectors)):
            right = vectors[j]
            rho = (
                None
                if np.ptp(left) == 0 or np.ptp(right) == 0
                else float(spearmanr(left, right).statistic)
            )
            pairs.append({"fold_a": i + 1, "fold_b": j + 1, "spearman_rho": rho})
    valid = [p["spearman_rho"] for p in pairs if p["spearman_rho"] is not None]
    return {
        "pairs": pairs,
        "mean_rho": float(np.mean(valid)) if valid else None,
        "interpretation": "descriptive rank agreement; folds share training data",
    }
