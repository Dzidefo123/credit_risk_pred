"""Proper scores, diagnostic calibration and paired facility/calendar uncertainty."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

SCORES = ("joint_log_loss", "default_brier", "payoff_brier")


def losses(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if (
        p.shape != (len(y), 3)
        or not np.isin(y, [0, 1, 2]).all()
        or (
            not np.isfinite(p).all()
            or (p < 0).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-10)
        )
    ):
        raise ValueError("Invalid coherent event probabilities")
    return np.column_stack(
        [
            -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)),
            (p[:, 1] - (y == 1)) ** 2,
            (p[:, 2] - (y == 2)) ** 2,
        ]
    )


def scores(data, p, suppress=False):
    y = data["event"]
    result = {n: float(v) for n, v in zip(SCORES, losses(y, p).mean(axis=0), strict=True)}
    result.update(
        intervals=len(y),
        facilities=len(np.unique(data["facility"])),
        defaults=int(np.count_nonzero(y == 1)),
        payoffs=int(np.count_nonzero(y == 2)),
    )
    for code, cause in [(1, "default"), (2, "payoff")]:
        supported = np.count_nonzero(y == code) >= 20 and np.count_nonzero(y != code) >= 20
        result[cause + "_auc"] = float(roc_auc_score(y == code, p[:, code])) if supported else None
        result[cause + "_status"] = "SUPPORTED" if supported else "SPARSE_CAUSE_SUPPRESSED"
        if suppress and not supported:
            result[cause + "_brier"] = None
    return result


def calibration(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    result = dict(
        observed_rate=float(y.mean()),
        mean_predicted=float(p.mean()),
        absolute_mean_rate_error=float(abs(p.mean() - y.mean())),
        evaluation_recalibration_applied=False,
    )
    if y.sum() < 20 or (1 - y).sum() < 20:
        result.update(status="SPARSE_CAUSE_SUPPRESSED", intercept=None, slope=None, reliability=[])
        return result
    x = logit(np.clip(p, 1e-12, 1 - 1e-12))

    def objective(ab):
        eta = ab[0] + ab[1] * x
        residual = expit(eta) - y
        return float(np.mean(np.logaddexp(0, eta) - y * eta)), np.array(
            [residual.mean(), (residual * x).mean()]
        )

    fitted = minimize(
        objective,
        np.array([0.0, 1.0]),
        jac=True,
        method="L-BFGS-B",
        options=dict(maxiter=1000, ftol=1e-12, gtol=1e-8),
    )
    bins = np.array_split(np.argsort(p, kind="stable"), 10)
    result.update(
        status="DIAGNOSTIC_ONLY" if fitted.success else "DIAGNOSTIC_FIT_FAILED",
        intercept=float(fitted.x[0]) if fitted.success else None,
        slope=float(fitted.x[1]) if fitted.success else None,
        reliability=[
            dict(
                n=len(i),
                events=int(y[i].sum()),
                observed=float(y[i].mean()),
                predicted=float(p[i].mean()),
            )
            for i in bins
        ],
    )
    return result


def auc_cache(y, p, groups):
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    return groups[order], np.asarray(y[order], float), starts


def weighted_auc(cache, weights):
    groups, y, starts = cache
    w = weights[groups]
    positives = np.add.reduceat(w * y, starts)
    negatives = np.add.reduceat(w * (1 - y), starts)
    positive, negative = positives.sum(), negatives.sum()
    if positive < 20 or negative < 20:
        return None
    return float(positives @ (np.cumsum(negatives) - negatives / 2) / (positive * negative))


def paired(data, one, two, seed=61035, draws=1000, unit="facility", ranking=True):
    grouping = data["facility"] if unit == "facility" else data["month"] // 12
    unique, index = np.unique(grouping, return_inverse=True)
    count = np.bincount(index).astype(float)
    difference = losses(data["event"], two) - losses(data["event"], one)
    sums = np.column_stack([np.bincount(index, weights=difference[:, i]) for i in range(3)])
    aucs = {}
    if ranking:
        for code, cause in [(1, "default"), (2, "payoff")]:
            aucs[cause] = (
                auc_cache(data["event"] == code, one[:, code], index),
                auc_cache(data["event"] == code, two[:, code], index),
            )
    rng = np.random.default_rng(seed)
    values = {n: [] for n in [*SCORES, *[c + "_auc" for c in aucs]]}
    for _ in range(draws):
        weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
        for name, v in zip(SCORES, weights @ sums / (weights @ count), strict=True):
            values[name].append(float(v))
        for cause, (a, b) in aucs.items():
            first, second = weighted_auc(a, weights), weighted_auc(b, weights)
            if first is not None and second is not None:
                values[cause + "_auc"].append(second - first)
    first_scores, second_scores = scores(data, one), scores(data, two)
    intervals = {}
    for n, vals in values.items():
        enough = len(vals) >= min(draws, 950)
        intervals[n] = dict(
            delta=second_scores[n] - first_scores[n] if first_scores[n] is not None else None,
            lower=float(np.quantile(vals, 0.025)) if enough else None,
            upper=float(np.quantile(vals, 0.975)) if enough else None,
            valid_replicates=len(vals),
            status="SUPPORTED" if enough else "INSUFFICIENT_REPLICATES",
        )
    return dict(
        unit=unit,
        clusters=len(unique),
        seed=seed,
        draws=draws,
        intervals=intervals,
        orientation="M2 minus M1; negative proper score is better",
        fixed_models=True,
        conditional_on_realized_calendar=unit == "facility",
    )
