"""Reference-only numeric bins, full-population PSI and separate missingness."""

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from credit_risk.monitoring.settings import MonitoringConfig


def numeric_vector(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or np.isinf(x).any():
        raise ValueError("Numeric monitoring vectors must be nonempty, with no infinity")
    return x


def population_stability_index(reference_counts, current_counts, smoothing=1e-6):
    """PSI uses additive proportion smoothing and renormalizes both distributions."""
    r = np.asarray(reference_counts, dtype=float)
    c = np.asarray(current_counts, dtype=float)
    if r.ndim != 1 or r.shape != c.shape or not len(r):
        raise ValueError("Counts must be aligned nonempty vectors")
    if not np.isfinite(r).all() or not np.isfinite(c).all() or (r < 0).any() or (c < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if r.sum() <= 0 or c.sum() <= 0 or not 0 < smoothing < 0.1:
        raise ValueError("Both populations and smoothing must be positive")
    p = (r / r.sum() + smoothing) / (1 + len(r) * smoothing)
    q = (c / c.sum() + smoothing) / (1 + len(c) * smoothing)
    contribution = (q - p) * np.log(q / p)
    return dict(
        psi=float(contribution.sum()),
        reference_shares=p.tolist(),
        current_shares=q.tolist(),
        contributions=contribution.tolist(),
    )


def fit_numeric_reference(values, bins=10):
    x = numeric_vector(values)
    finite = x[~np.isnan(x)]
    if not len(finite):
        kind, cuts = "all_missing", []
    elif np.min(finite) == np.max(finite):
        kind, cuts = "constant", [float(finite[0])]
    else:
        kind = "quantile"
        cuts = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1])).tolist()
    profile = dict(
        kind=kind,
        cuts=cuts,
        rows=len(x),
        numeric_values=np.sort(finite).tolist(),
        minimum=float(finite.min()) if len(finite) else None,
        maximum=float(finite.max()) if len(finite) else None,
        mean=float(finite.mean()) if len(finite) else None,
        std=float(finite.std()) if len(finite) else None,
        iqr=float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)) if len(finite) else None,
        missing_rate=float(np.isnan(x).mean()),
    )
    profile["counts"] = bucket_counts(x, profile)
    return profile


def bucket_counts(values, profile):
    x = numeric_vector(values)
    finite = x[~np.isnan(x)]
    if profile["kind"] == "all_missing":
        counts = [len(finite)]
    elif profile["kind"] == "constant":
        point = profile["cuts"][0]
        counts = [
            int((finite < point).sum()),
            int((finite == point).sum()),
            int((finite > point).sum()),
        ]
    elif profile["kind"] == "quantile":
        cuts = np.asarray(profile["cuts"], dtype=float)
        counts = np.bincount(
            np.searchsorted(cuts, finite, side="left"), minlength=len(cuts) + 1
        ).tolist()
    else:
        raise ValueError("Unknown reference bin kind")
    return [*counts, int(np.isnan(x).sum())]


def pd_to_score(probabilities, config):
    p = numeric_vector(probabilities)
    if np.isnan(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("Model PD must be finite and in [0,1]")
    clipped = np.clip(p, config.score_probability_epsilon, 1 - config.score_probability_epsilon)
    factor = config.score_points_to_double_odds / np.log(2)
    return config.score_base + factor * (
        np.log1p(-clipped) - np.log(clipped) - np.log(config.score_base_good_bad_odds)
    )


def monitoring_frame(features, probabilities, config):
    if (
        features.empty
        or features.columns.duplicated().any()
        or {"pd", "score"} & set(features.columns)
    ):
        raise ValueError("Features must be nonempty, uniquely named and exclude pd/score")
    p = numeric_vector(probabilities)
    if len(p) != len(features):
        raise ValueError("PD must align positionally with features")
    frame = features.copy()
    frame["pd"] = p
    frame["score"] = pd_to_score(p, config)
    return frame


def fit_reference(features, probabilities, config=None):
    config = config or MonitoringConfig()
    frame = monitoring_frame(features, probabilities, config)
    return dict(
        measurement_contract=config.measurement_contract(),
        feature_names=list(features.columns),
        rows=len(frame),
        profiles={name: fit_numeric_reference(frame[name], config.bins) for name in frame},
    )


def severity(value, threshold):
    if value is None:
        return "UNAVAILABLE"
    if value >= threshold.critical:
        return "CRITICAL"
    if value >= threshold.warning:
        return "WARNING"
    return "OK"


def compare_reference(reference, features, probabilities, config=None):
    config = config or MonitoringConfig()
    if reference["measurement_contract"] != config.measurement_contract():
        raise ValueError("Measurement settings changed; freeze a new reference explicitly")
    if list(features.columns) != reference["feature_names"]:
        raise ValueError("Current feature names/order differ from frozen reference")
    frame = monitoring_frame(features, probabilities, config)
    metrics = {}
    alerts = []
    for name in frame:
        profile = reference["profiles"][name]
        x = numeric_vector(frame[name])
        current = x[~np.isnan(x)]
        prior = np.asarray(profile["numeric_values"], dtype=float)
        counts = bucket_counts(x, profile)
        stability = population_stability_index(profile["counts"], counts, config.smoothing)
        missing = float(np.isnan(x).mean())
        missing_delta = missing - profile["missing_rate"]
        sufficient = min(len(prior), len(current)) >= config.minimum_numeric_rows
        ks_result = ks_2samp(prior, current, method="asymp") if sufficient else None
        distance = float(wasserstein_distance(prior, current)) if sufficient else None
        scale = profile["iqr"] or profile["std"]
        out_of_range = (
            float(((current < profile["minimum"]) | (current > profile["maximum"])).mean())
            if len(prior) and len(current)
            else None
        )
        mean = float(current.mean()) if len(current) else None
        mean_delta = (
            mean - profile["mean"] if mean is not None and profile["mean"] is not None else None
        )
        result = dict(
            channel="feature_csi_equivalent" if name in features else name,
            **stability,
            reference_counts=profile["counts"],
            current_counts=counts,
            bin_kind=profile["kind"],
            bin_cuts=profile["cuts"],
            missing_bucket="last",
            reference_missing_rate=profile["missing_rate"],
            current_missing_rate=missing,
            missing_rate_delta=missing_delta,
            reference_mean=profile["mean"],
            current_mean=mean,
            mean_delta=mean_delta,
            reference_numeric_rows=len(prior),
            current_numeric_rows=len(current),
            continuous_metrics_status="available" if sufficient else "insufficient_numeric_rows",
            ks_statistic=float(ks_result.statistic) if ks_result is not None else None,
            ks_pvalue_descriptive=float(ks_result.pvalue)
            if ks_result is not None and np.isfinite(ks_result.pvalue)
            else None,
            wasserstein_distance=distance,
            wasserstein_reference_scale=distance / scale
            if distance is not None and scale
            else None,
            out_of_reference_range_fraction=out_of_range,
        )
        checks = {
            "psi": (result["psi"], config.psi),
            "missingness_delta": (abs(missing_delta), config.missingness),
            "ks_statistic": (result["ks_statistic"], config.ks),
            "out_of_range": (out_of_range, config.out_of_range),
        }
        if name in ("pd", "score"):
            checks["mean_delta"] = (
                abs(mean_delta) if mean_delta is not None else None,
                config.pd_mean if name == "pd" else config.score_mean,
            )
        result["alerts"] = {key: severity(value, t) for key, (value, t) in checks.items()}
        for key, (value, t) in checks.items():
            level = severity(value, t)
            if level in ("WARNING", "CRITICAL"):
                alerts.append(
                    dict(
                        variable=name,
                        metric=key,
                        value=value,
                        severity=level,
                        warning_threshold=t.warning,
                        critical_threshold=t.critical,
                    )
                )
        metrics[name] = result
    sufficient_population = min(reference["rows"], len(frame)) >= config.minimum_population_rows
    level = (
        "CRITICAL"
        if any(a["severity"] == "CRITICAL" for a in alerts)
        else "WARNING"
        if alerts
        else "OK"
    )
    return dict(
        status=level if sufficient_population else "INSUFFICIENT_DATA",
        sufficient_population=sufficient_population,
        reference_rows=reference["rows"],
        current_rows=len(frame),
        alerts=alerts,
        metrics=metrics,
        outcome_validation={
            "status": "unavailable",
            "reason": "No mature dated outcomes; population drift does not establish concept drift",
        },
    )
