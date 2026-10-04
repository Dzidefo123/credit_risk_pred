"""Masked-outcome learning and IPW; synthetic truth is never an inference input."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.decisioning.reject_settings import RejectInferenceConfig

FEATURES = ("risk_factor", "debt_factor", "income_factor")
SCENARIOS = ("mar_overlap", "mnar_hidden", "deterministic_no_overlap")


@dataclass(frozen=True)
class SyntheticSelection:
    features: pd.DataFrame
    accepted: np.ndarray
    observed_outcome: np.ndarray
    oracle_outcome: np.ndarray
    true_propensity: np.ndarray
    train_positions: np.ndarray
    holdout_positions: np.ndarray


def generate_selection(
    config: RejectInferenceConfig, seed: int, scenario: str
) -> SyntheticSelection:
    """Generate potential outcomes under a common hypothetical loan for all applicants.

    MAR selection uses recorded X only; MNAR selection and outcomes share an
    unrecorded factor. Deterministic rejection creates structural support gaps.
    Truth is reserved for simulation evaluation or an explicitly unattainable oracle.
    """
    if scenario not in SCENARIOS:
        raise ValueError("Unknown selection scenario")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(config.samples, 3))
    hidden = rng.normal(size=config.samples)
    risk, debt, income = x.T
    outcome_pd = expit(
        config.outcome_intercept
        + risk
        + 0.5 * debt
        - 0.5 * income
        + config.nonlinear_risk_effect * risk**2
        + config.hidden_outcome_effect * hidden
    )
    y = (rng.random(config.samples) < outcome_pd).astype(int)
    selection_logit = (
        config.selection_intercept
        + config.selection_risk_coefficient * risk
        + config.selection_debt_coefficient * debt
        + config.selection_income_coefficient * income
    )
    if scenario == "mnar_hidden":
        selection_logit += config.hidden_selection_effect * hidden
    propensity = np.clip(
        expit(selection_logit), config.true_propensity_floor, 1 - config.true_propensity_floor
    )
    if scenario == "deterministic_no_overlap":
        propensity = (expit(selection_logit) >= config.deterministic_cutoff).astype(float)
    accepted = (rng.random(config.samples) < propensity).astype(int)
    observed = np.where(accepted == 1, y, np.nan)
    positions = rng.permutation(config.samples)
    n_holdout = int(config.samples * config.holdout_fraction)
    return SyntheticSelection(
        pd.DataFrame(x, columns=FEATURES),
        accepted,
        observed,
        y,
        propensity,
        positions[n_holdout:],
        positions[:n_holdout],
    )


def logistic_model(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def validate_observed(features, accepted, observed_outcome):
    x = features.loc[:, FEATURES].to_numpy(dtype=float)
    a = np.asarray(accepted)
    y = np.asarray(observed_outcome, dtype=float)
    if not len(x) or not np.isfinite(x).all() or a.shape != (len(x),) or y.shape != (len(x),):
        raise ValueError("Features and selection/outcome vectors must be finite and aligned")
    if not np.isin(a, [0, 1]).all():
        raise ValueError("Acceptance must be binary")
    mask = a == 1
    if not np.isnan(y[~mask]).all() or not np.isin(y[mask], [0, 1]).all():
        raise ValueError("Rejected outcomes must be masked; accepted outcomes must be binary")
    if len(np.unique(y[mask])) != 2:
        raise ValueError("Observed accepted outcomes require both classes")
    return a.astype(int), y, mask


def crossfit_propensity(features, accepted, folds: int, seed: int, c: float):
    """Estimate P(acceptance|X) from training-only applicant data, without outcomes."""
    a = np.asarray(accepted)
    if a.shape != (len(features),) or not np.isin(a, [0, 1]).all():
        raise ValueError("Propensity selection vector must be aligned and binary")
    if len(np.unique(a)) != 2 or min(np.bincount(a.astype(int))) < folds:
        raise ValueError("Both selection classes need at least one row per fold")
    probabilities = np.empty(len(features))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, validation in splitter.split(features, a):
        model = logistic_model(c)
        model.fit(features.iloc[train].loc[:, FEATURES], a[train])
        probabilities[validation] = model.predict_proba(features.iloc[validation].loc[:, FEATURES])[
            :, 1
        ]
    return probabilities


def inverse_probability_weights(accepted, propensity, floor: float, maximum: float):
    """Return accepted-row weights, normalized to mean one, and instability diagnostics.

    Clipping changes the estimand/fit. It cannot repair structural zero acceptance.
    """
    a = np.asarray(accepted)
    p = np.asarray(propensity, dtype=float)
    if p.ndim != 1 or a.shape != p.shape or not len(p) or not np.isin(a, [0, 1]).all():
        raise ValueError("Selection and propensity vectors must be aligned and nonempty")
    if not np.isfinite(p).all() or ((p <= 0) | (p > 1)).any():
        raise ValueError("Propensity must be in (0,1]; zero support cannot be repaired")
    if not 0 < floor <= 1 or not np.isfinite(maximum) or maximum < 1:
        raise ValueError("Invalid propensity floor or weight cap")
    mask = a == 1
    if not mask.any():
        raise ValueError("At least one accepted row is required")
    raw = 1 / p[mask]
    clipped = np.minimum(1 / np.maximum(p[mask], floor), maximum)
    normalized = clipped / clipped.mean()
    return normalized, dict(
        accepted_rows=int(mask.sum()),
        estimated_propensity_min=float(p.min()),
        estimated_propensity_max=float(p.max()),
        population_below_floor_fraction=float((p < floor).mean()),
        accepted_clipped_fraction=float((clipped < raw).mean()),
        raw_weight_max=float(raw.max()),
        clipped_weight_max=float(clipped.max()),
        effective_sample_size=float(clipped.sum() ** 2 / (clipped**2).sum()),
        normalization="accepted weights divided by their mean",
    )


def fit_observed_models(
    features, accepted, observed_outcome, config, seed, positivity_supported=True
):
    """Fit accepted-only logistic benchmarks. No access to hidden/reject truth."""
    a, y, mask = validate_observed(features, accepted, observed_outcome)
    baseline = logistic_model(config.logistic_c)
    baseline.fit(features.loc[mask, FEATURES], y[mask].astype(int))
    models = {"accepted_only": baseline}
    if not positivity_supported:
        return models, {"ipw_status": "disabled_structural_zero_support"}, None
    p = crossfit_propensity(features, a, config.propensity_folds, seed, config.logistic_c)
    weights, diagnostics = inverse_probability_weights(
        a, p, config.estimated_propensity_floor, config.maximum_weight
    )
    weighted = logistic_model(config.logistic_c)
    weighted.fit(
        features.loc[mask, FEATURES], y[mask].astype(int), logisticregression__sample_weight=weights
    )
    models["ipw"] = weighted
    diagnostics["ipw_status"] = "estimated_oof_propensity_clipped"
    return models, diagnostics, p


def reject_sensitivity(accepted, observed_outcome, probabilities, multipliers):
    """Vary assumed reject odds without assigning any new observed outcome labels."""
    a = np.asarray(accepted)
    y = np.asarray(observed_outcome, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if a.ndim != 1 or not len(a) or y.shape != a.shape or p.shape != a.shape:
        raise ValueError("Sensitivity vectors must be aligned and nonempty")
    if (
        not np.isin(a, [0, 1]).all()
        or not np.isnan(y[a == 0]).all()
        or not np.isin(y[a == 1], [0, 1]).all()
    ):
        raise ValueError("Outcomes must be observed only for accepted rows")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Sensitivity probabilities must be in [0,1]")
    rejected = a == 0
    output = []
    for multiplier in multipliers:
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("Odds multiplier must be positive and finite")
        adjusted = multiplier * p[rejected] / (1 - p[rejected] + multiplier * p[rejected])
        output.append(
            dict(
                reject_odds_multiplier=float(multiplier),
                assumed_reject_bad_rate=float(adjusted.mean()) if rejected.any() else None,
                population_bad_rate_assumption=float((np.nansum(y) + adjusted.sum()) / len(a)),
            )
        )
    return output


def population_risk_bounds(accepted, observed_outcome):
    """Worst-case aggregate bounds: every unobserved reject is good or bad."""
    a = np.asarray(accepted)
    y = np.asarray(observed_outcome, dtype=float)
    if a.ndim != 1 or not len(a) or y.shape != a.shape or not np.isin(a, [0, 1]).all():
        raise ValueError("Selection and observed outcomes must be aligned and nonempty")
    if not np.isnan(y[a == 0]).all() or not np.isin(y[a == 1], [0, 1]).all():
        raise ValueError("Rejected outcomes must be masked and accepted outcomes binary")
    observed_bads = np.nansum(y)
    return dict(
        lower=float(observed_bads / len(a)), upper=float((observed_bads + (a == 0).sum()) / len(a))
    )
