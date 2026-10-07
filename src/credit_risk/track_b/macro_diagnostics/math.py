"""Frozen logit/probability decomposition and explicitly optimistic oracle diagnostics."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from threadpoolctl import threadpool_limits

from credit_risk.track_b.macro_hazard.data import MACRO
from credit_risk.track_b.macro_hazard.metrics import losses


def summary(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(
            count=0, mean=None, sd=None, median=None, minimum=None, maximum=None, quantiles={}
        )
    return dict(
        count=len(x),
        mean=float(x.mean()),
        sd=float(x.std()),
        median=float(np.median(x)),
        minimum=float(x.min()),
        maximum=float(x.max()),
        quantiles={str(q): float(np.quantile(x, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
    )


def probabilities(logits):
    x = np.asarray(logits, float)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("Two finite cause-versus-none logits required")
    p = softmax(np.column_stack([np.zeros(len(x)), x]), axis=1)
    if not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("Diagnostic probability conservation failed")
    return p


def macro_contributions(bundle, data):
    encoder, model = bundle
    output = []
    for name in encoder.macro:
        i = encoder.names.index(name)
        z = (
            data["macro"][:, MACRO.index(name)] - encoder.parameters["means"][i]
        ) / encoder.parameters["scales"][i]
        beta = (model.coef_[1:] - model.coef_[0])[:, i]
        output.append(z[:, None] * beta)
    return np.stack(output, axis=1) if output else np.zeros((len(data), 0, 2))


def components(bundle, data):
    encoder, model = bundle
    contrast = model.coef_[1:] - model.coef_[0]
    intercept = model.intercept_[1:] - model.intercept_[0]
    x = encoder.transform(data)
    with threadpool_limits(limits=1):
        logits = np.asarray(x @ contrast.T) + intercept
    result = dict(
        intercept=np.broadcast_to(intercept, (len(data), 2)),
        macro=macro_contributions(bundle, data),
    )
    for group in ["mortgage", "duration", "cohort"]:
        indices = [
            i
            for i, n in enumerate(encoder.names)
            if (group == "duration" and n.startswith("duration:"))
            or (group == "cohort" and n.startswith("vintage:"))
            or (
                group == "mortgage"
                and n not in encoder.macro
                and not n.startswith(("duration:", "vintage:"))
            )
        ]
        result[group] = np.asarray(x[:, indices] @ contrast[:, indices].T)
    reconstructed = sum(
        result[n] for n in ["intercept", "mortgage", "duration", "cohort"]
    ) + result["macro"].sum(axis=1)
    if not np.allclose(reconstructed, logits, atol=1e-10, rtol=1e-10):
        raise ValueError("Frozen contribution/logit reconstruction failed")
    return logits, result


def one_at_a_time(m1_logits, contributions):
    """Reference attribution is nonadditive; no outcome metric is used here."""
    baseline = probabilities(m1_logits)
    return np.stack(
        [
            probabilities(m1_logits + contributions[:, i]) - baseline
            for i in range(contributions.shape[1])
        ],
        axis=1,
    )


def ablated(logits, contributions, index):
    return probabilities(logits - contributions[:, index])


def oracle_intercepts(logits, y):
    """Joint offsets preserve multinomial coherence; same-sample loss is optimistic."""
    y = np.asarray(y, int)
    if not np.isin(y, [0, 1, 2]).all() or set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Oracle requires all three event classes")

    def objective(offset):
        eta = np.column_stack([np.zeros(len(y)), logits + offset])
        p = softmax(eta, axis=1)
        value = np.mean(logsumexp(eta, axis=1) - eta[np.arange(len(y)), y])
        gradient = p[:, 1:].mean(axis=0) - np.array([(y == 1).mean(), (y == 2).mean()])
        return float(value), gradient

    fitted = minimize(
        objective,
        np.zeros(2),
        jac=True,
        method="L-BFGS-B",
        options=dict(maxiter=1000, ftol=1e-12, gtol=1e-9),
    )
    if not fitted.success:
        raise ValueError("Oracle intercept diagnostic failed to converge")
    p = probabilities(logits + fitted.x)
    return p, dict(
        label="POST_HOC_ORACLE_DIAGNOSTIC",
        offsets=fitted.x.tolist(),
        iterations=int(fitted.nit),
        observed_rates=[float((y == i).mean()) for i in [1, 2]],
        predicted_rates=p[:, 1:].mean(axis=0).tolist(),
        same_sample_optimism=True,
        independently_validated=False,
        model_applied_or_promoted=False,
    )


def score_difference(y, original, changed):
    keys = ["joint_log_loss", "default_brier", "payoff_brier"]
    original_loss = losses(y, original).mean(axis=0)
    changed_loss = losses(y, changed).mean(axis=0)
    return dict(
        original=dict(zip(keys, original_loss.tolist(), strict=True)),
        diagnostic=dict(zip(keys, changed_loss.tolist(), strict=True)),
        delta=dict(zip(keys, (changed_loss - original_loss).tolist(), strict=True)),
    )


def substitute(default, payoff):
    """Never renormalize a mathematically invalid substitution."""
    from credit_risk.track_b.survival.math import curves

    return curves(default, payoff)


def brier_accounting(y, probability):
    """Exact variance/covariance identity, not a binned reliability decomposition."""
    y = np.asarray(y, float)
    p = np.asarray(probability, float)
    event_variance = float(y.var())
    prediction_variance = float(p.var())
    covariance = float(np.mean((y - y.mean()) * (p - p.mean())))
    bias_squared = float((p.mean() - y.mean()) ** 2)
    reconstructed = event_variance + prediction_variance - 2 * covariance + bias_squared
    actual = float(np.mean((p - y) ** 2))
    if not np.isclose(actual, reconstructed, atol=1e-12, rtol=1e-10):
        raise ValueError("Brier accounting failed")
    return dict(
        event_variance=event_variance,
        prediction_variance=prediction_variance,
        covariance=covariance,
        mean_bias_squared=bias_squared,
        brier=actual,
        identity="Var(y)+Var(p)-2Cov(y,p)+(E[p]-E[y])^2",
        interpretation="Exact score accounting; covariance is not AUC or causal attribution",
    )
