"""Outcome-free support diagnostics and descriptive composition; no shift thresholds."""

import numpy as np
from scipy.stats import ks_2samp

from credit_risk.track_b.macro_hazard.data import CATEGORICAL, MACRO, NUMERIC
from credit_risk.track_b.macro_hazard.protocol import PRIMARY

from .math import summary


def macro_values(data):
    return data["macro"][:, [MACRO.index(n) for n in PRIMARY]]


def frozen_bins(values):
    inside = np.unique(np.quantile(values, np.linspace(0, 1, 11))[1:-1])
    return np.r_[-np.inf, inside, np.inf]


def bin_index(values, edges):
    return np.searchsorted(edges[1:-1], values, side="right")


def shift(development, evaluation):
    d, e = np.asarray(development, float), np.asarray(evaluation, float)
    edges = frozen_bins(d)
    dc = np.bincount(bin_index(d, edges), minlength=len(edges) - 1) / len(d)
    ec = np.bincount(bin_index(e, edges), minlength=len(edges) - 1) / len(e)
    a, b = dc + 1e-6, ec + 1e-6
    a, b = a / a.sum(), b / b.sum()
    lower, upper = np.quantile(d, [0.05, 0.95])
    outside = (e < d.min()) | (e > d.max())
    central = (e < lower) | (e > upper)
    return dict(
        development=summary(d),
        evaluation=summary(e),
        standardized_mean_difference=float((e.mean() - d.mean()) / d.std())
        if d.std() > 0
        else None,
        psi=float(np.sum((b - a) * np.log(b / a))),
        ks_distance=float(ks_2samp(d, e).statistic),
        outside_development_range=dict(count=int(outside.sum()), fraction=float(outside.mean())),
        inside_development_range=dict(
            count=int((~outside).sum()), fraction=float((~outside).mean())
        ),
        outside_development_central90=dict(
            count=int(central.sum()), fraction=float(central.mean())
        ),
        central_quantiles=[0.05, 0.95],
        development_bin_edges=[None if not np.isfinite(x) else float(x) for x in edges],
        bin_frequencies=dict(development=dc.tolist(), evaluation=ec.tolist()),
    )


def distinct_months(data):
    months, index = np.unique(data["month"], return_index=True)
    return months, macro_values(data[index])


def regimes(data):
    year = data["month"] // 12
    groups = {
        f"{a}_{b}": (year >= a) & (year <= b) for a, b in [(2010, 2012), (2013, 2015), (2016, 2017)]
    }
    groups.update(
        {str(y) + ("_partial" if y == 2026 else ""): year == y for y in range(2019, 2027)}
    )
    return {k: mask for k, mask in groups.items() if mask.any()}


def correlation(values):
    r = np.corrcoef(values, rowvar=False)
    rank = int(np.linalg.matrix_rank(r))
    eig = np.linalg.eigvalsh(r)
    condition = float(np.linalg.cond(r))
    return dict(
        matrix=r.tolist(),
        rank=rank,
        condition_number=condition if np.isfinite(condition) else None,
        smallest_eigenvalue=float(eig.min()),
        vif=np.diag(np.linalg.inv(r)).tolist() if rank == r.shape[0] else None,
        interpretation=(
            "Descriptive shared-calendar predictor geometry; no independent-row inference"
        ),
    )


def support_diagnostics(development, evaluation):
    dm, d = distinct_months(development)
    em, e = distinct_months(evaluation)
    dv, ev = macro_values(development), macro_values(evaluation)
    mean, sd = d.mean(axis=0), d.std(axis=0)
    if (sd == 0).any():
        raise ValueError("Constant macro prevents multivariate support diagnostic")
    z = (d - mean) / sd
    ze = (e - mean) / sd
    _, singular, axes = np.linalg.svd(z, full_matrices=False)
    covariance = np.cov(z, rowvar=False, bias=True)
    inverse = np.linalg.pinv(covariance, rcond=1e-10)
    dd = np.einsum("ij,jk,ik->i", z, inverse, z)
    ed = np.einsum("ij,jk,ik->i", ze, inverse, ze)
    reference = float(np.quantile(dd, 0.95))
    evaluation_month_weights = np.unique(evaluation["month"], return_counts=True)[1]
    return dict(
        label="POST_VALIDATION_DIAGNOSTIC",
        features=list(PRIMARY),
        interval_weighted={n: shift(dv[:, i], ev[:, i]) for i, n in enumerate(PRIMARY)},
        distinct_month_weighted={n: shift(d[:, i], e[:, i]) for i, n in enumerate(PRIMARY)},
        multivariate=dict(
            fitted_on="Distinct development months only, no outcomes",
            mean=mean.tolist(),
            sd=sd.tolist(),
            covariance_rank=int(np.linalg.matrix_rank(covariance)),
            pca_axes=axes[:2].tolist(),
            explained_variance_fraction=(singular**2 / np.sum(singular**2)).tolist(),
            development95_squared_distance=reference,
            development_distances=summary(dd),
            evaluation_distances=summary(ed),
            evaluation_outside_reference_months=int((ed > reference).sum()),
            evaluation_outside_reference_intervals=int(
                evaluation_month_weights[ed > reference].sum()
            ),
            development_projection=(z @ axes[:2].T).tolist(),
            evaluation_projection=(ze @ axes[:2].T).tolist(),
            development_months=dm.astype(int).tolist(),
            evaluation_months=em.astype(int).tolist(),
            evaluation_squared_distances=ed.tolist(),
        ),
        correlations=dict(
            development_months=correlation(d),
            evaluation_months=correlation(e),
            development_intervals=correlation(dv),
            evaluation_intervals=correlation(ev),
        ),
        regime_macro_summaries={
            split: {
                key: {n: summary(macro_values(data[mask])[:, i]) for i, n in enumerate(PRIMARY)}
                for key, mask in regimes(data).items()
            }
            for split, data in [("development", development), ("evaluation", evaluation)]
        },
    )


def composition(data):
    _, first = np.unique(data["facility"], return_index=True)
    result = {}
    for weighting, rows in [
        ("risk_intervals", data),
        ("unique_facility_first_split_row", data[first]),
    ]:
        result[weighting] = dict(
            records=len(rows),
            age=summary(rows["duration"]),
            numeric={
                n: summary(
                    np.expm1(rows["numeric"][:, i]) if n == "orig_upb" else rows["numeric"][:, i]
                )
                for i, n in enumerate(NUMERIC)
            },
            numeric_missing={
                n: int(np.isnan(rows["numeric"][:, i]).sum()) for i, n in enumerate(NUMERIC)
            },
            vintage={
                str(v): int(np.count_nonzero(rows["vintage"] == v))
                for v in np.unique(rows["vintage"])
            },
            categorical={
                n: {
                    str(v): int(c)
                    for v, c in zip(
                        *np.unique(rows["categories"][:, i], return_counts=True), strict=True
                    )
                }
                for i, n in enumerate(CATEGORICAL)
            },
            original_rate_minus_current_mortgage_rate=summary(
                rows["numeric"][:, 4] - rows["macro"][:, MACRO.index("mortgage_30y_level")]
            ),
        )
    return result
