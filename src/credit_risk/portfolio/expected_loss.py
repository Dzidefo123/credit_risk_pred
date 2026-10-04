"""Modular expected-loss arithmetic with aligned model probabilities and exposure."""

import numpy as np
import pandas as pd

from credit_risk.portfolio.loss_settings import ExpectedLossConfig


def _arrays(*values):
    arrays = []
    for value in values:
        array = np.asarray(value)
        if array.ndim > 1 or array.size == 0 or array.dtype.kind not in "iuf":
            raise ValueError("Inputs must be nonempty numeric scalars or vectors")
        array = array.astype(float)
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("Inputs must be finite and nonnegative")
        arrays.append(array)
    try:
        return np.broadcast_arrays(*arrays)
    except ValueError as exc:
        raise ValueError("Probability, loss severity and exposure shapes must align") from exc


def calculate_expected_loss(probability, lgd, ead):
    p, severity, exposure = _arrays(probability, lgd, ead)
    if (p > 1).any() or (severity > 1).any():
        raise ValueError("PD and LGD must be in [0,1]")
    return p * severity * exposure


def exposure_at_default(balance, credit_limit, credit_conversion_factor):
    drawn, limit, ccf = _arrays(balance, credit_limit, credit_conversion_factor)
    if (ccf > 1).any():
        raise ValueError("Credit conversion factor must be in [0,1]")
    with np.errstate(over="ignore"):
        ead = drawn + ccf * np.maximum(limit - drawn, 0)
    if not np.isfinite(ead).all():
        raise ValueError("Exposure calculation overflowed")
    return ead


def stress_probability(probability, odds_multiplier):
    p, multiplier = _arrays(probability, odds_multiplier)
    if (p > 1).any() or (multiplier <= 0).any():
        raise ValueError("Probability must be in [0,1] and odds multiplier positive")
    # Stable at exact PD 0/1; avoids constructing infinite odds or clipping PDs.
    result = p / (p + (1 - p) / multiplier)
    if not np.isfinite(result).all():
        raise ValueError("Scenario probability calculation overflowed")
    return result


def loss_table(snapshot, pd_scores, config=None):
    """Join model probabilities by account/date, never by incidental row position.

    Model scores must have the same population as snapshot. Defaulted inventory
    has PD 1 but is separated from nondefault forward expected loss. Its undrawn
    lines are assumed unavailable and residual loss is LGD x observed balance.
    """
    config = config or ExpectedLossConfig()
    keys = ["account_id", "observation_date"]
    required = {*keys, "balance", "credit_limit", "default_flag"}
    if not required.issubset(snapshot.columns) or not {*keys, "pd"}.issubset(pd_scores.columns):
        raise ValueError("Snapshot and model scores lack required loss fields")
    for table in (snapshot, pd_scores):
        if table.empty or table[keys].isna().any().any() or table.duplicated(keys).any():
            raise ValueError("Loss keys must be nonmissing and unique")
    if "pd" in snapshot.columns:
        raise ValueError("Snapshot must not contain an ambiguous existing PD")
    if not pd.api.types.is_bool_dtype(snapshot.default_flag) or snapshot.default_flag.isna().any():
        raise ValueError("default_flag must be observed boolean")
    joined = snapshot.merge(
        pd_scores[keys + ["pd"]], on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if not joined._merge.eq("both").all():
        raise ValueError("Snapshot and scored populations differ; missing or extra model scores")
    joined = joined.drop(columns="_merge").rename(columns={"pd": "base_pd"})
    calculate_expected_loss(joined.base_pd, 1.0, joined.balance)
    if not joined.loc[joined.default_flag, "base_pd"].eq(1).all():
        raise ValueError("Recorded defaults require PD 1 under this loss convention")
    results = []
    for scenario in config.scenarios:
        rows = joined.copy(deep=True)
        rows["scenario"] = scenario.name
        rows["pd"] = stress_probability(rows.base_pd, scenario.pd_odds_multiplier)
        rows["lgd"] = scenario.lgd
        rows["credit_conversion_factor"] = np.where(
            rows.default_flag, 0.0, scenario.credit_conversion_factor
        )
        rows["undrawn"] = np.maximum(rows.credit_limit - rows.balance, 0)
        rows["available_undrawn"] = np.where(rows.default_flag, 0.0, rows.undrawn)
        rows["ead"] = exposure_at_default(
            rows.balance, rows.credit_limit, rows.credit_conversion_factor
        )
        rows["forward_expected_loss"] = calculate_expected_loss(rows.pd, rows.lgd, rows.ead)
        rows.loc[rows.default_flag, "forward_expected_loss"] = 0.0
        rows["defaulted_loss_assumption"] = np.where(
            rows.default_flag, rows.lgd * rows.balance, 0.0
        )
        rows["combined_loss_proxy"] = rows.forward_expected_loss + rows.defaulted_loss_assumption
        results.append(rows)
    return pd.concat(results, ignore_index=True)


def portfolio_loss_summary(rows, top_n=10):
    """Keep performing forecasts and defaulted stock separate; reconcile additive totals."""
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be positive")
    summaries = []
    for scenario, group in rows.groupby("scenario", sort=False):
        nondefault = group.loc[~group.default_flag]
        total_ead = float(group.ead.sum())
        performing_ead = float(nondefault.ead.sum())
        loss = float(nondefault.forward_expected_loss.sum())
        shares = group.ead / total_ead if total_ead > 0 else group.ead * 0
        summary = {
            "scenario": scenario,
            "accounts": len(group),
            "nondefault_accounts": len(nondefault),
            "defaulted_accounts": int(group.default_flag.sum()),
            "balance": float(group.balance.sum()),
            "ead": total_ead,
            "nondefault_ead": performing_ead,
            "forward_expected_loss": loss,
            "defaulted_loss_assumption": float(group.defaulted_loss_assumption.sum()),
            "combined_loss_proxy": float(group.combined_loss_proxy.sum()),
            "expected_new_defaults": float(nondefault.pd.sum()),
            "mean_nondefault_pd": float(nondefault.pd.mean()) if len(nondefault) else None,
            "ead_weighted_nondefault_pd": float(
                (nondefault.pd * nondefault.ead).sum() / performing_ead
            )
            if performing_ead > 0
            else None,
            "forward_el_rate": loss / performing_ead if performing_ead > 0 else None,
            "ead_hhi": float((shares**2).sum()) if total_ead > 0 else None,
            "top_n_ead_share": float(group.ead.nlargest(top_n).sum() / total_ead)
            if total_ead > 0
            else None,
            "top_n_forward_el_share": float(
                nondefault.forward_expected_loss.nlargest(top_n).sum() / loss
            )
            if loss > 0
            else None,
        }
        if any(isinstance(value, float) and not np.isfinite(value) for value in summary.values()):
            raise ValueError("Portfolio aggregation overflowed or produced non-finite totals")
        summaries.append(summary)
    return pd.DataFrame(summaries)


def segment_loss_summary(rows, dimensions=("state", "origination_month"), top_n=10):
    summaries = []
    for dimension in dimensions:
        for (_, segment), group in rows.groupby(["scenario", dimension], sort=True, dropna=False):
            summary = portfolio_loss_summary(group, top_n=top_n).iloc[0].to_dict()
            summary.update(dimension=dimension, segment=str(segment))
            summaries.append(summary)
    return pd.DataFrame(summaries)
