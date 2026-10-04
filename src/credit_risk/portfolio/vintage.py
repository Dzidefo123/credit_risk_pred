"""Origination month x MOB snapshots and coverage-aware cumulative default curves."""

import numpy as np
import pandas as pd

from credit_risk.portfolio.common import prepare_history
from credit_risk.portfolio.settings import PortfolioAnalyticsConfig


def vintage_table(accounts, history, config=None, as_of=None):
    """Retain the original cohort denominator; incomplete cumulative rates are unknown.

    Snapshot rates use observed rows, explicitly reporting their coverage. An
    absent account is neither closed nor performing: no such status exists in
    the source contract. Recorded default differs from an analytic 90-DPD bad.
    """
    config = config or PortfolioAnalyticsConfig()
    data, cutoff = prepare_history(history, accounts, as_of)
    master = accounts.loc[accounts.origination_date <= cutoff].copy()
    master["origination_month"] = master.origination_date.dt.to_period("M")
    data["origination_month"] = data.origination_date.dt.to_period("M")
    data["bad"] = data.default_flag | (data.dpd >= config.bad_dpd_threshold)
    data["delinquent"] = data.default_flag | (data.dpd >= config.delinquency_dpd_threshold)
    # Before the first missing MOB, sorted observations must equal 0, 1, 2, ...
    sequence = data.groupby("account_id", sort=False).cumcount()
    contiguous = data.months_on_book.eq(sequence).groupby(data.account_id, sort=False).cummin()
    complete_through = data.loc[contiguous].groupby("account_id").months_on_book.max()
    first_default = data.loc[data.default_flag].groupby("account_id").months_on_book.min()
    aggregates = data.groupby(["origination_month", "months_on_book"]).agg(
        observed_accounts=("account_id", "size"),
        bad_count=("bad", "sum"),
        delinquent_count=("delinquent", "sum"),
        default_count=("default_flag", "sum"),
        balance_exposure=("balance", "sum"),
        credit_limit_exposure=("credit_limit", "sum"),
    )
    rows = []
    for cohort, group in master.groupby("origination_month", sort=True):
        n = len(group)
        coverage_ends = complete_through.reindex(group.account_id).fillna(-1).to_numpy()
        defaults = first_default.reindex(group.account_id).dropna().to_numpy()
        for mob in range(config.max_months_on_book + 1):
            observation = (cohort + mob).to_timestamp(how="end").normalize()
            matured = bool(observation <= cutoff)
            row = {
                "origination_month": str(cohort),
                "months_on_book": mob,
                "observation_date": observation,
                "cohort_accounts": n,
                "is_synthetic": bool(group.is_synthetic.iloc[0]),
                "matured": matured,
                "low_support": n < config.minimum_cohort_size,
                "observed_accounts": 0,
                "bad_count": 0,
                "delinquent_count": 0,
                "default_count": 0,
                "balance_exposure": 0.0,
                "credit_limit_exposure": 0.0,
            }
            if matured and (cohort, mob) in aggregates.index:
                for field, value in aggregates.loc[(cohort, mob)].items():
                    row[field] = (
                        int(value)
                        if field.endswith("count") or field == "observed_accounts"
                        else float(value)
                    )
            observed = row["observed_accounts"]
            complete = int((coverage_ends >= mob).sum()) if matured else None
            ever_default = int((defaults <= mob).sum()) if matured else None
            row.update(
                snapshot_coverage=observed / n if matured else np.nan,
                complete_history_accounts=complete,
                fully_observed=bool(matured and complete == n),
                bad_rate=row["bad_count"] / observed if observed else np.nan,
                delinquency_rate=row["delinquent_count"] / observed if observed else np.nan,
                recorded_default_rate=row["default_count"] / observed if observed else np.nan,
                cumulative_observed_default_count=ever_default,
                cumulative_default_lower_bound=ever_default / n if matured else np.nan,
                cumulative_default_rate=ever_default / n if matured and complete == n else np.nan,
            )
            # Future cells have no measured exposure/count, rather than zero balances.
            if not matured:
                for field in (
                    "observed_accounts",
                    "bad_count",
                    "delinquent_count",
                    "default_count",
                    "balance_exposure",
                    "credit_limit_exposure",
                ):
                    row[field] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)
