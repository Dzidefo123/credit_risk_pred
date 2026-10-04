"""Shared cutoff and provenance checks for longitudinal portfolio analytics."""

import pandas as pd

from credit_risk.data.validation import DataContractError, validate_history


def prepare_history(history, accounts=None, as_of=None):
    validate_history(history, accounts)
    flags = set(history.is_synthetic.unique())
    if accounts is not None:
        flags.update(accounts.is_synthetic.unique())
    if len(flags) != 1:
        raise DataContractError("Analyze real and synthetic portfolios separately")
    cutoff = history.observation_date.max() if as_of is None else pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise ValueError("as_of must be a timezone-naive date without a time component")
    ordered = (
        history.loc[history.observation_date <= cutoff]
        .sort_values(["account_id", "observation_date"])
        .reset_index(drop=True)
        .copy()
    )
    return ordered, cutoff
