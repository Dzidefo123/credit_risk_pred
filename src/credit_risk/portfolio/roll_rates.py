"""Consecutive-month migrations, coverage diagnostics and two weighting bases."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from credit_risk.data.validation import STATES
from credit_risk.portfolio.common import prepare_history


@dataclass(frozen=True)
class RollRateResult:
    pairs: pd.DataFrame
    count_matrix: pd.DataFrame
    probability_matrix: pd.DataFrame
    balance_matrix: pd.DataFrame
    balance_probability_matrix: pd.DataFrame
    state_summary: pd.DataFrame
    monthly_summary: pd.DataFrame
    diagnostics: dict


def transition_matrices(pairs):
    counts = pd.crosstab(pairs.from_state, pairs.to_state).reindex(
        index=STATES, columns=STATES, fill_value=0
    )
    balances = (
        pairs.groupby(["from_state", "to_state"])
        .origin_balance.sum()
        .unstack(fill_value=0)
        .reindex(index=STATES, columns=STATES, fill_value=0)
        .astype(float)
    )
    probabilities = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    balance_probabilities = balances.div(balances.sum(axis=1).replace(0, np.nan), axis=0)
    return counts, probabilities, balances, balance_probabilities


def roll_rates(history, as_of=None, accounts=None):
    """Use observed t -> t+1 pairs, never bridge gaps or infer terminal outcomes.

    Exposure weights are origin closing balances. Rows with no count support or
    zero origin balance have unknown corresponding transition probabilities.
    """
    data, cutoff = prepare_history(history, accounts, as_of)
    grouped = data.groupby("account_id", sort=False)
    next_date = grouped.observation_date.shift(-1)
    next_state = grouped.state.shift(-1)
    eligible = data.observation_date.add(pd.offsets.MonthEnd(1)) <= cutoff
    consecutive = next_date.eq(data.observation_date.add(pd.offsets.MonthEnd(1)))
    used = eligible & consecutive
    pairs = (
        data.loc[used, ["account_id", "observation_date", "state", "balance"]]
        .rename(
            columns={
                "observation_date": "origin_date",
                "state": "from_state",
                "balance": "origin_balance",
            }
        )
        .reset_index(drop=True)
    )
    pairs["destination_date"] = next_date.loc[used].to_numpy()
    pairs["to_state"] = next_state.loc[used].to_numpy()
    positions = {state: index for index, state in enumerate(STATES)}
    source = pairs.from_state.map(positions)
    destination = pairs.to_state.map(positions)
    pairs["roll_forward"] = destination > source
    pairs["roll_back"] = destination < source
    pairs["cure"] = (source > 0) & (source < len(STATES) - 1) & (destination == 0)
    pairs["new_default"] = (pairs.from_state != "DEFAULT") & (pairs.to_state == "DEFAULT")
    pairs["stay"] = destination == source
    counts, probabilities, balances, balance_probabilities = transition_matrices(pairs)
    summaries = []
    for state in STATES:
        subset = pairs.loc[pairs.from_state == state]
        n, exposure = len(subset), float(subset.origin_balance.sum())
        row = {"from_state": state, "pairs": n, "origin_balance": exposure}
        for field in ("roll_forward", "roll_back", "cure", "new_default", "stay"):
            row[field + "_rate"] = float(subset[field].mean()) if n else np.nan
            row[field + "_balance_rate"] = (
                float(subset.loc[subset[field], "origin_balance"].sum() / exposure)
                if exposure > 0
                else np.nan
            )
        summaries.append(row)
    origins = data.loc[eligible].copy()
    origins["paired"] = consecutive.loc[eligible]
    coverage = origins.groupby("observation_date").agg(
        eligible_origins=("account_id", "size"), paired_origins=("paired", "sum")
    )
    monthly = []
    for date, row in coverage.iterrows():
        subset = pairs.loc[pairs.origin_date == date]
        nondefault = subset.loc[subset.from_state != "DEFAULT"]
        delinquent = subset.loc[~subset.from_state.isin(["CURRENT", "DEFAULT"])]
        monthly.append(
            {
                "origin_date": date,
                "eligible_origins": int(row.eligible_origins),
                "paired_origins": int(row.paired_origins),
                "missing_next": int(row.eligible_origins - row.paired_origins),
                "pair_coverage": float(row.paired_origins / row.eligible_origins),
                "nondefault_pairs": len(nondefault),
                "delinquent_pairs": len(delinquent),
                "roll_forward_rate": float(nondefault.roll_forward.mean())
                if len(nondefault)
                else np.nan,
                "roll_back_rate": float(delinquent.roll_back.mean()) if len(delinquent) else np.nan,
                "cure_rate": float(delinquent.cure.mean()) if len(delinquent) else np.nan,
                "new_default_rate": float(nondefault.new_default.mean())
                if len(nondefault)
                else np.nan,
            }
        )
    diagnostics = {
        "as_of": str(cutoff.date()),
        "snapshots_as_of": len(data),
        "eligible_origins": int(eligible.sum()),
        "consecutive_pairs": len(pairs),
        "missing_next": int((eligible & ~consecutive).sum()),
        "not_yet_due_origins": int((~eligible).sum()),
        "gap_links_excluded": int((next_date.notna() & ~consecutive).sum()),
        "pair_coverage": float(used.sum() / eligible.sum()) if eligible.sum() else None,
        "weight_basis": "origin closing balance; count and balance rates reported separately",
    }
    return RollRateResult(
        pairs,
        counts,
        probabilities,
        balances,
        balance_probabilities,
        pd.DataFrame(summaries),
        pd.DataFrame(
            monthly,
            columns=[
                "origin_date",
                "eligible_origins",
                "paired_origins",
                "missing_next",
                "pair_coverage",
                "nondefault_pairs",
                "delinquent_pairs",
                "roll_forward_rate",
                "roll_back_rate",
                "cure_rate",
                "new_default_rate",
            ],
        ),
        diagnostics,
    )
