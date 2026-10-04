"""Pure policy functions: raw affordability inputs, no outcome-driven decisions."""

import numpy as np
import pandas as pd

from credit_risk.decisioning.settings import CreditStrategy

INPUTS = ("MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines")


def decide(applicants: pd.DataFrame, probabilities, policy: CreditStrategy) -> pd.DataFrame:
    """PD grades have inclusive upper bounds; approval is strictly below threshold.

    Limits are hypothetical in source income units. DebtRatio is a penalty proxy,
    not a verified cash-flow measure. Missing inputs never receive automatic offers.
    """
    pd_values = np.asarray(probabilities, dtype=float)
    if pd_values.shape != (len(applicants),) or not np.isfinite(pd_values).all():
        raise ValueError("Supply one finite PD per applicant in positional order")
    if ((pd_values < 0) | (pd_values > 1)).any():
        raise ValueError("PD must be in [0,1]")
    if not set(INPUTS).issubset(applicants.columns):
        raise ValueError("Policy requires raw income, DebtRatio and utilization columns")
    values = applicants.loc[:, INPUTS].to_numpy(dtype=float)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("Policy inputs must be nonnegative or missing")
    grades = np.searchsorted(policy.risk_grade_upper_bounds, pd_values, side="left")
    records = []
    for i, (income, debt, utilization) in enumerate(values):
        prob = pd_values[i]
        reasons = []
        if prob >= policy.decline_at_or_above_pd:
            decision = "DECLINE"
            reasons.append("PD_AT_OR_ABOVE_DECLINE")
        elif prob >= policy.approve_below_pd:
            decision = "MANUAL_REVIEW"
            reasons.append("PD_IN_REVIEW_BAND")
        else:
            decision = "APPROVE"
            reasons.append("PD_BELOW_APPROVAL")
        guards = []
        if not np.isfinite(income) or income <= 0:
            guards.append("INCOME_UNAVAILABLE_OR_ZERO")
        if not np.isfinite(debt):
            guards.append("DEBT_RATIO_MISSING")
        elif debt > policy.maximum_auto_debt_ratio:
            guards.append("DEBT_RATIO_ABOVE_AUTO_GUARD")
        if not np.isfinite(utilization):
            guards.append("UTILIZATION_MISSING")
        elif utilization > policy.maximum_auto_utilization:
            guards.append("UTILIZATION_ABOVE_AUTO_GUARD")
        cap = 0.0
        if not guards:
            base = min(policy.maximum_limit, income * policy.income_limit_multiplier)
            cap = base / (1 + debt) / (1 + policy.utilization_penalty * utilization)
            cap *= policy.grade_limit_factors[grades[i]] * (1 - prob)
            cap = float(np.floor(cap / policy.limit_increment) * policy.limit_increment)
            if cap < policy.minimum_limit or cap <= 0:
                guards.append("LIMIT_BELOW_MINIMUM")
        if guards and decision == "APPROVE":
            decision = "MANUAL_REVIEW"
        reasons.extend(guards)
        limit = cap if decision == "APPROVE" else 0.0
        ead = limit * policy.assumed_drawdown
        records.append(
            dict(
                pd=float(prob),
                risk_grade=f"G{grades[i] + 1}",
                decision=decision,
                reason_codes="|".join(reasons),
                recommended_limit=limit,
                indicative_cap=cap,
                assumed_ead=ead,
                expected_loss_proxy=float(prob * policy.lgd_assumption * ead),
            )
        )
    return pd.DataFrame(
        records,
        index=applicants.index,
        columns=[
            "pd",
            "risk_grade",
            "decision",
            "reason_codes",
            "recommended_limit",
            "indicative_cap",
            "assumed_ead",
            "expected_loss_proxy",
        ],
    )


def policy_summary(decisions: pd.DataFrame, labels=None) -> dict:
    approved = decisions.loc[decisions.decision == "APPROVE"]
    n = len(decisions)
    summary = dict(
        applicants=n,
        approved=len(approved),
        manual_review=int((decisions.decision == "MANUAL_REVIEW").sum()),
        declined=int((decisions.decision == "DECLINE").sum()),
        approval_rate=len(approved) / n if n else None,
        expected_bad_rate=float(approved.pd.mean()) if len(approved) else None,
        expected_bad_count=float(approved.pd.sum()),
        total_offered_limit=float(approved.recommended_limit.sum()),
        assumed_ead=float(approved.assumed_ead.sum()),
        expected_loss_proxy=float(approved.expected_loss_proxy.sum()),
        approved_grade_counts={
            str(k): int(v) for k, v in approved.risk_grade.value_counts().items()
        },
    )
    if labels is not None:
        y = np.asarray(labels)
        if y.shape != (n,) or not np.isin(y, [0, 1]).all():
            raise ValueError("Historical labels must be a complete binary positional vector")
        mask = (decisions.decision == "APPROVE").to_numpy()
        summary["historical_selected_bad_rate"] = float(y[mask].mean()) if mask.any() else None
    return summary
