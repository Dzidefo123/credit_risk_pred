"""Nominal-time landmarks and protocol outcomes; no model fitting or complete-case filter."""

from collections import Counter, defaultdict

import pandas as pd

from .schemas import AUDIT, FEATURES, OUTCOMES, empty_panel


def event_category(row, policy):
    if row.get("duplicate_month"):
        return "unknown"
    code = row["termination_code"]
    period = row["reporting_month"]
    if (code and row["termination_month"] != period) or (not code and row["termination_month"]):
        return "ambiguous"
    state = row["delinquency_state"]
    severe = state == policy["reo_state"] or (
        state.isdigit()
        and policy["numeric_delinquency_min"] <= int(state) <= policy["numeric_delinquency_max"]
    )
    default = severe or code in policy["credit_termination_codes"]
    payoff = code in policy["competing_payoff_codes"]
    if default and payoff:
        return "ambiguous"
    if default:
        return "default"
    if code in policy["administrative_exit_codes"]:
        return "administrative"
    if state in {"", policy["unknown_state"]}:
        return "unknown"
    return "payoff" if payoff else "none"


def outcome(t0, history, policy, horizon=12):
    observed = 0
    for offset in range(1, horizon + 1):
        period = t0 + offset
        if period not in history:
            return "right_censored", None, None, observed
        category = event_category(history[period], policy)
        if category == "ambiguous":
            return "ambiguous_event_order", None, offset, observed
        if category == "unknown":
            status = "insufficient_followup" if observed == 0 else "right_censored"
            return status, None, None, observed
        if category == "administrative":
            return "right_censored", None, offset, observed
        observed += 1
        if category == "default":
            return "positive_default", 1, offset, observed
        if category == "payoff":
            return "competing_payoff", 0, offset, observed
    return "negative_survived_horizon", 0, None, observed


def build_panel(origination, performance, protocol):
    grouped = defaultdict(list)
    for row in performance:
        grouped[row["loan_id"]].append(row)
    outputs = []
    findings = Counter()
    per_loan_rows = []
    for loan_id in sorted(origination):
        records = grouped.get(loan_id, [])
        if not records:
            findings["selected_origination_without_performance"] += 1
            continue
        months = defaultdict(list)
        for row in records:
            months[row["reporting_month"]].append(row)
        history = {}
        for period, same_month in months.items():
            if len(same_month) > 1:
                findings["duplicate_loan_months"] += 1
                fingerprints = {str(sorted(r.items())) for r in same_month}
                findings["exact_duplicate_rows"] += len(same_month) - len(fingerprints)
                findings["conflicting_duplicate_months"] += len(fingerprints) > 1
                # Keep raw records outside panel; placeholder is not a chosen/merged source row.
                history[period] = {"duplicate_month": True, "reporting_month": period}
            else:
                history[period] = same_month[0]
        ordered = sorted(history)
        per_loan_rows.append(len(ordered))
        findings["source_out_of_order_loans"] += [r["reporting_month"] for r in records] != sorted(
            r["reporting_month"] for r in records
        )
        previous = None
        clean_prefix = True
        terminal_seen = False
        for i, period in enumerate(ordered):
            row = history[period]
            if previous is not None and period != previous + 1:
                findings["gap_intervals"] += 1
                findings["missing_expected_months"] += period.ordinal - previous.ordinal - 1
                clean_prefix = False
            category = event_category(row, protocol["event"])
            post_terminal = terminal_seen
            if post_terminal:
                if row.get("defect_month"):
                    findings["documented_defect_post_terminal_rows"] += 1
                else:
                    findings["unexpected_post_terminal_rows"] += 1
                clean_prefix = False
            valid = (
                clean_prefix
                and category == "none"
                and i + 1 >= protocol["time"]["minimum_consecutive_pre_t0_months"]
            )
            reason = (
                "eligible"
                if valid
                else (
                    "insufficient_lookback"
                    if clean_prefix and category == "none"
                    else "prior_or_current_event_unknown_gap_or_terminal"
                )
            )
            features = {
                key: origination[loan_id].get(key, row.get(key))
                if key in origination[loan_id]
                else row.get(key)
                for key in FEATURES
            }
            if row.get("duplicate_month"):
                features = {
                    key: origination[loan_id].get(key) if key in origination[loan_id] else None
                    for key in FEATURES
                }
            features = {key: (None if value == "" else value) for key, value in features.items()}
            for key, missing in {"occupancy_status": "9", "property_type": "99"}.items():
                if features.get(key) == missing:
                    features[key] = None
            result = dict(
                zip(
                    AUDIT,
                    [
                        loan_id,
                        str(period),
                        valid,
                        reason,
                        "unverified_provider_vintage; nominal-time only",
                    ],
                    strict=True,
                )
            )
            result.update(features)
            if valid:
                values = outcome(
                    period, history, protocol["event"], protocol["time"]["horizon_months"]
                )
            else:
                values = ("not_incident_risk_eligible", None, None, 0)
            result.update(dict(zip(OUTCOMES, values, strict=True)))
            outputs.append(result)
            if category != "none":
                clean_prefix = False
            if row.get("termination_code"):
                terminal_seen = True
            previous = period
    panel = pd.DataFrame(outputs) if outputs else empty_panel()
    if not panel.empty:
        panel["binary_default_12m"] = pd.array(panel["binary_default_12m"], dtype="Int64")
        panel["event_offset"] = pd.array(panel["event_offset"], dtype="Int64")
    return panel, {
        "findings": dict(sorted(findings.items())),
        "observations_per_loan": per_loan_rows,
    }


def distribution(values):
    numeric = pd.to_numeric(pd.Series(values, dtype=float), errors="coerce").dropna()
    if numeric.empty:
        return {"available": 0, "min": None, "median": None, "max": None}
    return {
        "available": len(numeric),
        "min": float(numeric.min()),
        "median": float(numeric.median()),
        "max": float(numeric.max()),
    }


def summarize(panel, origination, performance, integrity, protocol):
    eligible = panel[panel["eligible"].eq(True)] if not panel.empty else panel
    statuses = Counter(eligible["outcome_status"])
    denominators = len(eligible)
    completed = statuses["negative_survived_horizon"]
    return {
        "selected_loans": len(origination),
        "origination_records": len(origination),
        "monthly_performance_records": len(performance),
        "ambiguous_records": sum(
            event_category(r, protocol["event"]) == "ambiguous" for r in performance
        ),
        "panel_observations": len(panel),
        "eligible_landmarks": denominators,
        "ineligible_landmarks": len(panel) - denominators,
        "reporting_period_range": [panel["t0"].min(), panel["t0"].max()]
        if not panel.empty
        else None,
        "observations_per_loan": distribution(integrity["observations_per_loan"]),
        "feature_missingness": {key: int(panel[key].isna().sum()) for key in FEATURES},
        "origination_missingness": {
            key: sum(r.get(key) is None or r.get(key) == "" for r in origination.values())
            for key in origination[next(iter(origination))]
        }
        if origination
        else {},
        "performance_missingness": {
            key: sum(r.get(key) is None or r.get(key) == "" for r in performance)
            for key in performance[0]
        }
        if performance
        else {},
        "delinquency_state_distribution": dict(
            sorted(Counter(r["delinquency_state"] or "missing" for r in performance).items())
        ),
        "termination_categories": dict(
            sorted(Counter(event_category(r, protocol["event"]) for r in performance).items())
        ),
        "balance_distribution": distribution([r["current_principal_balance"] for r in performance]),
        "outcomes_eligible_only": dict(sorted(statuses.items())),
        "followup": {
            "denominator_eligible_landmarks": denominators,
            "complete_event_free_horizons": completed,
            "complete_event_free_fraction": completed / denominators if denominators else None,
            "outcome_ascertained_including_events_and_payoff": sum(
                statuses[k]
                for k in ["positive_default", "negative_survived_horizon", "competing_payoff"]
            ),
            "status_counts": dict(sorted(statuses.items())),
        },
        "temporal_integrity": integrity["findings"],
    }
