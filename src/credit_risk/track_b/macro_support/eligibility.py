"""Calendar completeness intersects existing incident-risk history, not facility deletion."""

import hashlib
from enum import StrEnum

from credit_risk.track_b.survival.risk import monthly_text, ordinal

VERSION = "macro-support-v1.0.0"
REDUCED = (
    "unemployment_level",
    "unemployment_change_3m",
    "treasury_10y_level",
    "cpi_yoy",
    "gdp_qoq",
)


class Reason(StrEnum):
    ELIGIBLE_FULL_PIT = "ELIGIBLE_FULL_PIT"
    ELIGIBLE_REDUCED_PIT = "ELIGIBLE_REDUCED_PIT"
    NO_PIT_MORTGAGE_RATE = "NO_PIT_MORTGAGE_RATE"
    NO_PIT_HPI = "NO_PIT_HPI"
    NO_UNEMPLOYMENT_CHANGE_OPERAND = "NO_UNEMPLOYMENT_CHANGE_OPERAND"
    MACRO_PROVENANCE_INSUFFICIENT = "MACRO_PROVENANCE_INSUFFICIENT"
    MACRO_NOT_AVAILABLE_AT_T0 = "MACRO_NOT_AVAILABLE_AT_T0"
    MACRO_SOURCE_OBSERVATION_MISSING = "MACRO_SOURCE_OBSERVATION_MISSING"
    MACRO_STALE = "MACRO_STALE"
    MACRO_OPERAND_UNAVAILABLE = "MACRO_OPERAND_UNAVAILABLE"
    MACRO_METADATA_REGIME_MISMATCH = "MACRO_METADATA_REGIME_MISMATCH"
    MORTGAGE_INTERVAL_INELIGIBLE = "MORTGAGE_INTERVAL_INELIGIBLE"
    STUDY_CUTOFF_EXCEEDED = "STUDY_CUTOFF_EXCEEDED"


def complete_months(table, names):
    if len(table) != len({r["reporting_month"] for r in table}):
        raise ValueError("Duplicate macro month")
    if not names or len(names) != len(set(names)):
        raise ValueError("Invalid design feature set")
    return [
        r["reporting_month"]
        for r in sorted(table, key=lambda r: r["reporting_month"])
        if all(r["features"].get(n, {}).get("status") == "AVAILABLE" for n in names)
    ]


def support_windows(months):
    if months != sorted(set(months)):
        raise ValueError("Unsorted or duplicate support months")
    windows = []
    for month in months:
        if windows and ordinal(month) == ordinal(windows[-1]["last"]) + 1:
            windows[-1]["last"] = month
            windows[-1]["months"] += 1
        else:
            windows.append(dict(first=month, last=month, months=1))
    return windows


def macro_reasons(month, mapping, names, cutoff):
    if month > cutoff:
        return [Reason.STUDY_CUTOFF_EXCEEDED.value]
    if month not in mapping:
        return [Reason.MACRO_PROVENANCE_INSUFFICIENT.value]
    reasons = set()
    for name in names:
        feature = mapping[month]["features"].get(name, {})
        status = feature.get("status", "UNVERIFIED")
        if status == "AVAILABLE" and any(
            i.get("current_revised", False) for i in feature.get("inputs", [])
        ):
            status = "UNVERIFIED"
        if status == "AVAILABLE":
            continue
        if name == "mortgage_30y_level" or (
            name == "mortgage_treasury_spread"
            and mapping[month]["features"].get("mortgage_30y_level", {}).get("status")
            != "AVAILABLE"
        ):
            reasons.add(Reason.NO_PIT_MORTGAGE_RATE.value)
        if name == "hpi_yoy":
            reasons.add(Reason.NO_PIT_HPI.value)
        if name == "unemployment_change_3m" and status == "MISSING_OPERAND":
            reasons.add(Reason.NO_UNEMPLOYMENT_CHANGE_OPERAND.value)
        reasons.add(
            {
                "STALE": Reason.MACRO_STALE,
                "MISSING_OPERAND": Reason.MACRO_OPERAND_UNAVAILABLE,
                "METADATA_REGIME_MISMATCH": Reason.MACRO_METADATA_REGIME_MISMATCH,
                "UNAVAILABLE": Reason.MACRO_NOT_AVAILABLE_AT_T0,
            }.get(status, Reason.MACRO_PROVENANCE_INSUFFICIENT).value
        )
    return sorted(reasons)


def mortgage_reason(current, target, index, minimum_lookback=6):
    if not target or ordinal(target["reporting_month"]) != ordinal(current["reporting_month"]) + 1:
        return "NO_CONSECUTIVE_TARGET"
    if not current["analytical_prefix"] or current["event_category"] != "none":
        return "PRIOR_EVENT_OR_UNASCERTAINABLE_PREFIX"
    if index + 1 < minimum_lookback:
        return "INSUFFICIENT_LOOKBACK"
    if not target["analytical_prefix"]:
        return "TARGET_OUTSIDE_CONTIGUOUS_PREFIX"
    if target["event_category"] not in {"none", "default", "payoff"}:
        return "TARGET_" + target["event_category"].upper()
    if current["months_since_first_payment_proxy"] < 0:
        return "PRE_FIRST_PAYMENT_PROXY"
    return None


def eligible_intervals(history, mapping, names, cutoff, minimum_lookback=6):
    """All eligible later intervals of a seasoned loan survive calendar filtering."""
    if len({r["reporting_month"] for r in history}) != len(history):
        raise ValueError("Duplicate mortgage month")
    if history != sorted(history, key=lambda r: r["reporting_month"]):
        raise ValueError("Mortgage history must be sorted")
    intervals = []
    for i, (current, target) in enumerate(zip(history, history[1:], strict=False)):
        month = target["reporting_month"]
        if mortgage_reason(current, target, i, minimum_lookback):
            continue
        if macro_reasons(month, mapping, names, cutoff):
            continue
        intervals.append(
            dict(
                t0=current["reporting_month"],
                target_month=month,
                event=target["event_category"],
                provider_age=current["loan_age"],
                duration=current["months_since_first_payment_proxy"],
            )
        )
    if sum(r["event"] != "none" for r in intervals) > 1:
        raise ValueError("Repeated first endpoint")
    return intervals


def duration_band(age):
    if age is None or age < 0:
        return "UNAVAILABLE"
    lower = 0
    for upper in [12, 24, 36, 60, 84, 120, 180, 240]:
        if age <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return "241+"


def validation_role(vintage, loan, salt):
    bucket = int(hashlib.sha256(f"{salt}:{vintage}:{loan}".encode()).hexdigest(), 16) % 10
    return "development" if bucket <= 6 else "temporal_evaluation"


def exit_reason(history, intervals, names, mapping, cutoff):
    if not intervals:
        return "NO_ELIGIBLE_INTERVAL"
    last = intervals[-1]
    if last["event"] != "none":
        return last["event"]
    month = monthly_text(ordinal(last["target_month"]) + 1)
    if macro_reasons(month, mapping, names, cutoff):
        return "MACRO_SUPPORT_END"
    target = next((r for r in history if r["reporting_month"] == month), None)
    if target is None:
        return "OBSERVATION_END_OR_GAP"
    return {
        "administrative": "ADMINISTRATIVE_CENSOR",
        "ambiguous": "AMBIGUOUS_CENSOR",
        "unknown": "UNKNOWN_CENSOR",
    }.get(target["event_category"], "MORTGAGE_INELIGIBLE_CENSOR")
