"""Outcome-independent identity, missing-state and descriptive longitudinal primitives."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from credit_risk.track_b.data.freddie import NUMERIC, SENTINELS
from credit_risk.track_b.data.panel import event_category
from credit_risk.track_b.data.schemas import ORIGINATION, PERFORMANCE


def set_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def rank_ids(ids, vintage, n=20000):
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 20000:
        raise ValueError("Invalid sample size")
    if any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("Identifier-only sampling")
    return sorted(set(ids), key=lambda i: (rank_key(i, vintage), i))[:n]


def rank_key(loan, vintage):
    return hashlib.sha256(f"track-b-multivintage-v1:{vintage}:{loan}".encode()).hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalize(token, field, *, available=True, compatible=True, applicable=True):
    if not compatible:
        raise ValueError("SEMANTICALLY_INCOMPATIBLE")
    if not available:
        return None, "STRUCTURALLY_UNAVAILABLE"
    if not applicable:
        return None, "NOT_APPLICABLE"
    if not token or token == SENTINELS.get(field):
        return None, "MISSING_IN_SOURCE"
    if field == "net_sale_proceeds" and token.isalpha():
        return None, "MISSING_IN_SOURCE"
    if field in NUMERIC:
        try:
            value = float(token)
        except ValueError:
            return None, "PARSER_FAILURE"
        if not np.isfinite(value):
            return None, "PARSER_FAILURE"
        return value, "OBSERVED"
    return token, "OBSERVED"


def quantiles(values):
    if not values:
        return dict(n=0, minimum=None, p25=None, median=None, p75=None, maximum=None)
    q = np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist()
    return dict(
        n=len(values), **dict(zip(["minimum", "p25", "median", "p75", "maximum"], q, strict=True))
    )


def age_band(age):
    if age is None or age < 0:
        return "unavailable"
    for bound in [12, 24, 36, 60, 84, 120, 180, 240]:
        if age <= bound:
            return f"<= {bound}"
    return ">240"


def trajectory(rows, policy, horizons, rule):
    """Keep all source rows; first contiguous analytical prefix is never repaired/re-entered."""
    findings = Counter()
    months = Counter(r["reporting_month"] for r in rows)
    if list(months) != sorted(months):
        findings["source_out_of_order_facilities"] = 1
    findings["duplicate_facility_months"] = sum(n - 1 for n in months.values())
    ordered = sorted(rows, key=lambda r: r["reporting_month"])
    if not ordered:
        return dict(
            endpoint="active_or_unknown",
            observed_span=0,
            risk_exit=0,
            risk_event="missing",
            first=None,
            last=None,
            findings=dict(findings),
            annotations=[],
            support={h: False for h in horizons},
        )
    first, last = ordered[0]["reporting_month"], ordered[-1]["reporting_month"]
    cats = [event_category(r, policy) for r in ordered]
    endpoint = next(
        (c for c in cats if c in {"default", "payoff", "administrative", "ambiguous"}),
        "active_or_unknown",
    )
    previous = None
    entry_active = cats[0] == "none" and months[first] == 1
    findings["initial_nonactive_or_duplicate_facilities"] = int(not entry_active)
    prefix = entry_active
    first_event_seen = source_terminal_seen = False
    exit_time = 0
    risk_event = "observation_end"
    annotations = []
    for r, c in zip(ordered, cats, strict=True):
        month = r["reporting_month"]
        gap = previous is not None and month.ordinal > previous.ordinal + 1
        if gap:
            findings["gap_intervals"] += 1
            findings["missing_expected_months"] += month.ordinal - previous.ordinal - 1
        duplicate = months[month] > 1
        if first_event_seen:
            findings["post_research_endpoint_rows"] += 1
        if source_terminal_seen:
            findings["post_source_termination_rows"] += 1
        if r["termination_code"]:
            findings["source_termination_rows"] += 1
            if source_terminal_seen:
                findings["multiple_source_termination_rows"] += 1
        if r["current_principal_balance"] is None:
            findings["missing_current_balance_rows"] += 1
        if r["loan_age"] is not None and r["loan_age"] < 0:
            findings["negative_provider_age_rows"] += 1
        if previous is not None and month.ordinal < previous.ordinal:
            findings["impossible_sorted_time"] += 1
        active_prefix = prefix and not gap and not duplicate
        annotations.append(
            dict(
                event_category=c,
                post_research_endpoint=first_event_seen,
                post_source_termination=source_terminal_seen,
                analytical_prefix=active_prefix,
                pandemic_regime=month.year in {2020, 2021},
            )
        )
        if prefix:
            if gap or duplicate or c in {"unknown", "ambiguous", "administrative"}:
                risk_event = "gap" if gap else "duplicate" if duplicate else c
                prefix = False
            else:
                exit_time = month.ordinal - first.ordinal
                if c in {"default", "payoff"}:
                    risk_event = c
                    prefix = False
        first_event_seen |= c in {"default", "payoff", "administrative", "ambiguous"}
        source_terminal_seen |= bool(r["termination_code"])
        previous = month
    known = {
        h: entry_active
        and (exit_time >= h or (risk_event in {"default", "payoff"} and exit_time <= h))
        for h in horizons
    }
    at_risk = {h: entry_active and exit_time >= h for h in horizons}
    return dict(
        endpoint=endpoint,
        observed_span=last.ordinal - first.ordinal,
        recorded_months=len(months),
        risk_exit=exit_time,
        risk_event=risk_event,
        first=str(first),
        last=str(last),
        findings=dict(findings),
        annotations=annotations,
        support=known,
        at_risk=at_risk,
    )


def horizon_support(at_risk, known, rule):
    if at_risk >= rule["supported_min_at_risk"] and known >= rule["supported_min_known"]:
        return "SUPPORTED"
    if at_risk >= rule["limited_min_at_risk"] and known >= rule["limited_min_known"]:
        return "LIMITED"
    return "UNSUPPORTED"


def apc_diagnostic(cohorts, ages):
    """Linear-algebra demonstration only; exact synthetic/inferred clocks, no outcome fit."""
    c = np.repeat(np.asarray(cohorts, float), len(ages))
    a = np.tile(np.asarray(ages, float), len(cohorts))
    period = c + a
    design = np.column_stack([np.ones(len(a)), period, c, a])
    return dict(
        rows=len(a),
        columns=4,
        rank=int(np.linalg.matrix_rank(design)),
        null_vector=[0, 1, -1, -1],
        maximum_identity_residual=float(np.abs(period - c - a).max()),
        scope="Exact-clock illustration; does not assert exact mortgage origination month",
    )


def missing_tally(raw, fields, counts, facility_missing):
    for field, token in zip(fields, raw, strict=True):
        _, state = normalize(
            token.strip(),
            field,
            applicable=not (field == "termination_month" and not raw[8].strip()),
        )
        counts[field][state] += 1
        if (
            (field == "net_sale_proceeds" and token.strip().isalpha())
            or token.strip() == SENTINELS.get(field)
            or (field == "delinquency_state" and token.strip() in {"XX", "RA"})
        ):
            counts[field]["special_value:" + token.strip()] += 1
        if state != "OBSERVED":
            facility_missing.add(field)


def fresh_counts():
    return defaultdict(Counter)


FIELDS = {"origination": ORIGINATION, "performance": PERFORMANCE}
