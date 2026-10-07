"""First eligible window entry, monthly first-event exits, no future predictors."""

import hashlib
from collections import Counter

from credit_risk.track_b.data.panel import event_category


def ordinal(month):
    text = str(month).strip().replace("-", "")
    return int(text[:4]) * 12 + int(text[4:6]) - 1


def monthly_text(n):
    return f"{n // 12:04d}-{n % 12 + 1:02d}"


def raw_history(lines, policy):
    history = {}
    for raw in lines:
        tokens = raw.split("|")
        month = ordinal(tokens[1])
        date = ordinal(tokens[9]) if tokens[9].strip() else None
        row = dict(
            reporting_month=month,
            termination_month=date,
            termination_code=tokens[8].strip(),
            delinquency_state=tokens[3].strip(),
        )
        if month in history:
            raise ValueError("Duplicate monthly source record")
        history[month] = dict(category=event_category(row, policy), state=row["delinquency_state"])
    return history


def first_endpoint(history):
    return next(
        (
            history[t]["category"]
            for t in sorted(history)
            if history[t]["category"] in ["default", "payoff", "administrative", "ambiguous"]
        ),
        "active_or_unknown",
    )


def trajectory(frame, history, features, start=None, end=None):
    f = frame.sort_values("t0")
    first = f.loc[f.eligible.eq(True)].copy()
    if start is not None:
        first = first.loc[first.t0.ge(start)]
    if end is not None:
        first = first.loc[first.t0.le(end)]
    if first.empty:
        return None, [], "no_eligible_entry"
    entry = first.iloc[0]
    origin = ordinal(entry.t0)
    if any(history[t]["category"] != "none" for t in history if t <= origin):
        raise ValueError("Eligible entry contradicts prior event/unknown prefix")
    limit = ordinal(end) if end else max(history)
    indexed = {ordinal(row["t0"]): row for row in f.to_dict("records")}
    rows = []
    exit_time = 0
    event = 0
    reason = "observation_end"
    for now in range(origin, limit + 1):
        current = indexed.get(now)
        if current is None or not current["eligible"]:
            break
        if now + 1 > limit:
            reason = "calendar_censor" if end else "observation_end"
            break
        target = history.get(now + 1)
        if target is None:
            reason = "gap_or_observation_end"
            break
        kind = target["category"]
        if kind not in ["none", "default", "payoff"]:
            reason = kind
            break
        event = {"none": 0, "default": 1, "payoff": 2}[kind]
        row = {name: current[name] for name in features}
        row.update(
            loan_id=str(entry.loan_id),
            t0=monthly_text(now),
            target_month=monthly_text(now + 1),
            duration=now + 1 - origin,
            event_code=event,
            next_state=target["state"],
        )
        rows.append(row)
        exit_time = now + 1 - origin
        if event:
            reason = kind
            break
    subject = dict(
        loan_id=str(entry.loan_id),
        entry_month=str(entry.t0),
        entry_time=0,
        exit_time=exit_time,
        event_code=event,
        exit_reason=reason,
        entry_mortgage_age=float(entry.loan_age),
        **{name: entry[name] for name in features},
    )
    if exit_time <= 0:
        return subject, [], "zero_duration_quarantined"
    if any(r["duration"] <= 0 or r["t0"] >= r["target_month"] for r in rows):
        raise ValueError("Pre-entry/invalid interval")
    if sum(r["event_code"] > 0 for r in rows) > 1:
        raise ValueError("Repeated first event")
    return subject, rows, "included"


def fingerprint(frame, columns):
    return hashlib.sha256(
        frame.loc[:, columns]
        .sort_values(columns[:2])
        .to_csv(index=False, lineterminator="\n")
        .encode()
    ).hexdigest()


def effective(subjects, intervals=None):
    return dict(
        facilities=int(len(subjects)),
        risk_intervals=int(len(intervals)) if intervals is not None else None,
        default_facilities=int(subjects.event_code.eq(1).sum()),
        payoff_facilities=int(subjects.event_code.eq(2).sum()),
        censored_facilities=int(subjects.event_code.eq(0).sum()),
    )


def transition_table(risk):
    labels = risk.next_state.astype(str).copy()
    labels.loc[risk.event_code.eq(1)] = "default"
    labels.loc[risk.event_code.eq(2)] = "payoff/maturity"
    counts = Counter(zip(risk.delinquency_state.astype(str), labels, strict=True))
    totals = Counter(risk.delinquency_state.astype(str))
    return [
        dict(current=a, next=b, count=int(n), probability=n / totals[a])
        for (a, b), n in sorted(counts.items())
    ]


def check_feature_time(current, observed, event_month=None):
    if ordinal(observed) > ordinal(current):
        raise ValueError("Future feature prohibited")
    if event_month is not None and ordinal(current) >= ordinal(event_month):
        raise ValueError("Post-event feature prohibited")
