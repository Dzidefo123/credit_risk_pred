"""Policy B never coerces source values or generalizes the approved anomaly set."""

import hashlib

from credit_risk.track_b.data.annual import GateStop
from credit_risk.track_b.data.schemas import ORIGINATION, PERFORMANCE
from credit_risk.track_b.multivintage.schema import parse


def classify(line, vintage, quarter, number, registry):
    tokens = line.split("|")
    if len(tokens) != len(ORIGINATION):
        raise GateStop("Unexpected origination field count")
    approved = [
        r
        for r in registry
        if (r["vintage"], int(r["member"][-1]), r["line"]) == (vintage, quarter, number)
    ]
    if len(approved) > 1:
        raise GateStop("Duplicate anomaly authorization")
    if approved:
        row = approved[0]
        if (
            tokens[19] != row["loan_id"]
            or hashlib.sha256(line.encode()).hexdigest() != row["raw_record_sha256"]
            or line != row["raw"]
        ):
            raise GateStop("Authorized anomaly identity/content changed")
        return dict(eligible=False, raw=line, reason=row["reason_code"], audit_reference=row["key"])
    try:
        parsed = parse(tokens, "origination", vintage)
        if parsed["loan_id"][4] != str(quarter):
            raise ValueError("unresolved_quarter_membership")
    except ValueError as exc:
        raise GateStop(
            f"Unapproved origination anomaly at {vintage}Q{quarter} line {number}: {exc}"
        ) from exc
    return dict(eligible=True, raw=line, loan_id=parsed["loan_id"], reason=None)


class StructuralPerformanceReader:
    """All streamed columns checked; unchanged adapter types selected histories."""

    def __init__(self, reader):
        self.reader = reader
        self.vintage = reader.vintage

    def lines(self, quarter, kind):
        for number, line in enumerate(self.reader.lines(quarter, kind), 1):
            if kind == "performance" and line.count("|") != len(PERFORMANCE) - 1:
                raise GateStop(
                    f"Unexpected performance field count at {self.vintage}Q{quarter} line {number}"
                )
            yield line
