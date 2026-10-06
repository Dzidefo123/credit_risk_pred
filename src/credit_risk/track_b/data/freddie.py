"""Streaming, fail-closed official R47 pipe-file parsing; no network acquisition."""

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from .schemas import LAYOUT, ORIGINATION, PERFORMANCE

MONTHS = {
    "first_payment_month",
    "maturity_month",
    "reporting_month",
    "defect_month",
    "termination_month",
    "last_paid_due_month",
}
NUMERIC = {
    "orig_credit_score",
    "mi_percentage",
    "number_units",
    "orig_cltv",
    "orig_dti",
    "orig_upb",
    "orig_ltv",
    "orig_interest_rate",
    "original_loan_term",
    "number_of_borrowers",
    "vantage_score",
    "current_principal_balance",
    "loan_age",
    "remaining_legal_months",
    "current_interest_rate",
    "non_interest_upb",
    "mi_recoveries",
    "net_sale_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "preservation_costs",
    "taxes_insurance",
    "misc_expenses",
    "actual_loss",
    "cumulative_modification_costs",
    "estimated_ltv",
    "removal_upb",
    "delinquent_accrued_interest",
    "period_modification_costs",
    "interest_bearing_upb",
    "bankruptcy_cramdown_costs",
}
SENTINELS = {
    "orig_credit_score": "9999",
    "vantage_score": "9999",
    "mi_percentage": "999",
    "number_units": "99",
    "orig_cltv": "999",
    "orig_dti": "999",
    "orig_ltv": "999",
    "number_of_borrowers": "99",
    "estimated_ltv": "999",
}
INTEGER = {
    "orig_credit_score",
    "number_units",
    "original_loan_term",
    "number_of_borrowers",
    "vantage_score",
    "loan_age",
    "remaining_legal_months",
}
TERMINATIONS = {"", "01", "02", "03", "09", "15", "16", "96"}


@dataclass
class ParseStats:
    source_rows: int = 0
    valid_rows: int = 0
    malformed_rows: int = 0
    rejects: Counter = field(default_factory=Counter)

    def summary(self):
        return {
            "source_rows": self.source_rows,
            "valid_rows": self.valid_rows,
            "malformed_rows": self.malformed_rows,
            "rejects": dict(sorted(self.rejects.items())),
        }


def month(value):
    if not re.fullmatch(r"[0-9]{6}", value):
        raise ValueError("invalid_month")
    year, m = int(value[:4]), int(value[4:])
    if not 1900 <= year <= 2200 or not 1 <= m <= 12:
        raise ValueError("invalid_month")
    return pd.Period(f"{year:04d}-{m:02d}", freq="M")


def parse_row(tokens, kind):
    columns = ORIGINATION if kind == "origination" else PERFORMANCE
    if len(tokens) != len(columns):
        raise ValueError("wrong_field_count")
    values = [value.strip() for value in tokens]
    raw = dict(zip(columns, values, strict=True))
    if not re.fullmatch(r"F10Q[1-4][0-9]{7}", raw["loan_id"]):
        raise ValueError("missing_or_wrong_vintage_loan_id")
    result = dict(raw)
    for key, value in raw.items():
        if key in MONTHS:
            result[key] = None if not value else month(value)
        elif key in NUMERIC:
            if not value or value == SENTINELS.get(key):
                result[key] = None
            else:
                try:
                    number = float(value)
                except ValueError:
                    # Official net-proceeds layout is alpha-numeric. Keep token separately;
                    # never guess that an unexplained disclosure code means zero.
                    if key == "net_sale_proceeds" and re.fullmatch(r"[A-Za-z]+", value):
                        result[key] = None
                        result["net_sale_proceeds_disclosure_code"] = value
                        continue
                    raise ValueError("invalid_numeric") from None
                if not math.isfinite(number) or (key in INTEGER and not number.is_integer()):
                    raise ValueError("invalid_numeric")
                if (
                    key in {"orig_upb", "current_principal_balance", "non_interest_upb"}
                    and number < 0
                ):
                    raise ValueError("negative_principal")
                result[key] = number
    if kind == "origination":
        if raw["amortization_type"] != "FRM" or not raw["first_payment_month"]:
            raise ValueError("unsupported_product_or_missing_static_date")
    else:
        if not raw["reporting_month"]:
            raise ValueError("missing_reporting_month")
        state = raw["delinquency_state"]
        if state not in {"", "XX", "RA"} and not re.fullmatch(r"[0-9]{2}", state):
            raise ValueError("unrecognized_state")
        if raw["termination_code"] not in TERMINATIONS:
            raise ValueError("unrecognized_termination")
    return result


def records(handle, kind, stats, *, layout=LAYOUT):
    if layout != LAYOUT or kind not in {"origination", "performance"}:
        raise ValueError("Unsupported source layout")
    reader = csv.reader(handle, delimiter="|", strict=True)
    for tokens in reader:
        stats.source_rows += 1
        try:
            result = parse_row(tokens, kind)
        except ValueError as exc:
            stats.malformed_rows += 1
            stats.rejects[str(exc)] += 1
            continue
        stats.valid_rows += 1
        yield result
