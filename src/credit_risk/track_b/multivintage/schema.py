"""Explicit R47 adapter; preserves Task2 rules with a vintage-qualified identifier."""

import math
import re

from credit_risk.track_b.data.freddie import (
    INTEGER,
    MONTHS,
    NUMERIC,
    SENTINELS,
    TERMINATIONS,
    month,
)
from credit_risk.track_b.data.schemas import ORIGINATION, PERFORMANCE

VERSION = "multivintage-r47-v1.0.0"


def parse(tokens, kind, vintage):
    columns = ORIGINATION if kind == "origination" else PERFORMANCE
    if len(tokens) != len(columns):
        raise ValueError("wrong_field_count")
    values = [value.strip() for value in tokens]
    raw = dict(zip(columns, values, strict=True))
    if not re.fullmatch(rf"F{vintage % 100:02d}Q[1-4][0-9]{{7}}", raw["loan_id"]):
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
