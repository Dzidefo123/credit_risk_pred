"""Percentage-point original-coupon proxy, never a current-rate assertion."""

import numpy as np

from credit_risk.track_b.macro_hazard.data import MACRO, macro_join


def gap(data):
    contract = data["numeric"][:, 4]
    market = data["macro"][:, MACRO.index("mortgage_30y_level")]
    if not np.isfinite(contract).all() or not np.isfinite(market).all():
        raise ValueError("Missing contractual/PIT market rate; no implicit gap imputation")
    if ((contract <= 0) | (contract >= 100) | (market <= 0) | (market >= 100)).any():
        raise ValueError("Invalid percentage rate units/range")
    return contract - market


def representation(data, linear=False):
    value = gap(data)
    return (
        value[:, None] if linear else np.column_stack([np.maximum(value, 0), np.minimum(value, 0)])
    )


def checked_market(table, month):
    return macro_join(table, month, ("mortgage_30y_level",))[MACRO.index("mortgage_30y_level")]


def proxy_status(current_rate, original_rate, previously_modified):
    """Audit only: current/monthly source knowledge time is not historically verified."""
    if not current_rate:
        return "CURRENT_RATE_MISSING"
    try:
        rate = float(current_rate)
    except ValueError:
        return "CURRENT_RATE_MALFORMED"
    if not np.isfinite(rate) or not 0 < rate < 100:
        return "CURRENT_RATE_MALFORMED"
    if previously_modified:
        return "MODIFIED_ORIGINAL_COUPON_PROXY"
    return "COUPON_DIFFERS" if abs(rate - original_rate) > 1e-6 else "COUPON_MATCHES"
