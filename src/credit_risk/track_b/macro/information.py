"""Date-resolution point-in-time selection and transformation lineage."""

import hashlib
import json
import math
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    series_id: str = Field(min_length=1)
    geography: Literal["US"]
    reference_period: date  # inclusive period end; never FRED quarter-start label
    release_date: date  # original publication date for THIS reference observation
    vintage_date: date  # first availability of THIS value version in selected archive
    revision_date: date  # agency publication date of THIS version (initial = release)
    value: float = Field(allow_inf_nan=False)
    revision_sequence: int = Field(ge=0)
    source: str = Field(min_length=1)
    frequency: Literal["D", "W", "M", "Q"]
    units: str = Field(min_length=1)
    seasonal_adjustment: Literal["SA", "NSA"]
    transformation: Literal["level"] = "level"
    retrieved_at: datetime
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    representation: Literal["vintage", "current_revised"] = "vintage"
    date_evidence: Literal["exact", "conservative_upper_bound"] = "exact"

    @model_validator(mode="after")
    def chronology(self):
        import calendar

        d = self.reference_period
        if self.frequency in {"M", "Q"} and d.day != calendar.monthrange(d.year, d.month)[1]:
            raise ValueError("Normalize source labels to completed period-end")
        if self.frequency == "Q" and d.month not in {3, 6, 9, 12}:
            raise ValueError("Quarter period-end required")
        if self.release_date < self.reference_period:
            raise ValueError("Observed period must be complete before publication")
        if self.revision_date < self.release_date or self.vintage_date < self.revision_date:
            raise ValueError("Version chronology inconsistent")
        if self.revision_sequence == 0 and self.revision_date != self.release_date:
            raise ValueError("Initial release cannot have later revision date")
        if self.retrieved_at.tzinfo is None:
            raise ValueError("Retrieval timestamp needs timezone")
        return self


def select_asof(rows, series_id, t0, *, period=None, rule="latest_known", max_age_days=None):
    """Select newest completed period, then first release or latest admissible version.

    Dates mean end-of-day assessment. Date-only sources must NOT be used intraday.
    Unknown publication dates are not representable; no implicit lag substitutes.
    """
    if rule not in {"latest_known", "first_release"}:
        raise ValueError("Current-revised/descriptive representation is not a predictor rule")
    selected = [r for r in rows if r.series_id == series_id and r.geography == "US"]
    if any(r.representation != "vintage" for r in selected):
        raise ValueError("Current-revised history prohibited in point-in-time input")
    keys = [(r.reference_period, r.vintage_date, r.revision_sequence) for r in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate/ambiguous macro version")
    if len({(r.frequency, r.units, r.seasonal_adjustment, r.source) for r in selected}) > 1:
        raise ValueError("Metadata regime change requires harmonization")
    eligible = [
        r
        for r in selected
        if r.reference_period <= t0
        and r.release_date <= t0
        and r.vintage_date <= t0
        and r.revision_date <= t0
        and (period is None or r.reference_period == period)
        and (rule != "first_release" or r.revision_sequence == 0)
    ]
    if not eligible:
        return None
    if max_age_days is not None:
        if max_age_days < 0:
            raise ValueError("Negative freshness cap")
        eligible = [r for r in eligible if (t0 - r.reference_period).days <= max_age_days]
        if not eligible:
            return None
    newest = max(r.reference_period for r in eligible)
    versions = [r for r in eligible if r.reference_period == newest]
    return max(versions, key=lambda r: (r.vintage_date, r.revision_date, r.revision_sequence))


def national_geography(level, value=None):
    """National information applies to all facilities without reading their geography."""
    if level != "national":
        raise ValueError("State/MSA/postal features require a separately approved protocol")
    if value not in (None, "", "US"):
        raise ValueError("National mapping accepts only US or absent geography")
    return "US"


def transform_asof(rows, series_id, t0, *, months, kind, rule="latest_known", max_age_days=None):
    """Paired periods resolved in the SAME information set; no backfill or interpolation."""
    import calendar

    if months not in {1, 3, 12} or kind not in {"difference", "growth_pct"}:
        raise ValueError("Unregistered transformation")
    current = select_asof(rows, series_id, t0, rule=rule, max_age_days=max_age_days)
    if current is None:
        return None
    if current.frequency not in {"M", "Q"} or (current.frequency == "Q" and months % 3):
        raise ValueError("Transformation requires aligned monthly/quarterly endpoints")
    d = current.reference_period
    if d.day != calendar.monthrange(d.year, d.month)[1]:
        raise ValueError("Monthly/quarterly reference period must be period-end")
    if current.frequency == "Q" and d.month not in {3, 6, 9, 12}:
        raise ValueError("Quarter endpoint required")
    index = d.year * 12 + d.month - 1 - months
    year, month = divmod(index, 12)
    prior_period = date(year, month + 1, calendar.monthrange(year, month + 1)[1])
    prior = select_asof(rows, series_id, t0, period=prior_period, rule=rule)
    if prior is None:
        return None
    if kind == "growth_pct" and prior.value <= 0:
        raise ValueError("Growth denominator must be positive")
    value = (
        current.value - prior.value
        if kind == "difference"
        else 100 * (current.value / prior.value - 1)
    )
    if not math.isfinite(value):
        raise ValueError("Nonfinite feature")
    return dict(
        value=value,
        t0=t0.isoformat(),
        rule=rule,
        months=months,
        kind=kind,
        inputs=[r.model_dump(mode="json") for r in (current, prior)],
    )


def feature_hash(features):
    """Canonical processed JSON hash, independent of dictionary insertion order."""
    payload = json.dumps(features, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def release_lags(rows):
    """First-release lags only; archive/revision delay is a separate diagnostic."""
    import statistics

    days = [(r.release_date - r.reference_period).days for r in rows if r.revision_sequence == 0]
    return dict(
        n=len(days),
        median_days=statistics.median(days) if days else None,
        range_days=[min(days), max(days)] if days else None,
    )


def spread_asof(rows, left, right, t0, *, max_age_days=(None, None)):
    """Latest known rate differential in percentage points, with separate period lineage."""
    a, b = (
        select_asof(rows, sid, t0, max_age_days=cap)
        for sid, cap in zip((left, right), max_age_days, strict=True)
    )
    if a is None or b is None:
        return None
    if a.units != "Percent" or b.units != "Percent":
        raise ValueError("Spread requires rates in percent")
    return dict(
        value=a.value - b.value,
        units="percentage_points",
        t0=t0.isoformat(),
        alignment="latest available native observations; not common-period averages",
        inputs=[r.model_dump(mode="json") for r in (a, b)],
    )
