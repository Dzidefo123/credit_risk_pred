"""Auditable archive bounds without fabricating exact agency publication dates."""

import calendar
import math
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from credit_risk.track_b.macro.information import feature_hash

VERSION = "pit-macro-v1.0.0"


def period_end(label, frequency):
    d = date.fromisoformat(label)
    if frequency == "M":
        if d.day != 1:
            raise ValueError("Monthly provider label must be period start")
        return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])
    if frequency == "Q":
        if d.day != 1 or d.month not in {1, 4, 7, 10}:
            raise ValueError("Quarter provider label must be quarter start")
        month = d.month + 2
        return date(d.year, month, calendar.monthrange(d.year, month)[1])
    if frequency not in {"D", "W"}:
        raise ValueError("Unsupported frequency")
    return d


def previous_month_end(month):
    """Risk month m cannot see macro released during m."""
    d = date.fromisoformat(month + "-01")
    index = d.year * 12 + d.month - 2
    year, zero_month = divmod(index, 12)
    return date(year, zero_month + 1, calendar.monthrange(year, zero_month + 1)[1])


def lag_period(d, months):
    index = d.year * 12 + d.month - 1 - months
    year, zero_month = divmod(index, 12)
    return date(year, zero_month + 1, calendar.monthrange(year, zero_month + 1)[1])


class VintageValue(BaseModel):
    """Nullable exact dates stay distinct from certified conservative upper bounds.

    ALFRED real-time validity is an archive information set, not proof of an
    exact agency date. Initial release requires separate explicit evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    series_id: str = Field(min_length=1)
    reference_period: date
    value: float = Field(allow_inf_nan=False)
    archive_start: date
    archive_end: date
    publication_upper_bound: date
    revision_upper_bound: date
    release_date: date | None = None
    revision_date: date | None = None
    certified_initial: bool = False
    initial_evidence: str | None = None
    provenance: Literal["VINTAGE_AWARE_AVAILABLE", "RELEASE_DATE_RECONSTRUCTABLE"]
    date_evidence: Literal["exact", "conservative_archive_upper_bound"]
    source: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_at: datetime
    frequency: Literal["D", "W", "M", "Q"]
    units: str = Field(min_length=1)
    seasonal_adjustment: Literal["SA", "NSA"]
    geography: Literal["US"] = "US"
    measurement_regime: str = Field(min_length=1)
    representation: Literal["vintage", "current_revised"] = "vintage"

    @model_validator(mode="after")
    def dates(self):
        if self.archive_end < self.archive_start:
            raise ValueError("Reversed real-time validity")
        if self.publication_upper_bound < self.reference_period:
            raise ValueError("Uncompleted reference period")
        if not self.publication_upper_bound <= self.revision_upper_bound <= self.archive_start:
            raise ValueError("Knowledge-time bounds inconsistent")
        if (
            self.release_date
            and not self.reference_period <= self.release_date <= self.publication_upper_bound
        ):
            raise ValueError("Exact release inconsistent with bound")
        if (
            self.revision_date
            and not (self.release_date or self.reference_period)
            <= self.revision_date
            <= self.revision_upper_bound
        ):
            raise ValueError("Exact revision inconsistent with bound")
        if self.certified_initial and (
            not self.initial_evidence
            or not self.release_date
            or self.release_date != self.revision_date
        ):
            raise ValueError("Initial release requires exact independent evidence")
        if self.date_evidence == "exact" and (not self.release_date or not self.revision_date):
            raise ValueError("Unverified dates cannot be labeled exact")
        if self.retrieved_at.tzinfo is None or self.archive_start > self.retrieved_at.date():
            raise ValueError("Invalid retrieval chronology")
        d = self.reference_period
        if self.frequency in {"M", "Q"} and d.day != calendar.monthrange(d.year, d.month)[1]:
            raise ValueError("Completed reference period-end required")
        if self.frequency == "Q" and d.month not in {3, 6, 9, 12}:
            raise ValueError("Quarter endpoint required")
        return self


def validate_versions(rows):
    """Do not let duplicates, overlapping revisions or current history slip through."""
    groups = {}
    for row in rows:
        if row.representation != "vintage":
            raise ValueError("Current-revised observation prohibited")
        groups.setdefault((row.series_id, row.reference_period), []).append(row)
    for versions in groups.values():
        ordered = sorted(versions, key=lambda r: r.archive_start)
        for a, b in zip(ordered, ordered[1:], strict=False):
            if a.archive_end >= b.archive_start:
                raise ValueError("Duplicate or overlapping real-time versions")
        if len({(r.frequency, r.source, r.seasonal_adjustment) for r in ordered}) > 1:
            raise ValueError("Source metadata mismatch")


def select(rows, sid, t0, *, period=None, rule="latest_known", cap=None):
    if type(t0) is not date:
        raise ValueError("End-of-day dates only; intraday datetime unsupported")
    if rule not in {"latest_known", "initial_release"}:
        raise ValueError("Unsupported knowledge rule")
    selected = [r for r in rows if r.series_id == sid]
    validate_versions(selected)
    eligible = [
        r
        for r in selected
        if r.reference_period <= t0
        and r.publication_upper_bound <= t0
        and r.revision_upper_bound <= t0
        and r.archive_start <= t0 <= r.archive_end
        and (period is None or r.reference_period == period)
        and (rule != "initial_release" or r.certified_initial)
    ]
    # Initial release sensitivity intentionally retains the certified original
    # after subsequent revisions; its archive_end is not a knowledge expiry.
    if rule == "initial_release":
        eligible = [
            r
            for r in selected
            if r.certified_initial
            and r.reference_period <= t0
            and r.publication_upper_bound <= t0
            and r.revision_upper_bound <= t0
            and r.archive_start <= t0
            and (period is None or r.reference_period == period)
        ]
    if cap is not None:
        if cap < 0:
            raise ValueError("Negative freshness cap")
        eligible = [r for r in eligible if (t0 - r.reference_period).days <= cap]
    if not eligible:
        return None
    return max(eligible, key=lambda r: (r.reference_period, r.archive_start))


def engineer(rows, feature, t0, caps, rule="latest_known"):
    sid = feature["series_id"]
    current = select(rows, sid, t0, cap=caps[sid], rule=rule)
    unbounded = select(rows, sid, t0, rule=rule)
    if current is None:
        return dict(value=None, status="STALE" if unbounded else "UNAVAILABLE", inputs=[])
    inputs = [current]
    kind = feature["transformation"]
    if kind == "level":
        value = current.value
    elif kind == "spread":
        other_sid = feature["secondary_series"]
        other = select(rows, other_sid, t0, cap=caps[other_sid], rule=rule)
        if other is None:
            raw = select(rows, other_sid, t0, rule=rule)
            return dict(value=None, status="STALE" if raw else "MISSING_OPERAND", inputs=[])
        if current.units != "Percent" or other.units != "Percent":
            raise ValueError("Spread requires percent rate operands")
        inputs.append(other)
        value = current.value - other.value
    elif kind in {"difference", "growth_pct"}:
        lag = feature["lag_months"]
        if lag not in {3, 12} or current.frequency not in {"M", "Q"}:
            raise ValueError("Unregistered transformation alignment")
        prior = select(rows, sid, t0, period=lag_period(current.reference_period, lag), rule=rule)
        if prior is None:
            return dict(value=None, status="MISSING_OPERAND", inputs=[])
        if (
            current.units,
            current.frequency,
            current.seasonal_adjustment,
            current.measurement_regime,
        ) != (prior.units, prior.frequency, prior.seasonal_adjustment, prior.measurement_regime):
            return dict(value=None, status="METADATA_REGIME_MISMATCH", inputs=[])
        inputs.append(prior)
        if kind == "growth_pct" and prior.value <= 0:
            raise ValueError("Nonpositive growth denominator")
        value = (
            current.value - prior.value
            if kind == "difference"
            else 100 * (current.value / prior.value - 1)
        )
    else:
        raise ValueError("Unregistered feature transformation")
    if not math.isfinite(value):
        raise ValueError("Nonfinite engineered feature")
    result = dict(
        value=value,
        status="AVAILABLE",
        t0=t0.isoformat(),
        rule=rule,
        feature_version=VERSION,
        units=feature["units"],
        inputs=[r.model_dump(mode="json") for r in inputs],
    )
    result["lineage_sha256"] = feature_hash(result)
    return result


def month_table(rows, months, features, caps):
    if len(months) != len(set(months)):
        raise ValueError("Duplicate assessment month")
    validate_versions(rows)
    return [
        dict(
            reporting_month=month,
            t0=previous_month_end(month).isoformat(),
            pandemic_calendar_2020=month[:4] == "2020",
            pandemic_calendar_2021=month[:4] == "2021",
            features={
                f["name"]: engineer(rows, f, previous_month_end(month), caps) for f in features
            },
        )
        for month in sorted(months)
    ]


def left_join(records, table):
    """Iterator preserves input order/count and every facility, including missing months."""
    mapping = {row["reporting_month"]: row for row in table}
    if len(mapping) != len(table):
        raise ValueError("Duplicate macro-month key")
    for record in records:
        if "macro" in record:
            raise ValueError("Existing macro field cannot be overwritten")
        yield dict(record, macro=mapping.get(record["reporting_month"]))
