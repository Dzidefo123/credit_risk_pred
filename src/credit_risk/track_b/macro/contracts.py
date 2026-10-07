"""Future acquisition/scenario contracts; deliberately no download or hazard code."""

import hashlib
import math
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Coverage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: date
    end: date

    @model_validator(mode="after")
    def ordered(self):
        if self.start > self.end:
            raise ValueError("Reversed coverage")
        return self


class AcquisitionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    series: list[str] = Field(min_length=1)
    geography: Literal["US"]
    acquisition_time: datetime
    source_url: str = Field(pattern=r"^https://")
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    vintage_coverage: Coverage
    reference_coverage: Coverage
    release_coverage: Coverage
    units: str = Field(min_length=1)
    seasonal_adjustment: Literal["SA", "NSA"]
    license: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    vintage_status: Literal["VINTAGE-AWARE AVAILABLE", "RELEASE-DATE RECONSTRUCTABLE"]
    date_evidence: str = Field(min_length=1)
    raw_path: str = Field(min_length=1)
    complete_revision_history: bool

    @model_validator(mode="after")
    def archived(self):
        if self.acquisition_time.tzinfo is None:
            raise ValueError("Acquisition timezone missing")
        if self.vintage_coverage.end > self.acquisition_time.date():
            raise ValueError("Archive coverage extends beyond acquisition")
        path = PurePosixPath(self.raw_path)
        if (
            "\\" in self.raw_path
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[:4] != ("data", "track_b", "macro", "raw")
            or len(path.parts) < 5
        ):
            raise ValueError("Raw path must be relative within raw zone")
        return self


def verify_raw(path, expected):
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("Immutable raw response hash mismatch")
    return actual


def write_raw(path, payload):
    """Exclusive create; never overwrite even if existing bytes match."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return verify_raw(path, hashlib.sha256(payload).hexdigest())


class ScenarioPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    series_id: str = Field(min_length=1)
    geography: Literal["US"]
    period_end: date
    horizon: int = Field(ge=1)
    value: float = Field(allow_inf_nan=False)
    units: str = Field(min_length=1)
    frequency: Literal["M"]  # native quarterly/weekly conversion must be declared upstream
    source_kind: Literal["forecast", "hypothetical", "observed"]
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_reference_period: date | None = None
    observed_release_date: date | None = None


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_name: str = Field(min_length=1)
    kind: Literal["forecast", "hypothetical", "historical_replay"]
    as_of: date
    forecast_origin: date
    publication_date: date
    provider: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    consistency_review: str = Field(min_length=1)
    conversion_rule: str = Field(min_length=1)
    horizon: int = Field(ge=1)
    paths: list[ScenarioPoint] = Field(min_length=1)
    scenario_probability: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    probability_evidence: str | None = None

    @model_validator(mode="after")
    def valid(self):
        import calendar

        if not self.forecast_origin <= self.publication_date <= self.as_of:
            raise ValueError("Scenario forecast origin/publication must be known as of issuance")
        if self.scenario_probability is not None and not self.probability_evidence:
            raise ValueError("Probability needs defensible source/method")
        expected_kind = {
            "forecast": "forecast",
            "hypothetical": "hypothetical",
            "historical_replay": "observed",
        }[self.kind]
        for sid in {r.series_id for r in self.paths}:
            rows = [r for r in self.paths if r.series_id == sid]
            if [r.horizon for r in rows] != list(range(1, self.horizon + 1)):
                raise ValueError("Paths must be ordered, unique and complete")
            if len({r.units for r in rows}) != 1:
                raise ValueError("Path units inconsistent")
            for r in rows:
                index = self.as_of.year * 12 + self.as_of.month - 1 + r.horizon
                year, month = divmod(index, 12)
                expected = date(year, month + 1, calendar.monthrange(year, month + 1)[1])
                if r.period_end != expected or r.source_kind != expected_kind:
                    raise ValueError(
                        "Future observation masquerading as forecast or invalid horizon"
                    )
        if self.kind == "historical_replay":
            if self.scenario_probability is not None:
                raise ValueError("Historical replay is descriptive; probability weights prohibited")
            for r in self.paths:
                if r.original_reference_period is None or r.observed_release_date is None:
                    raise ValueError("Replay needs original reference/release dates")
                if not r.original_reference_period <= r.observed_release_date <= self.as_of:
                    raise ValueError("Historical replay source path was unavailable at issuance")
        elif any(
            r.original_reference_period is not None or r.observed_release_date is not None
            for r in self.paths
        ):
            raise ValueError("Observed source dates cannot masquerade as forecast evidence")
        return self


def validate_scenario_set(scenarios):
    if not scenarios:
        raise ValueError("Empty scenario set")
    if len({s.scenario_name for s in scenarios}) != len(scenarios):
        raise ValueError("Scenario names must be unique")
    if (
        len(
            {(s.as_of, s.horizon, tuple(sorted({r.series_id for r in s.paths}))) for s in scenarios}
        )
        > 1
    ):
        raise ValueError("Scenario set requires common origin, horizon and variables")
    weights = [s.scenario_probability for s in scenarios]
    if any(w is not None for w in weights):
        if any(w is None for w in weights) or not math.isclose(sum(weights), 1, abs_tol=1e-9):
            raise ValueError("Weights must be complete and sum to one")
    return True
