"""Strict JSON contracts; outcomes, serialized models and unknown fields are forbidden."""

from typing import Literal

from pydantic import Field

from credit_risk.utils.config import ConfigModel


class ApplicantFeatures(ConfigModel):
    RevolvingUtilizationOfUnsecuredLines: float | None = Field(default=None, ge=0)
    age: float | None = Field(default=None, ge=0, multiple_of=1)
    NumberOfTime30_59DaysPastDueNotWorse: int | None = Field(default=None, ge=0, le=2**53 - 1)
    DebtRatio: float | None = Field(default=None, ge=0)
    MonthlyIncome: float | None = Field(default=None, ge=0)
    NumberOfOpenCreditLinesAndLoans: int | None = Field(default=None, ge=0, le=2**53 - 1)
    NumberOfTimes90DaysLate: int | None = Field(default=None, ge=0, le=2**53 - 1)
    NumberRealEstateLoansOrLines: int | None = Field(default=None, ge=0, le=2**53 - 1)
    NumberOfTime60_89DaysPastDueNotWorse: int | None = Field(default=None, ge=0, le=2**53 - 1)
    NumberOfDependents: int | None = Field(default=None, ge=0, le=2**53 - 1)


class ApplicantRequest(ConfigModel):
    application_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    features: ApplicantFeatures


class ScoreResponse(ConfigModel):
    application_id: str
    pd: float = Field(ge=0, le=1)
    risk_grade: str
    model_version: str
    model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    package_version: str
    calibration_method: str
    target_semantics: str
    missing_features: list[str]
    suspicious_inputs: dict[str, int]
    research_only: Literal[True] = True


class DecisionResponse(ScoreResponse):
    decision: Literal["APPROVE", "MANUAL_REVIEW", "DECLINE"]
    recommended_limit: float = Field(ge=0)
    reason_codes: list[str]
    policy_name: str
    policy_sha256: str
    assumed_ead: float = Field(ge=0)
    expected_loss_proxy: float = Field(ge=0)
    limit_units: Literal["unverified source income units"] = "unverified source income units"


class HealthResponse(ConfigModel):
    status: Literal["ready", "unavailable"]
    package_version: str
    research_only: Literal[True] = True
    model_version: str | None = None
    reason: str | None = None
