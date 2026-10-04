"""Explicit portfolio metric definitions and reporting support thresholds."""

from pydantic import Field, model_validator

from credit_risk.utils.config import ConfigModel


class PortfolioAnalyticsConfig(ConfigModel):
    max_months_on_book: int = Field(default=35, ge=0, le=120)
    bad_dpd_threshold: int = Field(default=90, ge=1)
    delinquency_dpd_threshold: int = Field(default=30, ge=1)
    minimum_cohort_size: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def check_thresholds(self):
        if self.delinquency_dpd_threshold > self.bad_dpd_threshold:
            raise ValueError("Delinquency threshold must not exceed bad threshold")
        return self
