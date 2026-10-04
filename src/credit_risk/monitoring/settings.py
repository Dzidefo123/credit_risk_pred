"""Explicit governance thresholds, not universal statistical significance rules."""

from pydantic import Field, model_validator

from credit_risk.utils.config import ConfigModel


class AlertThreshold(ConfigModel):
    warning: float = Field(gt=0)
    critical: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.warning >= self.critical:
            raise ValueError("Warning must be below critical threshold")
        return self


class MonitoringConfig(ConfigModel):
    bins: int = Field(default=10, ge=2, le=50)
    smoothing: float = Field(default=1e-6, gt=0, lt=0.1)
    minimum_population_rows: int = Field(default=500, ge=1)
    minimum_numeric_rows: int = Field(default=50, ge=1)
    score_base: float = 600.0
    score_base_good_bad_odds: float = Field(default=20.0, gt=0)
    score_points_to_double_odds: float = Field(default=20.0, gt=0)
    score_probability_epsilon: float = Field(default=1e-6, gt=0, lt=0.5)
    psi: AlertThreshold = Field(default_factory=lambda: AlertThreshold(warning=0.1, critical=0.25))
    missingness: AlertThreshold = Field(
        default_factory=lambda: AlertThreshold(warning=0.05, critical=0.1)
    )
    ks: AlertThreshold = Field(default_factory=lambda: AlertThreshold(warning=0.1, critical=0.2))
    out_of_range: AlertThreshold = Field(
        default_factory=lambda: AlertThreshold(warning=0.05, critical=0.1)
    )
    pd_mean: AlertThreshold = Field(
        default_factory=lambda: AlertThreshold(warning=0.01, critical=0.03)
    )
    score_mean: AlertThreshold = Field(
        default_factory=lambda: AlertThreshold(warning=10.0, critical=25.0)
    )

    def measurement_contract(self):
        keys = (
            "bins",
            "smoothing",
            "score_base",
            "score_base_good_bad_odds",
            "score_points_to_double_odds",
            "score_probability_epsilon",
        )
        return {k: self.model_dump()[k] for k in keys}
