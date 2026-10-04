"""Assumptions for a separate synthetic selection-bias experiment."""

from pydantic import Field, model_validator

from credit_risk.utils.config import ConfigModel


class RejectInferenceConfig(ConfigModel):
    samples: int = Field(default=12000, ge=1000, le=1000000)
    seeds: list[int] = Field(default_factory=lambda: [91, 92, 93], min_length=1)
    holdout_fraction: float = Field(default=0.3, gt=0, lt=0.5)
    propensity_folds: int = Field(default=5, ge=2, le=10)
    selection_intercept: float = 0.4
    selection_risk_coefficient: float = Field(default=-1.2, lt=0)
    selection_debt_coefficient: float = -0.6
    selection_income_coefficient: float = 0.5
    true_propensity_floor: float = Field(default=0.05, gt=0, lt=0.5)
    outcome_intercept: float = -2.5
    nonlinear_risk_effect: float = Field(default=0.7, ge=0)
    hidden_outcome_effect: float = Field(default=0.9, ge=0)
    hidden_selection_effect: float = Field(default=-1.5, lt=0)
    deterministic_cutoff: float = Field(default=0.5, gt=0, lt=1)
    estimated_propensity_floor: float = Field(default=0.05, gt=0, lt=1)
    maximum_weight: float = Field(default=20.0, ge=1)
    logistic_c: float = Field(default=100.0, gt=0)
    reject_odds_multipliers: list[float] = Field(
        default_factory=lambda: [0.5, 1.0, 2.0], min_length=1
    )

    @model_validator(mode="after")
    def check_lists(self):
        if len(set(self.seeds)) != len(self.seeds) or any(not 0 <= x < 2**32 for x in self.seeds):
            raise ValueError("Seeds must be unique unsigned 32-bit integers")
        m = self.reject_odds_multipliers
        if any(x <= 0 for x in m) or len(set(m)) != len(m):
            raise ValueError("Sensitivity odds multipliers must be positive and unique")
        return self
