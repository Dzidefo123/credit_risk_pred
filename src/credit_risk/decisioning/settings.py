"""Configurable appetite and conservative proxy limit assumptions."""

from pydantic import Field, model_validator

from credit_risk.utils.config import ConfigModel, DecisionPolicyConfig


class CreditStrategy(DecisionPolicyConfig):
    name: str = Field(default="baseline", pattern=r"^[a-z][a-z0-9_]*$")
    grade_limit_factors: list[float] = Field(default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2])
    limit_increment: float = Field(default=100.0, gt=0)
    maximum_auto_debt_ratio: float = Field(default=1.0, ge=0)
    maximum_auto_utilization: float = Field(default=1.0, ge=0)
    utilization_penalty: float = Field(default=1.0, ge=0)
    assumed_drawdown: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def check_factors(self):
        f = self.grade_limit_factors
        if len(f) != len(self.risk_grade_upper_bounds) or any(not 0 <= x <= 1 for x in f):
            raise ValueError("Supply one factor in [0,1] per risk grade")
        if any(a < b for a, b in zip(f, f[1:], strict=False)):
            raise ValueError("Grade limit factors must be nonincreasing")
        return self


class PolicyComparisonConfig(ConfigModel):
    policies: list[CreditStrategy] = Field(default_factory=lambda: [CreditStrategy()], min_length=1)

    @model_validator(mode="after")
    def unique_names(self):
        if len({p.name for p in self.policies}) != len(self.policies):
            raise ValueError("Policy names must be unique")
        return self
