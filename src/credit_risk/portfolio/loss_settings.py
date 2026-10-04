"""Explicit educational PD horizon, LGD, conversion factors and scenario assumptions."""

from pydantic import Field, model_validator

from credit_risk.utils.config import ConfigModel


class LossScenario(ConfigModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    pd_odds_multiplier: float = Field(default=1.0, gt=0, le=100)
    lgd: float = Field(default=0.45, ge=0, le=1)
    credit_conversion_factor: float = Field(default=0.5, ge=0, le=1)


class ExpectedLossConfig(ConfigModel):
    horizon_months: int = Field(default=12, ge=1, le=60)
    minimum_state_pairs: int = Field(default=20, ge=1)
    require_complete_snapshot: bool = True
    top_n_concentration: int = Field(default=10, ge=1)
    scenarios: list[LossScenario] = Field(
        default_factory=lambda: [
            LossScenario(name="base"),
            LossScenario(
                name="adverse", pd_odds_multiplier=1.5, lgd=0.6, credit_conversion_factor=0.75
            ),
            LossScenario(
                name="severe", pd_odds_multiplier=2.5, lgd=0.75, credit_conversion_factor=1.0
            ),
        ],
        min_length=1,
    )

    @model_validator(mode="after")
    def check_scenarios(self):
        names = [scenario.name for scenario in self.scenarios]
        if len(names) != len(set(names)) or names[0] != "base":
            raise ValueError("Unique scenarios must start with base")
        if self.scenarios[0].pd_odds_multiplier != 1:
            raise ValueError("Base scenario must preserve model PD")
        return self
