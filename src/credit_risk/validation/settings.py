"""Prespecified calibration and validation choices, independent of base training."""

from typing import Literal

from pydantic import Field

from credit_risk.utils.config import ConfigModel


class ValidationConfig(ConfigModel):
    selection_metric: Literal["log_loss", "brier"] = "log_loss"
    probability_epsilon: float = Field(default=1e-6, gt=0, lt=0.5)
    calibration_bins: int = Field(default=10, ge=2, le=50)
    minimum_calibration_events: int = Field(default=20, ge=1)
    minimum_segment_rows: int = Field(default=100, ge=1)
    minimum_segment_events: int = Field(default=20, ge=1)
    bootstrap_samples: int = Field(default=200, ge=20, le=2000)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    bootstrap_seed: int = Field(default=43, ge=0, le=2**32 - 1)
