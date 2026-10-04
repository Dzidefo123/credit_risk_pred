"""Typed configuration contracts; business engines are implemented in later phases."""

from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class DataPaths(ConfigModel):
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    artifacts: str = "artifacts"

    def resolve(self, project_root: Path) -> dict[str, Path]:
        """Resolve explicit paths without creating directories or reading datasets."""
        return {name: (project_root / value).resolve() for name, value in self.model_dump().items()}


class DevelopmentConfig(ConfigModel):
    environment: Literal["development", "test", "production"] = "development"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    paths: DataPaths = Field(default_factory=DataPaths)


class LogisticConfig(ConfigModel):
    c: float = Field(default=1.0, gt=0)
    max_iter: int = Field(default=3000, ge=1)
    tolerance: float = Field(default=1e-5, gt=0)


class XGBoostConfig(ConfigModel):
    n_estimators: int = Field(default=250, ge=1)
    max_depth: int = Field(default=3, ge=1, le=16)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    reg_lambda: float = Field(default=5.0, ge=0)
    n_jobs: int = Field(default=2, ge=1)


class ModelConfig(ConfigModel):
    baseline: Literal["logistic_regression"] = "logistic_regression"
    challenger: Literal["xgboost"] = "xgboost"
    validation_fraction: float = Field(default=0.15, gt=0, lt=1)
    classification_threshold: float = Field(default=0.1, gt=0, lt=1)
    clip_lower_quantile: float = Field(default=0.001, ge=0, lt=1)
    clip_upper_quantile: float = Field(default=0.999, gt=0, le=1)
    logistic: LogisticConfig = Field(default_factory=LogisticConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    calibration_fraction: float = Field(default=0.2, gt=0, lt=1)
    calibration_methods: list[Literal["raw", "sigmoid", "isotonic"]] = Field(
        default_factory=lambda: ["raw", "sigmoid", "isotonic"], min_length=1
    )

    @model_validator(mode="after")
    def check_partitions(self) -> "ModelConfig":
        if self.test_fraction + self.calibration_fraction + self.validation_fraction >= 1:
            raise ValueError(
                "Training partition must remain after validation, test and calibration partitions"
            )
        if self.clip_lower_quantile >= self.clip_upper_quantile:
            raise ValueError("Lower clipping quantile must be below upper quantile")
        if len(set(self.calibration_methods)) != len(self.calibration_methods):
            raise ValueError("Calibration methods must be unique")
        return self


class DecisionPolicyConfig(ConfigModel):
    approve_below_pd: float = Field(default=0.03, ge=0, le=1)
    decline_at_or_above_pd: float = Field(default=0.1, ge=0, le=1)
    risk_grade_upper_bounds: list[float] = Field(
        default_factory=lambda: [0.01, 0.03, 0.06, 0.1, 1.0], min_length=1
    )
    lgd_assumption: float = Field(default=0.45, ge=0, le=1)
    minimum_limit: float = Field(default=500.0, ge=0)
    maximum_limit: float = Field(default=10000.0, ge=0)
    income_limit_multiplier: float = Field(default=2.0, gt=0)

    @model_validator(mode="after")
    def check_policy(self) -> "DecisionPolicyConfig":
        if self.approve_below_pd >= self.decline_at_or_above_pd:
            raise ValueError("Approval threshold must be below decline threshold")
        bounds = self.risk_grade_upper_bounds
        if any(not 0 < x <= 1 for x in bounds) or bounds[-1] != 1:
            raise ValueError("Risk grade bounds must be in (0, 1] and end at 1")
        if any(a >= b for a, b in zip(bounds, bounds[1:], strict=False)):
            raise ValueError("Risk grade bounds must be strictly increasing")
        if self.minimum_limit > self.maximum_limit:
            raise ValueError("Minimum limit must not exceed maximum limit")
        return self


T = TypeVar("T", bound=ConfigModel)


def load_config(path: Path | str, schema: type[T]) -> T:
    """Use safe YAML parsing, then reject unknown keys and invalid parameter values."""
    with Path(path).open(encoding="utf-8") as stream:
        try:
            payload = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return schema.model_validate(payload)
