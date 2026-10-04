"""Serving paths are explicit and resolved relative to the YAML file."""

from pydantic import Field

from credit_risk.utils.config import ConfigModel


class ServingConfig(ConfigModel):
    source_csv: str
    run_dir: str
    validation_dir: str
    policy_config: str
    policy_name: str = Field(default="baseline", min_length=1)
