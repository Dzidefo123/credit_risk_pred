"""Local SQLite tracking settings; no implicit external server or registry."""

from pydantic import Field

from credit_risk.utils.config import ConfigModel


class TrackingConfig(ConfigModel):
    database_path: str = "../artifacts/mlflow/tracking.db"
    artifact_root: str = "../artifacts/mlflow/files"
    experiment_name: str = Field(default="credit-risk-lab", min_length=1, max_length=100)
    include_models: bool = True
