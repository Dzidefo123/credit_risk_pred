"""Configuration failures must be detected before an experiment runs."""

import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from credit_risk.cli import main
from credit_risk.utils.config import (
    DecisionPolicyConfig,
    DevelopmentConfig,
    ModelConfig,
    load_config,
)
from credit_risk.utils.logging import JsonFormatter

ROOT = Path(__file__).resolve().parents[1]


def test_repository_configuration_and_relative_paths():
    directory = ROOT / "configs"
    development = load_config(directory / "development.yaml", DevelopmentConfig)
    load_config(directory / "model.yaml", ModelConfig)
    load_config(directory / "decision_policy.yaml", DecisionPolicyConfig)
    assert development.paths.resolve(directory.parent)["raw"] == ROOT / "data/raw"


@pytest.mark.parametrize("content", ["", "- a", "seed: [", "!!python/object:os.system {}"])
def test_invalid_yaml_is_rejected(tmp_path, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_config(path, DevelopmentConfig)


def test_unknown_parameter_does_not_silently_use_defaults(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("sead: 42")
    with pytest.raises(ValidationError, match="sead"):
        load_config(path, DevelopmentConfig)


@pytest.mark.parametrize(
    "settings",
    [
        {"test_fraction": 0.7, "calibration_fraction": 0.3},
        {"test_fraction": -0.1},
        {"calibration_methods": ["raw", "raw"]},
        {"validation_fraction": 0.7},
        {"clip_lower_quantile": 0.9, "clip_upper_quantile": 0.1},
        {"logistic": {"c": 0.0}},
        {"xgboost": {"subsample": 0.0}},
    ],
)
def test_invalid_experiment_partitions(settings):
    with pytest.raises(ValidationError):
        ModelConfig(**settings)


@pytest.mark.parametrize(
    "settings",
    [
        {"approve_below_pd": 0.2, "decline_at_or_above_pd": 0.1},
        {"risk_grade_upper_bounds": [0.1, 0.05, 1.0]},
        {"risk_grade_upper_bounds": [0.1, 0.5]},
        {"risk_grade_upper_bounds": [float("nan"), 1.0]},
        {"minimum_limit": 2000.0, "maximum_limit": 1000.0},
        {"lgd_assumption": 1.1},
        {"minimum_limit": float("nan")},
        {"income_limit_multiplier": float("inf")},
    ],
)
def test_invalid_policy_settings(settings):
    with pytest.raises(ValidationError):
        DecisionPolicyConfig(**settings)


def test_configuration_cli_success_and_missing_files(tmp_path, capsys):
    assert main(["check-config", "--config-dir", str(ROOT / "configs")]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "valid"
    assert isinstance(result["version"], str)
    assert json.loads(captured.err)["level"] == "INFO"
    assert main(["check-config", "--config-dir", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["level"] == "ERROR"


def test_logging_preserves_structured_details():
    record = logging.LogRecord("credit_risk", logging.INFO, "", 0, "run %s", ("ready",), None)
    record.details = {"seed": 42}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "run ready"
    assert payload["details"] == {"seed": 42}
    assert payload["timestamp"].endswith("+00:00")
