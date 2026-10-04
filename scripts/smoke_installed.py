"""Run with python -I from an isolated core-wheel environment, outside the checkout."""

import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pydantic import ValidationError

from credit_risk import __version__
from credit_risk.api.main import create_app
from credit_risk.api.schemas import ApplicantRequest
from credit_risk.api.service import ScoringService
from credit_risk.decisioning.settings import CreditStrategy
from credit_risk.tracking.mlflow import prepare_export


def main():
    root = Path(__file__).resolve().parents[1]
    expected = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    assert __version__ == expected
    origin = Path(importlib.util.find_spec("credit_risk").origin).resolve()
    assert not origin.is_relative_to(root / "src"), "Smoke test imported the editable checkout"
    assert importlib.util.find_spec("mlflow") is None and "mlflow" not in sys.modules
    model = SimpleNamespace(predict_proba=lambda frame: np.array([[0.99, 0.01]]))
    service = ScoringService(model, CreditStrategy(), "fixture", "sigmoid", "a" * 64, "test label")
    request = ApplicantRequest(
        application_id="installed-wheel",
        features={
            "age": 43.0,
            "MonthlyIncome": 5000.0,
            "DebtRatio": 0.2,
            "RevolvingUtilizationOfUnsecuredLines": 0.2,
        },
    )
    assert service.score(request).pd == 0.01
    assert service.decision(request).decision == "APPROVE"
    try:
        ApplicantRequest(application_id="invalid", features={"age": 43.5})
    except ValidationError:
        pass
    else:
        raise AssertionError("Fractional age accepted")
    assert create_app().state.service is None
    assert callable(prepare_export)
    print(
        json.dumps({"status": "passed", "package_version": __version__, "mlflow_installed": False})
    )


if __name__ == "__main__":
    main()
