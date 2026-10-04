"""Release gates prevent shipping private data and inherited serving artifacts."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "check_release", Path(__file__).resolve().parents[1] / "scripts/check_release.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.mark.parametrize(
    "member",
    [
        "credit_risk/model.joblib",
        "credit_risk/source.csv",
        "credit_risk/.env",
        "credit_risk/__pycache__/cache.pyc",
        "../escaped.py",
        "/absolute.py",
        "app.py",
    ],
)
def test_wheel_gate_rejects_private_generated_or_unexpected_files(member):
    names = ["credit_risk/api/main.py", "credit_risk/tracking/mlflow.py", member]
    with pytest.raises(ValueError):
        release.validate_members(names, wheel=True)


def test_source_gate_excludes_inherited_runtime_and_allows_checkout_api():
    with pytest.raises(ValueError, match="V1"):
        release.validate_members(["lab-0.12.0/main.py"])
    release.validate_members(["lab-0.12.0/api/main.py", "lab-0.12.0/src/credit_risk/api/main.py"])
