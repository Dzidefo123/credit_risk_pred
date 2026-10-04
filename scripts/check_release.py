"""Reject data, V1 code and generated environments in wheel/source releases."""

import argparse
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

FORBIDDEN_PARTS = {"artifacts", ".venv", "venv", "__pycache__", ".git", "mlruns", "mlartifacts"}
FORBIDDEN_SUFFIXES = {".csv", ".pkl", ".joblib", ".pyc", ".ipynb", ".h5", ".keras", ".db"}


def validate_members(names, wheel=False):
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive path: {name}")
        if FORBIDDEN_PARTS.intersection(path.parts) or path.suffix in FORBIDDEN_SUFFIXES:
            raise ValueError(f"Generated/private artifact in release: {name}")
        if path.name.startswith(".env") or "templates" in path.parts:
            raise ValueError(f"Secret or V1 template in release: {name}")
        if not wheel and len(path.parts) == 2 and path.name in ("app.py", "main.py"):
            raise ValueError(f"V1 serving code in source release: {name}")
        if wheel and path.parts[0] != "credit_risk" and not path.parts[0].endswith(".dist-info"):
            raise ValueError(f"Unexpected wheel content: {name}")
    if wheel:
        required = {"credit_risk/api/main.py", "credit_risk/tracking/mlflow.py"}
        if not required.issubset(names):
            raise ValueError("Missing packaged API/tracking modules")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    with zipfile.ZipFile(args.dist_dir / f"credit_risk_lab-{version}-py3-none-any.whl") as wheel:
        validate_members(wheel.namelist(), wheel=True)
    with tarfile.open(args.dist_dir / f"credit_risk_lab-{version}.tar.gz") as source:
        validate_members(source.getnames())
    print(f"PASS: {version} wheel and source archive contain only intended release files")


if __name__ == "__main__":
    main()
