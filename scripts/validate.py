"""Run calibration and validation: uv run python scripts/validate.py --help."""
import sys

from credit_risk.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["validate", *sys.argv[1:]]))
