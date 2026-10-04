"""Run the installed training CLI: uv run python scripts/train.py --help."""

import sys

from credit_risk.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
