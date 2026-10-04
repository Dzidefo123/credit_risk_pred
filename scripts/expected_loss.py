"""Expected loss: uv run python scripts/expected_loss.py --help."""

import sys

from credit_risk.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["expected-loss", *sys.argv[1:]]))
