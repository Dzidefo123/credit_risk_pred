"""Portfolio analytics: uv run python scripts/portfolio.py --help."""

import sys

from credit_risk.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["analyze-portfolio", *sys.argv[1:]]))
