"""Run the development policy comparison from the checkout."""
import sys
from credit_risk.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["compare-policies", *sys.argv[1:]]))
