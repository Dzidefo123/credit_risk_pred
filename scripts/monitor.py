"""Compare an applicant population with the frozen monitoring reference."""
import sys
from credit_risk.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["monitor",*sys.argv[1:]]))
