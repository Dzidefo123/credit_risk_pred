"""Run the separate synthetic reject-inference experiment."""
import sys
from credit_risk.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["reject-inference", *sys.argv[1:]]))
