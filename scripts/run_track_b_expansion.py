"""Task 4 explicit planning/empirical phases; no model fitting."""

import argparse
from pathlib import Path

from credit_risk.track_b.expansion.empirical import run
from credit_risk.track_b.expansion.planning import plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["A", "B"], required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase == "A":
        print(plan(root)["chosen_n"])
    elif args.archive:
        run(root, args.archive)
    else:
        parser.error("Phase B requires --archive")
