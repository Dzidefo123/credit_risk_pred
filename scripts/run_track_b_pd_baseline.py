"""Run Task 3 from the immutable local Task 2 panel; no downloads."""

import argparse
from pathlib import Path

from credit_risk.track_b.pd.research import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    result = run(Path(__file__).resolve().parents[1], args.archive)
    print(result["decision"])
    print(result["temporal_split"])
