"""Build Task9 aggregate evidence without fitting or changing mortgage panels."""

import argparse
from pathlib import Path

from credit_risk.track_b.pit_macro.reporting import build

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="task9_v1")
    args = parser.parse_args()
    print(build(Path(__file__).resolve().parents[1], args.run)["decision"])
