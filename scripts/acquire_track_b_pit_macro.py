"""Acquire official Task9 vintage inputs under the frozen registry; no model access."""

import argparse
from pathlib import Path

from credit_risk.track_b.pit_macro.acquisition import acquire

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="task9_api_v2")
    args = parser.parse_args()
    summary, _ = acquire(Path(__file__).resolve().parents[1], run=args.run)
    print("Dated versions:", sum(s["normalized_versions"] for s in summary["series"]))
