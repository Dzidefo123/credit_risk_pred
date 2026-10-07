"""Authorized Task8 mortgage-only study; no network client or model fitting."""

import argparse
from pathlib import Path

from credit_risk.track_b.multivintage.study import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    results = run(args.root, args.source_dir)
    if len(results) != 7 or any(
        r["status"] not in {"READY", "READY_WITH_LIMITATIONS"} for r in results
    ):
        raise SystemExit(2)
