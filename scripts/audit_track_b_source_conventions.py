"""Task8A origination-only census; no new sample or performance access."""

import argparse
from pathlib import Path

from credit_risk.track_b.reconciliation.evidence import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.source_dir)
