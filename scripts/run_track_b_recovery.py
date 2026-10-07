"""Task8C exact authorized quarantine and fail-closed recovery; no model or macro work."""

import argparse
from pathlib import Path

from credit_risk.track_b.recovery.study import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    results = run(Path(__file__).resolve().parents[1], args.source_dir)
    if len(results) != 7 or any(r["status"] != "READY_WITH_LIMITATIONS" for r in results):
        raise SystemExit(2)
