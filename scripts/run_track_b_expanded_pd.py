"""Execute the prespecified, one-shot expanded PD study."""

import argparse
from pathlib import Path

from credit_risk.track_b.expanded_pd.study import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.archive)
