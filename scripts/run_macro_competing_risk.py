"""Task10 explicit phases; never automatically fit or consume a prior ledger."""

import argparse
from pathlib import Path

from credit_risk.track_b.macro_hazard.protocol import freeze
from credit_risk.track_b.macro_hazard.study import evaluate, train
from credit_risk.track_b.macro_hazard.verification import preservation, replay
from credit_risk.track_b.macro_support.study import immutable_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "evaluate", "replay", "verify"])
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    private = root / "data/track_b/models/macro_hazard_v1"
    checked = preservation(root, args.source_dir)
    freeze(root)
    if args.phase == "train":
        immutable_json(private / "preservation_before_fit.json", checked)
        train(root)
    elif args.phase == "evaluate":
        evaluate(root)
    elif args.phase == "replay":
        replay(root)
    else:
        immutable_json(private / "preservation_after.json", checked)
        print("Task10 preservation PASSED")
