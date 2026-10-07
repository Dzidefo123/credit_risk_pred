"""Task12 explicit audit/freeze/train/explore/report/verify phases; no old ledger API."""

import argparse
from pathlib import Path

from credit_risk.track_b.macro_support.study import immutable_json
from credit_risk.track_b.refinancing.audit import PRIVATE, audit, verify
from credit_risk.track_b.refinancing.protocol import freeze

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["audit", "freeze", "train", "explore", "report", "verify"]
    )
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase == "audit":
        audit(root)
    elif args.phase == "freeze":
        freeze(root)
    elif args.phase == "verify":
        if args.source_dir is None:
            parser.error("Final preservation requires --source-dir")
        immutable_json(root / PRIVATE / "preservation_final.json", verify(root, args.source_dir))
        print("Task12 preservation PASSED")
    elif args.phase == "report":
        from render_refinancing_report import render

        from credit_risk.track_b.refinancing.reporting import make

        if not (root / "reports/track_b/refinancing_incentive_payoff_research.json").exists():
            make(root)
        render(root)
    else:
        from credit_risk.track_b.refinancing.study import explore, train

        (train if args.phase == "train" else explore)(root)
