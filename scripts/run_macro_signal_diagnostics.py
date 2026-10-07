"""Explicit Task11 phases; no Task10 ledger transitions or replacement models."""

import argparse
from pathlib import Path

from credit_risk.track_b.macro_diagnostics.protocol import PRIVATE, freeze
from credit_risk.track_b.macro_diagnostics.refits import run as refits
from credit_risk.track_b.macro_diagnostics.study import run as diagnose
from credit_risk.track_b.macro_diagnostics.verification import verify
from credit_risk.track_b.macro_support.study import immutable_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["refits", "diagnose", "verify", "report"])
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    freeze(root)
    if args.phase == "refits":
        refits(root)
    elif args.phase == "diagnose":
        diagnose(root)
    elif args.phase == "report":
        from credit_risk.track_b.macro_diagnostics.reporting import make

        make(root)
    else:
        if args.source_dir is None:
            parser.error("Final verification requires --source-dir")
        checked = verify(root, args.source_dir)
        immutable_json(root / PRIVATE / "preservation_final.json", checked)
        print("Task11 preservation PASSED")
