"""Report the Task8 audit and preservation gate; never runs ingestion or models."""

import argparse
import json
from pathlib import Path

from credit_risk.track_b.multivintage.core import write_json
from credit_risk.track_b.multivintage.reporting import build, figure, verify_preservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = build(root)
    preservation = verify_preservation(root, args.source_dir.resolve())
    evidence["preservation"] = preservation
    write_json(root / "reports/track_b/multi_vintage_data_audit.json", evidence)
    write_json(root / "docs/track_b/multi_vintage_preservation_manifest.json", preservation)
    figure(root, evidence)
    report = root / "reports/track_b/MULTI_VINTAGE_DATA_AUDIT.md"
    with report.open("a", encoding="utf-8") as stream:
        stream.write("\n## Preservation\n\n")
        stream.write(json.dumps(preservation, indent=2) + "\n")
        stream.write("\n## Aggregate diagnostics\n\n")
        stream.write("![Audited subset diagnostics](multi_vintage_diagnostics.png)\n")
        stream.write("\n## Reproduction\n\n")
        stream.write(
            "Ingestion: `python scripts/run_track_b_multivintage.py --source-dir SOURCE`. "
        )
        stream.write(
            "Reporting: `python scripts/report_track_b_multivintage.py --source-dir SOURCE`. "
        )
        stream.write("Partial failed stages require explicit recovery review; no silent rebuild.\n")
    print(evidence["decision"])
    if not evidence["decision"].startswith("MULTI-VINTAGE COHORT READY"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
