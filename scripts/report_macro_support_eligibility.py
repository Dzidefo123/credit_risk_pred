"""Task9A frozen support audit. No API access, estimation or holdout consumption."""

import argparse
from pathlib import Path

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.macro.information import feature_hash
from credit_risk.track_b.macro_support.study import (
    aggregate,
    freeze_spec,
    immutable_json,
    macro_support,
    read_json,
)
from credit_risk.track_b.macro_support.verification import verify
from credit_risk.track_b.multivintage.study import lf_hash


def run(root, source_dir, reproduce=False):
    private = root / "data/track_b/macro_support"
    if reproduce:
        support = read_json(private / "support.json")
        counts = read_json(private / "counts.json")
        spec = read_json(root / "docs/track_b/macro_support_eligibility_spec.json")
        freeze = read_json(private / "before_counts.json")
        for name, expected in freeze["public_lf_hashes"].items():
            if lf_hash(root / name) != expected:
                raise ValueError("Pre-count design/code changed: " + name)
        _, mapping = macro_support(root)
        reproduced = aggregate(root, support, mapping, spec, directory="reproduction_v1")
        if feature_hash(counts) != feature_hash(reproduced):
            raise ValueError("Empirical aggregate reproduction failed")
        private_hashes = {}
        for path in sorted((private / "v1").glob("*.jsonl")):
            actual = digest(path)
            if digest(private / "reproduction_v1" / path.name) != actual:
                raise ValueError("Private facility eligibility reproduction failed")
            private_hashes[path.name] = actual
        immutable_json(
            private / "reproduction.json",
            dict(
                status="PASSED",
                counts_sha256=feature_hash(counts),
                interval_key_sha256=counts["interval_key_sha256"],
                facility_files_sha256=private_hashes,
                actual_second_full_canonical_pass=True,
                no_predictions_created=True,
            ),
        )
    else:
        verification = verify(root, source_dir)
        immutable_json(private / "precheck.json", verification)
        support, mapping = macro_support(root)
        immutable_json(private / "support.json", support)
        spec = freeze_spec(root, support)
        names = [
            "docs/track_b/macro_support_eligibility_spec.json",
            "docs/track_b/macro_historical_coverage_resolution.json",
            "src/credit_risk/track_b/macro_support/eligibility.py",
            "src/credit_risk/track_b/macro_support/study.py",
        ]
        immutable_json(
            private / "before_counts.json",
            dict(
                frozen_before_mortgage_event_counts=True,
                public_lf_hashes={n: lf_hash(root / n) for n in names},
                specification_sha256_lf=lf_hash(root / names[0]),
                support_sha256=feature_hash(support),
                outcomes_used_to_select_design=False,
            ),
        )
        counts = aggregate(root, support, mapping, spec)
        immutable_json(private / "counts.json", counts)
    verification = verify(root, source_dir)
    immutable_json(private / "postcheck.json", verification)
    print("Task9A frozen evidence and eligibility audit PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reproduce", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.report_only:
        from credit_risk.track_b.macro_support.reporting import make_report

        make_report(project)
    else:
        run(project, args.source_dir, args.reproduce)
