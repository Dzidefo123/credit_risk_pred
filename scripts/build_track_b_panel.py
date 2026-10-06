"""Build a private Track B panel from an already-authorized local sample ZIP only."""

import argparse
from pathlib import Path

from credit_risk.track_b.data.annual_report import enrich
from credit_risk.track_b.data.annual_run import run_annual
from credit_risk.track_b.data.preflight import record_preflight
from credit_risk.track_b.data.workflow import IngestionRejected, blocked_audit, build, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--loans", type=int, default=1000)
    parser.add_argument("--publish-aggregates", action="store_true")
    parser.add_argument(
        "--preflight", action="store_true", help="Hash and inspect directories only"
    )
    parser.add_argument("--attest-official-download", action="store_true")
    parser.add_argument("--annual", action="store_true", help="Approved annual amendment workflow")
    parser.add_argument("--annual-report-only", action="store_true")
    args = parser.parse_args()
    if args.annual_report_only:
        import json

        path = args.root / "data/track_b/manifests/annual_2010_v1/run_audit.json"
        audit = json.loads(path.read_text(encoding="utf-8"))
        enrich(args.root, audit)
        print("Aggregate report rebuilt; no sample or outcome changes")
        return 0
    if args.annual:
        if args.source is None or not args.attest_official_download:
            parser.error("Annual run requires original ZIP and official-download attestation")
        result = run_annual(args.root, args.source)
        print(f"Annual gate: {result['feasibility']}")
        return 3 if result["feasibility"].startswith("STOP") else 0
    if args.preflight:
        if args.source is None:
            parser.error("Preflight requires the original ZIP path")
        result = record_preflight(
            args.root, args.source, official_attestation=args.attest_official_download
        )
        print(f"{result['status']}: metadata only; SHA256={result['sha256']}")
        return 3 if result["blockers"] else 0
    if args.source is None:
        blocked_audit(args.root)
        print("NOT_ACQUIRED: no authorized archive; empirical feasibility and modeling blocked.")
        return 2
    if args.authorization is None:
        parser.error(
            "Authorized local-source attestation required; no credentials or automatic download"
        )
    try:
        audit, _ = build(
            args.root,
            args.source,
            args.authorization,
            n=args.loans,
            publish_aggregates=args.publish_aggregates,
        )
    except IngestionRejected as exc:
        # Rejection evidence is private; no licensed row values are logged/published.
        write_json(args.root / "data/track_b/manifests/ingestion_rejection.json", exc.audit)
        parser.exit(1, f"Track B ingestion stopped: {exc}; private rejection counts recorded\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Track B ingestion stopped: {exc}\n")
    if audit["feasibility"] == "STOP — DATA UNSUITABLE":
        parser.exit(1, "Private audit written; integrity/cohort gate failed; modeling blocked\n")
    print(f"Private panel written; status={audit['status']}; no modeling or ECL.")


if __name__ == "__main__":
    raise SystemExit(main())
