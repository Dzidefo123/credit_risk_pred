"""Local authorized-archive interface, private outputs and honest aggregate audit."""

import io
import json
import shutil
import zipfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from .freddie import ParseStats, records
from .panel import build_panel, distribution, summarize
from .sampling import SALT, select_loans
from .schemas import (
    COLUMN_ROLES,
    LAYOUT,
    LAYOUT_SHA256,
    LAYOUT_URL,
    LOSS_FIELDS,
    ORIGINATION,
    PARSER_VERSION,
    PERFORMANCE,
    digest,
    load_protocol,
)

LIMITATIONS = [
    "Facility IDs are not borrower IDs; no cross-facility identity inference.",
    "Latest-release data gives nominal-time consistency, not verified historical availability.",
    "Delinquency bands are not exact DPD and the composite event is not regulatory default.",
    "Principal balance is an exposure proxy, not validated EAD.",
    "Loss fields are aggregate disclosures, not timed recoveries/workout LGD.",
    "All censoring/ineligible statuses retained; no complete-case cohort selection.",
    "No models, performance comparisons, calibration, feature selection or ECL calculated.",
]
AUTH_FIELDS = {
    "source_kind",
    "source_organization",
    "vintage",
    "source_release",
    "layout",
    "source_sha256",
    "acquired_at",
    "official_source_reference",
    "terms_accepted",
    "license_redistribution_status",
    "aggregate_publication_permitted",
}


def load_authorization(path, source):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or set(data) != AUTH_FIELDS
        or data["source_kind"]
        not in {
            "authorized_freddie",
            "synthetic_fixture",
        }
    ):
        raise ValueError("Missing/unsupported authorization fields; never supply secrets")
    if (
        data["vintage"] != 2010
        or data["source_release"] != "47"
        or data["layout"] != LAYOUT
        or data["terms_accepted"] is not True
        or not isinstance(data["aggregate_publication_permitted"], bool)
    ):
        raise ValueError("Source vintage/release/layout/authorization does not match protocol")
    string_fields = AUTH_FIELDS - {"vintage", "terms_accepted", "aggregate_publication_permitted"}
    if any(not isinstance(data[key], str) for key in string_fields):
        raise ValueError("Malformed source-attestation types")
    parsed = urlsplit(data["official_source_reference"])
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"www.freddiemac.com", "claritydownload.fmapps.freddiemac.com"}
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Use a public official reference, not tokens or session URLs")
    acquired = datetime.fromisoformat(data["acquired_at"])
    if acquired.tzinfo is None or acquired > datetime.now(UTC):
        raise ValueError("Acquisition time must be timezone-aware and not future")
    if data["license_redistribution_status"] not in {"prohibited", "restricted", "permitted"}:
        raise ValueError("Explicit redistribution status required")
    if data["source_kind"] == "authorized_freddie" and data["source_organization"] != "Freddie Mac":
        raise ValueError("Official provider attestation required")
    if digest(source) != data["source_sha256"]:
        raise ValueError("Authorized source hash mismatch; parsing blocked")
    return data


def archive_members(archive):
    names = archive.namelist()
    expected = {"sample_orig_2010.txt", "sample_perf_2010.txt"}
    if (
        len(names) != 2
        or set(names) != expected
        or any(i.flag_bits & 1 for i in archive.infolist())
    ):
        raise ValueError("Expected unencrypted 2010 sample pair; no guessed layout")
    if sum(i.file_size for i in archive.infolist()) > 2 * 1024**3:
        raise ValueError(
            "Uncompressed archive exceeds engineering budget; review before proceeding"
        )
    return "sample_orig_2010.txt", "sample_perf_2010.txt"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def audit_markdown(audit):
    title = "# Freddie 2010 Track B data audit"
    if audit["status"] == "NOT_ACQUIRED":
        return (
            title + "\n\n**No Freddie loan records acquired or inspected.** Official authenticated "
            "access/local archive unavailable. No bypass, alternate vintage or fabricated "
            "data used.\n\nThe ingestion interface and synthetic-fixture tests "
            "are engineering evidence only. Empirical feasibility is not established.\n\n"
            + "| Empirical question | Result |\n| --- | --- |\n"
            + "\n".join(
                f"| Q{i} | Not assessable without authorized records |" for i in range(1, 10)
            )
            + "\n\nNo cohort counts, follow-up percentages, balance distributions or model results "
            "are reported. The vintage and 1,000-loan cap remain unchanged.\n\n"
            + "Feasibility: **PROCEED WITH CONDITIONS** for finishing authorized engineering only. "
            "Acquisition and empirical validation remain blocked; do not proceed to modeling.\n"
        )
    q = audit["cohort"]
    label = (
        "SYNTHETIC FIXTURE ONLY"
        if audit["source_kind"] == "synthetic_fixture"
        else "AUTHORIZED SOURCE DESCRIPTIVE AUDIT"
    )
    rows = [
        (key, str(q[key]))
        for key in (
            "selected_loans",
            "origination_records",
            "monthly_performance_records",
            "panel_observations",
            "eligible_landmarks",
            "ineligible_landmarks",
            "reporting_period_range",
        )
    ]
    body = title + f"\n\n**{label}**. Not model results.\n\n| Quantity | Value |\n| --- | --- |\n"
    body += "\n".join(f"| {k} | {v} |" for k, v in rows)
    body += "\n\n## Follow-up completeness (all eligible landmarks, no maturity filtering)\n\n"
    body += "| Status | Count |\n| --- | --- |\n" + "\n".join(
        f"| {k} | {v} |" for k, v in q["followup"]["status_counts"].items()
    )
    body += (
        "\n\nComplete-event-free and ascertained outcomes differ. Payoff is a competing "
        "event, not an administratively censored negative. See JSON for exact denominators, "
        "missingness, balances, parser/linkage counts, loss availability and integrity.\n"
    )
    body += "\n## Limitations\n\n" + "\n".join("- " + x for x in LIMITATIONS) + "\n"
    return body


def blocked_audit(root):
    protocol, protocol_hash = load_protocol(root)
    audit = {
        "schema_version": 1,
        "status": "NOT_ACQUIRED",
        "source_kind": None,
        "acquisition": {
            "files_acquired": 0,
            "reason": "No officially authorized local 2010 archive supplied",
        },
        "parser_version": PARSER_VERSION,
        "source_layout": LAYOUT,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256_lf": protocol_hash,
        "sampling": {
            "salt": SALT,
            "maximum_loans": 1000,
            "vintage": 2010,
            "outcome_independent": True,
        },
        "cohort": None,
        "followup": None,
        "temporal_integrity": None,
        "outcomes": None,
        "empirical_questions": {f"Q{i}": "not_assessable" for i in range(1, 10)},
        "limitations": LIMITATIONS,
        "feasibility": "PROCEED WITH CONDITIONS",
        "modeling_gate": "BLOCKED: authorized-source empirical verification required",
    }
    report = Path(root) / "reports/track_b"
    write_json(report / "freddie_2010_data_audit.json", audit)
    (report / "FREDDIE_2010_DATA_AUDIT.md").write_text(audit_markdown(audit), encoding="utf-8")
    return audit


class IngestionRejected(ValueError):
    """Structured rejection counts without licensed row content."""

    def __init__(self, message, stats, **findings):
        super().__init__(message)
        self.audit = {
            "status": "SOURCE_REJECTED",
            "cohort": None,
            "parse_counts": {k: v.summary() for k, v in stats.items()},
            "findings": findings,
            "reason": message,
            "feasibility": "STOP — DATA UNSUITABLE",
            "modeling_gate": "BLOCKED",
        }


def build(root, source, authorization, *, n=1000, publish_aggregates=False):
    root, source = Path(root).resolve(), Path(source).resolve()
    protocol, protocol_hash = load_protocol(root)
    if (
        not source.is_file()
        or source.suffix.lower() != ".zip"
        or source.stat().st_size > 200 * 1024**2
    ):
        raise ValueError(
            "Supply one existing authorized sample ZIP within 200 MiB; no automatic acquisition"
        )
    auth = load_authorization(authorization, source)
    if publish_aggregates and (
        not auth["aggregate_publication_permitted"] or auth["source_kind"] == "synthetic_fixture"
    ):
        raise ValueError("Public aggregates need permission; fixtures are not empirical evidence")
    stats = {kind: ParseStats() for kind in ("origination", "performance")}
    original = set()
    duplicate_orig = 0
    selected = {}
    monthly = []
    perf_ids = set()
    unmatched = 0
    member_metadata = []
    with zipfile.ZipFile(source) as archive:
        orig_name, perf_name = archive_members(archive)
        for name in (orig_name, perf_name):
            h = sha256()
            with archive.open(name) as member:
                for block in iter(lambda: member.read(1024 * 1024), b""):
                    h.update(block)
            member_metadata.append(
                {
                    "original_filename": name,
                    "byte_size": archive.getinfo(name).file_size,
                    "sha256": h.hexdigest(),
                    "storage": "member of private source ZIP",
                }
            )
        with io.TextIOWrapper(
            archive.open(orig_name), encoding="utf-8-sig", errors="strict"
        ) as stream:
            for row in records(stream, "origination", stats["origination"]):
                if row["loan_id"] in original:
                    duplicate_orig += 1
                original.add(row["loan_id"])
                if len(original) > 60000:
                    raise ValueError("Archive exceeds single-vintage official sample scope")
        if duplicate_orig or stats["origination"].malformed_rows:
            raise ValueError("Origination rejects block sampling; no repair or outcome reselection")
        ids = select_loans(original, n)
        selected_ids = set(ids)
        # Second pass retains only selected static records; ID pool is metadata only.
        with io.TextIOWrapper(
            archive.open(orig_name), encoding="utf-8-sig", errors="strict"
        ) as stream:
            for row in records(stream, "origination", ParseStats()):
                if row["loan_id"] in selected_ids:
                    selected[row["loan_id"]] = row
        with io.TextIOWrapper(
            archive.open(perf_name), encoding="utf-8-sig", errors="strict"
        ) as stream:
            for row in records(stream, "performance", stats["performance"]):
                perf_ids.add(row["loan_id"])
                unmatched += row["loan_id"] not in original
                if row["loan_id"] in selected:
                    if len(monthly) >= 250000:
                        raise ValueError(
                            "Selected-history resource budget exceeded; review, never truncate"
                        )
                    monthly.append(row)
        if stats["performance"].malformed_rows or unmatched:
            raise IngestionRejected(
                "Performance schema/linkage rejects block panel; do not reinterpret protocol",
                stats,
                unmatched_performance_rows=unmatched,
            )
    if not selected:
        raise IngestionRejected("No valid source identifiers; no panel", stats)
    panel, integrity = build_panel(selected, monthly, protocol)
    cohort = summarize(panel, selected, monthly, integrity, protocol)
    private = root / "data/track_b"
    targets = [
        private / "raw" / source.name,
        private / "interim/selected_origination.csv",
        private / "interim/selected_performance.csv",
        private / "processed/panel.csv",
        private / "manifests/freddie_2010_manifest.json",
        private / "manifests/panel_manifest.json",
    ]
    if any(not p.resolve().is_relative_to(private) for p in targets) or any(
        p.exists() for p in targets[1:]
    ):
        raise ValueError("Private output escapes zone or already exists; refusing overwrite")
    if digest(source) != auth["source_sha256"]:
        raise ValueError("Source changed during parsing")
    for p in targets:
        p.parent.mkdir(parents=True, exist_ok=True)
    if targets[0] != source:
        if targets[0].exists() and digest(targets[0]) != auth["source_sha256"]:
            raise ValueError("Different raw archive already occupies destination")
        if not targets[0].exists():
            shutil.copyfile(source, targets[0])
    if digest(targets[0]) != auth["source_sha256"]:
        raise ValueError("Raw archive changed during storage; panel output blocked")
    pd.DataFrame(selected.values()).to_csv(targets[1], index=False)
    pd.DataFrame(monthly).to_csv(targets[2], index=False)
    panel.to_csv(targets[3], index=False)
    manifest = {
        **auth,
        "original_filename": source.name,
        "byte_size": source.stat().st_size,
        "local_storage": str(targets[0].relative_to(root)).replace("\\", "/"),
        "parser_version": PARSER_VERSION,
        "source_files": member_metadata,
        "layout_documentation": LAYOUT_URL,
        "layout_documentation_sha256": LAYOUT_SHA256,
        "expected_column_counts": {
            "origination": len(ORIGINATION),
            "performance": len(PERFORMANCE),
        },
        "field_mapping": {"origination": list(ORIGINATION), "performance": list(PERFORMANCE)},
        "loss_sign_convention": (
            "R47 gains/recoveries negative, losses/expenses positive; signed values retained"
        ),
        "missing_convention": "Blank/documented sentinels; unknown states never current",
        "parse_counts": {k: v.summary() for k, v in stats.items()},
        "duplicate_origination": duplicate_orig,
        "unmatched_performance_rows": unmatched,
        "unmatched_origination_records": len(set(original) - perf_ids),
        "selection_salt": SALT,
        "selected_id_set_sha256": sha256("\n".join(sorted(ids)).encode()).hexdigest(),
    }
    write_json(targets[4], manifest)
    panel_manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256_lf": protocol_hash,
        "source_sha256": auth["source_sha256"],
        "acquisition_manifest_sha256": digest(targets[4]),
        "sampling_salt": SALT,
        "unique_selected_loans": len(selected),
        "observations": len(panel),
        "date_range": cohort["reporting_period_range"],
        "transformation_version": PARSER_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "output_sha256": digest(targets[3]),
        "column_roles": COLUMN_ROLES,
        "historical_availability_verified": False,
    }
    panel_manifest["transformation_source_sha256_lf"] = {
        p.name: sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    write_json(targets[5], panel_manifest)
    blocking = (
        not cohort["eligible_landmarks"]
        or not cohort["followup"]["outcome_ascertained_including_events_and_payoff"]
        or any(
            integrity["findings"].get(k, 0)
            for k in [
                "selected_origination_without_performance",
                "conflicting_duplicate_months",
                "unexpected_post_terminal_rows",
            ]
        )
    )
    sign_flags = {
        "positive_recovery_values": sum(
            r[k] is not None and r[k] > 0
            for r in monthly
            for k in ["mi_recoveries", "net_sale_proceeds", "non_mi_recoveries"]
        ),
        "negative_expense_values": sum(
            r[k] is not None and r[k] < 0
            for r in monthly
            for k in [
                "total_expenses",
                "legal_costs",
                "preservation_costs",
                "taxes_insurance",
                "misc_expenses",
            ]
        ),
        "unexplained_net_proceeds_codes": sum(
            "net_sale_proceeds_disclosure_code" in r for r in monthly
        ),
    }
    audit = {
        "schema_version": 1,
        "status": "FIXTURE_ONLY" if auth["source_kind"] == "synthetic_fixture" else "ACQUIRED",
        "source_kind": auth["source_kind"],
        "provenance": manifest,
        "parser_version": PARSER_VERSION,
        "cohort": cohort,
        "loss_availability": {key: distribution([r[key] for r in monthly]) for key in LOSS_FIELDS},
        "protocol_sha256_lf": protocol_hash,
        "loss_sign_review_flags": sign_flags,
        "limitations": LIMITATIONS,
        "feasibility": "STOP — DATA UNSUITABLE" if blocking else "PROCEED WITH CONDITIONS",
        "modeling_gate": "BLOCKED for synthetic fixtures"
        if auth["source_kind"] == "synthetic_fixture"
        else (
            "BLOCKED: integrity findings require review"
            if blocking
            else "Requires cohort/censoring and feasibility review; no models in Task 2"
        ),
    }
    write_json(private / "manifests/freddie_2010_data_audit.json", audit)
    (private / "manifests/FREDDIE_2010_DATA_AUDIT.md").write_text(
        audit_markdown(audit), encoding="utf-8"
    )
    if publish_aggregates:
        (root / "reports/track_b").mkdir(parents=True, exist_ok=True)
        write_json(root / "reports/track_b/freddie_2010_data_audit.json", audit)
        (root / "reports/track_b/FREDDIE_2010_DATA_AUDIT.md").write_text(
            audit_markdown(audit), encoding="utf-8"
        )
    return audit, panel_manifest
