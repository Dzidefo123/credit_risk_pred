"""Annual-frame phases: origination, frozen sample, selected-only performance, audit."""

import csv
import json
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from .annual import AnnualReader, GateStop, load_amendment, peak_memory_bytes
from .freddie import parse_row
from .panel import build_panel, distribution, event_category, summarize
from .sampling import SALT, select_loans
from .schemas import COLUMN_ROLES, LOSS_FIELDS, ORIGINATION, PARSER_VERSION, PERFORMANCE, digest
from .workflow import LIMITATIONS, audit_markdown, write_json


def check_memory(limits):
    value = peak_memory_bytes()
    if value is not None and value > limits["max_peak_memory_bytes"]:
        raise GateStop("Process memory guard exceeded; no truncation or resampling")
    return value


def structural_gate(reader, q, kind, evidence):
    widths = Counter()
    rows = 0
    for line in reader.lines(q, kind):
        tokens = next(csv.reader([line], delimiter="|", strict=True))
        widths[len(tokens)] += 1
        rows += 1
        try:
            parse_row(tokens, kind)
        except ValueError as exc:
            result = {
                "quarter": q,
                "kind": kind,
                "rows_inspected": rows,
                "field_counts": dict(widths),
                "expected_fields": len(ORIGINATION if kind == "origination" else PERFORMANCE),
                "reason": str(exc),
                "delimiter": "pipe",
                "encoding": "UTF-8 compatible",
            }
            evidence.append(result)
            raise GateStop(
                "Actual structural layout incompatible with pinned parser", result
            ) from exc
        if rows >= 20:
            break
    if not rows:
        raise GateStop("Empty source member")
    evidence.append(
        {
            "quarter": q,
            "kind": kind,
            "rows_inspected": rows,
            "field_counts": dict(widths),
            "expected_fields": len(ORIGINATION if kind == "origination" else PERFORMANCE),
            "status": "compatible structural probe",
            "delimiter": "pipe",
            "encoding": "UTF-8 compatible",
        }
    )


def run_annual(root, source, *, progress=print, source_kind="authorized_freddie"):
    root, source = Path(root).resolve(), Path(source).resolve()
    protocol, protocol_hash, amendment, amendment_hash = load_amendment(root)
    limits = amendment["resources"]
    private = root / "data/track_b/manifests/annual_2010_v1"
    private.mkdir(parents=True, exist_ok=True)
    if (private / "sample_manifest.json").exists():
        raise GateStop("Existing frozen sample: refuse redraw; explicit resume required")
    started = time.monotonic()
    schema = []
    counts = {}
    universe = {}
    selected = {}
    monthly = []
    sample_manifest = None
    inventory = []
    resources = {}
    phase = "annual_preflight"
    public = {
        "schema_version": 2,
        "status": "RUNNING",
        "phase": phase,
        "source_sha256": amendment["source_sha256"],
        "original_filename": source.name,
        "amendment_id": amendment["amendment_id"],
        "amendment_sha256_lf": amendment_hash,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256_lf": protocol_hash,
        "initial_stop_preserved": amendment["initial_stop_evidence"],
        "parser_version": PARSER_VERSION,
        "schema_evidence": schema,
        "quarter_counts": counts,
        "cohort": None,
        "limitations": LIMITATIONS,
    }
    try:
        with AnnualReader(source, root, limits, amendment["source_sha256"]) as reader:
            inventory = reader.inventory
            public["annual_preflight"] = {
                "status": "passed",
                "inventory": inventory,
                "source_unchanged_before_scan": True,
            }
            phase = "origination_layout"
            for q in range(1, 5):
                structural_gate(reader, q, "origination", schema)
            phase = "origination_universe"
            for q in range(1, 5):
                scanned = valid = duplicate = 0
                for line in reader.lines(q, "origination"):
                    scanned += 1
                    try:
                        row = parse_row(
                            next(csv.reader([line], delimiter="|", strict=True)), "origination"
                        )
                    except ValueError as exc:
                        raise GateStop(
                            "Origination malformed row; sampling blocked",
                            {"quarter": q, "row_number": scanned, "reason": str(exc)},
                        ) from exc
                    loan = row["loan_id"]
                    if loan[4] != str(q):
                        raise GateStop("Loan quarter does not match origination member")
                    if loan in universe:
                        duplicate += 1
                        raise GateStop(
                            "Duplicate ID within/across origination quarters",
                            {"quarter": q, "duplicates": duplicate},
                        )
                    universe[loan] = q
                    valid += 1
                    if len(universe) > limits["max_universe_ids"]:
                        raise GateStop("Universe metadata limit exceeded")
                    if scanned % 100000 == 0:
                        check_memory(limits)
                        progress(f"Origination Q{q}: {scanned:,} rows scanned")
                counts[str(q)] = {
                    "origination_rows": scanned,
                    "valid_ids": valid,
                    "malformed_rows": 0,
                    "missing_ids": 0,
                    "duplicates_within_or_across": duplicate,
                }
                progress(f"Origination Q{q} complete: {scanned:,} valid IDs")
            phase = "sample_freeze"
            ids = select_loans(universe, 1000)
            fingerprint = sha256("\n".join(sorted(ids)).encode()).hexdigest()
            sample_manifest = {
                "source_sha256": amendment["source_sha256"],
                "protocol_sha256_lf": protocol_hash,
                "amendment_sha256_lf": amendment_hash,
                "salt": SALT,
                "algorithm": "SHA256(salt:ID), then ID; no quotas",
                "eligible_annual_universe": len(universe),
                "selected_count": len(ids),
                "sample_set_sha256": fingerprint,
                "selected_by_quarter": dict(sorted(Counter(str(universe[i]) for i in ids).items())),
                "frozen_before_performance_access": True,
                "frozen_at": datetime.now(UTC).isoformat(),
            }
            write_json(private / "sample_manifest.json", sample_manifest)
            (private / "selected_ids.txt").write_text(
                "\n".join(sorted(ids)) + "\n", encoding="utf-8"
            )
            reader.freeze(ids)
            progress(f"Sample frozen: {len(ids)} IDs from {len(universe):,}; SHA256={fingerprint}")
            phase = "selected_origination"
            for q in range(1, 5):
                for line in reader.lines(q, "origination"):
                    # Keep only the already-frozen identifiers; static records cannot replace them.
                    tokens = next(csv.reader([line], delimiter="|", strict=True))
                    if tokens[19].strip() in reader.sample:
                        row = parse_row(tokens, "origination")
                        selected[row["loan_id"]] = row
            phase = "performance_layout"
            for q in range(1, 5):
                structural_gate(reader, q, "performance", schema)
            phase = "performance_stream"
            for q in range(1, 5):
                scanned = retained = unmatched = 0
                for line in reader.lines(q, "performance"):
                    scanned += 1
                    loan = line.partition("|")[0].strip()
                    if loan not in universe:
                        unmatched += 1
                        raise GateStop(
                            "Performance ID absent from annual origination universe",
                            {"quarter": q, "row_number": scanned},
                        )
                    if universe[loan] != q:
                        raise GateStop("Performance quarter linkage mismatch")
                    if loan not in reader.sample:
                        if scanned % 1000000 == 0:
                            check_memory(limits)
                            progress(
                                f"Performance Q{q}: {scanned:,} scanned, {retained:,} retained"
                            )
                        continue
                    try:
                        row = parse_row(
                            next(csv.reader([line], delimiter="|", strict=True)), "performance"
                        )
                    except ValueError as exc:
                        raise GateStop(
                            "Selected performance row malformed",
                            {"quarter": q, "row_number": scanned, "reason": str(exc)},
                        ) from exc
                    monthly.append(row)
                    retained += 1
                    if len(monthly) > limits["max_retained_performance_rows"]:
                        raise GateStop("Selected history budget exceeded; sample unchanged")
                counts[str(q)].update(
                    performance_rows_scanned=scanned,
                    selected_rows_retained=retained,
                    unmatched_ids=unmatched,
                    malformed_selected_rows=0,
                )
                progress(f"Performance Q{q} complete: {scanned:,} scanned, {retained:,} retained")
            resources = {
                "elapsed_scan_seconds": time.monotonic() - started,
                "peak_memory_bytes": peak_memory_bytes(),
                "temporary_bytes_peak": reader.temp_bytes_peak,
                "temporary_files_created": reader.temp_files_created,
                "full_text_extracted": False,
                "non_selected_attributes_retained": False,
            }
        if digest(source) != amendment["source_sha256"]:
            raise GateStop("Root source changed during run")
        phase = "longitudinal_audit"
        panel, integrity = build_panel(selected, monthly, protocol)
        cohort = summarize(panel, selected, monthly, integrity, protocol)
        check_memory(limits)
        values = Counter()
        loan_events = Counter()
        for loan in ids:
            records = sorted(
                [r for r in monthly if r["loan_id"] == loan], key=lambda r: r["reporting_month"]
            )
            categories = [event_category(r, protocol["event"]) for r in records]
            first = next(
                (
                    c
                    for c in categories
                    if c in {"default", "payoff", "administrative", "ambiguous"}
                ),
                "active_or_unknown",
            )
            loan_events[first] += 1
            if "default" in categories:
                values["selected_loans_with_observed_qualifying_record"] += 1
        eligible = panel[panel.eligible]
        status_table = []
        for status, number in cohort["followup"]["status_counts"].items():
            status_table.append(
                {
                    "status": status,
                    "count": number,
                    "fraction_of_all_eligible": number / len(eligible),
                }
            )
        calendar = {}
        for year, group in eligible.groupby(eligible.t0.str[:4]):
            calendar[year] = {
                "eligible": len(group),
                "statuses": dict(Counter(group.outcome_status)),
            }
        public.update(
            status="ACQUIRED" if source_kind == "authorized_freddie" else "FIXTURE_ONLY",
            source_kind=source_kind,
            phase="complete",
            annual_universe=len(universe),
            sample=sample_manifest,
            cohort=cohort,
            followup_status_table=status_table,
            followup_by_year=calendar,
            loan_first_observed_events=dict(loan_events),
            default_record_loans=dict(values),
            resources=resources,
            exposure_eligible=distribution(eligible.current_principal_balance),
            loss_availability={k: distribution([r[k] for r in monthly]) for k in LOSS_FIELDS},
            assumptions={
                "row_layout": "CONFIRMED WITH QUALIFICATION",
                "historical_availability": "NOT TESTABLE",
                "monthly_principal": "CONFIRMED WITH QUALIFICATION",
                "timed_workout_LGD": "NOT TESTABLE",
            },
        )
        blocking = any(
            integrity["findings"].get(k, 0)
            for k in [
                "selected_origination_without_performance",
                "conflicting_duplicate_months",
                "unexpected_post_terminal_rows",
            ]
        )
        blocking = (
            blocking
            or not len(eligible)
            or not cohort["followup"]["outcome_ascertained_including_events_and_payoff"]
        )
        public["feasibility"] = (
            "STOP — DATA/PROTOCOL INCOMPATIBLE" if blocking else "PROCEED WITH CONDITIONS"
        )
        public["modeling_gate"] = (
            "Blocked by integrity findings"
            if blocking
            else "Task 3 cohort/censoring review required; no models fitted"
        )
        output = root / "data/track_b/processed/annual_2010_v1/panel.csv"
        if output.exists():
            raise GateStop("Prior processed panel exists; refusing overwrite")
        output.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(output, index=False)
        write_json(
            private / "panel_manifest.json",
            {
                "source_sha256": amendment["source_sha256"],
                "protocol_sha256_lf": protocol_hash,
                "amendment_sha256_lf": amendment_hash,
                "sample_manifest_sha256": digest(private / "sample_manifest.json"),
                "sample_set_sha256": fingerprint,
                "output_sha256": digest(output),
                "observations": len(panel),
                "loans": int(panel.loan_id.nunique()),
                "schema_version": 1,
                "parser_version": PARSER_VERSION,
                "column_roles": COLUMN_ROLES,
                "date_range": cohort["reporting_period_range"],
                "created_at": datetime.now(UTC).isoformat(),
                "transformation_source_sha256_lf": {
                    p.name: sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                    for p in Path(__file__).parent.glob("*.py")
                },
            },
        )
    except (GateStop, ValueError, zipfile.BadZipFile) as exc:
        public.update(
            status="EMPIRICAL_GATE_STOPPED",
            phase=phase,
            stop_reason=str(exc),
            stop_evidence=getattr(exc, "evidence", {}),
            sample=sample_manifest,
            annual_universe=len(universe) if universe else None,
            resources={
                "elapsed_seconds": time.monotonic() - started,
                "peak_memory_bytes": peak_memory_bytes(),
            },
            feasibility="STOP — DATA/PROTOCOL INCOMPATIBLE",
            modeling_gate="BLOCKED",
        )
        progress(f"STOP at {phase}: {exc}")
    write_json(private / "run_audit.json", public)
    write_json(root / "reports/track_b/freddie_2010_data_audit.json", public)
    if public["status"] in {"ACQUIRED", "FIXTURE_ONLY"}:
        text = audit_markdown(public)
        text += (
            "\n## Annual sample and scan\n\n"
            + json.dumps({"sample": sample_manifest, "resources": resources}, indent=2)
            + "\n"
        )
    else:
        text = "# Freddie 2010 Track B audit - empirical gate stopped\n\n"
        text += f"Phase: **{public['phase']}**. Reason: **{public['stop_reason']}**.\n\n"
        text += f"Source SHA-256: `{public['source_sha256']}`.\n\n"
        text += "The approved annual amendment passed only as far as recorded in JSON. "
        text += "No scientific rules changed. No adaptive redraw, models or ECL.\n\n"
        text += (
            "[Initial stop evidence](FREDDIE_2010_INITIAL_PREFLIGHT_STOP.json) is preserved.\n\n"
        )
        text += "Decision: **STOP — DATA/PROTOCOL INCOMPATIBLE**. No completion commit.\n"
    (root / "reports/track_b/FREDDIE_2010_DATA_AUDIT.md").write_text(text, encoding="utf-8")
    return public
