"""Exact-record quarantine, full origination gates, freeze, then legacy history audit."""

import hashlib
import heapq
import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from credit_risk.track_b.data.annual import GateStop, peak_memory_bytes
from credit_risk.track_b.data.schemas import digest, load_protocol
from credit_risk.track_b.multivintage.audit import audit
from credit_risk.track_b.multivintage.core import rank_key, set_hash, write_json
from credit_risk.track_b.multivintage.reader import VintageReader
from credit_risk.track_b.multivintage.schema import VERSION
from credit_risk.track_b.multivintage.study import checkpoint_valid, lf_hash, retain, source_code

from .eligibility import StructuralPerformanceReader, classify


def attestation_code(root):
    return {
        **source_code(root),
        **{
            p.relative_to(root).as_posix(): lf_hash(p)
            for p in (root / "src/credit_risk/track_b/recovery").glob("*.py")
            if p.name != "reporting.py"
        },
    }


def counterfactual(root, vintage, census):
    path = root / f"data/track_b/multivintage/reconciliation_v1/identifiers_{vintage}.sqlite"
    if digest(path) != census["identifier_database_sha256"]:
        raise GateStop("Counterfactual identifier evidence changed")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        return heapq.nsmallest(
            20000,
            (r[0] for r in db.execute("SELECT loan FROM identifiers")),
            key=lambda loan: (rank_key(loan, vintage), loan),
        )


def sample(root, reader, output, plan, identity, registry, census):
    frozen = output / "sample_manifest.json"
    dbpath = output / "origination.sqlite"
    if frozen.exists():
        manifest = json.loads(frozen.read_text(encoding="utf-8"))
        checkpoint_valid(manifest, {**identity, "sample_sha256": manifest["sample_sha256"]})
        ids = (output / "selected_ids.txt").read_text(encoding="utf-8").splitlines()
        if (
            digest(dbpath) != manifest["origination_sha256"]
            or set_hash(ids) != manifest["sample_sha256"]
        ):
            raise GateStop("Frozen recovery sample changed")
        reader.freeze(ids)
        return ids, manifest
    if dbpath.exists():
        raise GateStop("Partial recovery origination stage: explicit review required")
    vintage = reader.vintage
    counts, seen = {}, []
    with sqlite3.connect(dbpath) as db:
        db.execute(
            "CREATE TABLE origination(loan TEXT PRIMARY KEY,rank TEXT NOT NULL,raw TEXT NOT NULL)"
        )
        eligible = inspected = 0
        for quarter in range(1, 5):
            batch = []
            count = 0
            for number, line in enumerate(reader.lines(quarter, "origination"), 1):
                inspected += 1
                if inspected > plan["limits"]["max_universe_ids"]:
                    raise GateStop("Origination universe resource cap")
                decision = classify(line, vintage, quarter, number, registry)
                if not decision["eligible"]:
                    seen.append(decision["audit_reference"])
                    continue
                loan = decision["loan_id"]
                batch.append((loan, rank_key(loan, vintage), line))
                eligible += 1
                count += 1
                if len(batch) >= 5000:
                    db.executemany("INSERT INTO origination VALUES (?,?,?)", batch)
                    db.commit()
                    batch = []
                    memory = peak_memory_bytes()
                    if memory and memory > plan["limits"]["max_peak_memory_bytes"]:
                        raise GateStop("Origination memory cap")
                    if dbpath.stat().st_size > plan["limits"]["max_cache_bytes"]:
                        raise GateStop("Origination cache cap")
            if batch:
                db.executemany("INSERT INTO origination VALUES (?,?,?)", batch)
                db.commit()
            counts[str(quarter)] = count
            print(f"{vintage} full origination Q{quarter}: {count:,} eligible", flush=True)
        expected = [r["key"] for r in registry if r["vintage"] == vintage]
        if sorted(seen) != sorted(expected):
            raise GateStop("Approved quarantine detection count mismatch")
        if inspected != census["unique_identifiers"] or eligible != inspected - len(expected):
            raise GateStop("Origination universe differs from counterfactual expectation")
        db.execute("CREATE INDEX ranking ON origination(rank,loan)")
        ids = [
            r[0] for r in db.execute("SELECT loan FROM origination ORDER BY rank,loan LIMIT 20000")
        ]
        if len(ids) != 20000:
            raise GateStop("Insufficient eligible IDs for prespecified sample")
        hypothetical = counterfactual(root, vintage, census)
        delta = len(set(ids) ^ set(hypothetical))
        if delta:
            raise GateStop("Quarantine sample impact differs from Task8B expectation")
        universe_digest = hashlib.sha256()
        for index, (loan,) in enumerate(db.execute("SELECT loan FROM origination ORDER BY loan")):
            universe_digest.update(("\n" if index else "").encode() + loan.encode())
        db.commit()
    ledger = [r for r in plan["quarantine"] if r["vintage"] == vintage]
    write_json(output / "quarantine.json", ledger, True)
    (output / "selected_ids.txt").write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
    manifest = dict(
        **identity,
        sample_sha256=set_hash(ids),
        facilities=len(ids),
        universe=eligible,
        inspected=inspected,
        quarantine_count=len(seen),
        universe_sha256=universe_digest.hexdigest(),
        origination_rows_by_quarter=counts,
        origination_sha256=digest(dbpath),
        namespace=plan["namespace"],
        sample_symmetric_difference=delta,
        counterfactual_inclusion_sample_sha256=set_hash(hypothetical),
        frozen_at=datetime.now(UTC).isoformat(),
        frozen_before_performance_access=True,
        full_origination_gate_passed=True,
    )
    write_json(frozen, manifest, True)
    reader.freeze(ids)
    print(f"{vintage}: sample frozen; zero impact; {manifest['sample_sha256']}", flush=True)
    return ids, manifest


def reuse(root, source_dir, vintage, oldplan):
    output = root / f"data/track_b/multivintage/processed/v1/{vintage}"
    result = json.loads((output / "completed.json").read_text(encoding="utf-8"))
    if digest(source_dir / f"historical_data_{vintage}.zip") != result["source_sha256"]:
        raise GateStop("Reused cohort source changed")
    ids_path = (
        root / "data/track_b/manifests/expansion_v1/selected_ids.txt"
        if vintage == 2010
        else output / "selected_ids.txt"
    )
    ids = ids_path.read_text(encoding="utf-8").splitlines()
    checkpoint_valid(
        result,
        dict(
            source_sha256=result["source_sha256"],
            parser_version=VERSION,
            protocol_sha256=lf_hash(root / "docs/track_b/multi_vintage_protocol.json"),
            sample_sha256=set_hash(ids),
            code_sha256=source_code(root),
        ),
    )
    if len(ids) != 20000 or result["sample_sha256"] != (
        oldplan["existing_2010_sample_sha256"] if vintage == 2010 else result["sample_sha256"]
    ):
        raise GateStop("Reused sample identity changed")
    for name, expected in result["output_hashes"].items():
        if digest(output / name) != expected:
            raise GateStop("Reused panel changed")
    print(f"{vintage}: completed checkpoint verified; no performance rescan", flush=True)
    return result


def run(root, source_dir):
    root, source_dir = Path(root).resolve(), Path(source_dir).resolve()
    base = root / "data/track_b/multivintage/recovery_v1"
    planpath = root / "docs/track_b/multi_vintage_recovery_protocol.json"
    plan = json.loads(planpath.read_text(encoding="utf-8"))
    registry_path = base / "private_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if digest(registry_path) != plan["quarantine_registry_sha256"]:
        raise GateStop("Authorized private registry changed")
    if {r["key"] for r in registry} != {"A1", "A2", "A3", "A4"} or len(registry) != 4:
        raise GateStop("Exact four-record authorization required")
    for entry in registry:
        approved = next(r for r in plan["quarantine"] if r["audit_reference"] == entry["key"])
        if any(entry[k] != approved[k] for k in ["vintage", "reason_code", "raw_record_sha256"]):
            raise GateStop("Registry differs from public authorization")
    if (
        lf_hash(root / "docs/track_b/multi_vintage_protocol.json")
        != plan["extends_protocol_sha256_lf"]
    ):
        raise GateStop("Original protocol compatibility changed")
    _, event_hash = load_protocol(root)
    if event_hash != plan["base_event_protocol_sha256_lf"]:
        raise GateStop("Frozen event protocol changed")
    code = attestation_code(root)
    attested = json.loads((base / "test_attestation.json").read_text(encoding="utf-8"))
    if (
        not attested["passed"]
        or attested["code_sha256"] != code
        or attested["protocol_sha256"] != lf_hash(planpath)
    ):
        raise GateStop("Pre-access recovery test attestation failed")
    oldplan = json.loads(
        (root / "docs/track_b/multi_vintage_protocol.json").read_text(encoding="utf-8")
    )
    reports = [reuse(root, source_dir, y, oldplan) for y in plan["reused_vintages"]]
    census = json.loads(
        (root / "data/track_b/multivintage/reconciliation_v1/census.json").read_text(
            encoding="utf-8"
        )
    )
    # All four origination universes pass and samples freeze before any new performance access.
    start = time.monotonic()
    try:
        for vintage in plan["recovered_vintages"]:
            output = base / str(vintage)
            output.mkdir(exist_ok=True)
            pre = json.loads(
                (root / f"data/track_b/multivintage/manifests/preflight_{vintage}.json").read_text(
                    encoding="utf-8"
                )
            )
            identity = dict(
                source_sha256=pre["sha256"],
                parser_version=VERSION,
                protocol_sha256=lf_hash(planpath),
                code_sha256=code,
                eligibility_version=plan["eligibility_version"],
            )
            if any(
                r["source_sha256"] != pre["sha256"] for r in registry if r["vintage"] == vintage
            ):
                raise GateStop("Quarantine source identity mismatch")
            with VintageReader(
                source_dir / f"historical_data_{vintage}.zip",
                root,
                plan["limits"],
                pre["sha256"],
                vintage,
            ) as reader:
                sample(
                    root,
                    reader,
                    output,
                    plan,
                    identity,
                    registry,
                    next(c for c in census if c["vintage"] == vintage),
                )
        for vintage in plan["recovered_vintages"]:
            output = base / str(vintage)
            if (output / "completed.json").exists():
                old = json.loads((output / "completed.json").read_text(encoding="utf-8"))
                expected = json.loads((output / "sample_manifest.json").read_text(encoding="utf-8"))
                checkpoint_valid(old, expected)
                for name, h in old["output_hashes"].items():
                    if digest(output / name) != h:
                        raise GateStop("Completed recovery output changed")
                reports.append(old)
                continue
            pre = json.loads(
                (root / f"data/track_b/multivintage/manifests/preflight_{vintage}.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads((output / "sample_manifest.json").read_text(encoding="utf-8"))
            ids = (output / "selected_ids.txt").read_text(encoding="utf-8").splitlines()
            vintage_start = time.monotonic()
            with VintageReader(
                source_dir / f"historical_data_{vintage}.zip",
                root,
                plan["limits"],
                pre["sha256"],
                vintage,
            ) as reader:
                reader.freeze(ids)
                cache, counts = retain(
                    StructuralPerformanceReader(reader), output, ids, manifest, plan
                )
                result = audit(root, output, cache, ids, vintage, plan)
                if result["selected_ids_missing"] or result["integrity"].get(
                    "duplicate_facility_months", 0
                ):
                    raise GateStop("Selected linkage or duplicate-month audit failed")
                result.update(
                    **manifest,
                    source=pre,
                    status="READY_WITH_LIMITATIONS",
                    observed_layout=dict(
                        origination_columns=31,
                        performance_columns=35,
                        delimiter="|",
                        encoding="UTF8 compatible",
                        field_order="Unchanged pinned R47 adapter; selected histories typed",
                    ),
                    scan=counts,
                    seconds=time.monotonic() - vintage_start,
                    peak_memory_bytes=peak_memory_bytes(),
                    temporary_disk_peak=reader.temp_bytes_peak,
                    output_bytes=sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
                    output_hashes={
                        n: digest(output / n) for n in ["origination.csv", "monthly.csv"]
                    },
                )
                write_json(output / "completed.json", result, True)
                reports.append(result)
                print(f"{vintage}: audit completed, {result['rows']:,} rows", flush=True)
    except (ValueError, sqlite3.Error) as exc:
        failure = dict(
            vintage=vintage,
            status="DATA_QUALITY_STOP",
            error=str(exc),
            seconds=time.monotonic() - start,
            stage="performance" if (output / "performance.sqlite").exists() else "origination",
        )
        write_json(base / "failure.json", failure)
        reports.append(failure)
        print(f"STOP {vintage}: {exc}", flush=True)
    write_json(base / "results.json", sorted(reports, key=lambda r: r["vintage"]))
    return reports
