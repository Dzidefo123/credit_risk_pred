"""Sequential bounded acquisition with per-vintage sample and quarter checkpoints."""

import hashlib
import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from credit_risk.track_b.data.annual import GateStop, peak_memory_bytes
from credit_risk.track_b.data.schemas import ORIGINATION, PERFORMANCE, digest, load_protocol

from .core import rank_key, set_hash, write_json
from .reader import VintageReader
from .schema import VERSION, parse


def lf_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_code(root):
    return {
        p.relative_to(root).as_posix(): lf_hash(p)
        for p in (root / "src/credit_risk/track_b/multivintage").glob("*.py")
        if p.name != "reporting.py"
    }


def checkpoint_valid(checkpoint, expected):
    for field in [
        "source_sha256",
        "parser_version",
        "protocol_sha256",
        "sample_sha256",
        "code_sha256",
    ]:
        if checkpoint.get(field) != expected.get(field):
            raise GateStop("Checkpoint identity mismatch: " + field)
    return True


def sample(root, reader, vintage, output, plan, identity):
    """Full valid origination universe on disk; digest rank has no outcome inputs."""
    frozen = output / "sample_manifest.json"
    dbpath = output / "origination.sqlite"
    if frozen.exists():
        manifest = json.loads(frozen.read_text(encoding="utf-8"))
        expected = {**identity, "sample_sha256": manifest["sample_sha256"]}
        checkpoint_valid(manifest, expected)
        ids = (output / "selected_ids.txt").read_text(encoding="utf-8").splitlines()
        if (
            set_hash(ids) != manifest["sample_sha256"]
            or digest(dbpath) != manifest["origination_sha256"]
        ):
            raise GateStop("Frozen origination/sample corrupted")
        reader.freeze(ids)
        return ids, manifest
    if dbpath.exists():
        raise GateStop("Uncompleted origination stage; explicit recovery review required")
    db = sqlite3.connect(dbpath)
    db.execute(
        "CREATE TABLE origination(loan TEXT PRIMARY KEY,rank TEXT NOT NULL,raw TEXT NOT NULL)"
    )
    counts = {}
    batch = []
    for q in range(1, 5):
        count = 0
        for line in reader.lines(q, "origination"):
            tokens = line.split("|")
            row = parse(tokens, "origination", vintage)
            loan = row["loan_id"]
            if loan[4] != str(q):
                raise GateStop("Origination quarter mismatch")
            batch.append((loan, rank_key(loan, vintage), line))
            count += 1
            if len(batch) >= 5000:
                db.executemany("INSERT INTO origination VALUES (?,?,?)", batch)
                db.commit()
                batch = []
        if batch:
            db.executemany("INSERT INTO origination VALUES (?,?,?)", batch)
            db.commit()
            batch = []
        counts[str(q)] = count
        if sum(counts.values()) > plan["limits"]["max_universe_ids"]:
            raise GateStop("Universe resource cap")
        print(f"{vintage} origination Q{q}: {count:,} IDs", flush=True)
    db.execute("CREATE INDEX ranking ON origination(rank,loan)")
    ids = [
        r[0]
        for r in db.execute(
            "SELECT loan FROM origination ORDER BY rank,loan LIMIT ?", (plan["n_per_vintage"],)
        )
    ]
    if not ids:
        raise GateStop("Empty eligible universe")
    h = hashlib.sha256()
    for n, (loan,) in enumerate(db.execute("SELECT loan FROM origination ORDER BY loan")):
        h.update(("\n" if n else "").encode() + loan.encode())
    universe_hash = h.hexdigest()
    db.close()
    (output / "selected_ids.txt").write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
    manifest = {
        **identity,
        "sample_sha256": set_hash(ids),
        "facilities": len(ids),
        "universe": sum(counts.values()),
        "universe_sha256": universe_hash,
        "origination_rows_by_quarter": counts,
        "origination_sha256": digest(dbpath),
        "namespace": f"track-b-multivintage-v1:{vintage}:ID",
        "frozen_at": datetime.now(UTC).isoformat(),
        "frozen_before_performance_access": True,
    }
    write_json(frozen, manifest, True)
    reader.freeze(ids)
    print(f"{vintage}: {len(ids):,} sample IDs frozen; {manifest['sample_sha256']}", flush=True)
    return ids, manifest


def retain(reader, output, ids, manifest, plan):
    path = output / "performance.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS performance(loan TEXT,raw TEXT,quarter INTEGER)")
    counts = []
    selected = set(ids)
    total = 0
    for q in range(1, 5):
        marker = output / f"quarter_{q}.json"
        if marker.exists():
            m = json.loads(marker.read_text(encoding="utf-8"))
            checkpoint_valid(m, manifest)
            if (
                db.execute("SELECT COUNT(*) FROM performance WHERE quarter=?", (q,)).fetchone()[0]
                != m["retained"]
            ):
                raise GateStop("Quarter cache row count mismatch")
            # Full retained-quarter content checksum, not just cardinality.
            h = hashlib.sha256()
            for (raw,) in db.execute(
                "SELECT raw FROM performance WHERE quarter=? ORDER BY rowid", (q,)
            ):
                h.update((raw + "\n").encode())
            if h.hexdigest() != m["retained_sha256"]:
                raise GateStop("Quarter cache content mismatch")
            counts.append(m)
            total += m["retained"]
            continue
        if db.execute("SELECT COUNT(*) FROM performance WHERE quarter=?", (q,)).fetchone()[0]:
            raise GateStop("Partial quarter present; recovery cannot silently overwrite")
        scanned = kept = 0
        batch = []
        h = hashlib.sha256()
        start = time.monotonic()
        for line in reader.lines(q, "performance"):
            scanned += 1
            loan = line.partition("|")[0].strip()
            if loan not in selected:
                continue
            row = parse(line.split("|"), "performance", reader.vintage)
            if row["loan_id"][4] != str(q):
                raise GateStop("Selected performance quarter mismatch")
            batch.append((loan, line, q))
            kept += 1
            total += 1
            h.update((line + "\n").encode())
            if total > plan["limits"]["max_retained_performance_rows"]:
                raise GateStop("Retained row budget; no truncation")
            if len(batch) >= 5000:
                db.executemany("INSERT INTO performance VALUES (?,?,?)", batch)
                db.commit()
                batch = []
            if kept % 100000 == 0:
                memory = peak_memory_bytes()
                if memory and memory > plan["limits"]["max_peak_memory_bytes"]:
                    raise GateStop("Memory cap")
                if path.stat().st_size > plan["limits"]["max_cache_bytes"]:
                    raise GateStop("Cache disk cap")
        if batch:
            db.executemany("INSERT INTO performance VALUES (?,?,?)", batch)
            db.commit()
        m = {
            **manifest,
            "quarter": q,
            "scanned": scanned,
            "retained": kept,
            "retained_sha256": h.hexdigest(),
            "seconds": time.monotonic() - start,
            "malformed_selected_rows": 0,
            "origination_members": None,
        }
        write_json(marker, m, True)
        counts.append(m)
        print(
            f"{reader.vintage} performance Q{q}: {scanned:,} scanned; {kept:,} retained", flush=True
        )
    db.execute("CREATE INDEX IF NOT EXISTS by_loan ON performance(loan)")
    db.commit()
    db.close()
    return path, counts


def existing_2010(root, source, output, identity, plan, reader):
    ids = (
        (root / "data/track_b/manifests/expansion_v1/selected_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    if len(ids) != 20000 or set_hash(ids) != plan["existing_2010_sample_sha256"]:
        raise GateStop("Existing2010 sample changed")
    if (
        identity["source_sha256"]
        != "a82bc0f1efffccfb494b2b33f61877428bb4a6443c1a73d44bc7ee24d77aa45d"
    ):
        raise GateStop("Existing2010 source changed")
    manifest = {
        **identity,
        "sample_sha256": set_hash(ids),
        "facilities": 20000,
        "namespace": "Existing frozen freddie_mortgage_research_v1; no redraw",
        "frozen_before_performance_access": True,
        "reused_existing_sample": True,
    }
    frozen = output / "sample_manifest.json"
    if not frozen.exists():
        write_json(frozen, manifest, True)
    else:
        checkpoint_valid(json.loads(frozen.read_text(encoding="utf-8")), manifest)
    reader.freeze(ids)
    dbpath = output / "origination.sqlite"
    if not dbpath.exists():
        db = sqlite3.connect(dbpath)
        db.execute("CREATE TABLE origination(loan TEXT PRIMARY KEY,rank TEXT,raw TEXT)")
        for q in range(1, 5):
            for line in reader.lines(q, "origination"):
                tokens = line.split("|")
                if len(tokens) != len(ORIGINATION):
                    raise GateStop("2010 origination layout changed")
                if tokens[19] in reader.sample:
                    row = parse(tokens, "origination", 2010)
                    db.execute(
                        "INSERT INTO origination VALUES (?,?,?)", (row["loan_id"], "reused", line)
                    )
        db.commit()
        db.close()
    cache = root / "data/track_b/processed/expansion_v1/retained_performance.sqlite"
    return ids, manifest, cache, [dict(reused_cache=True, scanned=0, retained=None)]


def run(root, source_dir):
    from .audit import audit

    root, source_dir = Path(root).resolve(), Path(source_dir).resolve()
    planpath = root / "docs/track_b/multi_vintage_protocol.json"
    plan = json.loads(planpath.read_text(encoding="utf-8"))
    ph = lf_hash(planpath)
    _, oldph = load_protocol(root)
    if oldph != plan["base_event_protocol_sha256_lf"]:
        raise GateStop("Event protocol changed")
    attested = json.loads(
        (root / "data/track_b/multivintage/manifests/test_attestation.json").read_text(
            encoding="utf-8"
        )
    )
    code = source_code(root)
    if (
        attested["protocol_sha256"] != ph
        or attested["code_sha256"] != code
        or not attested["passed"]
    ):
        raise GateStop("Pre-access test/code gate failed")
    reports = []
    for vintage in plan["vintages"]:
        source = source_dir / f"historical_data_{vintage}.zip"
        pre = json.loads(
            (root / f"data/track_b/multivintage/manifests/preflight_{vintage}.json").read_text(
                encoding="utf-8"
            )
        )
        output = root / f"data/track_b/multivintage/processed/v1/{vintage}"
        output.mkdir(parents=True, exist_ok=True)
        identity = dict(
            source_sha256=pre["sha256"],
            parser_version=VERSION,
            protocol_sha256=ph,
            code_sha256=code,
        )
        done = output / "completed.json"
        if done.exists():
            old = json.loads(done.read_text(encoding="utf-8"))
            checkpoint_valid(old, {**identity, "sample_sha256": old["sample_sha256"]})
            for name, h in old["output_hashes"].items():
                if digest(output / name) != h:
                    raise GateStop("Completed output modified")
            reports.append(old)
            print(f"{vintage}: reused verified completed checkpoint", flush=True)
            continue
        start = time.monotonic()
        try:
            with VintageReader(source, root, plan["limits"], pre["sha256"], vintage) as reader:
                if vintage == 2010:
                    ids, manifest, cache, counts = existing_2010(
                        root, source, output, identity, plan, reader
                    )
                else:
                    ids, manifest = sample(root, reader, vintage, output, plan, identity)
                    cache, counts = retain(reader, output, ids, manifest, plan)
                result = audit(root, output, cache, ids, vintage, plan)
                result.update(
                    **manifest,
                    source=pre,
                    status="READY_WITH_LIMITATIONS",
                    observed_layout=dict(
                        delimiter="|",
                        encoding="UTF-8 compatible",
                        origination_columns=len(ORIGINATION),
                        performance_columns=len(PERFORMANCE),
                        field_order="Pinned R47 positions; retained rows typed; no shifts",
                    ),
                    scan=counts,
                    seconds=time.monotonic() - start,
                    peak_memory_bytes=peak_memory_bytes(),
                    temporary_disk_peak=reader.temp_bytes_peak,
                    output_bytes=sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
                    output_hashes={
                        n: digest(output / n) for n in ["origination.csv", "monthly.csv"]
                    },
                )
                write_json(done, result, True)
                reports.append(result)
                print(f"{vintage} completed: {result['rows']:,} retained rows", flush=True)
        except (ValueError, sqlite3.Error) as exc:
            result = dict(
                vintage=vintage,
                status="SCHEMA_INCOMPATIBLE" if "field_count" in str(exc) else "DATA_QUALITY_STOP",
                error=str(exc),
                source=pre,
                seconds=time.monotonic() - start,
            )
            write_json(output / "failure.json", result)
            reports.append(result)
            print(f"{vintage} STOP: {exc}", flush=True)
    write_json(root / "data/track_b/multivintage/manifests/results.json", reports)
    return reports
