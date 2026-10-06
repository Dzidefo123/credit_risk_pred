"""Hash immutable sources and inspect ZIP directories without reading loan payloads."""

import io
import json
import struct
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from .schemas import LAYOUT, PARSER_VERSION, digest, load_protocol


class StoredSlice(io.RawIOBase):
    """Bounded seekable view of a ZIP_STORED member; never extract or copy it."""

    def __init__(self, path, start, size):
        self.source = Path(path).open("rb")
        self.start, self.size, self.position = start, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (
            offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        )
        if position < 0:
            raise ValueError("Negative ZIP view offset")
        self.position = position
        return position

    def read(self, size=-1):
        count = min(self.size - self.position, size if size >= 0 else self.size - self.position)
        if count <= 0:
            return b""
        self.source.seek(self.start + self.position)
        result = self.source.read(count)
        self.position += len(result)
        return result

    def close(self):
        self.source.close()
        super().close()


def member_info(info):
    return {
        "name": info.filename,
        "compressed_bytes": info.compress_size,
        "uncompressed_bytes": info.file_size,
        "crc32": f"{info.CRC:08x}",
        "compression_type": info.compress_type,
        "encrypted": bool(info.flag_bits & 1),
        "zip_member_timestamp": list(info.date_time),
        "timestamp_is_not_release_or_acquisition_proof": True,
    }


def inspect_zip(source):
    source = Path(source)
    # Root provenance is computed before any ZIP-directory inspection.
    original_hash = digest(source)
    before = source.stat()
    inventory = []
    with zipfile.ZipFile(source) as outer:
        for info in outer.infolist():
            item = member_info(info)
            if info.filename.endswith(".zip") and info.compress_type == zipfile.ZIP_STORED:
                with source.open("rb") as handle:
                    handle.seek(info.header_offset)
                    header = handle.read(30)
                if header[:4] != b"PK\x03\x04":
                    raise ValueError("Invalid nested ZIP local header")
                name_length, extra_length = struct.unpack_from("<HH", header, 26)
                start = info.header_offset + 30 + name_length + extra_length
                with (
                    StoredSlice(source, start, info.file_size) as view,
                    zipfile.ZipFile(view) as inner,
                ):
                    item["members"] = [member_info(child) for child in inner.infolist()]
                item["nested_inspection"] = "central directory only; no loan-member payloads"
            elif info.filename.endswith(".zip"):
                item["nested_inspection"] = (
                    "not inspected: compressed wrapper unsupported without separate review"
                )
            inventory.append(item)
    after_hash = digest(source)
    if original_hash != after_hash or before.st_size != source.stat().st_size:
        raise ValueError("Original source changed during metadata inspection")
    return {
        "original_filename": source.name,
        "byte_size": before.st_size,
        "sha256": original_hash,
        "source_unchanged": True,
        "member_count": len(inventory),
        "archive_inventory": inventory,
        "loan_payloads_read": False,
        "records_parsed": 0,
        "extracted": False,
        "copied": False,
    }


def record_preflight(root, source, *, official_attestation=False):
    root, source = Path(root).resolve(), Path(source).resolve()
    protocol, protocol_hash = load_protocol(root)
    evidence = inspect_zip(source)
    names = {r["name"] for r in evidence["archive_inventory"]}
    nested = any("members" in r for r in evidence["archive_inventory"])
    flat_sample = (
        names == {"sample_orig_2010.txt", "sample_perf_2010.txt"} and evidence["member_count"] == 2
    )
    blockers = []
    if not flat_sample:
        blockers.append("Not the prespecified sample pair; frame/packaging amendment required")
    if evidence["byte_size"] > 200 * 1024**2:
        blockers.append(
            "Original ZIP exceeds current 200 MiB engineering budget; no automatic budget increase"
        )
    if not official_attestation:
        blockers.append("Official acquisition attestation missing")
    uncompressed = (
        sum(
            c["uncompressed_bytes"]
            for r in evidence["archive_inventory"]
            for c in r.get("members", [])
        )
        if nested
        else sum(r["uncompressed_bytes"] for r in evidence["archive_inventory"])
    )
    if uncompressed > 2 * 1024**3:
        blockers.append("Underlying payload exceeds current 2 GiB expanded-data budget")
    manifest = {
        **evidence,
        "schema_version": 1,
        "official_dataset": "Freddie Mac Single-Family Loan-Level Dataset",
        "dataset_family": "Standard Dataset (user attestation; records not checked)",
        "vintage": 2010,
        "acquisition_source": "Authenticated Freddie Mac/Clarity download (user attestation)"
        if official_attestation
        else "unverified",
        "acquisition_date": None,
        "acquisition_date_status": "Retrieval time unknown; ZIP/filesystem times not substituted",
        "acquisition_reported_on": datetime.now(UTC).date().isoformat(),
        "inventory_recorded_at": datetime.now(UTC).isoformat(),
        "original_local_source": str(source),
        "official_source_reference": "https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset",
        "source_release": None,
        "release_verified": False,
        "expected_layout": LAYOUT,
        "parser_version": PARSER_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256_lf": protocol_hash,
        "redistribution_status": "Raw redistribution not authorized; original retained privately",
        "apparent_structure": "annual nested quarter archives" if nested else "flat ZIP",
        "underlying_member_uncompressed_bytes": uncompressed,
        "status": "ACQUIRED_PREFLIGHT_STOPPED" if blockers else "ACQUIRED_METADATA_ONLY",
        "blockers": blockers,
        "actual_column_counts": None,
        "delimiter_verified": False,
        "encoding_verified": False,
        "loss_signs_verified_on_records": False,
    }
    private = root / "data/track_b/manifests/freddie_2010_manifest.json"
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    public = {key: value for key, value in manifest.items() if key != "original_local_source"}
    public.update(
        cohort=None,
        followup=None,
        outcomes=None,
        eligibility_universe=None,
        selected_loans=None,
        performance_rows_scanned=0,
        assumptions={
            "official_2010_sample_structure": "CONTRADICTED",
            "compatible_release_47_record_layout": "NOT TESTABLE",
            "quarter_names_indicate_2010": "CONFIRMED WITH QUALIFICATION",
            "default_followup_exposure_and_loss_semantics": "NOT TESTABLE",
        },
        feasibility="STOP — DATA/PROTOCOL INCOMPATIBLE" if blockers else "PROCEED WITH CONDITIONS",
        modeling_gate="BLOCKED: record-level schema and empirical gates not passed",
    )
    report = root / "reports/track_b"
    report.mkdir(parents=True, exist_ok=True)
    (report / "freddie_2010_data_audit.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Freddie 2010 Track B data audit",
        "## Acquisition verified; processing stopped at preflight",
        "The original authorized ZIP was hashed before inspection and rehashed afterward. "
        "Only outer/nested ZIP central-directory metadata was inspected. No loan rows were read, "
        "No extraction/copy/rename occurred. No panel or outcome statistics were generated.",
        (
            f"Original filename: `{manifest['original_filename']}`. "
            f"Byte size: **{manifest['byte_size']:,}**."
        ),
        f"Root SHA-256: `{manifest['sha256']}`.",
        "Authenticated Freddie Mac/Clarity acquisition is user-attested. Retrieval time "
        "was not supplied; ZIP/filesystem timestamps are not release or acquisition proof.",
        "## Verified member inventory",
        "| Member | Compressed bytes | Uncompressed bytes |",
        "| --- | --- | --- |",
    ]
    for item in manifest["archive_inventory"]:
        lines.append(
            f"| {item['name']} | {item['compressed_bytes']:,} | {item['uncompressed_bytes']:,} |"
        )
        for child in item.get("members", []):
            lines.append(
                f"| {item['name']} / {child['name']} | "
                f"{child['compressed_bytes']:,} | {child['uncompressed_bytes']:,} |"
            )
    lines += [
        "",
        f"Underlying text members total **{uncompressed:,} uncompressed bytes**. "
        "No documentation/readme member was found in this inventory. CRC values in JSON are ZIP "
        "integrity metadata, not independent member SHA-256 hashes; no extracted files exist.",
        "## Blockers",
        *["- " + b for b in blockers],
        "",
        "Task 1 fixed the official sample, at most 1,000 hashed IDs and a pinned layout. "
        "This annual bundle is a different input frame. Its orig_/perf_ filenames resemble current "
        "naming, but names/timestamps do not prove actual 31/35-column Release 47 compatibility. "
        "No positions, events, vintage, sample selection or budgets were changed.",
        "## Empirical gate results",
        "| Gate | Result |",
        "| --- | --- |",
        "| A - provenance/release | ZIP hashed; acquisition attested; release unverified |",
        "| B - schema | STOP: frame/budget incompatible; row layout uninspected |",
        "| C - linkage | Not evaluated |",
        "| D - time | Not evaluated |",
        "| E - outcomes | Not evaluated |",
        "| F - follow-up | Not evaluated |",
        "| G - leakage | Fixture-tested only; not exercised on real records |",
        "| H - exposure | Not evaluated |",
        "",
        "## Scientific decision",
        "**STOP — DATA/PROTOCOL INCOMPATIBLE.** This interface mismatch is not evidence "
        "of intrinsically unsuitable records. Eligible/selected loans and defaults, "
        "payoff rates, gaps, follow-up percentages and loss availability remain unknown.",
        "No Task 2 completion commit or modeling is justified. Next task: a separately approved, "
        "annual-bundle frame/parser/resource amendment before inspecting loan rows. "
        "Track A and the prespecified event/horizon logic remain unchanged.",
    ]
    import re

    text = "\n\n".join(lines) + "\n"
    text = re.sub(r"(\|[^\n]*\|)\n\n(?=\|)", r"\1\n", text)
    # Keep prose readable and the inventory/gate tables contiguous.
    (report / "FREDDIE_2010_DATA_AUDIT.md").write_text(text, encoding="utf-8")
    return public
