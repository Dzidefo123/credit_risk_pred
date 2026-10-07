"""Task13 structural inspection only; no acquisition, event extraction or modeling."""

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_member(name):
    """Only the authorized quarter's documented merged file, never extract paths."""
    return bool(re.fullmatch(r"2010Q1\.csv", name))


def structural_line(line, maximum=65536):
    """Count slots and validate encoding without decoding or returning field values."""
    if len(line) > maximum or not line.endswith(b"\n"):
        raise ValueError("Invalid or oversized record boundary; values withheld")
    try:
        line.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Invalid UTF-8; values withheld") from None
    if b"\x00" in line or b"|" not in line:
        raise ValueError("Invalid delimiter/encoding; values withheld")
    return {
        "field_count": line.rstrip(b"\r\n").count(b"|") + 1,
        "line_ending": "CRLF" if line.endswith(b"\r\n") else "LF",
        "ascii": line.isascii(),
    }


def inspect_archive(path, policy):
    """Read a bounded prefix; all field contents remain opaque and are discarded."""
    if policy["status"] != "SCHEMA_ONLY_AUTHORIZED" or Path(path).name != "2010Q1.zip":
        raise ValueError("Archive outside schema-only authorization")
    limit = policy["maximum_records_per_member"]
    if not 1 <= limit <= 512 or not 1 <= policy["maximum_record_bytes"] <= 65536:
        raise ValueError("Unbounded inspection forbidden")
    result = {
        "scope": "SCHEMA_ONLY_BOUNDED_PREFIX",
        "archive_name": "2010Q1.zip",
        "archive_bytes": Path(path).stat().st_size,
        "archive_sha256": sha256(path),
        "outcome_values_interpreted": False,
        "full_member_crc_verified": False,
        "full_population_schema_validated": False,
        "members": [],
    }
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1 or not safe_member(members[0].filename):
            raise ValueError("Unrecognized archive structure; no payload opened")
        member = members[0]
        if member.flag_bits & 1:
            raise ValueError("Encrypted archive unsupported")
        counts, endings, ascii_only, checked = set(), set(), True, 0
        with archive.open(member) as stream:
            for _ in range(limit):
                line = stream.readline(policy["maximum_record_bytes"] + 1)
                if not line:
                    break
                info = structural_line(line, policy["maximum_record_bytes"])
                counts.add(info["field_count"])
                endings.add(info["line_ending"])
                ascii_only &= info["ascii"]
                checked += 1
        if not checked or len(counts) != 1:
            raise ValueError("Empty or inconsistent schema prefix; values withheld")
        result["members"].append(
            dict(
                name=member.filename,
                compressed_bytes=member.compress_size,
                uncompressed_bytes=member.file_size,
                compression_method=member.compress_type,
                zip_timestamp=list(member.date_time),
                timestamp_is_release_proof=False,
                records_structurally_checked=checked,
                field_counts=sorted(counts),
                delimiter="pipe",
                encoding="UTF-8 compatible; ASCII prefix" if ascii_only else "UTF-8 prefix",
                line_endings=sorted(endings),
                header="Not interpreted; official FAQ documents headerless files",
            )
        )
    return result


def redact(text):
    """Redact facility IDs and row-like content before any diagnostic publication."""
    semantic_tokens = {"MORTGAGE30US", "facility95CI", "Percentile95"}
    text = re.sub(r"(?<![A-Za-z0-9])[0-9]{12}(?![A-Za-z0-9])", "[REDACTED_ID]", text)
    text = re.sub(
        r"(?<![A-Za-z0-9])[A-Za-z0-9]{12}(?![A-Za-z0-9])",
        lambda match: (
            "[REDACTED_ID]"
            if match[0] not in semantic_tokens and any(c.isdigit() for c in match[0])
            else match[0]
        ),
        text,
    )
    text = re.sub(
        r'("(?:loan_id|facility_id|borrower_id)"\s*:\s*")[A-Za-z0-9]{12}("\s*[,}])',
        r"\1[REDACTED_ID]\2",
        text,
    )
    text = re.sub(r"\bF[0-9]{2}Q[1-4][A-Za-z0-9]{7}\b", "[REDACTED_ID]", text)
    return "\n".join(
        "[REDACTED_ROW]" if line.count("|") >= 20 else line for line in text.splitlines()
    )


def public_path(name):
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", name):
        return False
    return not (
        path.parts[:1] == ("data",)
        or path.suffix.lower()
        in {".zip", ".csv", ".tsv", ".txt", ".parquet", ".gz", ".pdf", ".jsonl", ".7z", ".tar"}
        or path.name.startswith(".env")
    )


def release_adapter(release, count):
    """Only the pinned current documentation is validated; never guess a legacy layout."""
    if release != "glossary-2026-09-10" or count != 114:
        raise ValueError("Matching release/layout documentation required")
    return "merged-114"


def event_state(delinquency, code, reporting_month, zero_month):
    """Draft semantics for synthetic tests only; never called by structural inspection."""
    valid = {"", "01", "02", "03", "06", "09", "15", "16", "96"}
    if code not in valid:
        raise ValueError("Unknown Primary termination code; values withheld")
    if (code and zero_month != reporting_month) or (not code and zero_month):
        return "AMBIGUOUS_EXIT"
    severe = bool(re.fullmatch(r"[0-9]{2}", delinquency)) and int(delinquency) >= 3
    if severe and code == "01":
        return "AMBIGUOUS_EXIT"
    if severe or code in {"02", "03", "09"}:
        return "DEFAULT_PROXY"
    if code == "01":
        return "PAYOFF_OR_MATURITY"
    if code in {"06", "16", "96"}:
        return "ADMINISTRATIVE_EXIT"
    if code == "15" or not re.fullmatch(r"[0-9]{2}", delinquency):
        return "AMBIGUOUS_EXIT"
    return "AT_RISK"


def check_contracts(root):
    docs = root / "docs/track_b"
    crosswalk = json.loads(
        (docs / "freddie_fannie_field_crosswalk.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (docs / "fannie_external_replication_protocol.json").read_text(encoding="utf-8")
    )
    names = [x["canonical"] for x in crosswalk["fields"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate canonical field")
    classifications = set(crosswalk["classifications"])
    for field in crosswalk["fields"]:
        required = {
            "canonical",
            "freddie",
            "fannie",
            "description",
            "units",
            "frequency",
            "missing",
            "modification_behavior",
            "availability",
            "classification",
            "transformation",
            "citations",
            "notes",
        }
        if not required <= field.keys() or field["classification"] not in classifications:
            raise ValueError("Invalid crosswalk entry")
        if not field["citations"] or not field["availability"]:
            raise ValueError("Unversioned crosswalk")
        units = field["units"]
        if field["classification"] == "DIRECTLY_COMPATIBLE" and units["freddie"] != units["fannie"]:
            raise ValueError("Incompatible direct units")
        if any(not 1 <= n <= 114 for n in field["fannie"]["positions"]):
            raise ValueError("Invalid documented position")
    if protocol["status"] != "DRAFT_NOT_YET_AUTHORIZED" or protocol["outcome_access_authorized"]:
        raise ValueError("Replication is not authorized")
    if set(protocol["events"]["states"]) != {
        "AT_RISK",
        "DEFAULT_PROXY",
        "PAYOFF_OR_MATURITY",
        "ADMINISTRATIVE_EXIT",
        "AMBIGUOUS_EXIT",
    }:
        raise ValueError("Incomplete event states")
    if set(protocol["events"]["termination_mapping"]) != {
        "",
        "01",
        "02",
        "03",
        "06",
        "09",
        "15",
        "16",
        "96",
    }:
        raise ValueError("Incomplete Primary termination coverage")
    predictors = protocol["predictor_mapping"]
    allowed = {"EXACT_REPLICATION", "TRANSFORMED_REPLICATION", "PROXY_REPLICATION", "UNAVAILABLE"}
    if any(p["classification"] not in allowed for p in predictors.values()):
        raise ValueError("Invalid predictor classification")
    for model in protocol["models"].values():
        if not set(model) <= predictors.keys():
            raise ValueError("Unmapped replication predictor")
    payload = (root / protocol["mapping_file"]).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(payload).hexdigest() != protocol["mapping_sha256_lf"]:
        raise ValueError("Draft mapping hash changed")
    return {"status": "PASSED", "crosswalk_fields": len(names), "models": len(protocol["models"])}


def preservation(root):
    manifest = json.loads(
        (root / "docs/track_b/fannie_preservation_manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in manifest["public_lf_hashes"].items():
        actual = hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError("Frozen public evidence changed: " + name)
    for name, expected in manifest["private_byte_hashes"].items():
        if sha256(root / name) != expected:
            raise ValueError("Frozen private evidence changed: " + name)
    return dict(
        status="PASSED",
        public=len(manifest["public_lf_hashes"]),
        private=len(manifest["private_byte_hashes"]),
    )


def governance(root):
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    # Historical public files have separate existing governance. This gate covers Task13 outputs.
    candidates = [n for n in names if "fannie" in n.lower()]
    for name in candidates:
        if not public_path(name):
            raise ValueError("Licensed-data path prohibited")
        text = (root / name).read_text(encoding="utf-8")
        if redact(text) != text.rstrip("\n"):
            raise ValueError("Possible row/identifier exposure")
    return dict(status="PASSED", task13_paths_checked=len(candidates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["schema", "verify"])
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "schema":
        policy = json.loads(
            (root / "docs/track_b/fannie_schema_inspection_policy.json").read_text(encoding="utf-8")
        )
        result = inspect_archive(args.archive, policy)
        output = root / "reports/track_b/fannie_schema_validation.json"
    else:
        result = dict(
            contracts=check_contracts(root),
            preservation=preservation(root),
            governance=governance(root),
        )
        output = root / "reports/track_b/fannie_verification.json"
    if output.exists():
        raise ValueError("Existing evidence is immutable; do not overwrite")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
