"""Verify Task 14 paper claim bindings and terminology without empirical calculation."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Task 16V permits only these additive literature files; frozen hashes remain pinned.
POST_FREEZE_ALLOWED_FILES = {
    "docs/paper/literature/" + name
    for name in (
        "task16_reference_specs.json",
        "crossref_metadata.json",
        "primary_verification_notes.json",
        "reference_registry.json",
        "citation_gap_resolution.json",
        "novelty_dimensions.json",
        "task16_preservation_manifest.json",
    )
}


def check_paper_files(root):
    for path in (root / "docs/paper").iterdir():
        if path.is_dir():
            if path != root / "docs/paper/literature" or path.is_symlink():
                raise ValueError("Unreviewed paper directory")
            for addition in path.iterdir():
                name = addition.relative_to(root).as_posix()
                if (
                    name not in POST_FREEZE_ALLOWED_FILES
                    or not addition.is_file()
                    or addition.is_symlink()
                ):
                    raise ValueError("Unreviewed post-freeze paper file")
                check_public_text(addition.read_text(encoding="utf-8"))
            continue
        if path.suffix not in {".json", ".md"}:
            raise ValueError("Unreviewed paper file type")
        check_public_text(path.read_text(encoding="utf-8"))


def check_frozen_hash(root, name, expected):
    actual = digest(root / name)
    if actual == expected:
        return
    if name == "scripts/check_paper_evidence.py":
        amendments = read(root, "reports/paper/task16_compatibility_amendment.json")
        approved = amendments["approved_code_hashes"][name]
        if expected == approved["original_sha256_lf"] and actual == approved["amended_sha256_lf"]:
            return
    raise ValueError("Paper evidence freeze changed: " + name)


CLASSES = {
    "SUPPORTED_EMPIRICAL",
    "SUPPORTED_METHODOLOGICAL",
    "EXPLORATORY_DIAGNOSTIC",
    "EXPLORATORY_HYPOTHESIS",
    "PROPOSED_EXTERNAL_REPLICATION",
    "LIMITATION",
    "UNSUPPORTED",
    "PROHIBITED",
}
VERIFIED = {"VERIFIED_EXACT", "VERIFIED_WITH_QUALIFICATION"}
REQUIRED = {
    "claim_id",
    "short_name",
    "claim_class",
    "claim_strength",
    "candidate_claim",
    "allowed_language",
    "prohibited_language",
    "population",
    "dataset",
    "provider",
    "origination_vintage_scope",
    "observation_window",
    "development_window",
    "purge_window",
    "evaluation_window",
    "unit_of_analysis",
    "independence_unit",
    "event_definition",
    "competing_event_definition",
    "censoring_definition",
    "model_or_comparison",
    "metric",
    "point_estimate",
    "uncertainty",
    "statistical_method",
    "source_artifacts",
    "source_commit",
    "frozen_status",
    "exploratory_status",
    "known_limitations",
    "figure_candidate",
    "table_candidate",
    "manuscript_sections",
    "verification_status",
}


def read(root, name):
    return json.loads((root / name).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def pointer(value, address):
    for token in address.strip("/").split("/") if address else []:
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def check_ref(root, ref):
    name = ref["path"]
    if not name.startswith(("docs/track_b/", "reports/track_b/")) or ".." in Path(name).parts:
        raise ValueError("Evidence reference outside public frozen evidence")
    if digest(root / name) != ref["sha256_lf"]:
        raise ValueError("Frozen evidence hash mismatch: " + name)
    return pointer(read(root, name), ref["json_pointer"])


def check_claims(root, claims):
    ids = [c["claim_id"] for c in claims]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate claim ID")
    for claim in claims:
        if not REQUIRED <= claim.keys() or claim["claim_class"] not in CLASSES:
            raise ValueError("Incomplete claim schema or taxonomy")
        if claim["claim_strength"] not in {"PRIMARY", "SECONDARY", "CONTEXTUAL"}:
            raise ValueError("Invalid claim strength")
        if claim["claim_strength"] == "PRIMARY":
            if not claim["source_artifacts"] or claim["verification_status"] not in VERIFIED:
                raise ValueError("Primary claim lacks verified evidence")
            if claim["claim_class"] not in {"SUPPORTED_EMPIRICAL", "SUPPORTED_METHODOLOGICAL"}:
                raise ValueError("Exploratory/prohibited primary claim")
        if claim["quantitative"] and not claim["metric"]:
            raise ValueError("Quantitative claim missing metric")
        roles = {}
        for ref in claim["source_artifacts"]:
            roles[ref["role"]] = check_ref(root, ref)
        if claim["quantitative"]:
            if roles.get("point_estimate") != claim["point_estimate"]:
                raise ValueError("Point estimate differs from frozen evidence")
            if (
                claim["uncertainty"] is not None
                and roles.get("uncertainty") != claim["uncertainty"]
            ):
                raise ValueError("Uncertainty differs from frozen evidence")
            if claim["verification_status"] not in VERIFIED:
                raise ValueError("Quantitative claim is not verified")
        if claim["task15_admission"] != "BLOCKED":
            language = claim["allowed_language"].lower()
            if re.search(
                r"\b(causes?|caused|causal mechanism established|"
                r"regulator validated|audit approved)\b",
                language,
            ):
                raise ValueError("Unsupported causal/regulatory assertion")
            if re.search(
                r"\b(?:is|an?|validated)\s+(?:an?\s+)?(?:ifrs\s?9|irb|"
                r"regulatory\s+(?:pd|lgd|ead)|production bank)\s+model\b",
                language,
            ):
                raise ValueError("Unsupported regulatory-model assertion")
            if "fannie" in language and re.search(
                r"\b(sealed|pre-registered|registered|replicated|externally validated)\b", language
            ):
                raise ValueError("Invalid Fannie terminology")
            if re.search(
                r"(external validation|external replication|cross-provider transport) "
                r"(completed|demonstrated)",
                language,
            ):
                raise ValueError("Invalid completed external assertion")
        if claim["provider"] == "Fannie Mae" and claim["quantitative"]:
            raise ValueError("Fannie empirical result prohibited")
        if claim["experiment_id"] == "B12" and claim["claim_strength"] == "PRIMARY":
            raise ValueError("Refinancing primary confirmatory claim prohibited")
        if claim["experiment_id"] == "B12" and claim["exploratory_status"] != "EXPLORATORY_ONLY":
            raise ValueError("Refinancing exploratory label missing")
    return set(ids)


def check_public_text(text):
    patterns = [
        r"(?<![\d.])\b\d{12}\b(?![\d.])",
        r"\bF\d{2}Q[1-4]\d{7}\b",
        r"C:\\\\Users\\\\",
        r"(?i)authorization\s*:\s*bearer\s+\S+",
        r"(?i)(?:password|api_key|session_token)\s*[=:]\s*[\"'][^\"']+[\"']",
        r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----",
    ]
    if any(re.search(pattern, text) for pattern in patterns):
        raise ValueError("Possible private record/identifier/credential exposure")
    if any(line.count("|") >= 25 for line in text.splitlines()):
        raise ValueError("Possible raw loan row")


def verify(root=ROOT):
    registry = read(root, "docs/paper/track_b_claim_evidence_registry.json")
    ids = check_claims(root, registry["claims"])
    manifest = read(root, "docs/paper/paper_evidence_manifest.json")
    for group in [
        "source_artifact_hashes_lf",
        "paper_artifact_hashes_lf",
        "audit_code_hashes_lf",
        "audit_report_hashes_lf",
    ]:
        for name, expected in manifest.get(group, {}).items():
            check_frozen_hash(root, name, expected)
    for filename, key in [
        ("figure_registry.json", "figures"),
        ("table_registry.json", "tables"),
        ("experiment_design_freeze.json", "experiments"),
        ("manuscript_section_map.json", "sections"),
        ("headline_claim_audit.json", "headlines"),
    ]:
        for entry in read(root, "docs/paper/" + filename)[key]:
            if not set(entry["claim_ids"]) <= ids:
                raise ValueError("Unresolved claim reference")
    facts = read(root, "docs/paper/dataset_facts_registry.json")
    for fact in facts["facts"]:
        if check_ref(root, fact["source_artifacts"][0]) != fact["value"]:
            raise ValueError("Dataset fact differs from source")
    fannie = read(root, "reports/track_b/fannie_source_provenance_closure.json")
    if (
        fannie["protocol_status"] != "DRAFT_NOT_YET_AUTHORIZED"
        or fannie["scope"]["task13a_authorized"]
    ):
        raise ValueError("Fannie governance boundary changed")
    boundaries = read(root, "docs/paper/generalization_boundaries.json")
    if len(boundaries["boundaries"]) < 11 or len(boundaries["regulatory_prohibited_claims"]) != 8:
        raise ValueError("Generalization/regulatory boundaries missing")
    check_paper_files(root)
    if manifest["new_empirical_calculations"]:
        raise ValueError("New empirical calculations prohibited")
    conflicts = read(root, "docs/paper/evidence_conflicts.json")
    if conflicts["unresolved_primary_conflicts"]:
        raise ValueError("Primary evidence conflicts require reconciliation")
    return {
        "status": "PASSED",
        "claims": len(ids),
        "dataset_facts": len(facts["facts"]),
        "new_empirical_calculations": False,
        "task15_scientific_evidence_gate": "READY_WITH_MATERIAL_LIMITATIONS",
        "publication_review": "REQUIRED_BEFORE_PUBLICATION",
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
