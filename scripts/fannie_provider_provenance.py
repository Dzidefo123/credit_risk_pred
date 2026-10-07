"""Read-only Task 13T source hashes and governance; never open loan member bodies."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "provider_source_verification", ROOT / "scripts/fannie_source_provenance.py"
)
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)
digest = source.digest
verify_archive = source.verify_archive
verify_preservation = source.verify_preservation


def verify_documents(source_dir, registry):
    documents = registry["documents"]
    if len(documents) != 3 or len({d["document_id"] for d in documents}) != 3:
        raise ValueError("Exactly three distinct provider documents required")
    for document in documents:
        name = document["filename"]
        if Path(name).name != name:
            raise ValueError("Document filename must exclude private paths")
        path = source_dir / name
        if path.stat().st_size != document["byte_size"] or digest(path) != document["sha256"]:
            raise ValueError("Provider source document changed: " + name)
    return {"status": "PASSED", "documents": 3, "source_documents_modified": False}


def assessment_contract(assessment, response):
    if response["archive_specific_direct_confirmation"]:
        raise ValueError("Resource-directed response is not archive-specific confirmation")
    if assessment["evidence_type"] != "PROVIDER_DIRECTED_DOCUMENTARY_INFERENCE":
        raise ValueError("Unsupported direct-confirmation claim")
    if assessment["exact_archive_binding"] != "PARTIALLY_SUPPORTED":
        raise ValueError("Exact provider release not established by current evidence")
    if assessment["acceptance_status"] != "ACCEPTANCE_VERSION_UNRESOLVED":
        raise ValueError("Historical terms acceptance version remains unresolved")
    if assessment["publication_review_status"] != "PUBLICATION_REVIEW_REQUIRED":
        raise ValueError("No publication permission conclusion established")
    if (
        assessment["task13a_authorized"]
        or assessment["protocol_status"] != "DRAFT_NOT_YET_AUTHORIZED"
    ):
        raise ValueError("Task 13A activation prohibited")
    if assessment["readiness_matrix"]["Task 13A authorization"] != "FAIL":
        raise ValueError("Authorization/readiness inconsistency")
    if assessment["readiness_matrix"]["Exact archive release binding"] == "PASS":
        raise ValueError("No exact archive release proof")
    if any(assessment["scope"].values()):
        raise ValueError("Provenance-only scientific scope violated")
    return {"status": "PASSED", "task13a_authorized": False}


def verify(root, source_dir, archive):
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    return {
        "documents": verify_documents(
            source_dir, read("docs/track_b/fannie_provider_document_registry.json")
        ),
        "archive": verify_archive(
            archive, read("reports/track_b/fannie_release_archive_structure.json")
        ),
        "preservation": verify_preservation(
            root, read("docs/track_b/fannie_provider_preservation_manifest.json")
        ),
        "assessment": assessment_contract(
            read("reports/track_b/fannie_provenance_final_assessment.json"),
            read("reports/track_b/fannie_provider_response_evidence.json"),
        ),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(ROOT, args.source_dir, args.archive)))
