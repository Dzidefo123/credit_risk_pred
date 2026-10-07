"""Task 13T uses documentation metadata and synthetic fixtures, never loan outcomes."""

import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "provider_provenance", ROOT / "scripts/fannie_provider_provenance.py"
)
provider = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provider)
RELEASE_SPEC = importlib.util.spec_from_file_location(
    "provider_release", ROOT / "scripts/fannie_release_layout.py"
)
release = importlib.util.module_from_spec(RELEASE_SPEC)
RELEASE_SPEC.loader.exec_module(release)


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


@pytest.fixture
def assessment():
    return read("reports/track_b/fannie_provenance_final_assessment.json")


@pytest.fixture
def response():
    return read("reports/track_b/fannie_provider_response_evidence.json")


def test_three_source_document_identities():
    registry = read("docs/track_b/fannie_provider_document_registry.json")
    assert len(registry["documents"]) == 3
    assert {d["document_id"] for d in registry["documents"]} == {"layout", "calendar", "terms"}
    assert all(len(d["sha256"]) == 64 and d["byte_size"] > 0 for d in registry["documents"])
    layout = next(d for d in registry["documents"] if d["document_id"] == "layout")
    assert layout["sha256"] == "debc2a9ae2573ae73b57e4e52d99ca6825b90062208ffcab792a95190ddc43e5"
    assert not registry["raw_source_documents_committed"]


def test_document_hash_mutation_is_detected(tmp_path):
    docs = []
    for i in range(3):
        path = tmp_path / f"document-{i}.txt"
        body = f"synthetic documentation {i}".encode()
        path.write_bytes(body)
        docs.append(
            {
                "document_id": str(i),
                "filename": path.name,
                "sha256": hashlib.sha256(body).hexdigest(),
                "byte_size": len(body),
            }
        )
    assert provider.verify_documents(tmp_path, {"documents": docs})["status"] == "PASSED"
    (tmp_path / docs[0]["filename"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="document changed"):
        provider.verify_documents(tmp_path, {"documents": docs})
    with pytest.raises(ValueError, match="Exactly three"):
        provider.verify_documents(tmp_path, {"documents": docs[:2]})


def test_provider_body_recorded_without_identity(response):
    assert response["responding_organization"] == "Investor Relations"
    assert response["response_date"] is None
    assert len(response["provider_directed_resources"]) == 3
    assert "CRT Terms" in response["provider_response_transcription"]
    assert not any(response["sensitive_information_exclusion"].values())
    assert "Hello" not in response["provider_response_transcription"]
    assert "Washington" not in response["provider_response_transcription"]


def test_direct_confirmation_and_inference_separated(assessment, response):
    assert not response["archive_specific_direct_confirmation"]
    assert assessment["evidence_type"] == "PROVIDER_DIRECTED_DOCUMENTARY_INFERENCE"
    assert assessment["provider_release"] is None
    assert provider.assessment_contract(assessment, response)["status"] == "PASSED"
    response["archive_specific_direct_confirmation"] = True
    with pytest.raises(ValueError, match="not archive-specific"):
        provider.assessment_contract(assessment, response)


def test_exact_archive_identity_unchanged(assessment):
    assert assessment["archive_identity"] == read(
        "reports/track_b/fannie_release_archive_structure.json"
    )
    assert assessment["archive_identity"]["archive_sha256"] == (
        "09735f1dd50b3a3046117a6cdab12fe003d7aebd43f9a058bdeafbd24789c415"
    )


def test_113_structure_has_no_padding(assessment):
    registry = read("docs/track_b/fannie_release_layout_registry.json")
    identifier = assessment["layout_evidence"]["candidate_113_layout_id"]
    opaque = b"|".join([b"opaque"] * 113) + b"\n"
    assert release.validate_structure(opaque, identifier, registry)["field_count"] == 113
    with pytest.raises(ValueError, match="no padding"):
        release.validate_structure(opaque.rstrip(b"\n") + b"|opaque\n", identifier, registry)
    assert assessment["layout_evidence"]["no_position_114_padding"]
    assert assessment["physical_113_status"] == "YES_BY_DOCUMENTED_VERSION_RECONSTRUCTION"


def test_calendar_not_archive_release_proof(assessment):
    calendar = assessment["refresh_calendar_evidence"]
    assert calendar["frequency"] == "Quarterly"
    assert calendar["exact_2026_publication_dates"] is None
    assert not calendar["archive_specific_receipt"]
    assert calendar["variation_warning"]
    assert not assessment["local_download_timing"]["provider_publication_proof"]
    assert not assessment["local_download_timing"]["timestamps_modified"]
    assert assessment["exact_archive_binding"] == "PARTIALLY_SUPPORTED"


def test_crt_terms_not_sflpd_license_or_acceptance_proof(assessment):
    terms = assessment["terms_evidence"]
    assert terms["dated_as_of"] == "2017-10-20"
    assert not terms["same_as_prior_sflpd_document"]
    assert not terms["historical_accepted_version_bound"]
    assert not terms["publication_permission_determined"]
    assert assessment["acceptance_status"] == "ACCEPTANCE_VERSION_UNRESOLVED"
    assert not assessment["authenticated_portal_evidence"]["acceptance_receipt_present"]


@pytest.mark.parametrize(
    "flag",
    [
        "outcomes_inspected",
        "sample_created",
        "new_quarterly_archives_acquired",
        "models_fitted",
        "loan_values_reinterpreted",
        "holdout_consumed",
        "model_artifacts_regenerated",
        "task14_modified",
    ],
)
def test_no_new_research_or_paper_mutation(assessment, response, flag):
    assert not assessment["scope"][flag]
    assessment["scope"][flag] = True
    with pytest.raises(ValueError, match="scope violated"):
        provider.assessment_contract(assessment, response)


def test_terms_and_public_evidence_are_non_sensitive():
    for name in [
        "docs/track_b/fannie_provider_document_registry.json",
        "reports/track_b/fannie_provider_response_evidence.json",
        "reports/track_b/fannie_provenance_final_assessment.json",
        "scripts/fannie_provider_provenance.py",
        "tests/test_fannie_provider_provenance.py",
    ]:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert release.task13.public_path(name)
        assert release.task13.redact(text) == text.rstrip("\n"), name
        assert "C:\\Users\\" not in text
        if name.endswith(".json"):
            assert "WS_TRACKING" not in text
    assert (
        subprocess.run(
            ["git", "check-ignore", "data/track_b/fannie/provider_response/crt terms.txt"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )


def test_authorization_matches_failed_acceptance_gate(assessment, response):
    assert assessment["readiness_matrix"]["Acceptance provenance"] == "FAIL"
    assert assessment["readiness_matrix"]["Task 13A authorization"] == "FAIL"
    assert assessment["protocol_status"] == "DRAFT_NOT_YET_AUTHORIZED"
    assert not assessment["task13a_authorized"]
    changed = copy.deepcopy(assessment)
    changed["task13a_authorized"] = True
    with pytest.raises(ValueError, match="activation prohibited"):
        provider.assessment_contract(changed, response)


def test_task14_and_all_prior_public_files_preserved():
    manifest = read("docs/track_b/fannie_provider_preservation_manifest.json")
    assert len(manifest["public_lf_hashes"]) == 534
    for name, expected in manifest["public_lf_hashes"].items():
        assert provider.preserved_public_digest(ROOT / name) == expected, name
    assert len([n for n in manifest["public_lf_hashes"] if n.startswith("docs/paper/")]) == 20


def test_seven_vintage_scope_is_not_download_coverage(assessment):
    assert [v["year"] for v in assessment["seven_vintage_availability"]] == [
        2006,
        2008,
        2010,
        2014,
        2018,
        2020,
        2022,
    ]
    assert all(
        not v["downloadable_quarterly_coverage_verified"]
        for v in assessment["seven_vintage_availability"]
    )
