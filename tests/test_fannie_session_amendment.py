"""Later portal observations amend, rather than rewrite, frozen Task 13T evidence."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_current_acceptance_is_qualified_not_historical_consent():
    a = read("reports/track_b/fannie_provenance_session_assessment.json")
    assert a["acceptance_status"] == "ACCEPTANCE_EVIDENCED_WITH_MATERIAL_LIMITATIONS"
    assert a["historical_acceptance_version_status"] == "ACCEPTANCE_VERSION_UNRESOLVED"
    assert a["portal_notice"]["observed_directly"]
    assert a["portal_notice"]["use_deemed_review_and_agreement"]
    assert not a["task13a_authorized"]


def test_two_fresh_resources_and_prior_terms_separated():
    r = read("docs/track_b/fannie_provider_document_session_amendment.json")
    old = read("docs/track_b/fannie_provider_document_registry.json")
    assert len(r["fresh_authenticated_resource_downloads"]) == 2
    for item in r["fresh_authenticated_resource_downloads"]:
        original = next(d for d in old["documents"] if d["filename"] == item["filename"])
        assert item["sha256"] == original["sha256"]
        assert item["bytes"] == original["byte_size"]
    assert not r["third_resource"]["fresh_document_download_succeeded"]
    assert r["third_resource"]["current_view_version"] is None
    assert r["third_resource"]["current_view_effective_date"] is None
    assert "NOT_VERIFIED" in r["third_resource"]["prior_user_document"]["status"]


def test_original_task13t_and_task14_public_evidence_immutable():
    m = read("docs/track_b/fannie_session_preservation_manifest.json")
    assert len(m["public_lf_hashes"]) == 542
    spec = importlib.util.spec_from_file_location(
        "paper_compatibility", ROOT / "scripts/check_literature.py"
    )
    compatibility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compatibility)
    for name, expected in m["public_lf_hashes"].items():
        assert compatibility.preserved_digest(ROOT, name) == expected, name


def test_width_does_not_identify_archive_release():
    a = read("reports/track_b/fannie_provenance_session_assessment.json")
    assert not a["direct_archive_confirmation"]
    assert a["exact_archive_binding"] == "PARTIALLY_SUPPORTED"
    assert a["physical_113_status"] == "YES_BY_DOCUMENTED_VERSION_RECONSTRUCTION"
    assert a["no_position_114_padding"]
    assert a["protocol_status"] == "DRAFT_NOT_YET_AUTHORIZED"
    assert a["readiness_matrix"]["Task 13A authorization"] == "FAIL"


@pytest.mark.parametrize(
    "flag",
    [
        "outcome_access",
        "new_quarterly_archive",
        "sample_creation",
        "model_fitting",
        "email_sent",
        "task13a_started",
        "task15_started",
    ],
)
def test_scientific_and_communication_boundary(flag):
    a = read("reports/track_b/fannie_provenance_session_assessment.json")
    assert a["scope"][flag] is False


def test_session_evidence_contains_no_authentication_details():
    a = read("reports/track_b/fannie_provenance_session_assessment.json")
    text = json.dumps(a)
    assert "auth.pingone" not in text
    assert "state=" not in text
    assert "@" not in text
    assert "C:\\Users\\" not in text
    assert "cookie" not in text.lower()
    r = read("docs/track_b/fannie_provider_document_session_amendment.json")
    assert len(r["private_screenshot_hashes"]) == 3
    assert not r["raw_documents_or_screenshots_committed"]
