"""Task 13S metadata and preservation tests; no private loan values."""

import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "source_provenance", ROOT / "scripts/fannie_source_provenance.py"
)
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)
RELEASE_SPEC = importlib.util.spec_from_file_location(
    "source_release", ROOT / "scripts/fannie_release_layout.py"
)
release = importlib.util.module_from_spec(RELEASE_SPEC)
RELEASE_SPEC.loader.exec_module(release)


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


@pytest.fixture
def closure():
    return read("reports/track_b/fannie_source_provenance_closure.json")


def test_prior_archive_identity_unchanged(closure):
    assert closure["archive_structural_identity"] == read(
        "reports/track_b/fannie_release_archive_structure.json"
    )
    assert closure["archive_sha256"] == closure["archive_structural_identity"]["archive_sha256"]


def test_synthetic_archive_directory_only(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("opaque.txt", "not loan data")
    with zipfile.ZipFile(path) as archive:
        member = archive.infolist()[0]
        expected = {
            "archive_name": path.name,
            "archive_bytes": path.stat().st_size,
            "archive_sha256": source.digest(path),
            "members": [
                {
                    "name": member.filename,
                    "compressed_bytes": member.compress_size,
                    "uncompressed_bytes": member.file_size,
                    "compression_method": member.compress_type,
                    "zip_timestamp": list(member.date_time),
                }
            ],
        }

    def forbidden(*args, **kwargs):
        raise AssertionError("Member bodies must never be read")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden)
    assert source.verify_archive(path, expected)["member_bodies_opened"] is False
    expected["archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash changed"):
        source.verify_archive(path, expected)


def test_registry_and_exact_width_no_padding(closure):
    registry = read("docs/track_b/fannie_release_layout_registry.json")
    assert release.registry_contract(registry)["status"] == "PASSED"
    candidate = next(x for x in registry["layouts"] if x["layout_id"] == closure["layout_id"])
    assert candidate["ordering_sha256"] == closure["layout_ordering_sha256"]
    assert [f["position"] for f in candidate["fields"]] == list(range(1, 114))
    opaque = b"|".join([b"opaque"] * 113) + b"\n"
    assert release.validate_structure(opaque, closure["layout_id"], registry)["field_count"] == 113
    with pytest.raises(ValueError, match="no padding"):
        release.validate_structure(opaque, "fannie-sflpd-2026-10-114", registry)


def test_binding_cannot_be_inferred_from_width(closure):
    assert closure["release_identity_status"] == "RELEASE_IDENTITY_PARTIALLY_SUPPORTED"
    assert closure["layout_binding_status"] == "CANDIDATE_ONLY_NOT_ARCHIVE_BOUND"
    assert closure["provider_release"] is None
    assert not closure["layout_evidence"]["provider_release_verified"]
    assert not closure["layout_evidence"]["zip_timestamp_is_release_proof"]
    assert closure["download_date"] == "UNVERIFIED"


@pytest.mark.parametrize(
    "flag",
    [
        "loan_values_interpreted",
        "outcome_analysis",
        "additional_performance_archives_acquired",
        "sample_creation",
        "models_fitted",
        "model_artifacts_changed",
        "holdout_consumed",
        "task13a_authorized",
    ],
)
def test_research_scope_stays_closed(closure, flag):
    assert closure["scope"][flag] is False


def test_public_license_not_assumed_accepted(closure):
    license_info = closure["license"]
    assert license_info["document_status"] == "LICENSE_DOCUMENT_IDENTIFIED"
    assert not license_info["applicable_accepted_document_verified"]
    assert license_info["acceptance_evidence_status"] == "ACCEPTANCE_USER_ATTESTED_ONLY"
    assert "LICENSE_VERSION_UNRESOLVED" in license_info["additional_statuses"]
    assert license_info["legal_conclusion"] is None


def test_support_request_is_unsent_and_readiness_closed(closure):
    request = (ROOT / "reports/track_b/FANNIE_SOURCE_PROVIDER_SUPPORT_REQUEST.md").read_text()
    assert "not sent" in request
    assert closure["archive_sha256"] in request
    assert "Download date: not recorded" in request
    assert closure["readiness_matrix"]["Task 13A authorization"] == "NO"
    assert closure["readiness_matrix"]["Protocol frozen"] == "FAIL"


def test_public_artifacts_pass_existing_identifier_guard():
    names = [
        "reports/track_b/fannie_source_provenance_closure.json",
        "reports/track_b/FANNIE_SOURCE_PROVENANCE_CLOSURE.md",
        "reports/track_b/FANNIE_SOURCE_PROVIDER_SUPPORT_REQUEST.md",
        "scripts/fannie_source_provenance.py",
        "tests/test_fannie_source_provenance.py",
    ]
    for name in names:
        value = (ROOT / name).read_text(encoding="utf-8")
        assert release.task13.public_path(name)
        assert release.task13.redact(value) == value.rstrip("\n"), name
        assert "C:\\Users\\" not in value
    assert (
        subprocess.run(
            ["git", "check-ignore", "data/track_b/fannie/provenance-private.json"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )


def test_preservation_manifest_covers_all_prior_track13r_files():
    manifest = read("docs/track_b/fannie_source_preservation_manifest.json")
    baseline = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", manifest["base_commit"]], cwd=ROOT, text=True
    ).splitlines()
    assert set(baseline) == set(manifest["public_lf_hashes"])
    assert source.verify_preservation(ROOT, manifest)["status"] == "PASSED"


def test_preservation_detects_mutation(tmp_path):
    path = tmp_path / "frozen.txt"
    path.write_bytes(b"evidence\r\n")
    manifest = {
        "public_lf_hashes": {"frozen.txt": source.digest(path, normalize=True)},
        "private_byte_hashes": {},
        "git_refs": {},
    }
    path.write_bytes(b"evidence\n")
    assert source.verify_preservation(tmp_path, manifest)["status"] == "PASSED"
    path.write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="Prior public file changed"):
        source.verify_preservation(tmp_path, manifest)
