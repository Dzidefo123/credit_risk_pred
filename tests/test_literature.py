"""Regression gates for additive literature grounding; offline public evidence only."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("literature", ROOT / "scripts/check_literature.py")
literature = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(literature)


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def refs():
    return read("docs/paper/literature/reference_registry.json")["references"]


def test_complete_literature_contract():
    result = literature.verify()
    assert result["references"] == 31
    assert result["closest_papers"] == 16
    assert result["numeric_bindings"] == 123
    assert result["evidence_queries"] == []


@pytest.mark.parametrize("field", ["verification_status", "authors", "verification_sources"])
def test_unverified_or_incomplete_reference_rejected(field):
    records = copy.deepcopy(refs())
    records[0][field] = "UNVERIFIED" if field == "verification_status" else []
    with pytest.raises(ValueError):
        literature.validate_references(records)


@pytest.mark.parametrize("field", ["doi", "reference_id", "title"])
def test_duplicate_reference_rejected(field):
    records = copy.deepcopy(refs())
    records[1][field] = records[0][field]
    with pytest.raises(ValueError, match="Duplicate"):
        literature.validate_references(records)


def test_preprint_cannot_be_called_peer_reviewed():
    records = copy.deepcopy(refs())
    record = next(r for r in records if r["reference_id"] == "OPSurv2024")
    record["peer_reviewed"] = True
    with pytest.raises(ValueError, match="Preprint"):
        literature.validate_references(records)


@pytest.mark.parametrize("gap", ["CG01", "CG02", "CG03", "CG04", "CG05", "CG06"])
def test_each_citation_gap_has_scoped_resolution(gap):
    g = next(
        g
        for g in read("docs/paper/literature/citation_gap_resolution.json")["gaps"]
        if g["gap_id"] == gap
    )
    assert g["reference_ids"]
    assert g["status"] in {"PARTIALLY_RESOLVED", "RESOLVED_CONTEXTUAL_SUPPORT"}
    assert g["residual"]


@pytest.mark.parametrize(
    "key",
    [
        "Deng1996",
        "Deng2000",
        "Bu2026",
        "Peng2026",
        "Stanton1995",
        "Gneiting2007",
        "Gama2014",
        "Bianchi2026",
        "Roschewitz2025",
    ],
)
def test_required_classic_recent_and_foundational_sources(key):
    assert key in {r["reference_id"] for r in refs()}


@pytest.mark.parametrize(
    "mutation", ["number", "population", "split", "fannie", "first", "citation"]
)
def test_manuscript_corruption_rejected(mutation):
    old = (ROOT / "paper/main.md").read_text(encoding="utf-8")
    new = (ROOT / "paper/main_v0.2.md").read_text(encoding="utf-8")
    if mutation == "number":
        new = new.replace("0.09015<!-- NUM: N0001 -->", "0.90000<!-- NUM: N0001 -->")
    elif mutation == "population":
        new = new.replace("140,000<!-- NUM: N0010 -->", "141,000<!-- NUM: N0010 -->")
    elif mutation == "split":
        new = new.replace("### 5.2 Temporal separation", "### 5.2 Random separation")
    elif mutation == "fannie":
        new = new.replace("DRAFT_NOT_YET_AUTHORIZED", "AUTHORIZED")
    elif mutation == "first":
        new += "\nThis is the first paper to model mortgage competing risks.\n"
    else:
        new += "\n[@Fabricated2026]\n"
    with pytest.raises(ValueError):
        literature.validate_manuscript(old, new, {r["reference_id"] for r in refs()})


def test_bu_unknowns_are_not_absence_claims():
    rows = read("docs/paper/closest_paper_matrix.json")["papers"]
    row = next(r for r in rows if r["paper"] == "Bu2026")
    for field in [
        "PIT_macro",
        "revision_aware",
        "temporal_holdout",
        "proper_scores",
        "calibration",
        "distribution_shift_analysis",
    ]:
        assert row[field] == "UNKNOWN"


def test_empirical_claims_do_not_use_literature_as_result_evidence():
    for c in read("docs/paper/claim_literature_map.json")["claims"]:
        if c["claim_class"] == "SUPPORTED_EMPIRICAL":
            assert not c["supporting_reference_ids"]


def test_frozen_boundary_and_compatibility_exception():
    result = literature.preservation()
    assert result["prior_public_files"] == 562
    assert result["compatibility_exceptions"] == 6
    assert result["tags_unchanged"] == 2


def test_allowed_literature_still_passes_task14_contract():
    paper = literature.module("check_paper_evidence")
    assert paper.verify(ROOT)["status"] == "PASSED"


@pytest.mark.parametrize(
    "name",
    [
        "track_b_claim_evidence_registry.json",
        "experiment_design_freeze.json",
        "generalization_boundaries.json",
        "figure_registry.json",
        "table_registry.json",
    ],
)
def test_frozen_task14_scientific_mutation_still_fails(tmp_path, name):
    paper = literature.module("check_paper_evidence")
    relative = "docs/paper/" + name
    expected = read("docs/paper/paper_evidence_manifest.json")["paper_artifact_hashes_lf"][relative]
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / relative).read_bytes())
    paper.check_frozen_hash(tmp_path, relative, expected)
    target.write_bytes(target.read_bytes() + b"\n ")
    with pytest.raises(ValueError, match="Paper evidence freeze changed"):
        paper.check_frozen_hash(tmp_path, relative, expected)


@pytest.mark.parametrize("addition", ["unlisted.json", "nested", "unlisted.csv"])
def test_unlisted_literature_additions_fail(tmp_path, addition):
    paper = literature.module("check_paper_evidence")
    folder = tmp_path / "docs/paper/literature"
    folder.mkdir(parents=True)
    if addition == "nested":
        (folder / addition).mkdir()
    else:
        (folder / addition).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unreviewed post-freeze"):
        paper.check_paper_files(tmp_path)


def test_allowlisted_file_still_checked_for_private_text(tmp_path):
    paper = literature.module("check_paper_evidence")
    target = tmp_path / "docs/paper/literature/reference_registry.json"
    target.parent.mkdir(parents=True)
    target.write_text('"123456789012"', encoding="utf-8")
    with pytest.raises(ValueError, match="private"):
        paper.check_paper_files(tmp_path)


def test_approved_checker_hash_does_not_allow_further_edits(tmp_path):
    paper = literature.module("check_paper_evidence")
    relative = "scripts/check_paper_evidence.py"
    expected = read("docs/paper/paper_evidence_manifest.json")["audit_code_hashes_lf"][relative]
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / relative).read_bytes())
    amendment = tmp_path / "reports/paper/task16_compatibility_amendment.json"
    amendment.parent.mkdir(parents=True)
    amendment.write_bytes((ROOT / amendment.relative_to(tmp_path)).read_bytes())
    paper.check_frozen_hash(tmp_path, relative, expected)
    target.write_bytes(target.read_bytes() + b"\n# unapproved edit\n")
    with pytest.raises(ValueError, match="Paper evidence freeze changed"):
        paper.check_frozen_hash(tmp_path, relative, expected)
    with pytest.raises(ValueError, match="Unapproved compatibility"):
        literature.preserved_digest(tmp_path, relative)
