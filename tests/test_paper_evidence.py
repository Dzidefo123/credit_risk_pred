"""Task 14 scientific-claim and frozen-evidence regressions; no empirical research."""

import ast
import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paper_check", ROOT / "scripts/check_paper_evidence.py"
)
paper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper)
SOURCE_SPEC = importlib.util.spec_from_file_location(
    "paper_preservation", ROOT / "scripts/fannie_source_provenance.py"
)
preserve = importlib.util.module_from_spec(SOURCE_SPEC)
SOURCE_SPEC.loader.exec_module(preserve)


def read(name):
    return json.loads((ROOT / "docs/paper" / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def claims():
    return read("track_b_claim_evidence_registry.json")["claims"]


def test_full_paper_evidence_contract():
    assert paper.verify(ROOT)["status"] == "PASSED"


def test_duplicate_claims_fail(claims):
    with pytest.raises(ValueError, match="Duplicate claim"):
        paper.check_claims(ROOT, [claims[0], claims[0]])


def test_exact_metric_mutation_fails(claims):
    changed = copy.deepcopy(claims[0])
    changed["point_estimate"] = 0.99
    with pytest.raises(ValueError, match="differs from frozen"):
        paper.check_claims(ROOT, [changed])


def test_missing_quantitative_metric_fails(claims):
    changed = copy.deepcopy(claims[0])
    changed["metric"] = None
    with pytest.raises(ValueError, match="missing metric"):
        paper.check_claims(ROOT, [changed])


def test_primary_requires_verified_source(claims):
    changed = copy.deepcopy(next(c for c in claims if c["claim_strength"] == "PRIMARY"))
    changed["verification_status"] = "INSUFFICIENT_EVIDENCE"
    with pytest.raises(ValueError, match="Primary claim"):
        paper.check_claims(ROOT, [changed])
    changed["verification_status"] = "VERIFIED_EXACT"
    changed["source_artifacts"] = []
    with pytest.raises(ValueError, match="Primary claim"):
        paper.check_claims(ROOT, [changed])


def test_exploratory_cannot_be_primary(claims):
    changed = copy.deepcopy(next(c for c in claims if c["experiment_id"] == "B12"))
    changed["claim_strength"] = "PRIMARY"
    with pytest.raises(ValueError, match="Exploratory/prohibited primary"):
        paper.check_claims(ROOT, [changed])


@pytest.mark.parametrize(
    "language",
    [
        "Fannie is sealed",
        "Fannie is pre-registered",
        "Fannie externally validated these results",
        "Fannie independently replicated the finding",
        "External validation completed",
        "Macroeconomic variables caused the temporal failure",
        "This is an IFRS 9 model",
        "This is an IRB model",
        "This is a regulatory PD model",
        "This is a production bank model",
        "The model is regulator validated",
    ],
)
def test_unsafe_positive_language_fails(claims, language):
    changed = copy.deepcopy(claims[0])
    changed["allowed_language"] = language
    with pytest.raises(ValueError, match="assertion|terminology"):
        paper.check_claims(ROOT, [changed])


def test_fannie_draft_and_no_results(claims):
    fannie = paper.read(ROOT, "reports/track_b/fannie_source_provenance_closure.json")
    assert fannie["protocol_status"] == "DRAFT_NOT_YET_AUTHORIZED"
    protocol = paper.read(ROOT, "docs/track_b/fannie_external_replication_protocol.json")
    assert protocol["status"] == "DRAFT_NOT_YET_AUTHORIZED"
    assert fannie["scope"]["task13a_authorized"] is False
    assert fannie["scope"]["outcome_analysis"] is False
    future = next(c for c in claims if c["claim_id"] == "FANNIE_PROPOSED")
    assert future["claim_class"] == "PROPOSED_EXTERNAL_REPLICATION"
    assert future["point_estimate"] is None
    assert future["quantitative"] is False
    changed = copy.deepcopy(future)
    changed["quantitative"] = True
    changed["metric"] = "default_rate"
    with pytest.raises(ValueError, match="Fannie empirical"):
        paper.check_claims(ROOT, [changed])


@pytest.mark.parametrize(
    "filename,key",
    [
        ("figure_registry.json", "figures"),
        ("table_registry.json", "tables"),
        ("experiment_design_freeze.json", "experiments"),
        ("headline_claim_audit.json", "headlines"),
        ("manuscript_section_map.json", "sections"),
    ],
)
def test_all_planning_claim_references_resolve(claims, filename, key):
    ids = {c["claim_id"] for c in claims}
    for entry in read(filename)[key]:
        assert set(entry["claim_ids"]) <= ids


def test_generalization_and_regulatory_boundaries():
    boundaries = read("generalization_boundaries.json")
    assert len(boundaries["boundaries"]) == 13
    assert len(boundaries["regulatory_prohibited_claims"]) == 8
    assert "conditional_vs_lifetime" in {b["boundary_id"] for b in boundaries["boundaries"]}


def test_all_primary_and_quantitative_evidence_pinned(claims):
    for claim in claims:
        assert paper.REQUIRED <= claim.keys()
        if claim["quantitative"]:
            assert claim["metric"]
            assert claim["verification_status"] in paper.VERIFIED
        if claim["claim_strength"] == "PRIMARY":
            assert claim["source_artifacts"]
            assert claim["verification_status"] in paper.VERIFIED


def test_evidence_hash_and_private_reference_fail_closed(claims):
    ref = copy.deepcopy(claims[0]["source_artifacts"][0])
    ref["sha256_lf"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        paper.check_ref(ROOT, ref)
    ref["path"] = "data/track_b/private.csv"
    with pytest.raises(ValueError, match="outside public"):
        paper.check_ref(ROOT, ref)


def test_figure_generation_uses_aggregates_only():
    for figure in read("figure_registry.json")["figures"]:
        assert figure["can_generate_from_existing_frozen_artifacts"]
        assert not figure["requires_raw_data"]
        assert not figure["requires_new_calculation"]
        assert figure["status"] == "PLAN_ONLY_NOT_GENERATED"
    assert read("figure_registry.json")["raw_only_candidates"][0]["status"] == "DO_NOT_GENERATE"


def test_no_empirical_pipeline_imports_or_archive_reads():
    for name in ["build_paper_evidence.py", "check_paper_evidence.py"]:
        tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        assert imports <= {"hashlib", "json", "subprocess", "pathlib", "re"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in {
                    "fit",
                    "predict",
                    "predict_proba",
                    "read_csv",
                    "bootstrap",
                    "resample",
                    "extractall",
                }


def test_statistical_units_and_exposure():
    units = read("statistical_unit_audit.json")
    assert units["intervals_are_not_independent_observations"]
    calendar = [x for x in units["units"] if x["cluster_unit"] == "calendar_year"]
    assert all(x["number_of_clusters"] == 8 for x in calendar)
    design = read("experiment_design_freeze.json")["experiments"]
    for exp in design:
        if exp["experiment_id"] in {"B5", "B6", "B10", "B11", "B12"}:
            assert exp["prior_exposure"]


def test_precision_conflict_and_historical_stop_retained():
    conflicts = read("evidence_conflicts.json")
    assert not conflicts["unresolved_primary_conflicts"]
    assert {c["conflict_id"] for c in conflicts["conflicts"]} == {
        "C_MANIFEST",
        "C_BRIER_PRECISION",
        "C_FANNIE_TERMINOLOGY",
    }
    assert all(c["resolution"].startswith("RESOLVED_") for c in conflicts["conflicts"])


def test_all_prior_research_files_and_tag_preserved():
    manifest = read("task14_preservation_manifest.json")
    assert preserve.verify_preservation(ROOT, manifest)["public"] == 509


def test_builder_does_not_overwrite_existing_evidence():
    spec = importlib.util.spec_from_file_location(
        "paper_builder", ROOT / "scripts/build_paper_evidence.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    with pytest.raises(ValueError, match="already exists"):
        builder.write("track_b_claim_evidence_registry.json", {})


@pytest.mark.parametrize(
    "text", ['"123456789012"', "Authorization: Bearer synthetic", "|".join(["opaque"] * 113)]
)
def test_private_identifiers_credentials_and_rows_rejected(text):
    with pytest.raises(ValueError, match="private|loan row"):
        paper.check_public_text(text)


def test_decimal_metrics_are_not_loan_identifiers():
    paper.check_public_text("0.123456789012 and (-0.000000000001, 0.02]")


def test_claim_section_assignments_resolve(claims):
    sections = {x["section_id"] for x in read("manuscript_section_map.json")["sections"]}
    assert all(set(c["manuscript_sections"]) <= sections for c in claims)


def test_source_commits_bind_exact_public_artifacts(claims):
    unique = {
        (r["path"], r["source_commit"], r["sha256_lf"])
        for c in claims
        for r in c["source_artifacts"]
    }
    import hashlib

    for path, commit, expected in unique:
        body = subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT)
        assert hashlib.sha256(body.replace(b"\r\n", b"\n")).hexdigest() == expected
