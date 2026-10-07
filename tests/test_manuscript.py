"""Manuscript traceability checks use frozen public aggregates, never research records."""

import ast
import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("manuscript", ROOT / "scripts/check_manuscript.py")
manuscript = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manuscript)


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_complete_draft_contract():
    result = manuscript.verify()
    assert result["status"] == "PASSED"
    assert 6000 <= result["body_word_count"] <= 9000
    assert 200 <= result["abstract_word_count"] <= 300
    assert result["bindings"] == 123


def test_all_numeric_display_values_bind_to_verified_claims():
    claims = {
        c["claim_id"]: c for c in read("docs/paper/track_b_claim_evidence_registry.json")["claims"]
    }
    for binding in read("docs/paper/manuscript_numeric_bindings.json")["bindings"]:
        c = claims[binding["claim_id"]]
        assert c["verification_status"] in {"VERIFIED_EXACT", "VERIFIED_WITH_QUALIFICATION"}
        assert (
            manuscript.builder.claim_value(c, binding["claim_pointer"])
            == binding["exact_frozen_value"]
        )
        assert (
            format(binding["exact_frozen_value"], binding["display_format"])
            == binding["displayed_value"]
        )


def test_unsupported_claim_cannot_be_rendered():
    c = {
        "claim_class": "PROHIBITED",
        "verification_status": "VERIFIED_EXACT",
        "point_estimate": 0.99,
    }
    with pytest.raises(ValueError, match="prohibited numeric claim"):
        manuscript.builder.render("{{BAD||.2f}}", {"BAD": c})


def test_unverified_claim_cannot_be_rendered():
    c = {
        "claim_class": "SUPPORTED_EMPIRICAL",
        "verification_status": "INSUFFICIENT_EVIDENCE",
        "point_estimate": 0.99,
    }
    with pytest.raises(ValueError, match="Unverified numeric claim"):
        manuscript.builder.render("{{BAD||.2f}}", {"BAD": c})


@pytest.mark.parametrize(
    "literal", ["AUC 0.9999", "We evaluated 123,456 facilities", "We evaluated 123456 facilities"]
)
def test_unbound_quantitative_result_fails(literal):
    with pytest.raises(ValueError, match="Unbound quantitative"):
        manuscript.check_unbound_numbers(literal)


@pytest.mark.parametrize(
    "language",
    [
        "Fannie is sealed",
        "Fannie is pre-registered",
        "We completed external validation",
        "This model is an IRB model",
        "This model is an IFRS 9 production model",
        "Unemployment caused the failure",
        "We study 140,000 unique borrowers",
    ],
)
def test_overclaim_language_fails(language):
    with pytest.raises(ValueError):
        manuscript.check_language(language)


@pytest.mark.parametrize(
    "qualification",
    [
        "conditional entry",
        "research default proxy",
        "EXPLORATORY_ONLY",
        "post-hoc diagnostic analysis",
        "DRAFT_NOT_YET_AUTHORIZED",
        "PUBLICATION TERMS REVIEW REQUIRED",
    ],
)
def test_required_qualifications_present(qualification):
    assert qualification.lower() in (ROOT / "paper/main.md").read_text(encoding="utf-8").lower()


def test_citation_placeholders_not_fabricated_bibliography():
    text = (ROOT / "paper/main.md").read_text(encoding="utf-8")
    gaps = read("docs/paper/citation_gaps.json")["gaps"]
    assert len(gaps) == 6
    assert len({g["gap_id"] for g in gaps}) == 6
    assert not re.search(r"\b10\.\d{4,9}/\S+", text)
    assert "\\cite{" not in text


def test_figure_plan_and_appendix_placeholders():
    text = (ROOT / "paper/main.md").read_text(encoding="utf-8")
    figures = read("docs/paper/figure_registry.json")["figures"]
    assert all(f"[FIGURE {f['figure_id']} HERE]" in text for f in figures)
    assert all(f"Appendix {letter}." in text for letter in "ABCDEFGH")
    assert not list((ROOT / "paper").glob("*.csv"))


def test_frozen_task14_task13t_and_all_prior_public_artifacts():
    manifest = read("docs/paper/task15_preservation_manifest.json")
    assert len(manifest["public_lf_hashes"]) == 548
    for name, expected in manifest["public_lf_hashes"].items():
        assert (
            hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            == expected
        )


def test_no_fitting_scoring_or_provider_data_imports():
    for name in ["build_manuscript.py", "check_manuscript.py"]:
        tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
        imports = {
            a.name.split(".")[0]
            for n in ast.walk(tree)
            if isinstance(n, ast.Import)
            for a in n.names
        } | {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert imports <= {"hashlib", "json", "re", "subprocess", "pathlib", "importlib"}
        assert not any(
            isinstance(n, ast.Attribute)
            and n.attr in {"fit", "predict", "predict_proba", "read_csv", "resample", "bootstrap"}
            for n in ast.walk(tree)
        )


def test_display_format_is_presentation_only():
    c = {
        "claim_class": "SUPPORTED_EMPIRICAL",
        "verification_status": "VERIFIED_EXACT",
        "point_estimate": 0.33673487196277013,
        "source_artifacts": [],
    }
    original = copy.deepcopy(c)
    rendered, bindings = manuscript.builder.render("{{CIF||.1%}}", {"CIF": c})
    assert rendered.startswith("33.7%")
    assert c == original
    assert bindings[0]["exact_frozen_value"] == original["point_estimate"]
