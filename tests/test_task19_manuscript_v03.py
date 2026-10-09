"""Public v0.3 traceability and scope regressions; no empirical execution."""

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("task19", ROOT / "scripts/check_task19_manuscript.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def draft():
    return (ROOT / "paper/main_v0.3.md").read_text(encoding="utf-8")


def bindings():
    return json.loads(
        (ROOT / "reports/paper/task19_claim_evidence_map.json").read_text(encoding="utf-8")
    )["bindings"]


def test_complete_v03_contract():
    result = check.verify()
    assert result["quantitative_claims"] == result["mapped"] == 257
    assert result["orphan_numeric_claims"] == 0
    assert result["preservation"]["public"] == 603
    assert result["legacy_numeric_bindings"] == 123
    assert result["new_empirical_analysis"] is False


def test_all_prior_scientific_manuscript_and_review_files_preserved():
    assert check.preservation() == {
        "public": 603,
        "private": 0,
        "refs": 2,
        "engineering_exceptions": 2,
    }


def test_changed_frozen_artifact_rejected(tmp_path, monkeypatch):
    # Exercise the preservation gate without ever altering a research artifact.
    target = tmp_path / "paper/main_v0.2.md"
    target.parent.mkdir()
    target.write_text("deliberately changed", encoding="utf-8")
    frozen = check.read("reports/paper/task19_preservation_manifest.json")
    manifest = {
        "public_lf_hashes": {
            "paper/main_v0.2.md": frozen["public_lf_hashes"]["paper/main_v0.2.md"]
        },
        "private_byte_hashes": {},
        "git_refs": {},
    }
    monkeypatch.setattr(check, "ROOT", tmp_path)
    monkeypatch.setattr(check, "read", lambda _: manifest)
    with pytest.raises(ValueError, match="Prior scientific"):
        check.preservation()


@pytest.mark.parametrize(
    "corruption", ["number", "pointer", "source_hash", "extra_number", "marker"]
)
def test_numeric_or_source_corruption_fails(corruption):
    text, records = draft(), copy.deepcopy(bindings())
    if corruption == "number":
        text = text.replace(records[0]["displayed_value"], "0.99999", 1)
    elif corruption == "pointer":
        records[0]["source_field"] = "/primary/M2/scores/joint_log_loss"
    elif corruption == "source_hash":
        records[0]["source_sha256_lf"] = "0" * 64
    elif corruption == "marker":
        text = text.replace("<!-- Q19: Q0001 -->", "")
    else:
        text = text.replace("## References", "Additional AUC is 0.9999.\n\n## References")
    with pytest.raises(ValueError):
        check.validate_bindings(text, records)


@pytest.mark.parametrize(
    "claim",
    [
        "COVID caused the failure.",
        "This is the first study to do this.",
        "Within-month improvement was structurally impossible.",
        "Within-year contribution is 10.5%.",
        "M2 is closer than M1 at the shortest horizon.",
        "The contrast has an identified numerical cause.",
        "Fannie completed external replication.",
        "We have fully verified provider release timestamps.",
    ],
)
def test_corrected_and_unsupported_claims_rejected(claim):
    with pytest.raises(ValueError):
        check.validate_language(draft() + "\n" + claim)


@pytest.mark.parametrize(
    "phrase",
    [
        "ROBUST_FACILITY_ONLY",
        "DRAFT_NOT_YET_AUTHORIZED",
        "not independent corroboration",
        "release lags were not certified",
        "Current delinquency",
        "CG03 implementation review remains open",
    ],
)
def test_material_qualification_removal_fails(phrase):
    with pytest.raises(ValueError):
        check.validate_language(draft().replace(phrase, "removed"))


def test_corrected_denominators_and_pair_shares_present():
    text = draft()
    assert "1.81%" in text and "98.19%" in text
    assert "+0.00217" in text and "+0.00172" in text
    assert "M1 is closer than M2" in text
    assert "26<!--" in text and "29<!--" in text


def test_local_evidence_is_admitted_without_reexecution():
    result = check.admission()
    assert result == {"admitted_files": 5, "rerun": False}
    text = draft()
    assert "eligible-month population only" in text
    assert "post-hoc local closure" in text


def test_erratum_is_additive_and_original_review_unchanged():
    text = (ROOT / "reports/paper/TASK17_ERRATUM.md").read_text(encoding="utf-8")
    assert "argument is withdrawn" in text
    assert "audit error, not an error" in text
    assert check.preservation()["public"] == 603


def test_verified_reference_keys_only():
    ids = {
        r["reference_id"]
        for r in check.read("docs/paper/literature/reference_registry.json")["references"]
    }
    assert check.module("check_literature").citation_keys(draft()) <= ids


def test_no_model_scoring_or_data_pipeline_calls():
    for name in ("build_task19_manuscript.py", "check_task19_manuscript.py"):
        tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
        imports = {
            a.name.split(".")[0]
            for n in ast.walk(tree)
            if isinstance(n, ast.Import)
            for a in n.names
        }
        imports |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert imports <= {
            "argparse",
            "hashlib",
            "importlib",
            "json",
            "re",
            "subprocess",
            "pathlib",
        }
        assert not any(
            isinstance(n, ast.Attribute)
            and n.attr
            in {"fit", "predict", "predict_proba", "read_csv", "load", "bootstrap", "resample"}
            for n in ast.walk(tree)
        )


def test_pinned_engineering_amendment_rejects_further_edits(tmp_path, monkeypatch):
    import hashlib

    name = "tests/test_hostile_review.py"
    target = tmp_path / name
    target.parent.mkdir()
    old, approved = b"old historical assertion", b"scoped historical assertion"
    original_hash = hashlib.sha256(old).hexdigest()
    amended_hash = hashlib.sha256(approved).hexdigest()
    target.write_bytes(approved)
    manifest = {
        "public_lf_hashes": {name: original_hash},
        "private_byte_hashes": {},
        "git_refs": {},
        "engineering_compatibility_changes": {
            name: {"original_sha256_lf": original_hash, "amended_sha256_lf": amended_hash}
        },
    }
    monkeypatch.setattr(check, "ROOT", tmp_path)
    monkeypatch.setattr(check, "read", lambda _: manifest)
    assert check.preservation()["engineering_exceptions"] == 1
    target.write_bytes(approved + b"unapproved change")
    with pytest.raises(ValueError, match="Prior scientific"):
        check.preservation()


def test_scientific_artifact_cannot_receive_engineering_exception(monkeypatch):
    manifest = {
        "public_lf_hashes": {},
        "private_byte_hashes": {},
        "git_refs": {},
        "engineering_compatibility_changes": {
            "reports/track_b/macro_competing_risk_validation.json": {}
        },
    }
    monkeypatch.setattr(check, "read", lambda _: manifest)
    with pytest.raises(ValueError, match="Scientific evidence"):
        check.preservation()
