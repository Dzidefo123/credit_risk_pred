"""Documentation-contract checks; no Track B data acquisition or model implementation."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from credit_risk.reporting.evidence import Evidence
from credit_risk.validation.holdout_registry import verify_repository_registry

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/track_b"
FILES = [
    "LONGITUDINAL_DATA_CONTRACT.md",
    "DATASET_SUITABILITY_FRAMEWORK.md",
    "DATASET_SEARCH_STRATEGY.md",
]


def text(name):
    return (DOCS / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", FILES)
def test_track_b_document_links_and_research_scope(name):
    body = text(name)
    assert body.startswith("# Track B")
    assert "Task 0" in body
    assert "Track A" in body
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", body):
        parts = urlsplit(target)
        if parts.scheme:
            assert parts.scheme == "https" and parts.netloc in {"www.bis.org", "www.ifrs.org"}
        else:
            path = (DOCS / parts.path).resolve()
            assert path.is_relative_to(ROOT) and path.is_file(), target
    assert "C:\\" not in body


def test_every_registered_field_has_complete_contract_and_valid_status():
    body = text(FILES[0])
    entities = re.split(r"^### Entity: ", body, flags=re.MULTILINE)[1:]
    assert len(entities) == 10
    count = 0
    required_headers = [
        "Field",
        "Definition",
        "Type",
        "Temporal meaning",
        "Requirement",
        "Nullability",
        "Purpose/components",
        "Leakage control",
        "Source expectation",
    ]
    for entity in entities:
        lines = entity.splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith("|"))
        rows = []
        for line in lines[first:]:
            if not line.startswith("|"):
                break
            rows.append(line)
        assert [v.strip() for v in rows[0].strip("|").split("|")] == required_headers
        names = []
        for row in rows[2:]:
            cells = [v.strip() for v in row.strip("|").split("|")]
            assert len(cells) == len(required_headers) and all(cells)
            assert cells[4] in {"REQUIRED", "CONDITIONALLY REQUIRED", "OPTIONAL"}
            assert cells[5] not in {"unknown", "TBD", ""}
            names.append(cells[0])
            count += 1
        assert len(names) == len(set(names))
    assert count >= 60
    for field in (
        "borrower_id",
        "facility_id",
        "available_at",
        "effective_from",
        "revision_id",
        "default_definition_id",
        "default_derivation",
        "followup_end_date",
        "exit_reason",
        "exposure_at_default",
        "cash_flow_date",
        "macro_vintage_id",
        "forecast_as_of",
    ):
        assert f"| {field} |" in body


def test_point_in_time_labels_and_probability_meanings_are_explicit():
    body = text(FILES[0])
    for concept in (
        "t0 = observation / scoring date",
        "known at t0",
        "future payments",
        "Knowledge time",
        "Positive",
        "Negative",
        "Censored",
        "Insufficient follow-up",
        "Not incident-risk eligible",
        "(t0, H]",
        "competing",
        "left truncation",
        "conditional interval hazard",
        "marginal first-default mass",
        "cumulative risk",
        "available_at",
    ):
        assert concept in body, concept
    assert "365 days" in body and "never silently negative" in body
    assert "Future drawdown is an outcome" in body
    assert "not calculated losses" in body


def test_loss_and_accounting_boundaries_are_preserved():
    body = text(FILES[0])
    for concept in (
        "Observed workout LGD",
        "Simplified research LGD",
        "Regulatory/downturn LGD",
        "revolving",
        "nonpositive denominators",
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "POCI",
        "simplified approaches",
        "accounting policy",
        "institution-specific",
        "scenario_weight",
        "Sum to one",
        "discount",
        "independent bank validation",
    ):
        assert concept.lower() in body.lower(), concept
    assert "Performing status alone" in body
    assert "research default" in body.lower()


def test_component_specific_gates_no_overall_score_or_assumed_acceptance():
    body = text(FILES[1])
    for component in (
        "12-month PD",
        "Lifetime PD",
        "LGD",
        "EAD",
        "SICR",
        "IFRS 9 staging",
        "ECL",
        "Stress testing",
    ):
        assert f"| {component} |" in body
    for status in ("SUPPORTED", "PARTIALLY SUPPORTED", "UNSUPPORTED", "UNASSESSED"):
        assert status in body
    assert "no overall score" in body
    assert "A missing core gate cannot" in body
    assert "Missing exposures rejects EAD" in body
    assert "missing recovery history rejects LGD" in body
    assert "no actual candidate ratings" in body
    assert "Synthetic data must NEVER be used to claim empirical model performance" in body


def test_discovery_plan_and_proposed_architecture_do_not_create_modules():
    body = text(FILES[2])
    for category in (
        "Public credit datasets",
        "Academic/research panels",
        "Central-bank/supervisory sources",
        "Mortgage performance sources",
        "Consumer-credit performance panels",
        "Peer-to-peer lending sources",
        "Public macro/release archives",
        "Synthetic sources",
    ):
        assert category in body
    for n in range(1, 9):
        assert f"RQ-B{n}" in body
    assert "No dataset search/selection/download was performed" in body
    assert "no empty package/modules are created" in body
    # Task 3 authorizes the PD baseline; survival/accounting modules remain absent.
    for component in ("survival", "lgd", "ead", "sicr", "staging", "ecl"):
        assert not (ROOT / "src/credit_risk/track_b" / component).exists()
    assert "not one observed portfolio's empirical ECL" in body
    assert "Candidate Dataset Discovery and Comparative Suitability Assessment" in body


def test_track_a_evidence_frozen_sources_and_registry_still_match():
    # Aggregate/source hashes only: no raw data, bundle deserialization or fitting.
    Evidence(ROOT)
    verify_repository_registry(ROOT / "reports/holdout_registry.json")
    manifest = json.loads(
        (ROOT / "reports/model_validation/model_validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for record in manifest["source_evidence"]:
        payload = (ROOT / record["path"]).read_bytes()
        if Path(record["path"]).suffix == ".png":
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        else:
            assert (
                hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest() == record["sha256_lf"]
            )
    report = (ROOT / manifest["report_path"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(report).hexdigest() == manifest["report_sha256"]
    metrics = manifest["quantitative_evidence"]["historical"]["metrics"]
    assert {k: round(metrics[k], 6) for k in ("roc_auc", "brier", "log_loss")} == {
        "roc_auc": 0.868152,
        "brier": 0.048545,
        "log_loss": 0.176030,
    }
