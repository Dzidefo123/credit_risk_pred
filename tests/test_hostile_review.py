"""Regression gates for the Task 17 hostile scientific review.

These tests assert that the review covers every required audit area, attacks and
classifies every PRIMARY claim, and that it changed no empirical evidence. They
also re-derive the one piece of arithmetic the review relies on, so a drift in
the frozen artifacts would fail here rather than silently invalidate the report.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "reports/paper/review"
REPORT = ROOT / "reports/paper/TASK17_HOSTILE_SCIENTIFIC_REVIEW.md"
VERIFICATION = ROOT / "reports/paper/task17_verification.json"

BASE_COMMIT = "a4479ba4be115bc9a1d4b53ab77e258b40bd57c7"


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report_text():
    return REPORT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def attack():
    return read_json("reports/paper/review/task17_claim_attack_registry.json")


@pytest.fixture(scope="module")
def survival():
    return read_json("reports/paper/review/task17_claim_survival_matrix.json")


@pytest.fixture(scope="module")
def primary_claims():
    registry = read_json("docs/paper/track_b_claim_evidence_registry.json")
    return {c["claim_id"] for c in registry["claims"] if c.get("claim_strength") == "PRIMARY"}


def test_all_review_outputs_exist():
    for name in (
        "task17_claim_attack_registry.json",
        "task17_claim_survival_matrix.json",
        "task17_manuscript_change_register.json",
        "task17_sensitivity_register.json",
    ):
        assert (REVIEW_DIR / name).is_file(), name
    assert REPORT.is_file()
    assert VERIFICATION.is_file()


def test_five_reviewer_personas_present(report_text):
    for persona in (
        "R1 — Credit-risk modeller",
        "R2 — Survival / competing-risk statistician",
        "R3 — Machine-learning reviewer",
        "R4 — Mortgage economist",
        "R5 — Reproducibility / model-risk reviewer",
    ):
        assert persona in report_text, persona


def test_every_persona_has_a_recommendation(report_text):
    allowed = ("ACCEPT", "WEAK_ACCEPT", "BORDERLINE", "WEAK_REJECT", "REJECT")
    assert any(verdict in report_text for verdict in allowed)
    # The score table carries one recommendation per persona.
    row = [line for line in report_text.splitlines() if line.startswith("| **Recommendation**")]
    assert len(row) == 1
    cells = [c.strip() for c in row[0].strip("|").split("|")[1:]]
    assert len(cells) == 5
    assert all(c in allowed for c in cells), cells


def test_all_primary_claims_attacked(attack, primary_claims):
    attacked = {c["claim_id"] for c in attack["claims"]}
    assert attacked == primary_claims
    assert len(attacked) == 19


def test_attack_records_are_complete(attack):
    required = {
        "claim_id",
        "claim_text",
        "evidence_status",
        "reviewer_attack",
        "alternative_explanation",
        "estimand_concern",
        "selection_concern",
        "uncertainty_concern",
        "literature_concern",
        "severity",
        "survives_attack",
        "required_remedy",
        "remedy_requires_new_experiment",
        "manuscript_action",
    }
    severities = {"FATAL", "MAJOR", "MODERATE", "MINOR", "EDITORIAL"}
    for claim in attack["claims"]:
        assert required <= set(claim), claim["claim_id"]
        assert claim["severity"] in severities, claim["claim_id"]
        for field in ("reviewer_attack", "alternative_explanation", "required_remedy"):
            assert claim[field].strip(), f"{claim['claim_id']}:{field}"


def test_all_primary_claims_classified_in_survival_matrix(survival, primary_claims):
    classified = {c["claim_id"] for c in survival["claims"]}
    assert classified == primary_claims
    allowed = set(survival["legend"])
    for row in survival["claims"]:
        assert row["verdict"] in allowed, row["claim_id"]
        assert row["rationale"].strip()


def test_payoff_auc_claim_requires_new_analysis(survival):
    row = next(c for c in survival["claims"] if c["claim_id"] == "MACRO_PRIMARY_M2_PAYOFF_AUC")
    assert row["verdict"] == "REQUIRES_NEW_ANALYSIS"
    assert row["requires_new_experiment"] is True


def test_cross_cutting_findings_are_traceable(attack):
    for key, finding in attack["cross_cutting_findings"].items():
        assert finding["severity"] in {"MAJOR", "MODERATE", "MINOR"}, key
        assert finding["evidence"].strip(), key
        assert finding["derivation"].strip(), key
        assert finding["reviewers"], key
    referenced = {f for c in attack["claims"] for f in c["findings"]}
    assert referenced <= set(attack["cross_cutting_findings"])


@pytest.mark.parametrize(
    "topic",
    [
        "Conditional entry",
        "Left truncation",
        "IPCW",
        "Temporal split and facility overlap",
        "Prior outcome exposure",
        "APC and macro identification",
        "Macro pseudoreplication",
        "Calibration and proper scores",
        "CIF interpretation",
        "Payoff AUC interpretation",
        "Refinancing experiment",
        "Novelty",
        "Estimand audit",
        "Practical significance",
        "Minimum sufficient revision",
        "Meta-review",
    ],
)
def test_required_audit_sections_present(report_text, topic):
    assert topic in report_text, topic


def test_citation_gaps_cg03_and_cg06_addressed(report_text):
    assert "CG03" in report_text
    assert "CG06" in report_text


def test_within_time_discrimination_issue_reviewed(report_text):
    assert "between-period" in report_text
    assert "0.5906" in report_text and "0.5908" in report_text


def test_cif_forecast_interpretation_reviewed(report_text):
    assert "retrospective sequential mapping" in report_text
    assert "not a prospective forecast" in report_text


def test_bu_unknown_remains_unknown(report_text):
    assert "UNKNOWN stays UNKNOWN" in report_text
    assert "IMPORTANT, not BLOCKING" in report_text


def test_task12_stays_exploratory(report_text):
    assert "EXPLORATORY_ONLY" in report_text
    matrix = read_json("docs/paper/experiment_design_freeze.json")
    b12 = next(e for e in matrix["experiments"] if e["experiment_id"] == "B12")
    assert "EXPLORATORY_ONLY" in b12["description"]


def test_fannie_stays_proposed_and_unauthorized(report_text):
    freeze = read_json("docs/paper/experiment_design_freeze.json")
    b13 = next(e for e in freeze["experiments"] if e["experiment_id"] == "B13_13S")
    assert b13["protocol_status"] == "DRAFT_NOT_YET_AUTHORIZED"
    assert b13["task13a_authorized"] is False
    assert "DRAFT_NOT_YET_AUTHORIZED" in report_text


def test_decision_and_readiness_recorded(report_text):
    assert "MANUSCRIPT CORE SURVIVES WITH MAJOR REVISION" in report_text
    assert "ARXIV_PREPRINT_REQUIRES_ADDITIONAL_ANALYSIS" in report_text
    assert "PEER_REVIEW_REQUIRES_ADDITIONAL_ANALYSIS" in report_text


def test_no_manuscript_v0_3_created():
    assert not (ROOT / "paper/main_v0.3.md").exists()


def test_decomposition_reconstructs_frozen_delta():
    """The one arithmetic claim the review depends on must re-derive exactly."""
    macro = read_json("reports/track_b/macro_competing_risk_validation.json")
    calendar = macro["stability"]["calendar"]
    total = sum(calendar[y]["M1"]["intervals"] for y in calendar)
    assert total == macro["split_counts"]["evaluation_seen"]["intervals"]
    reconstructed = sum(
        calendar[y]["M1"]["intervals"]
        / total
        * (calendar[y]["M2"]["joint_log_loss"] - calendar[y]["M1"]["joint_log_loss"])
        for y in calendar
    )
    frozen = macro["paired_facility"]["intervals"]["joint_log_loss"]["delta"]
    assert reconstructed == pytest.approx(frozen, abs=1e-12)

    share_2020 = (
        calendar["2020"]["M1"]["intervals"]
        / total
        * (calendar["2020"]["M2"]["joint_log_loss"] - calendar["2020"]["M1"]["joint_log_loss"])
    ) / reconstructed
    assert share_2020 == pytest.approx(0.893, abs=5e-4)


def test_unseen_vintage_reversal_is_real():
    macro = read_json("reports/track_b/macro_competing_risk_validation.json")
    unseen = macro["unseen_vintage"]
    assert unseen["M2"]["scores"]["joint_log_loss"] < unseen["M1"]["scores"]["joint_log_loss"]


def test_primary_payoff_brier_calendar_interval_includes_zero():
    macro = read_json("reports/track_b/macro_competing_risk_validation.json")
    interval = macro["paired_calendar"]["intervals"]["payoff_brier"]
    assert interval["lower"] < 0 < interval["upper"]


def test_pit_release_lags_remain_unmeasured():
    audit = read_json("reports/track_b/pit_macro_data_audit_task9_api_v5.json")
    for series, record in audit["release_lags"].items():
        assert record["count"] == 0, series
        assert record["evidence_quality"] == "UNMEASURED_EXACT_PROVIDER_DATES", series


def test_no_empirical_values_changed():
    verification = read_json("reports/paper/task17_verification.json")
    assert verification["no_experiments_rerun"] is True
    assert verification["no_model_artifacts_changed"] is True
    assert verification["no_frozen_metrics_changed"] is True
    assert verification["no_new_fannie_outcomes"] is True
    assert verification["no_invented_result"] is True
    assert verification["unknown_remains_unknown"] is True
    assert verification["base_commit"] == BASE_COMMIT


def test_preservation_recorded():
    verification = read_json("reports/paper/task17_verification.json")
    for task in ("task14", "task15", "task16"):
        assert verification["preserved"][task] is True
    assert verification["preserved"]["numeric_bindings"] == 123


def test_every_criticism_traces_to_evidence(attack):
    for key, finding in attack["cross_cutting_findings"].items():
        evidence = finding["evidence"]
        for ref in evidence.split(" ; "):
            path = ref.split("#")[0].strip()
            if path and path != "Whole-manuscript":
                assert (ROOT / path).exists(), f"{key} -> {path}"


def test_change_and_sensitivity_registers_well_formed():
    changes = read_json("reports/paper/review/task17_manuscript_change_register.json")
    assert changes["applied"] is False
    severities = {"FATAL", "MAJOR", "MODERATE", "MINOR", "EDITORIAL"}
    ids = set()
    for change in changes["changes"]:
        assert change["severity"] in severities
        assert change["status"] == "PROPOSED"
        assert isinstance(change["blocking_preprint"], bool)
        assert isinstance(change["blocking_peer_review"], bool)
        ids.add(change["change_id"])
    assert len(ids) == len(changes["changes"])

    sensitivities = read_json("reports/paper/review/task17_sensitivity_register.json")
    assert sensitivities["executed"] is False
    priorities = {"SUBMISSION_BLOCKING", "STRONGLY_RECOMMENDED", "OPTIONAL", "NOT_NEEDED"}
    for analysis in sensitivities["analyses"]:
        assert analysis["priority"] in priorities
        assert analysis["expected_interpretation_if_positive"].strip()
        assert analysis["expected_interpretation_if_negative"].strip()
    assert any(a["priority"] == "SUBMISSION_BLOCKING" for a in sensitivities["analyses"])
