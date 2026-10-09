"""Regression gates for the Task 20 hostile manuscript review.

Per the brief, tests here verify the REVIEW ARTIFACTS only. They additionally
re-derive the arithmetic the review depends on, so drift in a frozen artifact or in
the manuscript fails this suite rather than silently invalidating the review.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "reports/paper/review"
REPORT = REVIEW / "TASK20_HOSTILE_MANUSCRIPT_REVIEW.md"
MANUSCRIPT = ROOT / "paper/main_v0.3.md"

BASE_COMMIT = "1a42d505c6652fecf384d00b2d19d737c4b8524a"
MANUSCRIPT_SHA = "dd7738c5726b27b703747b253f5e641bb9e3b32ba8507a18d2ba375ba765f17a"


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def raw_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def verification():
    return read("reports/paper/review/task20_review_verification.json")


@pytest.fixture(scope="module")
def macro():
    return read("reports/track_b/macro_competing_risk_validation.json")


@pytest.fixture(scope="module")
def report_text():
    return REPORT.read_text(encoding="utf-8")


# ------------------------------------------------------------------- artifacts


def test_all_review_outputs_exist():
    for name in (
        "TASK20_HOSTILE_MANUSCRIPT_REVIEW.md",
        "task20_estimand_audit.json",
        "task20_abstract_audit.json",
        "task20_claim_sample_audit.json",
        "task20_analysis_triage.json",
        "task20_review_verification.json",
    ):
        assert (REVIEW / name).is_file(), name


def test_review_target_is_the_hashed_manuscript(verification):
    assert raw_sha(MANUSCRIPT) == MANUSCRIPT_SHA
    block = verification["repository_state_verification"]["author_supplied_hashes_verified"]
    for path, record in block.items():
        assert record["matches"] is True, path
        assert record["actual"] == raw_sha(ROOT / path)
    assert all(verification["repository_state_verification"]["required_files_present"].values())


def test_independence_is_disclosed_everywhere(verification, report_text):
    ind = verification["independence"]
    assert ind["status"] == "COMPROMISED_AND_DISCLOSED"
    assert ind["brief_section_2_provisional_cold_read"] == "NOT_SATISFIABLE_BY_THIS_REVIEWER"
    assert ind["disclosed_prominently_in_report"] is True
    # the report must carry the declaration before any finding
    head = report_text[: report_text.index("## 1. Summary")]
    assert "not independent" in head.lower()
    assert "NOT_SATISFIABLE_BY_THIS_REVIEWER" in head
    for name in ("task20_estimand_audit.json", "task20_abstract_audit.json", "task20_analysis_triage.json"):
        assert read(f"reports/paper/review/{name}")["independence"]["independence_status"] == (
            "COMPROMISED_AND_DISCLOSED"
        )


def test_review_changed_nothing(verification):
    compliance = verification["review_only_compliance"]
    for flag, value in compliance.items():
        assert value is False, flag
    assert verification["task21_not_started"] is True
    assert not (ROOT / "paper/main_v0.4.md").exists()


# ------------------------------------------------------- required brief sections


@pytest.mark.parametrize(
    "heading",
    [
        "## 1. Summary",
        "## 2. Overall recommendation",
        "## 3. Central contribution",
        "## 4. Major strengths",
        "## 5. Major concerns",
        "## 6. Minor concerns",
        "## 7. Statistical design",
        "## 8. Temporal design",
        "## 9. Population and transport",
        "## 10. Pooled discrimination",
        "## 11. Probability quality",
        "## 12. CIF analysis",
        "## 13. Uncertainty",
        "## 14. Literature and novelty",
        "## 15. Governance and reproducibility",
        "## 16. Regulatory framing",
        "## 17. Abstract audit",
        "## 18. Title audit",
        "## 19. Fatal-flaw assessment",
        "## 20. Required revisions",
        "## 21. Recommended additional analyses",
        "## 22. Preprint readiness",
        "## 23. Peer-review readiness",
    ],
)
def test_report_has_every_required_section(report_text, heading):
    assert heading in report_text, heading


def test_two_separate_decisions_recorded(verification, report_text):
    decisions = verification["decisions"]
    assert decisions["preprint"] in {
        "READY_FOR_ARXIV", "READY_AFTER_MINOR_REVISION", "REQUIRES_MAJOR_REVISION",
        "NOT_READY_ANALYSIS_GAP", "NOT_DEFENSIBLE",
    }
    assert decisions["peer_review"] in {
        "STRONG_ACCEPT", "ACCEPT", "WEAK_ACCEPT", "BORDERLINE", "WEAK_REJECT", "REJECT",
    }
    assert decisions["calibrated_to_each_other"] is False
    assert decisions["preprint"] in report_text
    assert decisions["peer_review"] in report_text


def test_fatal_flaw_assessment_is_explicit(verification, report_text):
    assert verification["findings"]["fatal_flaw"] is False
    assert "NO FATAL FLAW" in report_text


# --------------------------------------------------- arithmetic the review rests on


def test_default_auc_concern_is_real(macro):
    """M1: the omitted result must actually be larger than the reported one and robust."""
    facility = macro["paired_facility"]["intervals"]
    default_delta = facility["default_auc"]["delta"]
    payoff_delta = facility["payoff_auc"]["delta"]
    assert abs(default_delta) > abs(payoff_delta)
    assert facility["default_auc"]["upper"] < 0  # excludes zero
    unseen = macro["unseen_vintage"]
    assert unseen["M2"]["scores"]["default_auc"] < unseen["M1"]["scores"]["default_auc"]
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    abstract = manuscript[manuscript.index("## Abstract"): manuscript.index("## 1. Introduction")]
    assert "0.622831" not in abstract  # the concern: absent from the abstract


def test_no_calendar_interval_exists_for_either_auc(macro):
    """M2: the internal inconsistency must be a fact about the frozen artifacts."""
    calendar = macro["paired_calendar"]["intervals"]
    facility = macro["paired_facility"]["intervals"]
    assert "payoff_auc" not in calendar
    assert "default_auc" not in calendar
    assert "payoff_auc" in facility and "default_auc" in facility
    assert macro["paired_facility"]["conditional_on_realized_calendar"] is True


def test_contribution_share_is_resolution_dependent(macro):
    """M3: year-level and month-level pair weights must genuinely differ."""
    calendar = macro["stability"]["calendar"]
    total_intervals = sum(calendar[y]["M1"]["intervals"] for y in calendar)
    total_payoffs = sum(calendar[y]["M1"]["payoffs"] for y in calendar)
    total_pairs = total_payoffs * (total_intervals - total_payoffs)
    within_year = sum(
        calendar[y]["M1"]["payoffs"] * (calendar[y]["M1"]["intervals"] - calendar[y]["M1"]["payoffs"])
        for y in calendar
    )
    year_share = within_year / total_pairs
    closure = read("reports/paper/task19_evidence/local_closure_output.json")
    month_share = closure["SA01_month"]["pair_structure"]["within_pair_share"]
    assert year_share == pytest.approx(0.17195, abs=5e-5)
    assert month_share == pytest.approx(0.019793, abs=5e-6)
    assert month_share < year_share  # finer strata shrink the within share


def test_default_calibration_is_absent_from_manuscript_but_present_in_evidence(macro):
    """M6: the omission and the underlying numbers must both be real."""
    for population in (macro["primary"], macro["unseen_vintage"]):
        for model in ("M1", "M2"):
            assert "default" in population[model]["calibration"]
    unseen = macro["unseen_vintage"]
    m1_slope = unseen["M1"]["calibration"]["default"]["slope"]
    m2_slope = unseen["M2"]["calibration"]["default"]["slope"]
    assert m1_slope > m2_slope
    observed = unseen["M2"]["calibration"]["default"]["observed_rate"]
    predicted = unseen["M2"]["calibration"]["default"]["mean_predicted"]
    assert observed / predicted > 3.0  # M2 under-predicts default by >3x on unseen
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    assert "0.000258" not in manuscript
    assert "1.0091" not in manuscript


def test_claim_sample_audit_resolved_every_binding():
    audit = read("reports/paper/review/task20_claim_sample_audit.json")
    pass1 = audit["pass_1_independent_rederivation"]
    pass2 = audit["pass_2_map_verification"]
    assert pass1["mismatched"] == 0 and pass1["matched"] >= 25
    assert pass2["failed"] == 0
    assert pass2["bindings"] == pass2["resolved_exactly"] == 257
    assert pass2["orphan_tokens"] == [] and pass2["unused_bindings"] == []
    assert pass2["all_carry_source_hash_and_commit"] is True
    assert audit["classification_tally"]["VALUE_ERROR"] == 0
    assert audit["classification_tally"]["SOURCE_ERROR"] == 0


def test_every_binding_still_resolves_against_its_own_artifact():
    """Re-run the map verification rather than trusting the recorded result."""
    bindings = read("reports/paper/task19_claim_evidence_map.json")["bindings"]
    assert len(bindings) == 257
    for entry in bindings:
        document = read(entry["source_artifact"])
        cursor = document
        for part in [p for p in entry["source_field"].split("/") if p]:
            part = part.replace("~1", "/").replace("~0", "~")
            cursor = cursor[int(part)] if isinstance(cursor, list) else cursor[part]
        expected = entry["exact_value"]
        if isinstance(cursor, (int, float)) and isinstance(expected, (int, float)):
            assert cursor == pytest.approx(expected, rel=1e-12, abs=1e-15), entry["binding_id"]
        else:
            assert cursor == expected, entry["binding_id"]


def test_manuscript_tokens_have_no_orphans():
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    tokens = set(re.findall(r"Q19: (Q\d{4})", manuscript))
    ids = {e["binding_id"] for e in read("reports/paper/task19_claim_evidence_map.json")["bindings"]}
    assert tokens == ids
    assert len(tokens) == 257


# ----------------------------------------------------------- audit artifact shape


def test_estimand_audit_covers_every_headline_and_names_shifts():
    audit = read("reports/paper/review/task20_estimand_audit.json")
    required = {
        "id", "headline", "population", "statistical_unit", "prediction_target", "horizon",
        "conditioning", "evaluation_window", "comparator", "metric", "uncertainty_unit", "nature",
    }
    assert len(audit["estimands"]) >= 8
    for row in audit["estimands"]:
        assert required <= set(row), row.get("id")
        assert row["nature"].isupper()
    assert len(audit["estimand_shifts_found"]) >= 4
    assert any("NOT stated" in s for s in audit["estimand_shifts_found"])


def test_abstract_audit_classifies_every_sentence():
    audit = read("reports/paper/review/task20_abstract_audit.json")
    allowed = {"SUPPORTED", "TOO_BROAD", "AMBIGUOUS", "MISSING_QUALIFICATION", "GOOD"}
    assert len(audit["sentences"]) >= 14
    for sentence in audit["sentences"]:
        assert sentence["class"] in allowed, sentence["n"]
        if sentence["class"] != "SUPPORTED" and sentence["class"] != "GOOD":
            assert sentence.get("note", "").strip(), sentence["n"]
    assert sum(audit["tally"].values()) == len(audit["sentences"])
    assert audit["material_omissions"]


def test_triage_classifies_every_analysis():
    triage = read("reports/paper/review/task20_analysis_triage.json")
    allowed = set(triage["classes"])
    for analysis in triage["analyses"]:
        assert analysis["class"] in allowed, analysis["analysis"]
        assert analysis["why"].strip()
    assert sum(triage["summary"].values()) == len(triage["analyses"])
    assert triage["summary"]["ARXIV_BLOCKER"] >= 1


def test_terminology_recommendations_are_concrete(verification):
    terms = verification["terminology_recommendations"]
    assert terms["pooled_decomposition_statistic"] == "cross-period concordance"
    assert "confidence" not in terms["eight_block_resampling_output"].lower().split(",")[0]
    assert terms["rejected"]
    assert set(terms["title_terms_to_replace"]) == {"Regime-Dependent", "Transport"}


def test_major_concerns_are_enumerated_and_resolvable(verification, report_text):
    findings = verification["findings"]
    assert findings["major_concerns"] == len(findings["major_concern_ids"]) == 7
    assert findings["all_major_concerns_resolvable_by_manuscript_editing"] is True
    for identifier in findings["major_concern_ids"]:
        tag = identifier.split()[0]
        assert f"### {tag}." in report_text, tag
    assert findings["new_relative_to_tasks_17_18"]


def test_prior_task_artifacts_untouched():
    assert read("reports/paper/review/task17_claim_survival_matrix.json")["claims"].__len__() == 19
    assert read("reports/paper/task18_verification.json")["no_frozen_metrics_changed"] is True
    assert read("reports/paper/task19_verification.json")
    assert (ROOT / "reports/paper/TASK17_ERRATUM.md").is_file()
