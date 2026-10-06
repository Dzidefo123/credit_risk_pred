"""Prespecified mortgage decision and event-contract integrity, without data or models."""

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/track_b"


def protocol():
    return json.loads((DOCS / "mortgage_research_protocol.json").read_text(encoding="utf-8"))


def test_selection_is_formal_but_does_not_authorize_data_or_models():
    data = protocol()
    assert data["status"] == "prespecified_before_loan_records"
    assert data["selection"]["dataset_selected"] is True
    assert "Standard" in data["selection"]["development_source"]
    for flag in (
        "record_level_data_accessed",
        "downloads_authorized_by_task1",
        "models_implemented",
    ):
        assert data["selection"][flag] is False
    assert data["track_a"]["reinterpretation_permitted"] is False
    # Task 2 may implement data engineering; predictive/accounting modules remain absent.
    for component in ("pd", "survival", "lgd", "ead", "sicr", "staging", "ecl"):
        assert not (ROOT / "src/credit_risk/track_b" / component).exists()


def test_decision_has_exactly_ten_distinct_scoped_components():
    decisions = protocol()["components"]
    assert len(decisions) == 10 and len({r["component"] for r in decisions}) == 10
    rows = {r["component"]: r for r in decisions}
    assert rows["12-month mortgage PD"]["decision"] == "PROCEED"
    assert rows["Borrower-level portfolio modeling"]["decision"] == "UNSUPPORTED"
    assert rows["Revolving CCF/EAD"]["decision"] == "OUT_OF_SCOPE"
    assert rows["ECL"]["decision"] == "LATER"
    assert "true timed" in rows["Workout LGD"]["scope"]
    for r in decisions:
        assert r["scope"]


def test_default_payoff_and_administrative_causes_are_disjoint():
    event = protocol()["event"]
    sets = [
        set(event[k])
        for k in ("credit_termination_codes", "competing_payoff_codes", "administrative_exit_codes")
    ]
    assert sets == [{"02", "03", "09"}, {"01"}, {"15", "16", "96"}]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1 :])
    assert event["numeric_delinquency_min"] == 3 and event["numeric_delinquency_max"] == 99
    assert event["reo_state"] == "RA" and event["unknown_state"] == "XX"
    assert event["missing_state_implies_current"] is False
    assert event["terminal_code_15_implies_default"] is False
    assert "quarantine" in event["default_and_payoff_same_month"]
    assert "fail closed" in event["unrecognized_codes"]


def test_calendar_horizon_and_unknown_labels_are_not_silently_negatives():
    data = protocol()
    time = data["time"]
    assert time["unit"] == "calendar_month" and time["outcome_offsets"] == [1, 12]
    assert time["horizon_months"] == 12
    labels = data["labels"]
    assert labels["positive_default"]["binary_default_12m"] == 1
    assert labels["negative_survived_horizon"]["binary_default_12m"] == 0
    assert labels["competing_payoff"]["binary_default_12m"] == 0
    for status in (
        "right_censored",
        "insufficient_followup",
        "ambiguous_event_order",
        "not_incident_risk_eligible",
    ):
        assert labels[status]["binary_default_12m"] is None
    assert "facility risk ends" in labels["competing_payoff"]["rule"]
    assert time["real_time_pit_claim_permitted"] is False
    assert time["first_payment_month_is_origination_date"] is False
    assert time["primary_reentry_after_gap"] is False


def test_small_pilot_is_prespecified_without_outcome_selection():
    data = protocol()
    pilot = data["pilot_plan"]
    assert pilot["vintage_year"] == 2010 and pilot["maximum_retained_loans"] == 1000
    assert "SHA256" in pilot["selection"] and "no outcome-based" in pilot["selection"]
    assert pilot["event_count_can_change_selection"] is False
    assert "all available monthly histories" in pilot["retain"]
    assert "separate acquisition authorization" in pilot["archive_scope"]
    assert "no empirical performance claim" in pilot["purpose"]


def test_crosswalk_and_contract_versions_remain_anchored():
    data = protocol()
    for key in ("crosswalk", "contract"):
        payload = (ROOT / data["evidence"][key + "_path"]).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == data["evidence"][key + "_sha256_lf"]
    assert data["track_a"]["target_commit"] == "0d3dc5f9dd1be6a40fc03fa91a1fe8b367a28b8c"
    assert data["track_a"]["tag_object"] == "20be4bf1bd291afb5dfad5b0befbe761651f1de1"


@pytest.mark.parametrize("name", ["DATASET_SELECTION_DECISION.md", "MORTGAGE_RESEARCH_DESIGN.md"])
def test_design_document_links_and_explicit_limitations(name):
    body = (DOCS / name).read_text(encoding="utf-8")
    for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", body):
        if link.startswith("https://"):
            continue
        assert (DOCS / link).resolve().is_file(), link
    assert "No" in body and "research" in body
    assert "C:\\" not in body


def test_design_covers_event_order_censoring_exposure_and_future_leakage():
    body = (DOCS / "MORTGAGE_RESEARCH_DESIGN.md").read_text(encoding="utf-8")
    for term in (
        "ambiguous_event_order",
        "k=12",
        "k=13",
        "delayed entry",
        "cumulative incidence",
        "h_D(k)",
        "principal-exposure proxy",
        "disposition-loss severity proxy",
        "future payments",
        "before each interval",
    ):
        assert term.lower() in body.lower(), term
    assert "later cure does not erase" in body
    assert "Censored/unknown outcomes are not training zeros" in body
    assert "not a second development/tuning source" in (
        DOCS / "DATASET_SELECTION_DECISION.md"
    ).read_text(encoding="utf-8")
