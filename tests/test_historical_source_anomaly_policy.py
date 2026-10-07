"""Standalone synthetic contracts for an unapplied governance design.

Reference operations below live only in tests; they are not production adapters.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads(
    (ROOT / "docs/track_b/historical_source_anomaly_policy.json").read_text(encoding="utf-8")
)


def rank(identifier, vintage):
    namespace = POLICY["sampling"]["namespace"].format(vintage=vintage, loan_id=identifier)
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest(), identifier


def counterfactual(rows, anomalies, vintage, n):
    # Eligibility is computed before ranking, from frozen keys only. Outcomes
    # cannot enter this reference operation, even when attached to toy rows.
    eligible = {r["id"] for r in rows if r["id"] not in anomalies}
    return sorted(eligible, key=lambda identifier: rank(identifier, vintage))[:n]


def hypothetical_field_quarantine(raw):
    return {
        "raw_token": raw,
        "value": None,
        "usable": False,
        "reason": POLICY["candidate_policies"]["C"]["reason_code"],
    }


def hypothetical_amendment(prior, response, version, parser_version):
    if not response["authority_verified"] or not response["response_date"]:
        raise ValueError("Authoritative dated evidence required")
    if response["response_sha256"] != hashlib.sha256(response["text"].encode()).hexdigest():
        raise ValueError("Response content hash mismatch")
    if tuple(map(int, version.split("."))) <= tuple(map(int, prior["version"].split("."))):
        raise ValueError("New amendment version required")
    if response["behavior_changes"] and parser_version == prior["parser_version"]:
        raise ValueError("New parser version required")
    return {
        "version": version,
        "parser_version": parser_version,
        "prior_audit_sha256": hashlib.sha256(
            json.dumps(prior, sort_keys=True).encode()
        ).hexdigest(),
        "old_rule": prior["rule"],
        "new_rule": response["rule"],
        "response_sha256": response["response_sha256"],
        "execution_authorized": False,
    }


def test_policy_is_design_only_and_does_not_resume_processing():
    assert POLICY["status"] == "DESIGN_ONLY_NOT_AUTHORIZED"
    for field in [
        "execution_authorized",
        "task8_resumption_authorized",
        "parser_change_authorized",
        "new_sample_freeze_authorized",
        "performance_access_authorized",
        "macro_acquisition_authorized",
        "model_fitting_authorized",
    ]:
        assert POLICY[field] is False
    assert all(p["authorized"] is False for p in POLICY["candidate_policies"].values())


def test_quarantine_precedes_ranking_and_eligibility_is_deterministic():
    rows = [{"id": f"synthetic{i}"} for i in range(40)]
    anomalies = {"synthetic2", "synthetic7"}
    expected = sorted(
        {r["id"] for r in rows} - anomalies, key=lambda identifier: rank(identifier, 2006)
    )[:10]
    assert counterfactual(rows, anomalies, 2006, 10) == expected
    assert counterfactual(rows[::-1] + rows, anomalies, 2006, 10) == expected
    assert not anomalies.intersection(expected)


def test_outcomes_and_predictors_cannot_change_eligibility_or_ranking():
    rows = [{"id": f"toy{i}", "default": i % 2, "rate": i} for i in range(30)]
    expected = counterfactual(rows, {"toy3"}, 2020, 10)
    for row in rows:
        row.update(default=1 - row["default"], rate=None, payoff=True)
    assert counterfactual(rows, {"toy3"}, 2020, 10) == expected


def test_hash_namespace_is_stable_and_archive_year_is_not_reassigned():
    identifier = "F09Q10000001"  # Synthetic fixture, not a source identifier.
    expected = hashlib.sha256(f"track-b-multivintage-v1:2008:{identifier}".encode()).hexdigest()
    assert rank(identifier, 2008) == (expected, identifier)
    assert rank(identifier, 2008) != rank(identifier, 2009)


def test_anomaly_outside_sample_has_zero_composition_effect():
    rows = [{"id": f"toy{i}"} for i in range(40)]
    ordered = counterfactual(rows, set(), 2006, 40)
    anomaly = ordered[-1]
    all_ids = set(counterfactual(rows, set(), 2006, 10))
    valid_ids = set(counterfactual(rows, {anomaly}, 2006, 10))
    assert all_ids == valid_ids and anomaly not in all_ids


def test_inside_sample_exclusion_has_one_entrant_one_departure_not_outcome_replacement():
    rows = [{"id": f"toy{i}"} for i in range(40)]
    anomaly = counterfactual(rows, set(), 2006, 40)[0]
    all_ids = set(counterfactual(rows, set(), 2006, 10))
    valid_ids = set(counterfactual(rows, {anomaly}, 2006, 10))
    assert all_ids - valid_ids == {anomaly}
    assert len(valid_ids - all_ids) == 1 and len(valid_ids ^ all_ids) == 2


def test_field_quarantine_preserves_raw_token_and_propagates_unresolved_reason():
    quarantine = hypothetical_field_quarantine(".")
    derived_toy_row = {"id": "toy", "original_rate_audit": copy.deepcopy(quarantine)}
    assert derived_toy_row["original_rate_audit"]["raw_token"] == "."
    assert derived_toy_row["original_rate_audit"]["value"] is None
    assert derived_toy_row["original_rate_audit"]["usable"] is False
    assert derived_toy_row["original_rate_audit"]["reason"] == (
        "UNRESOLVED_SOURCE_ANOMALY:ORIGINAL_RATE_DOT"
    )
    assert "MISSING_IN_SOURCE" not in quarantine.values()


def test_annual_agreement_is_distinct_from_quarter_agreement():
    anomalies = {a["key"]: a for a in POLICY["frozen_anomalies"]}
    for key in ["A3", "A4"]:
        a = anomalies[key]
        assert str(a["vintage"]) == a["embedded_quarter"][:4]
        assert a["member"] != a["embedded_quarter"]
        assert a["annual_dispute"] is False
    assert str(anomalies["A2"]["vintage"]) != anomalies["A2"]["embedded_quarter"][:4]
    assert anomalies["A2"]["annual_dispute"] is True


def test_empirical_rank_diagnostics_have_no_created_sample_or_physical_identifiers():
    expected = {"A1": 474260, "A2": 378990, "A3": 1413681, "A4": 646011}
    for row in POLICY["sampling_impact"]:
        assert row["raw_namespace_rank"] == expected[row["anomaly"]]
        assert row["within_first_20000"] is False
        assert row["symmetric_difference"] == 0
        assert row["samples_frozen"] is False and row["performance_accessed"] is False
        assert row["full_schema_eligibility_certified"] is False
        assert "loan_id" not in row
        a, b, c, d = row["policies"]
        assert a["selected_id_symmetric_difference"] is None
        assert b["candidate_eligible_universe"] == row["identifier_universe"] - 1
        assert b["anomaly_eligible"] is False
        assert b["selected_id_symmetric_difference"] == 0
        if row["anomaly"] == "A2":
            assert d["applicability"] == "blocked_annual_cohort_unresolved"
            assert d["candidate_eligible_universe"] is None
        if row["anomaly"] != "A1":
            assert c["candidate_eligible_universe"] is None


def test_provider_status_does_not_fabricate_contact_or_response():
    for request in POLICY["clarifications"]:
        assert request["submission_status"] == "NO_SUBMISSION_RECORDED"
        assert request["date_submitted"] is None
        assert request["response_date"] is None and request["response_sha256"] is None
        assert request["response_text"] is None


def response_fixture():
    text = "Synthetic authoritative clarification fixture, not an actual provider answer."
    return {
        "text": text,
        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "authority_verified": True,
        "response_date": "2026-10-07",
        "behavior_changes": True,
        "rule": "synthetic reviewed rule",
    }


def test_provider_amendment_is_new_version_without_rewriting_prior_audit():
    prior = {"version": "1.0.0", "parser_version": "1.0.0", "rule": "unresolved"}
    original = copy.deepcopy(prior)
    amendment = hypothetical_amendment(prior, response_fixture(), "1.1.0", "1.1.0")
    assert prior == original
    assert amendment["old_rule"] == "unresolved" and amendment["new_rule"] != "unresolved"
    assert amendment["execution_authorized"] is False
    assert amendment["prior_audit_sha256"]


@pytest.mark.parametrize("defect", ["authority", "hash", "date", "version", "parser_version"])
def test_provider_amendment_fails_closed_on_missing_authority_or_version(defect):
    prior = {"version": "1.0.0", "parser_version": "1.0.0", "rule": "unresolved"}
    response = response_fixture()
    version = parser_version = "1.1.0"
    if defect == "authority":
        response["authority_verified"] = False
    elif defect == "hash":
        response["response_sha256"] = "0" * 64
    elif defect == "date":
        response["response_date"] = None
    elif defect == "version":
        version = "1.0.0"
    else:
        parser_version = "1.0.0"
    with pytest.raises(ValueError):
        hypothetical_amendment(prior, response, version, parser_version)


def test_materiality_includes_effects_beyond_frequency():
    assert len(POLICY["materiality_dimensions"]) == 8
    assert POLICY["prevalence"]["fraction"] == 4 / 12169807
    assert POLICY["sampling"]["criteria_use_outcomes"] is False
    assert POLICY["robustness_design"]["models_run"] is False
