"""Regression gates for Task 18 frozen-array sensitivity closure.

These tests assert that Task 18 executed only the authorized analyses, produced no
empirical change, and that every derived figure still re-derives from the frozen
artifacts. Arithmetic the report depends on is recomputed here, so drift in a frozen
artifact fails this suite rather than silently invalidating the report.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reports/paper"
REPORT = PAPER / "TASK18_FROZEN_ARRAY_SENSITIVITY_CLOSURE.md"
REGISTRATION = PAPER / "task18_analysis_registration.json"
CLOSURE_SCRIPT = ROOT / "scripts/task18_local_closure.py"

BASE_COMMIT = "b64c20f2ab7f8779e2b1eb233deebd35da2c2645"
REGISTRATION_SHA = "b78aa2b0f63c620d6890efe203d6da811a887bd8a2697dd5a64d2a7bcfcefbed"
AUTHORIZED = ["SA01", "SA02", "SA03", "SA06"]
FORBIDDEN = ["SA04", "SA05", "SA07", "SA08", "SA09", "SA10", "SA11"]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def macro():
    return read("reports/track_b/macro_competing_risk_validation.json")


@pytest.fixture(scope="module")
def registration():
    return read("reports/paper/task18_analysis_registration.json")


@pytest.fixture(scope="module")
def results():
    return read("reports/paper/task18_results.json")


@pytest.fixture(scope="module")
def report_text():
    return REPORT.read_text(encoding="utf-8")


# --------------------------------------------------------------------- scoping


def test_all_task18_outputs_exist(registration):
    for relative in registration["expected_outputs"]:
        assert (ROOT / relative).is_file(), relative


def test_registration_hash_is_stable_and_referenced(registration):
    import hashlib

    payload = REGISTRATION.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(payload).hexdigest() == REGISTRATION_SHA
    assert registration["written_before_metric_production"] is True
    assert registration["base_commit"] == BASE_COMMIT
    for name in (
        "task18_sa01_within_period_auc.json",
        "task18_sa02_calendar_decomposition.json",
        "task18_sa03_unseen_vintage.json",
        "task18_sa06_cif_entry_distribution.json",
    ):
        assert read(f"reports/paper/{name}")["registration_sha256_lf"] == REGISTRATION_SHA


def test_registration_scopes_authorized_and_prohibited(registration):
    assert registration["analyses_authorized"] == AUTHORIZED
    assert registration["analyses_prohibited"] == FORBIDDEN


def test_only_authorized_analyses_executed(results):
    assert results["analyses_executed"] == AUTHORIZED
    assert results["analyses_not_executed"] == FORBIDDEN
    assert set(results["statuses"]) == set(AUTHORIZED)
    allowed = {"CONFIRMED", "REFUTED", "QUALIFIED", "INCONCLUSIVE"}
    for analysis, status in results["statuses"].items():
        assert status in allowed, f"{analysis}:{status}"


@pytest.mark.parametrize("analysis", FORBIDDEN)
def test_forbidden_analyses_produced_no_output(analysis):
    for path in PAPER.glob("task18_*.json"):
        blob = json.dumps(read(f"reports/paper/{path.name}"))
        hits = [m for m in re.finditer(analysis, blob)]
        for hit in hits:
            window = blob[max(0, hit.start() - 240) : hit.end() + 240]
            assert any(
                token in window
                for token in (
                    "analyses_prohibited",
                    "analyses_not_executed",
                    "NOT_EXECUTED",
                    "Do NOT execute",
                    "remaining",
                    "peer-review",
                    "STRONGLY_RECOMMENDED",
                    "OPTIONAL",
                    "NOT_NEEDED",
                )
            ), f"{analysis} referenced outside a prohibition/deferral context in {path.name}"


def test_no_fitting_or_regeneration_claimed(registration, results):
    for forbidden in (
        "fitting or refitting any model",
        "regenerating or mutating predictions",
        "fitting any calibrator",
        "computing AIC/BIC or complexity adjustment",
        "facility-weighted rescoring",
        "inspecting any Fannie outcome",
    ):
        assert forbidden in registration["prohibited_operations"], forbidden
    sa01 = read("reports/paper/task18_sa01_within_period_auc.json")
    assert "nothing was fitted" in sa01["method"]
    assert "No prediction array was read" in sa01["method"]


def test_no_fannie_outcome_accessed():
    freeze = read("docs/paper/experiment_design_freeze.json")
    b13 = next(e for e in freeze["experiments"] if e["experiment_id"] == "B13_13S")
    assert b13["protocol_status"] == "DRAFT_NOT_YET_AUTHORIZED"
    assert b13["task13a_authorized"] is False
    for path in PAPER.glob("task18_*.json"):
        blob = json.dumps(read(f"reports/paper/{path.name}")).lower()
        assert "fannie_source_provenance_closure" not in blob or "unauthorized" in blob


def test_no_manuscript_version_created_or_edited():
    assert not (ROOT / "paper/main_v0.3.md").exists()
    consequences = read("reports/paper/task18_manuscript_consequences.json")
    assert consequences["applied"] is False


# ------------------------------------------------------------------ SA01 gates


def test_sa01_eligibility_rule_frozen_and_no_auc_imputed():
    sa01 = read("reports/paper/task18_sa01_within_period_auc.json")
    el = sa01["eligibility"]
    assert el["imputed_auc_count"] == 0
    assert el["excluded_single_class_or_sparse"] == 0
    assert el["eligible_strata"] == el["total_strata"] == 8
    assert "SUPPORTED" in el["rule"]
    assert sa01["interpretation"] in {
        "WITHIN_PERIOD_GAIN_PRESENT",
        "WITHIN_PERIOD_GAIN_NEGLIGIBLE",
        "WITHIN_PERIOD_GAIN_REVERSED",
        "INCONCLUSIVE_DUE_TO_EVENT_SUPPORT",
    }


def test_sa01_auc_decomposition_identity_holds(macro):
    """Re-derive the within/between split from frozen aggregates."""
    sa01 = read("reports/paper/task18_sa01_within_period_auc.json")
    calendar = macro["stability"]["calendar"]
    total_n = sum(calendar[y]["M1"]["intervals"] for y in calendar)
    total_k = sum(calendar[y]["M1"]["payoffs"] for y in calendar)
    total_pairs = total_k * (total_n - total_k)
    within_pairs = sum(
        calendar[y]["M1"]["payoffs"] * (calendar[y]["M1"]["intervals"] - calendar[y]["M1"]["payoffs"])
        for y in calendar
    )
    assert sa01["pair_structure"]["total_case_control_pairs"] == total_pairs
    assert sa01["pair_structure"]["within_year_pairs"] == within_pairs

    for model in ("M1", "M2"):
        within = (
            sum(
                calendar[y]["M1"]["payoffs"]
                * (calendar[y]["M1"]["intervals"] - calendar[y]["M1"]["payoffs"])
                * calendar[y][model]["payoff_auc"]
                for y in calendar
            )
            / within_pairs
        )
        pooled = macro["primary"][model]["scores"]["payoff_auc"]
        between = (pooled * total_pairs - within_pairs * within) / (total_pairs - within_pairs)
        assert sa01["aggregates"][model]["within_year_pair_weighted"] == pytest.approx(within, abs=1e-12)
        assert sa01["aggregates"][model]["between_year_solved"] == pytest.approx(between, abs=1e-12)
        assert 0.0 <= between <= 1.0


def test_sa01_between_period_dominates():
    sa01 = read("reports/paper/task18_sa01_within_period_auc.json")
    gains = sa01["gains"]
    assert gains["between_year_solved"] > gains["within_year_pair_weighted"]
    assert gains["within_share_of_pooled_gain"] < 0.25
    assert sa01["pair_structure"]["between_year_pair_share"] > 0.8


def test_sa01_structural_claim_recorded_as_refuted(report_text):
    sa01 = read("reports/paper/task18_sa01_within_period_auc.json")
    assert sa01["structural_check"]["verdict"] == "TASK17_STRUCTURAL_CLAIM_REFUTED"
    # the specification half must still be verified, only the inference falls
    spec = sa01["structural_check"]["specification_verified"]
    assert spec["all_added_terms_are_month_constant"] is True
    assert spec["interactions_prohibited"] is True
    assert "TASK 17 CLAIM REFUTED" in report_text


def test_sa01_structural_counterexample_is_arithmetically_correct():
    """h_P under a multinomial softmax is not rank-invariant to a month-constant shift."""
    import math

    def h_payoff(a, b, g_p, g_d):
        e_p, e_d = math.exp(a + g_p), math.exp(b + g_d)
        return e_p / (1.0 + e_d + e_p)

    f1, f2 = (0.0, 0.0), (-1.0, -10.0)
    assert h_payoff(*f1, 0.0, 0.0) > h_payoff(*f2, 0.0, 0.0)
    assert h_payoff(*f1, 0.0, 10.0) < h_payoff(*f2, 0.0, 10.0)
    # contrast: a single binary logit IS rank invariant
    sigmoid = lambda z: 1 / (1 + math.exp(-z))
    for shift in (0.0, 10.0):
        assert sigmoid(0.0 + shift) > sigmoid(-1.0 + shift)


# ------------------------------------------------------------------ SA02 gates


def test_sa02_calendar_decomposition_reconstructs_headline_delta(macro):
    sa02 = read("reports/paper/task18_sa02_calendar_decomposition.json")
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
    assert sa02["reconciliation"]["reconciles"] is True
    assert sa02["reconciliation"]["reconstructed_delta"] == pytest.approx(reconstructed, abs=1e-15)


def test_sa02_two_thousand_twenty_share(macro):
    sa02 = read("reports/paper/task18_sa02_calendar_decomposition.json")
    calendar = macro["stability"]["calendar"]
    total = sum(calendar[y]["M1"]["intervals"] for y in calendar)
    delta = macro["paired_facility"]["intervals"]["joint_log_loss"]["delta"]
    contribution = (
        calendar["2020"]["M1"]["intervals"]
        / total
        * (calendar["2020"]["M2"]["joint_log_loss"] - calendar["2020"]["M1"]["joint_log_loss"])
    )
    assert sa02["concentration"]["calendar_2020_share_of_delta"] == pytest.approx(
        contribution / delta, abs=1e-12
    )
    assert contribution / delta > 0.85


def test_sa02_distinguishes_the_two_ex_2020_quantities(macro):
    sa02 = read("reports/paper/task18_sa02_calendar_decomposition.json")
    pair = sa02["two_ex_2020_quantities"]
    residual = pair["contribution_residual"]["value"]
    renormalised = pair["renormalised_ex_2020_evaluation_delta"]["value"]
    assert residual != renormalised
    calendar = macro["stability"]["calendar"]
    kept = {y: calendar[y] for y in calendar if y != "2020"}
    weight = sum(kept[y]["M1"]["intervals"] for y in kept)
    expected = sum(
        kept[y]["M1"]["intervals"]
        / weight
        * (kept[y]["M2"]["joint_log_loss"] - kept[y]["M1"]["joint_log_loss"])
        for y in kept
    )
    assert renormalised == pytest.approx(expected, abs=1e-12)
    assert "correction" in pair


def test_sa02_respects_the_pandemic_interpretation_rule():
    sa02 = read("reports/paper/task18_sa02_calendar_decomposition.json")
    interpretation = sa02["interpretation"]
    assert any("not exclusive to 2020" in s for s in interpretation["permitted_and_supported"])
    assert any("COVID caused the failure" in s for s in interpretation["prohibited_and_not_asserted"])
    assert "NOT_SUPPORTED" in interpretation["task11_distinction_preserved"]
    diagnostics = read("reports/track_b/macro_signal_attribution_stability.json")
    assert diagnostics["assessment"]["hypothesis_register"]["H7"]["status"] == "NOT_SUPPORTED"


def test_sa02_ex_2020_interval_caveat_present():
    sa02 = read("reports/paper/task18_sa02_calendar_decomposition.json")
    ex = sa02["year_block_sensitivity"]["ex_2020"]
    assert ex["blocks"] == 7
    assert "coverage" in ex["coverage_caveat"].lower()
    assert ex["includes_zero"] is True


# ------------------------------------------------------------------ SA03 gates


def test_sa03_metrics_match_frozen_source_exactly(macro):
    sa03 = read("reports/paper/task18_sa03_unseen_vintage.json")
    for metric, row in sa03["metrics"].items():
        for population, block in (("seen", macro["primary"]), ("unseen", macro["unseen_vintage"])):
            for model in ("M1", "M2"):
                assert row[f"{population}_{model}"] == block[model]["scores"][metric]


def test_sa03_reversal_is_real_and_classified(macro):
    sa03 = read("reports/paper/task18_sa03_unseen_vintage.json")
    unseen = macro["unseen_vintage"]
    assert unseen["M2"]["scores"]["joint_log_loss"] < unseen["M1"]["scores"]["joint_log_loss"]
    assert sa03["primary_decision_metric"]["reversal_confirmed"] is True
    assert sa03["study_level_classification"].startswith(
        ("A. CONSISTENT", "B. TEMPORAL", "C. MIXED", "D. NO_DEFENSIBLE")
    )


def test_sa03_explanations_are_labelled_hypotheses():
    sa03 = read("reports/paper/task18_sa03_unseen_vintage.json")
    assert sa03["hypotheses_for_the_reversal"]
    for item in sa03["hypotheses_for_the_reversal"]:
        assert item["status"].startswith("HYPOTHESIS"), item["hypothesis"]
    assert sa03["calendar_decomposition"]["status"] == "NOT_EXECUTABLE"


# ------------------------------------------------------------------ SA06 gates


def test_sa06_entry_counts_reconcile(macro):
    sa06 = read("reports/paper/task18_sa06_cif_entry_distribution.json")
    assert sa06["cohort"]["landmarks"] == macro["cif"]["landmarks"]
    counts = sa06["corroboration_lower_bound"]["frozen_facility_counts_by_year"]
    for year, value in counts.items():
        assert value == macro["stability"]["calendar"][year]["M1"]["facilities"]
    at_least = sa06["corroboration_lower_bound"]["facilities_entering_in_2019_at_least"]
    assert at_least == macro["stability"]["calendar"]["2019"]["M1"]["facilities"]
    assert sa06["corroboration_lower_bound"]["share_entering_in_2019_at_least"] == pytest.approx(
        at_least / macro["cif"]["landmarks"], abs=1e-12
    )
    assert sa06["corroboration_lower_bound"]["facilities_possibly_entering_later_at_most"] == (
        macro["cif"]["landmarks"] - at_least
    )


def test_sa06_entry_bound_matches_the_frozen_exclusion_mask(macro):
    """Bound must use entry + h - 1 <= cutoff, as cif.py does."""
    sa06 = read("reports/paper/task18_sa06_cif_entry_distribution.json")

    def ordinal(token):
        text = str(token).replace("-", "")
        return int(text[:4]) * 12 + int(text[4:6]) - 1

    def label(n):
        return f"{n // 12:04d}-{n % 12 + 1:02d}"

    cutoff = ordinal("2026-02")
    per_horizon = sa06["derivation_upper_bound"]["per_horizon_latest_permitted_entry"]
    for horizon, expected in per_horizon.items():
        assert label(cutoff - int(horizon) + 1) == expected, horizon
    assert sa06["derivation_upper_bound"]["entry_bound"].endswith(per_horizon["60"])
    start = ordinal("2019-01")
    assert sa06["derivation_upper_bound"]["permitted_entry_months"] == (
        cutoff - 60 + 1 - start + 1
    )
    for horizon in ("12", "24", "36", "60"):
        assert macro["cif"]["horizons"][horizon]["calendar_truncated_landmarks"] == 0


def test_sa06_all_four_horizons_reported_for_both_causes(macro):
    sa06 = read("reports/paper/task18_sa06_cif_entry_distribution.json")
    horizons = sa06["all_frozen_horizons"]
    assert set(horizons) == {"12", "24", "36", "60"}
    for horizon, row in horizons.items():
        frozen = macro["cif"]["horizons"][horizon]["models"]
        for model in ("M0", "M1", "M2"):
            assert row[f"{model}_payoff"] == frozen[model]["payoff"]["mean_predicted_cif"]
            assert row[f"{model}_default"] == frozen[model]["default"]["mean_predicted_cif"]
        assert row["observed_payoff_cif"] == frozen["M1"]["payoff"]["observed_default_cif"]
        assert row["observed_default_cif"] == frozen["M1"]["default"]["observed_default_cif"]


def test_sa06_path_classification_and_language_verdict():
    sa06 = read("reports/paper/task18_sa06_cif_entry_distribution.json")
    assert sa06["path_classification"] in {
        "SINGLE_OR_NEAR_SINGLE_HISTORICAL_PATH",
        "LIMITED_SET_OF_HISTORICAL_PATHS",
        "DISPERSED_HISTORICAL_ROLLING_PATHS",
    }
    assert sa06["manuscript_language_assessment"]["supported"] is False
    assert sa06["cif_2020_calendar_alignment"]["scope"].startswith("This is a calendar-alignment")


# ------------------------------------------------- uncertainty, PIT, integrity


def test_payoff_brier_intervals_verified_and_classified(macro, results):
    block = results["payoff_brier_uncertainty"]
    facility = macro["paired_facility"]["intervals"]["payoff_brier"]
    calendar = macro["paired_calendar"]["intervals"]["payoff_brier"]
    assert block["facility"]["lower"] == facility["lower"]
    assert block["calendar"]["lower"] == calendar["lower"]
    assert facility["lower"] > 0
    assert calendar["lower"] < 0 < calendar["upper"]
    assert block["classification"] == "ROBUST_FACILITY_ONLY"


def test_facility_and_calendar_uncertainty_distinguished(macro, results):
    hierarchy = results["uncertainty_hierarchy"]
    assert macro["paired_facility"]["conditional_on_realized_calendar"] is True
    assert macro["paired_calendar"]["conditional_on_realized_calendar"] is False
    assert "conditional on the realised calendar" in hierarchy["verified_wording"]
    assert "Neither establishes broad macroeconomic sampling uncertainty" in hierarchy["verified_wording"]
    assert hierarchy["level_3_macro_sampling"]["estimates"].startswith("Not estimated")


def test_pit_terminology_respects_release_lag_limitation(results):
    pit = results["pit_terminology"]
    audit = read("reports/track_b/pit_macro_data_audit_task9_api_v5.json")
    for series, record in audit["release_lags"].items():
        assert record["count"] == 0, series
        assert record["evidence_quality"] == "UNMEASURED_EXACT_PROVIDER_DATES", series
        assert pit["what_is_not_established"]["release_lag_counts"][series] == 0
    assert pit["verdict"] != "POINT_IN_TIME"
    assert pit["verdict"] == "VINTAGE_AND_REVISION_AWARE"


def test_manuscript_consequences_are_complete():
    consequences = read("reports/paper/task18_manuscript_consequences.json")
    required = {
        "finding_id",
        "source_analysis",
        "verified_result",
        "affected_task17_change_ids",
        "affected_claim_ids",
        "required_manuscript_change",
        "allowed_language",
        "prohibited_language",
        "preprint_blocking_resolved",
    }
    registry = {
        claim["claim_id"]
        for claim in read("docs/paper/track_b_claim_evidence_registry.json")["claims"]
    }
    ids = set()
    for finding in consequences["findings"]:
        assert required <= set(finding), finding.get("finding_id")
        assert finding["allowed_language"] and finding["prohibited_language"]
        for claim_id in finding["affected_claim_ids"]:
            assert claim_id in registry, claim_id
        ids.add(finding["finding_id"])
    assert len(ids) == len(consequences["findings"])


def test_local_closure_script_is_read_only_and_aggregate_only():
    source = CLOSURE_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("np.save", ".fit(", "joblib.dump", "to_csv", "shutil", "os.remove", "unlink"):
        assert forbidden not in source, forbidden
    assert "roc_auc_score" in source
    assert 'np.load' in source
    assert "no_loan_level_values_emitted" in source
    assert "facility_keys" not in source  # never resolves facility identifiers


def test_preservation_recorded_and_nothing_empirical_changed():
    verification = read("reports/paper/task18_verification.json")
    assert verification["base_commit"] == BASE_COMMIT
    for flag in (
        "no_experiments_rerun",
        "no_models_fitted",
        "no_predictions_regenerated",
        "no_frozen_metrics_changed",
        "no_new_fannie_outcomes",
        "no_manuscript_version_created",
        "tracked_files_unchanged",
    ):
        assert verification[flag] is True, flag
    for task in ("task10", "task11", "task12", "task14", "task15", "task16", "task17"):
        assert verification["preserved"][task] is True, task
    assert verification["preserved"]["numeric_bindings"] == 123


def test_task17_artifacts_unchanged():
    survival = read("reports/paper/review/task17_claim_survival_matrix.json")
    assert len(survival["claims"]) == 19
    assert (PAPER / "TASK17_HOSTILE_SCIENTIFIC_REVIEW.md").is_file()
    assert read("reports/paper/task17_verification.json")["no_invented_result"] is True


def test_preprint_gate_recorded(results, report_text):
    assert results["preprint_analysis_gate"] in {
        "PREPRINT_ANALYSIS_GAPS_CLOSED",
        "PREPRINT_ANALYSIS_GAPS_PARTIALLY_CLOSED",
        "PREPRINT_ANALYSIS_GAPS_NOT_CLOSED",
    }
    assert results["preprint_analysis_gate"] in report_text
    assert "does not mean the manuscript is ready" in results["gate_caveat"]
