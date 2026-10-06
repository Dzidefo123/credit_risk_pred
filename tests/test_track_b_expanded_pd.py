"""Task5 safety/math tests use synthetic facilities, not licensed observations."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from credit_risk.track_b.expanded_pd import core
from credit_risk.track_b.expanded_pd.study import lf_hash
from credit_risk.track_b.pd.baseline import cohort, counts, split

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "docs/track_b/expanded_pd_feature_registry.json").read_text())
DESIGN = json.loads((ROOT / "docs/track_b/expanded_pd_design.json").read_text())


def sample(n=120):
    rows = []
    for i in range(n):
        y = int(i % 7 == 0)
        row = {k: float(i % 13 + 1) for k in core.NUMERIC}
        row.update(
            orig_credit_score=float(600 + i),
            orig_ltv=float(60 + i % 30),
            loan_purpose="P" if i % 2 else "R",
            occupancy_status="P" if i % 3 else "I",
            delinquency_state=str(i % 3),
            loan_id=f"synthetic-{i}",
            t0="2013-06",
            eligible=True,
            outcome_status="positive_default" if y else "negative_survived_horizon",
            binary_default_12m=y,
            event_offset=1 if y else np.nan,
            observed_followup_months=12,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_grouped_development_split_is_label_blind_deterministic_disjoint():
    f = sample()
    a = core.internal_split(f)
    changed = f.copy()
    changed.binary_default_12m = 1 - changed.binary_default_12m
    b = core.internal_split(changed.sample(frac=1, random_state=2))
    assert sum(len(g) for g in a) == len(f)
    for i in range(3):
        assert set(a[i].loan_id) == set(b[i].loan_id)
        for j in range(i + 1, 3):
            assert not set(a[i].loan_id) & set(a[j].loan_id)


def test_frozen_dates_purge_and_same_outer_hash_rule():
    assert DESIGN["development_end"] == "2014-12" and DESIGN["evaluation_start"] == "2016-01"
    assert DESIGN["purged"] == "2015"
    f = sample()
    f["t0"] = "2015-06"
    assert all(g.empty for g in split(cohort(f)))
    assert (
        DESIGN["sample_sha256"]
        == "e719a6b4d23caca8ac55eeaa6fae54da23c95887dd83b3ac22cd925d63902832"
    )
    assert (
        DESIGN["panel_sha256"] == "7583f24d1dfcd9d11e13fe0a76af154285edc21e844cf15e65500b2881211dad"
    )


@pytest.mark.parametrize(
    "field",
    [
        "future_delinquency",
        "future_balance",
        "future_payment_status",
        "future_modification",
        "future_default",
        "actual_loss",
        "event_offset",
        "loan_id",
        "t0",
    ],
)
def test_future_outcome_audit_columns_cannot_enter_predictors(field):
    registry = copy.deepcopy(REGISTRY)
    registry["selected"].append(field)
    with pytest.raises(ValueError, match="firewall"):
        core.frame(sample(), registry)


def test_exact_t0_delinq_allowed_next_month_rejected():
    core.check_time("delinquency_state", "2013-06", "2013-06")
    with pytest.raises(ValueError, match="Future"):
        core.check_time("delinquency_state", "2013-07", "2013-06")
    assert "delinquency_state" in core.frame(sample(), REGISTRY)
    assert not {"delinquency_state", "loan_age"} & set(core.frame(sample(), REGISTRY, static=True))


def test_development_only_preprocessing_and_deterministic_logistic():
    f = sample()
    f.loc[0, "orig_dti"] = np.nan
    m = core.fit_model("logistic", f, REGISTRY, DESIGN["logistic"])
    numeric = m[0].named_transformers_["numeric"]
    assert numeric[0].statistics_[2] == f.orig_dti.median()
    before = numeric[1].mean_.copy()
    evaluation = sample(5)
    evaluation["orig_dti"] = 10000.0
    m.predict_proba(core.frame(evaluation, REGISTRY))
    np.testing.assert_array_equal(before, numeric[1].mean_)
    repeat = core.fit_model("logistic", f, REGISTRY, DESIGN["logistic"])
    np.testing.assert_allclose(m[-1].coef_, repeat[-1].coef_)


def test_hazard_risk_events_exit_and_unknown_followup():
    f = sample(6)
    f["loan_id"] = ["a", "a", "a", "b", "b", "c"]
    f["eligible"] = [True, True, False, True, False, True]
    f["outcome_status"] = [
        "positive_default",
        "positive_default",
        "not_incident_risk_eligible",
        "competing_payoff",
        "not_incident_risk_eligible",
        "ambiguous_event_order",
    ]
    f["event_offset"] = [2, 1, np.nan, 1, np.nan, 1]
    f["observed_followup_months"] = [2, 1, 0, 1, 0, 0]
    risk = core.hazard_periods(f)
    assert risk.loan_id.tolist() == ["a", "a", "b"]
    assert risk.hazard_event.tolist() == [0, 1, 0]
    assert counts(risk)["default_loans"] == 1
    double = pd.concat([risk, risk.iloc[[1]]])
    with pytest.raises(ValueError, match="Duplicated"):
        core.hazard_periods(double)


def test_survival_product_endpoints_not_sum():
    h = np.full((1, 12), 0.1)
    assert core.survival_pd(h)[0] == pytest.approx(1 - 0.9**12)
    assert core.survival_pd(h)[0] != pytest.approx(1.2)
    assert core.survival_pd(np.zeros((1, 12)))[0] == 0
    assert core.survival_pd(np.ones((1, 12)))[0] == 1
    with pytest.raises(ValueError):
        core.survival_pd(np.zeros((2, 11)))
    with pytest.raises(ValueError):
        core.survival_pd(np.full((1, 12), np.nan))


def test_hazard_forecast_uses_deterministic_clock_not_future_observations():
    seen = []

    class Model:
        def predict_proba(self, x):
            seen.append(x.copy())
            return np.tile([0.99, 0.01], (len(x), 1))

    f = sample(3)
    p = core.hazard_forecast(Model(), f, REGISTRY)
    np.testing.assert_allclose(p, 1 - 0.99**12)
    for i, x in enumerate(seen):
        np.testing.assert_array_equal(x.loan_age, f.loan_age + i)
        np.testing.assert_array_equal(x.delinquency_state, f.delinquency_state.astype(float))


def test_bounded_xgboost_configuration_and_no_eval_set():
    assert len(DESIGN["xgboost_candidates"]) == 2
    assert all(
        c["max_depth"] <= 3 and c["n_estimators"] <= 180 for c in DESIGN["xgboost_candidates"]
    )
    config = {**DESIGN["xgboost_fixed"], **DESIGN["xgboost_candidates"][0]}
    m = core.pipeline("xgboost", config)
    assert m[-1].random_state == 51005 and m[-1].get_params()["scale_pos_weight"] is None


@pytest.mark.parametrize(
    "probabilities", [[0.1, 0.1, 0.8, 0.5, 0.8], [0.0, 0.0, 1.0, 0.2, 0.7], [0.1] * 5]
)
def test_weighted_fast_metrics_match_reference_with_ties(probabilities):
    y = np.array([0, 1, 1, 0, 1])
    w = np.array([2, 0, 1, 3, 1])
    m = core.WeightedMetrics(y, probabilities).evaluate(w)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, probabilities, sample_weight=w))
    assert m["average_precision"] == pytest.approx(
        average_precision_score(y, probabilities, sample_weight=w)
    )
    assert m["brier"] == pytest.approx(brier_score_loss(y, probabilities, sample_weight=w))
    assert m["log_loss"] == pytest.approx(
        log_loss(y, probabilities, sample_weight=w, labels=[0, 1])
    )


def test_paired_cluster_samples_shared_and_deterministic():
    f = sample(20)
    f["loan_id"] = [f"synthetic-{i // 2}" for i in range(20)]
    predictions = {"logistic": np.linspace(0.01, 0.1, 20), "xgboost": np.linspace(0.01, 0.1, 20)}
    r = core.clustered(f, predictions, draws=20, seed=5)
    assert r == core.clustered(f, predictions, draws=20, seed=5)
    assert r["unit"] == "loan_id"
    for v in r["paired"]["xgboost-minus-logistic"].values():
        assert v["point"] == v["lower"] == v["upper"] == 0


def test_single_class_discrimination_explicit_and_probability_metrics_valid():
    r = core.WeightedMetrics([0, 0], [0.1, 0.2]).evaluate([1, 1])
    assert r["roc_auc"] is None and r["average_precision"] is None
    assert r["brier"] == pytest.approx(0.025)


def test_ledger_freeze_access_once_and_specification_integrity(tmp_path):
    ledger = core.Ledger(tmp_path)
    ledger.freeze({"configuration": "fixed"})
    with pytest.raises(ValueError, match="reevaluation"):
        ledger.freeze({})
    ledger.open()
    ledger.complete(["AP", "Brier"])
    assert ledger.data["state"] == "CONSUMED" and ledger.data["prediction_generation_count"] == 1
    with pytest.raises(ValueError, match="unopened"):
        ledger.open()
    other = core.Ledger(tmp_path / "other")
    other.freeze({"configuration": "fixed"})
    saved = json.loads(other.path.read_text())
    saved["specification"]["configuration"] = "changed"
    other.path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="integrity"):
        other.open()


def test_old_task4_and_design_protocol_preservation():
    report = json.loads((ROOT / "reports/track_b/sample_expansion_feasibility.json").read_text())
    assert report["event_support"]["development"]["default_loans"] == 246
    assert report["event_support"]["evaluation"]["default_loans"] == 95
    assert report["sample"]["sample_set_sha256"] == DESIGN["sample_sha256"]
    assert REGISTRY["historical_knowledge_time"] == "UNVERIFIED"
    assert len(lf_hash(ROOT / "docs/track_b/expanded_pd_design.json")) == 64
    assert all(
        REGISTRY["fields"][n]["classification"]
        in ["STATIC_AT_ORIGINATION", "TIME_VARYING_KNOWN_AT_T0"]
        for n in REGISTRY["selected"]
    )


def test_flat_explanation_ranks_are_explicitly_undefined():
    from credit_risk.track_b.expanded_pd.study import finite_stat

    assert finite_stat(np.nan) is None
    assert finite_stat(0.5) == 0.5


def test_completed_evaluation_and_frozen_model_code_integrity():
    r = json.loads((ROOT / "reports/track_b/expanded_pd_validation.json").read_text())
    ledger = r["evaluation_ledger"]
    assert ledger["state"] == "CONSUMED" and ledger["prediction_generation_count"] == 1
    assert ledger["post_evaluation_model_change"] is False
    for name, expected in r["model_specifications"]["source_code_sha256_lf"].items():
        assert lf_hash(ROOT / name.replace("\\", "/")) == expected
    assert r["cohort_reproduction"]["cohorts"]["primary"] == dict(
        landmarks=1241045, loans=19590, positive_landmarks=7153, default_loans=618
    )
    assert r["cohort_reproduction"]["cohorts"]["evaluation"]["default_loans"] == 95
    assert r["clustered_uncertainty"]["draws"] == 1000
    assert r["clustered_uncertainty"]["invalid_single_class_draws"] == 0
    assert r["sample_sha256"] == DESIGN["sample_sha256"]


def test_logged_diagnostic_supplement_does_not_change_primary_evaluation():
    r = json.loads((ROOT / "reports/track_b/expanded_pd_validation.json").read_text())
    audit = r["post_evaluation_diagnostic_supplement"]
    assert audit["models_modified"] is False and audit["predictions_regenerated"] is False
    assert audit["primary_metrics_recomputed"] is False
    assert audit["frozen_xgboost_split_counts"]["delinquency_state"] == 0
    assert audit["source_sha256_lf"] == lf_hash(
        ROOT / "src/credit_risk/track_b/expanded_pd/audit.py"
    )
    assert r["champion"] == "logistic"
    assert r["hazard"]["events_once_per_facility"] is True


def test_discrete_drift_audit_handles_concentrated_zero_states():
    from credit_risk.track_b.expanded_pd.audit import categorical_psi

    a = pd.Series(["00"] * 99 + ["01"])
    b = pd.Series(["00"] * 90 + ["01"] * 10)
    assert categorical_psi(a, b) > 0
    assert categorical_psi(a, a) == 0
