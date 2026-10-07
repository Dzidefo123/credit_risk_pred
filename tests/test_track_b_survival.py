"""Task6 toy risk sets, competing-risk identities and censor-adjusted metrics."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from credit_risk.track_b.survival.ledger import Ledger
from credit_risk.track_b.survival.math import aj, curves, horizon_metrics, support
from credit_risk.track_b.survival.models import DYNAMIC, STATIC, fit, forecast, frame, probabilities
from credit_risk.track_b.survival.risk import (
    check_feature_time,
    ordinal,
    trajectory,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "docs/track_b/survival_feature_registry.json").read_text())
PROTOCOL = json.loads((ROOT / "docs/track_b/survival_competing_risk_protocol.json").read_text())


def subjects(exit, event, entry=None):
    return pd.DataFrame(
        dict(
            entry_time=np.zeros(len(exit), int) if entry is None else entry,
            exit_time=exit,
            event_code=event,
        )
    )


def test_constant_competing_risk_identity_and_net_risk_difference():
    result = curves(np.full((1, 12), 0.1), np.full((1, 12), 0.2))[0]
    np.testing.assert_allclose(result.sum(axis=1), 1)
    assert result[-1, 0] == pytest.approx(0.7**12)
    assert result[-1, 1] == pytest.approx((1 - 0.7**12) / 3)
    assert result[-1, 2] == pytest.approx(2 * (1 - 0.7**12) / 3)
    assert result[-1, 1] < 1 - 0.9**12
    assert (np.diff(result[:, 1:3], axis=0) >= 0).all()


def test_zero_hazard_and_exhausted_risk_mass():
    r = curves(np.zeros((2, 3)), np.zeros((2, 3)))
    np.testing.assert_array_equal(r[:, :, 0], 1)
    np.testing.assert_array_equal(r[:, :, 1:], 0)
    r = curves(np.array([[1.0, 0]]), np.zeros((1, 2)))
    np.testing.assert_array_equal(r[0, :, 1], 1)
    with pytest.raises(ValueError):
        curves(np.array([[0.8]]), np.array([[0.3]]))


def test_toy_AJ_KM_and_tied_events():
    f = subjects([1, 1, 2, 3], [1, 2, 0, 1])
    r = aj(f, 3)
    assert r[0]["survival"] == 0.5 and r[0]["default_cif"] == 0.25 and r[0]["payoff_cif"] == 0.25
    assert r[1]["survival"] == 0.5
    assert r[2]["default_cif"] == 0.75 and r[2]["payoff_cif"] == 0.25
    assert r[2]["naive_net_default"] > r[2]["default_cif"]
    assert all(
        row["survival"] + row["default_cif"] + row["payoff_cif"] == pytest.approx(1) for row in r
    )


def test_delayed_entry_joins_only_after_entry_and_before_exit():
    f = subjects([26, 27], [1, 2], entry=[24, 25])
    r = aj(f, 27)
    assert all(row["at_risk"] == 0 for row in r[:24])
    assert r[24]["at_risk"] == 1 and r[25]["at_risk"] == 2
    with pytest.raises(ValueError):
        aj(subjects([1], [1], entry=[1]), 2)


def test_synthetic_competing_events_estimate_known_probabilities():
    # Exact empirical distribution: 100 facilities, first month 10 default /20 payoff.
    r = aj(subjects([1] * 30 + [2] * 70, [1] * 10 + [2] * 20 + [0] * 70), 1)[0]
    assert r["default_cif"] == 0.1 and r["payoff_cif"] == 0.2 and r["survival"] == 0.7


def test_uncensored_IPCW_equals_standard_Brier_and_competing_AUC():
    f = subjects([1, 1, 3, 3], [1, 2, 1, 0])
    p = np.array([0.8, 0.1, 0.2, 0.1])
    m = horizon_metrics(f, p, 1, minimum_events=1)
    y = [1, 0, 0, 0]
    assert m["ipcw_brier"] == pytest.approx(brier_score_loss(y, p))
    assert m["cumulative_dynamic_auc"] == roc_auc_score(y, p)
    assert m["known_status_facilities"] == 4


def test_censor_last_known_boundary_and_IPCW_denominator():
    f = subjects([1, 2, 3, 3], [0, 1, 2, 0])
    p = np.array([0.2, 0.7, 0.1, 0.2])
    atone = horizon_metrics(f, p, 1, minimum_events=1)
    assert atone["known_status_facilities"] == 4
    # At2: first subject censored before horizon; other three get inverse .75 survival.
    m = horizon_metrics(f, p, 2, minimum_events=1)
    expected = ((0.7 - 1) ** 2 + 0.1**2 + 0.2**2) / 0.75 / 4
    assert m["ipcw_brier"] == pytest.approx(expected)
    assert m["known_status_facilities"] == 3
    with pytest.raises(ValueError):
        horizon_metrics(subjects([1], [0], entry=[0.5]), [0.1], 1)


def test_rare_events_tail_and_training_support_explicit():
    m = horizon_metrics(subjects([1, 2], [1, 0]), [0.8, 0.1], 1)
    assert m["cumulative_dynamic_auc"] is None
    assert (
        support({"at_risk": 199, "censor_survival_before": 1}, 36, 50)
        == "INSUFFICIENT TAIL SUPPORT"
    )
    assert (
        support({"at_risk": 300, "censor_survival_before": 1}, 60, 54)
        == "INSUFFICIENT TRAINING-TIME SUPPORT"
    )


def toy_trajectory():
    rows = []
    for i, month in enumerate(["2012-01", "2012-02", "2012-03", "2012-04"]):
        r = {n: 1.0 for n in [*STATIC, *DYNAMIC]}
        r.update(
            loan_id="synthetic",
            t0=month,
            eligible=i >= 1,
            loan_age=23.0 + i,
            delinquency_state="00",
            loan_purpose="P",
            occupancy_status="P",
        )
        rows.append(r)
    history = {ordinal(r["t0"]): dict(category="none", state="00") for r in rows}
    return pd.DataFrame(rows), history


@pytest.mark.parametrize("cause,code", [("default", 1), ("payoff", 2)])
def test_entry_delayed_clock_first_event_and_no_post_event_rows(cause, code):
    f, h = toy_trajectory()
    h[ordinal("2012-03")] = dict(category=cause, state="03" if code == 1 else "00")
    subject, rows, status = trajectory(f, h, list(dict.fromkeys([*STATIC, *DYNAMIC])))
    assert status == "included" and subject["entry_month"] == "2012-02"
    assert subject["entry_mortgage_age"] == 24
    assert subject["exit_time"] == 1 and subject["event_code"] == code
    assert len(rows) == 1 and rows[0]["t0"] == "2012-02" and rows[0]["target_month"] == "2012-03"


@pytest.mark.parametrize("kind", ["administrative", "ambiguous", "unknown"])
def test_unresolved_endpoints_censor_without_labels(kind):
    f, h = toy_trajectory()
    h[ordinal("2012-04")] = dict(category=kind, state="XX")
    subject, rows, status = trajectory(f, h, list(dict.fromkeys([*STATIC, *DYNAMIC])))
    assert subject["event_code"] == 0 and subject["exit_reason"] == kind
    assert subject["exit_time"] == 1 and len(rows) == 1


def test_zero_exposure_and_prior_default_quarantined():
    f, h = toy_trajectory()
    h[ordinal("2012-03")] = dict(category="ambiguous", state="XX")
    assert trajectory(f, h, list(STATIC))[2] == "zero_duration_quarantined"
    h[ordinal("2012-01")] = dict(category="default", state="03")
    with pytest.raises(ValueError, match="prior"):
        trajectory(f, h, list(STATIC))


def test_development_calendar_censor_no_2015_intervals():
    f, h = toy_trajectory()
    f["t0"] = ["2014-10", "2014-11", "2014-12", "2015-01"]
    h = {ordinal(t): dict(category="none", state="00") for t in f.t0}
    h[ordinal("2015-01")] = dict(category="default", state="03")
    subject, rows, _ = trajectory(f, h, list(STATIC), end="2014-12")
    assert subject["event_code"] == 0 and subject["exit_reason"] == "calendar_censor"
    assert all(r["target_month"] <= "2014-12" for r in rows)


def test_feature_time_boundaries_and_event_exits():
    check_feature_time("2016-01", "2016-01", event_month="2016-02")
    with pytest.raises(ValueError, match="Future"):
        check_feature_time("2016-01", "2016-02")
    with pytest.raises(ValueError, match="Post-event"):
        check_feature_time("2016-02", "2016-02", event_month="2016-02")


@pytest.mark.parametrize(
    "forbidden", ["next_state", "exit_time", "event_code", "next_balance", "future_state_path"]
)
def test_predictor_firewall(forbidden):
    r = copy.deepcopy(REGISTRY)
    r["dynamic"].append(forbidden)
    f, _ = toy_trajectory()
    f["duration"] = 1
    with pytest.raises(ValueError, match="firewall"):
        frame(f, r, "dynamic")


def test_joint_logits_development_preprocessing_and_probability_conservation():
    f, _ = toy_trajectory()
    f = pd.concat([f] * 30, ignore_index=True)
    f["duration"] = np.arange(len(f)) % 36 + 1
    f["event_code"] = np.arange(len(f)) % 3
    f["orig_dti"] = np.arange(len(f), dtype=float)
    m = fit(f, REGISTRY, "structural")
    before = m[0].named_transformers_["numeric"][1].mean_.copy()
    q = probabilities(m, f, REGISTRY, "structural")
    np.testing.assert_allclose(q.sum(axis=1), 1)
    future = f.copy()
    future["orig_dti"] = 100000.0
    probabilities(m, future, REGISTRY, "structural")
    np.testing.assert_array_equal(before, m[0].named_transformers_["numeric"][1].mean_)
    curve = forecast(m, f.head(2), REGISTRY, 12)
    np.testing.assert_allclose(curve.sum(axis=2), 1)


def test_ledger_protocol_first_model_freeze_and_one_consumption(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", {"protocol": "fixed"})
    assert ledger.data["state"] == "PROTOCOL_FROZEN"
    with pytest.raises(ValueError):
        ledger.open()
    ledger.freeze({"models": "fixed"})
    ledger.open()
    ledger.complete()
    assert ledger.data["state"] == "CONSUMED"
    with pytest.raises(ValueError):
        ledger.open()
    with pytest.raises(ValueError):
        Ledger(tmp_path / "ledger.json", {})


def test_frozen_sampling_and_calendar_protocol():
    assert (
        PROTOCOL["sample_sha256"]
        == "e719a6b4d23caca8ac55eeaa6fae54da23c95887dd83b3ac22cd925d63902832"
    )
    assert (
        PROTOCOL["split"]["development_end"] == "2014-12"
        and PROTOCOL["split"]["purged_year"] == "2015"
    )
    assert PROTOCOL["split"]["evaluation_start"] == "2016-01"
    assert PROTOCOL["horizons"] == [12, 24, 36, 60]


def test_completed_first_event_audit_conservation_and_tail_rule():
    from credit_risk.track_b.survival.study import lf_hash

    r = json.loads((ROOT / "reports/track_b/survival_competing_risk_validation.json").read_text())
    assert r["survival_audit"]["raw_first_endpoints"] == dict(
        default=623, payoff=18147, administrative=29, ambiguous=33, active_or_unknown=1168
    )
    assert r["survival_audit"]["global_cohort"]["facilities"] == 19606
    assert r["survival_audit"]["partitions"]["evaluation"]["default_facilities"] == 95
    assert r["evaluation_ledger"]["state"] == "CONSUMED"
    assert r["evaluation_ledger"]["prediction_generation_count"] == 1
    assert r["evaluation_ledger"]["post_evaluation_model_change"] is False
    assert r["model_specifications"]["max_training_duration"] == 54
    assert r["horizon_results"]["60"]["status"] == "INSUFFICIENT TRAINING-TIME SUPPORT"
    assert "metrics" not in r["horizon_results"]["60"]
    for row in r["nonparametric_evaluation"]:
        assert row["survival"] + row["default_cif"] + row["payoff_cif"] == pytest.approx(1)
    for name, expected in r["model_specifications"]["source_code_sha256_lf"].items():
        assert lf_hash(ROOT / name.replace("\\", "/")) == expected


def test_post_evaluation_support_audit_uses_cached_predictions_only():
    from credit_risk.track_b.survival.study import lf_hash

    r = json.loads((ROOT / "reports/track_b/survival_competing_risk_validation.json").read_text())
    audit = r["time_support_audit"]
    assert audit["models_modified"] is False and audit["predictions_regenerated"] is False
    assert audit["primary_horizon_metrics_recomputed"] is False
    assert (
        audit["all_intervals"]
        == audit["supported_intervals"] + audit["outside_training_duration_intervals"]
    )
    assert audit["source_sha256_lf"] == lf_hash(ROOT / "src/credit_risk/track_b/survival/audit.py")
    assert r["task5_bridge"]["cached_Task5_scores_only"] is True
    assert r["task5_bridge"]["Task5_ledger_unchanged"] is True


def test_gap_censors_without_reentry_after_observations_resume():
    f, h = toy_trajectory()
    del h[ordinal("2012-04")]
    resumed = f.iloc[-1].copy()
    resumed["t0"] = "2012-05"
    f = pd.concat([f, pd.DataFrame([resumed])], ignore_index=True)
    h[ordinal("2012-05")] = dict(category="none", state="00")
    subject, rows, status = trajectory(f, h, list(STATIC))
    assert status == "included" and len(rows) == 1
    assert subject["exit_reason"] == "gap_or_observation_end" and subject["event_code"] == 0
