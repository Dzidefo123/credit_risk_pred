"""Threshold, affordability, loss conservation and frozen scoring isolation."""

import json

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.validation import ORIGINATION_FEATURES, ORIGINATION_TARGET
from credit_risk.decisioning import runner
from credit_risk.decisioning.policy import decide, policy_summary
from credit_risk.decisioning.settings import CreditStrategy, PolicyComparisonConfig


def inputs(n=1, **overrides):
    values = dict(MonthlyIncome=5000.0, DebtRatio=0.2, RevolvingUtilizationOfUnsecuredLines=0.3)
    values.update(overrides)
    return pd.DataFrame({k: [v] * n for k, v in values.items()})


def test_boundaries_and_inclusive_grades():
    result = decide(inputs(6), [0, 0.01, 0.03, 0.06, 0.1, 1], CreditStrategy())
    assert result.risk_grade.tolist() == ["G1", "G1", "G2", "G3", "G4", "G5"]
    assert result.decision.tolist() == [
        "APPROVE",
        "APPROVE",
        "MANUAL_REVIEW",
        "MANUAL_REVIEW",
        "DECLINE",
        "DECLINE",
    ]
    assert (result.loc[result.decision != "APPROVE", "recommended_limit"] == 0).all()


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("MonthlyIncome", np.nan, "INCOME_UNAVAILABLE_OR_ZERO"),
        ("MonthlyIncome", 0.0, "INCOME_UNAVAILABLE_OR_ZERO"),
        ("DebtRatio", np.nan, "DEBT_RATIO_MISSING"),
        ("DebtRatio", 1.01, "DEBT_RATIO_ABOVE_AUTO_GUARD"),
        ("RevolvingUtilizationOfUnsecuredLines", np.nan, "UTILIZATION_MISSING"),
        ("RevolvingUtilizationOfUnsecuredLines", 1.01, "UTILIZATION_ABOVE_AUTO_GUARD"),
        ("MonthlyIncome", 100.0, "LIMIT_BELOW_MINIMUM"),
    ],
)
def test_no_automatic_offer_when_affordability_unsupported(field, value, reason):
    result = decide(inputs(**{field: value}), [0.01], CreditStrategy()).iloc[0]
    assert result.decision == "MANUAL_REVIEW"
    assert result.recommended_limit == result.assumed_ead == result.expected_loss_proxy == 0
    assert reason in result.reason_codes
    assert decide(inputs(**{field: value}), [0.2], CreditStrategy()).iloc[0].decision == "DECLINE"


@pytest.mark.parametrize("pd_values", [[np.nan], [np.inf], [-0.1], [1.1], [], [[0.1]]])
def test_invalid_probability_vectors(pd_values):
    with pytest.raises(ValueError):
        decide(inputs(), pd_values, CreditStrategy())


@pytest.mark.parametrize("value", [-1.0, np.inf])
def test_invalid_raw_inputs(value):
    with pytest.raises(ValueError):
        decide(inputs(MonthlyIncome=value), [0.01], CreditStrategy())


def test_monotone_limits_and_rounding_do_not_force_minimum():
    policy = CreditStrategy(approve_below_pd=0.9, decline_at_or_above_pd=1.0)
    limits = decide(inputs(5), [0.005, 0.02, 0.04, 0.08, 0.2], policy).recommended_limit
    assert np.all(np.diff(limits) <= 0)
    for field, values in [
        ("MonthlyIncome", [1000.0, 2000.0, 5000.0]),
        ("DebtRatio", [0.0, 0.5, 1.0]),
        ("RevolvingUtilizationOfUnsecuredLines", [0.0, 0.5, 1.0]),
    ]:
        results = [
            decide(inputs(**{field: v}), [0.01], policy).recommended_limit.iloc[0] for v in values
        ]
        assert (
            np.all(np.diff(results) >= 0)
            if field == "MonthlyIncome"
            else np.all(np.diff(results) <= 0)
        )
    result = decide(inputs(), [0.01], policy).iloc[0]
    assert result.recommended_limit <= 5000 * 2 / (1 + 0.2) / (1 + 0.3) * (1 - 0.01)
    assert result.recommended_limit % 100 == 0


def test_outcomes_never_influence_decisions_and_loss_conserves():
    frame = inputs(3)
    frame[ORIGINATION_TARGET] = [0, 0, 0]
    a = decide(frame, [0.01, 0.02, 0.2], CreditStrategy())
    frame[ORIGINATION_TARGET] = [1, 1, 1]
    pd.testing.assert_frame_equal(a, decide(frame, [0.01, 0.02, 0.2], CreditStrategy()))
    s = policy_summary(a, [1, 0, 1])
    assert s["approved"] + s["manual_review"] + s["declined"] == 3
    assert s["expected_bad_rate"] == pytest.approx(0.015)
    assert s["historical_selected_bad_rate"] == 0.5
    assert s["expected_loss_proxy"] == pytest.approx((a.pd * a.assumed_ead * 0.45).sum())
    assert s["assumed_ead"] == pytest.approx(0.5 * s["total_offered_limit"])
    with pytest.raises(ValueError):
        policy_summary(a, [0, 1])


def test_empty_and_zero_approval_rates_are_undefined():
    assert policy_summary(decide(inputs(0), [], CreditStrategy()))["approval_rate"] is None
    assert (
        policy_summary(decide(inputs(), [1.0], CreditStrategy()), [1])["expected_bad_rate"] is None
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grade_limit_factors": [1.0]},
        {"grade_limit_factors": [1.0, 0.8, 0.9, 0.4, 0.2]},
        {"grade_limit_factors": [1.0, 0.8, 0.6, 0.4, -0.1]},
        {"assumed_drawdown": 1.1},
        {"limit_increment": 0.0},
        {"name": "../bad"},
        {"unknown": True},
    ],
)
def test_config_rejects_inconsistent_assumptions(kwargs):
    with pytest.raises(ValueError):
        CreditStrategy(**kwargs)


def test_duplicate_policy_names_rejected():
    with pytest.raises(ValueError):
        PolicyComparisonConfig(policies=[CreditStrategy(), CreditStrategy()])


def test_runner_only_scores_development_without_fit(tmp_path, monkeypatch):
    frame = pd.DataFrame({k: np.arange(4, dtype=float) for k in ORIGINATION_FEATURES})
    frame["MonthlyIncome"] = 5000.0
    frame["DebtRatio"] = 0.2
    frame["RevolvingUtilizationOfUnsecuredLines"] = 0.3
    frame[ORIGINATION_TARGET] = [0, 1, 0, 1]
    monkeypatch.setattr(runner, "verify_calibration_lock", lambda *args: None)
    seen = []

    class FrozenModel:
        def predict_proba(self, x):
            seen.extend(x.index.tolist())
            return np.array([[0.99, 0.01], [0.8, 0.2]])

    monkeypatch.setattr(
        runner,
        "verify_experiment",
        lambda *args: (
            {"source_sha256": "abc"},
            None,
            frame,
            {"development": np.array([0, 2]), "test": np.array([1, 3])},
            None,
        ),
    )
    selection = {"preferred_candidate": "xgboost"}
    validation = {
        "source_sha256": "abc",
        "target_semantics": "two-year delinquency",
        "artifacts_sha256": {"xgboost_selected.joblib": "modelhash"},
    }
    monkeypatch.setattr(
        runner, "load_selected_model", lambda *args: (FrozenModel(), selection, validation)
    )
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    (validation_dir / "selection.json").write_text("{}")
    result = runner.run_policy_comparison("unused", "unused", validation_dir, tmp_path / "output")
    assert seen == [0, 2]
    assert not result["final_test_scored"]
    assert result["policies"][0]["applicants"] == 2
    with pytest.raises(FileExistsError):
        runner.run_policy_comparison("unused", "unused", validation_dir, tmp_path / "output")


def test_integrity_rejection_happens_before_deserialization(tmp_path, monkeypatch):
    (tmp_path / "selection.json").write_text("{}")
    (tmp_path / "validation.json").write_text(
        json.dumps({"artifacts_sha256": {"selection.json": "bad"}})
    )

    def forbidden(*args):
        raise AssertionError("Must not deserialize")

    monkeypatch.setattr(runner.joblib, "load", forbidden)
    with pytest.raises(ValueError, match="checksum"):
        runner.load_selected_model(tmp_path)


def test_selected_model_checksum_checked_before_load(tmp_path, monkeypatch):
    selection = {"test_used_for_selection": False, "preferred_candidate": "xgboost"}
    selected_path = tmp_path / "selection.json"
    selected_path.write_text(json.dumps(selection), encoding="utf-8")
    (tmp_path / "xgboost_selected.joblib").write_bytes(b"corrupt")
    manifest = {
        "selection": selection,
        "artifacts_sha256": {
            "selection.json": runner.digest(selected_path),
            "xgboost_selected.joblib": "wrong",
        },
    }
    (tmp_path / "validation.json").write_text(json.dumps(manifest), encoding="utf-8")

    def forbidden(*args):
        raise AssertionError("Must verify before deserializing")

    monkeypatch.setattr(runner.joblib, "load", forbidden)
    with pytest.raises(ValueError, match="model checksum"):
        runner.load_selected_model(tmp_path)


def test_calibration_lock_checks_code_source_and_choices(tmp_path):
    import inspect
    from hashlib import sha256

    lock = {
        "source_sha256": "source",
        "choices": {"xgboost": "sigmoid"},
        "preferred_candidate": "xgboost",
        "model_hashes": {"xgboost": "base"},
        "calibration_code_sha256": runner.digest(inspect.getfile(runner.ProbabilityCalibrator)),
    }
    hashed = sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
    (tmp_path / "test_consumption.json").write_text(
        json.dumps({"selection": lock, "selection_sha256": hashed}), encoding="utf-8"
    )
    experiment = {"source_sha256": "source", "artifacts_sha256": {"xgboost.joblib": "base"}}
    selection = {"selected_methods": {"xgboost": "sigmoid"}, "preferred_candidate": "xgboost"}
    runner.verify_calibration_lock(tmp_path, experiment, {"selection_sha256": hashed}, selection)
    with pytest.raises(ValueError, match="provenance"):
        runner.verify_calibration_lock(
            tmp_path,
            {**experiment, "source_sha256": "different"},
            {"selection_sha256": hashed},
            selection,
        )
    with pytest.raises(ValueError, match="lock"):
        runner.verify_calibration_lock(
            tmp_path, experiment, {"selection_sha256": "different"}, selection
        )


def test_missing_policy_columns_fail_with_contract_error():
    with pytest.raises(ValueError, match="requires raw"):
        decide(pd.DataFrame({"MonthlyIncome": [5000.0]}), [0.01], CreditStrategy())
