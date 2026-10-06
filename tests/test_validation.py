"""Final-holdout isolation, tamper checks, reliability and dated maturity."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.pipeline import Pipeline

from credit_risk.data.validation import ORIGINATION_FEATURES, ORIGINATION_TARGET
from credit_risk.models.calibration import ProbabilityCalibrator
from credit_risk.models.pd import run_origination_experiment, split_origination
from credit_risk.utils.config import ModelConfig, XGBoostConfig
from credit_risk.validation.backtesting import backtest_predictions, segment_diagnostics
from credit_risk.validation.diagnostics import (
    calibration_edges,
    group_bootstrap_comparison,
    reliability_diagnostics,
)
from credit_risk.validation.holdout_registry import HoldoutRegistry
from credit_risk.validation.runner import run_validation, verify_experiment
from credit_risk.validation.settings import ValidationConfig


def test_reliability_includes_endpoints_empty_bins_and_hand_calculated_ece():
    result = reliability_diagnostics([0, 1, 0, 1], [0, 0.2, 0.8, 1], [0, 0.4, 0.6, 1])
    assert [row["rows"] for row in result["bins"]] == [2, 0, 2]
    assert result["ece"] == pytest.approx(0.4)
    assert result["observed_expected_ratio"] == 1
    assert result["bins"][1]["observed_bad_rate"] is None
    assert np.array_equal(calibration_edges(np.full(30, 0.2)), [0, 0.2, 1])
    for row in result["bins"]:
        if row["rows"]:
            assert row["wilson_lower"] <= row["observed_bad_rate"] <= row["wilson_upper"]
    with pytest.raises(ValueError, match="Edges"):
        reliability_diagnostics([0, 1], [0.2, 0.8], [0, 0.5, 0.5, 1])


def test_paired_group_bootstrap_deterministic_identical_models_zero_difference():
    labels = np.tile([0, 1], 30)
    p = np.tile([0.1, 0.9], 30)
    # Each cluster contains both outcomes, so every bootstrap replicate is evaluable.
    groups = np.repeat(np.arange(30), 2)
    a = group_bootstrap_comparison(labels, {"a": p, "b": p}, groups, samples=20)
    assert a == group_bootstrap_comparison(labels, {"a": p, "b": p}, groups, samples=20)
    assert a["valid_samples"] == 20
    assert a["intervals"]["a"]["roc_auc"] == {"lower": 1.0, "upper": 1.0}
    for bounds in a["paired_differences"]["intervals"].values():
        assert bounds == {"lower": 0.0, "upper": 0.0}


def test_bootstrap_single_class_replicates_counted():
    result = group_bootstrap_comparison([0, 1], {"a": np.array([0.2, 0.8])}, [0, 1], samples=100)
    assert result["skipped_single_class"] > 0
    assert result["valid_samples"] + result["skipped_single_class"] == 100


def test_dated_backtest_excludes_immature_and_unknown_outcomes_without_mutating():
    frame = pd.DataFrame(
        {
            "observation_date": ["2024-01-01"] * 4,
            "performance_end": ["2025-01-01", "2025-01-01", "2025-02-01", "2025-01-01"],
            "label": [0, 1, 1, np.nan],
            "probability": [0.1, 0.8, 0.9, 0.2],
        }
    )
    before = frame.copy(deep=True)
    result = backtest_predictions(frame, "2025-01-01")
    assert result.iloc[0]["rows"] == 2 and result.iloc[0]["roc_auc"] == 1
    assert backtest_predictions(frame, "2024-12-31").empty
    assert_frame_equal(frame, before)
    with pytest.raises(ValueError, match="Dated"):
        backtest_predictions(frame.drop(columns="observation_date"), "2025-01-01")
    frame.loc[0, "performance_end"] = "2023-01-01"
    with pytest.raises(ValueError, match="precedes"):
        backtest_predictions(frame, "2025-01-01")


def test_segments_missing_age_and_low_support_explicit():
    frame = pd.DataFrame(
        {
            "age": [np.nan, 30, 60, 60],
            "MonthlyIncome": [1, np.nan, 2, 2],
            "RevolvingUtilizationOfUnsecuredLines": [0, 1.2, 0.5, 0.5],
        }
    )
    rows = segment_diagnostics(frame, [0, 1, 0, 1], [0.1, 0.8, 0.2, 0.7], ValidationConfig())
    assert any(row["segment"] == "missing" and row["dimension"] == "age" for row in rows)
    assert all(row["low_support"] for row in rows)


@pytest.fixture
def frozen_run(tmp_path):
    HoldoutRegistry.initialize(tmp_path / "holdout_registry.json")
    rng = np.random.default_rng(24)
    n = 400
    frame = pd.DataFrame(
        {name: rng.integers(0, 6, n).astype(float) for name in ORIGINATION_FEATURES}
    )
    frame["age"] = rng.integers(21, 80, n)
    frame["MonthlyIncome"] = rng.lognormal(8, 0.5, n)
    frame["DebtRatio"] = rng.uniform(0, 2, n)
    frame["RevolvingUtilizationOfUnsecuredLines"] = rng.uniform(0, 1.5, n)
    frame[ORIGINATION_TARGET] = (
        (frame.NumberOfTimes90DaysLate >= 4) & (rng.random(n) > 0.3)
    ).astype(int)
    source, run = tmp_path / "source.csv", tmp_path / "run"
    frame.to_csv(source, index=False)
    model_config = ModelConfig(xgboost=XGBoostConfig(n_estimators=12, n_jobs=1))
    run_origination_experiment(source, run, model_config)
    splits = split_origination(pd.read_csv(source), model_config)
    config = ValidationConfig(bootstrap_samples=20, minimum_calibration_events=2)
    return source, run, splits, config


def test_calibrators_fit_only_calibration_and_test_scored_after_selection_lock(
    frozen_run, tmp_path, monkeypatch
):
    source, run, splits, config = frozen_run
    output = tmp_path / "validation"
    fit_labels, score_rows = [], []
    original_fit, original_predict = ProbabilityCalibrator.fit, Pipeline.predict_proba

    def traced_fit(self, p, y):
        fit_labels.append(set(y.index))
        return original_fit(self, p, y)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Base models must remain frozen")

    def traced_predict(self, X, **kwargs):
        score_rows.append(set(X.index))
        if set(X.index) == set(splits["test"]):
            assert (output / "selection.json").exists()
            assert (run / "test_consumption.json").exists()
            assert (
                HoldoutRegistry(tmp_path / "holdout_registry.json").read().entries[0].status
                == "consumed"
            )
            assert (
                json.loads((output / "selection.json").read_text())["test_used_for_selection"]
                is False
            )
        return original_predict(self, X, **kwargs)

    monkeypatch.setattr(ProbabilityCalibrator, "fit", traced_fit)
    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    monkeypatch.setattr(Pipeline, "predict_proba", traced_predict)
    result = run_validation(
        source, run, output, config, registry_path=tmp_path / "holdout_registry.json"
    )
    assert fit_labels == [set(splits["calibration"])] * 6
    assert (
        score_rows
        == [set(splits["calibration"]), set(splits["development"])] * 2 + [set(splits["test"])] * 2
    )
    assert result["fresh_test_access"] is True
    saved = joblib.load(output / "xgboost_selected.joblib")
    preds = pd.read_csv(output / "test_predictions.csv")
    frame = pd.read_csv(source)
    p = saved.predict_proba(frame.iloc[splits["test"]].loc[:, ORIGINATION_FEATURES])[:, 1]
    method = result["selection"]["selected_methods"]["xgboost"]
    assert np.allclose(p, preds[f"xgboost__{method}"])
    with pytest.raises(FileExistsError):
        run_validation(
            source, run, output, config, registry_path=tmp_path / "holdout_registry.json"
        )


def test_test_consumption_guard_blocks_repeat_before_loading(frozen_run, tmp_path, monkeypatch):
    source, run, _, config = frozen_run
    registry_path = tmp_path / "holdout_registry.json"
    first = run_validation(source, run, tmp_path / "first", config, registry_path=registry_path)
    assert first["fresh_test_access"] is True
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("No repeat loading/scoring"))
    for repeated_config in (config, config.model_copy(update={"probability_epsilon": 0.01})):
        with pytest.raises(ValueError, match="already consumed"):
            run_validation(
                source, run, tmp_path / "repeat", repeated_config, registry_path=registry_path
            )


@pytest.mark.parametrize("target", ["source", "model", "splits"])
def test_tampered_artifacts_rejected_before_deserialization(frozen_run, monkeypatch, target):
    source, run, _, _ = frozen_run
    path = {
        "source": source,
        "model": run / "xgboost.joblib",
        "splits": run / "split_assignments.csv",
    }[target]
    with path.open("ab") as handle:
        handle.write(b"tampered")
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("Must verify before loading"))
    with pytest.raises(ValueError, match="checksum"):
        verify_experiment(source, run)


def test_interrupted_final_access_stays_consumed(frozen_run, tmp_path, monkeypatch):
    from credit_risk.validation import runner

    source, run, splits, config = frozen_run
    original = runner.predict_probability

    def interrupted(pipeline, frame):
        if set(frame.index) == set(splits["test"]):
            raise RuntimeError("interrupted first final prediction")
        return original(pipeline, frame)

    monkeypatch.setattr(runner, "predict_probability", interrupted)
    registry = tmp_path / "holdout_registry.json"
    with pytest.raises(RuntimeError, match="interrupted first final"):
        run_validation(source, run, tmp_path / "interrupted", config, registry_path=registry)
    assert HoldoutRegistry(registry).read().entries[0].status == "consumed"
    with pytest.raises(ValueError, match="already consumed"):
        run_validation(source, run, tmp_path / "retry", config, registry_path=registry)
