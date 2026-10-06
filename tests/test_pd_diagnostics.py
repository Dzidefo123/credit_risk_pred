"""Scientific diagnostics and enforced isolation from consumed final-test samples."""

from hashlib import sha256

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.special import expit, logit

from credit_risk.data.validation import ORIGINATION_FEATURES, ORIGINATION_TARGET
from credit_risk.utils.config import ModelConfig, XGBoostConfig
from credit_risk.validation import pd_diagnostics as workflow
from credit_risk.validation.calibration import calibration_diagnostics, probability_diagnostics
from credit_risk.validation.diagnostics import group_bootstrap_comparison
from credit_risk.validation.holdout_registry import HoldoutRegistry
from credit_risk.validation.thresholds import (
    select_threshold,
    threshold_analysis,
    threshold_diagnostics,
)


def known_calibration(intercept, slope):
    rates = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    p = expit((logit(rates) - intercept) / slope)
    labels = np.concatenate(
        [np.r_[np.ones(int(1000 * r)), np.zeros(1000 - int(1000 * r))] for r in rates]
    )
    return labels, np.repeat(p, 1000)


@pytest.mark.parametrize("intercept,slope", [(0, 1), (0.7, 1), (-0.4, 0.6), (0.2, 1.5)])
def test_known_joint_calibration_and_fixed_slope_intercept(intercept, slope):
    y, p = known_calibration(intercept, slope)
    result = calibration_diagnostics(y, p)
    assert result["joint_intercept"] == pytest.approx(intercept, abs=1e-5)
    assert result["calibration_slope"] == pytest.approx(slope, abs=1e-5)
    if slope == 1:
        assert result["calibration_intercept"] == pytest.approx(intercept, abs=1e-5)
    else:
        assert abs(result["calibration_intercept"] - result["joint_intercept"]) > 0.01
    assert result["confidence_intervals"] is None


def test_boundaries_constant_and_single_class():
    result = calibration_diagnostics([0, 1, 0, 1], [0, 0, 1, 1])
    assert result["clipped_rows"] == 4
    assert np.isfinite(result["calibration_intercept"])
    assert np.isfinite(result["calibration_slope"])
    assert calibration_diagnostics([0, 1], [0.4, 0.4])["calibration_slope"] is None
    assert calibration_diagnostics([0, 0], [0.2, 0.3])["calibration_intercept"] is None
    assert "separated" in calibration_diagnostics([0, 1], [0.1, 0.9])["slope_status"]


@pytest.mark.parametrize(
    "y,p",
    [
        ([], []),
        ([0, 2], [0.2, 0.4]),
        ([0, 1], [np.nan, 0.4]),
        ([0, 1], [-0.1, 0.5]),
        ([0, 1], [0.1]),
    ],
)
def test_invalid_inputs(y, p):
    with pytest.raises(ValueError):
        calibration_diagnostics(y, p)
    with pytest.raises(ValueError):
        threshold_diagnostics(y, p, threshold=0.5)


@pytest.mark.parametrize("epsilon", [0, 1e-30, 0.5, -1, np.nan])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):
        calibration_diagnostics([0, 1], [0.2, 0.8], epsilon=epsilon)


def test_hand_calculated_threshold_metrics_and_equality():
    result = threshold_diagnostics([0, 0, 0, 1, 1], [0.1, 0.2, 0.6, 0.5, 0.3], threshold=0.5)
    assert result["confusion_matrix"] == [[2, 1], [1, 1]]
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    assert result["specificity"] == pytest.approx(2 / 3)
    assert result["balanced_accuracy"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert threshold_diagnostics([0, 0], [0, 1], threshold=0)["balanced_accuracy"] is None
    assert threshold_diagnostics([1, 1], [0, 1], threshold=1)["specificity"] is None
    with pytest.raises(ValueError):
        threshold_analysis([0, 1], [0.2, 0.8], thresholds=[])


@pytest.mark.parametrize("threshold", [-0.01, 1.01, np.nan])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        threshold_diagnostics([0, 1], [0.2, 0.8], threshold=threshold)


@pytest.mark.parametrize("objective", ["f1", "youden_j"])
def test_threshold_selection_objective_ties_and_partition(objective):
    result = select_threshold(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        thresholds=[0.3, 0.5, 0.9],
        objective=objective,
        partition="development",
    )
    assert result["threshold"] == 0.5
    assert result["objective_value"] == 1
    with pytest.raises(ValueError, match="train/development"):
        select_threshold(
            [0, 1], [0.2, 0.8], thresholds=[0.5], objective=objective, partition="test"
        )
    with pytest.raises(ValueError, match="both outcomes"):
        select_threshold(
            [0, 0], [0.2, 0.8], thresholds=[0.5], objective=objective, partition="train"
        )


def test_invalid_objective():
    with pytest.raises(ValueError, match="objective"):
        select_threshold([0, 1], [0.2, 0.8], thresholds=[0.5], objective="auc", partition="train")


def synthetic_frame():
    rng = np.random.default_rng(24)
    frame = pd.DataFrame(
        {name: rng.integers(0, 6, 120).astype(float) for name in ORIGINATION_FEATURES}
    )
    frame["age"] = rng.integers(21, 80, len(frame))
    frame["MonthlyIncome"] = rng.lognormal(8, 0.5, len(frame))
    frame[ORIGINATION_TARGET] = np.tile([0, 1], 60)
    return pd.concat([frame, frame.iloc[:4]], ignore_index=True)


def test_cv_deterministic_no_mutation_groups_and_bootstrap():
    frame = synthetic_frame()
    before = frame.copy(deep=True)
    config = ModelConfig(xgboost=XGBoostConfig(n_estimators=3, n_jobs=1))
    a = workflow.cross_validate_candidates(frame, config, folds=3, bootstrap_samples=20)
    b = workflow.cross_validate_candidates(frame, config, folds=3, bootstrap_samples=20)
    assert a == b
    assert_frame_equal(frame, before)
    assert sum(f["evaluation_rows"] for f in a["fold_design"]) == len(frame)
    assert all(f["group_overlap"] == 0 for f in a["fold_design"])
    assert a["uncertainty"]["valid_samples"] == 20
    for candidate in a["candidates"].values():
        assert len(candidate["cv_summary"]["roc_auc"]["fold_values"]) == 3
        assert sum(b["rows"] for b in candidate["oof"]["reliability"]["bins"]) == len(frame)
    with pytest.raises(ValueError, match="both outcomes"):
        workflow.cross_validate_candidates(frame.assign(SeriousDlqin2yrs=0), config)
    with pytest.raises(ValueError, match="groups"):
        workflow.cross_validate_candidates(frame.iloc[:4], config, folds=5)


def test_probability_and_bootstrap_inputs_not_mutated():
    y, p = known_calibration(0.2, 0.7)
    before = p.copy()
    probability_diagnostics(y, p)
    groups = np.arange(len(y))
    a = group_bootstrap_comparison(y, {"a": p, "b": p}, groups, samples=20, seed=5)
    assert a == group_bootstrap_comparison(y, {"a": p, "b": p}, groups, samples=20, seed=5)
    assert all(
        v == {"lower": 0.0, "upper": 0.0} for v in a["paired_differences"]["intervals"].values()
    )
    np.testing.assert_array_equal(p, before)


@pytest.fixture
def training_source(tmp_path, monkeypatch):
    frame = synthetic_frame().iloc[:8].copy()
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)
    frame = pd.read_csv(source)
    groups = pd.util.hash_pandas_object(frame.loc[:, ORIGINATION_FEATURES], index=False)
    assignments = tmp_path / "split_assignments.csv"
    pd.DataFrame(
        {
            "row_position": range(8),
            "partition": ["train"] * 4 + ["test"] * 4,
            "feature_group": groups,
        }
    ).to_csv(assignments, index=False)
    registry = HoldoutRegistry(tmp_path / "registry.json")
    HoldoutRegistry.initialize(registry.path)
    source_hash = workflow.file_digest(source)
    reservation = registry.reserve(
        source_hash, [f"legacy-pandas-v1:{int(g):016x}" for g in groups.iloc[4:]], "fixture"
    )
    registry.consume(reservation)
    monkeypatch.setattr(workflow, "verify_repository_registry", lambda path: registry.read())
    monkeypatch.setattr(workflow, "HISTORICAL_SOURCE_SHA256", source_hash)
    monkeypatch.setattr(workflow, "ASSIGNMENTS_SHA256", workflow.file_digest(assignments))
    monkeypatch.setattr(
        workflow, "TRAIN_POSITIONS_SHA256", sha256(np.arange(4, dtype="<i8").tobytes()).hexdigest()
    )
    monkeypatch.setattr(workflow, "TRAIN_ROWS", 4)
    monkeypatch.setattr(workflow, "TRAIN_EVENTS", 2)
    return source, assignments, registry, groups


def test_loader_excludes_consumed_holdout_at_parse_time(training_source, monkeypatch):
    source, assignments, registry, groups = training_source
    before = registry.path.read_bytes()
    read_csv = pd.read_csv
    parsed = []

    def guarded(path, **kwargs):
        if path == source:
            skip = kwargs["skiprows"]
            assert not skip(0)
            assert all(not skip(row + 1) for row in range(4))
            assert all(skip(row + 1) for row in range(4, 8))
            parsed.append(True)
        return read_csv(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded)
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("Frozen bundle access"))
    frame, provenance = workflow.load_training_only(source, assignments.parent, registry.path)
    assert len(frame) == 4 and parsed == [True]
    assert provenance["partition"] == "original train only"
    assert registry.path.read_bytes() == before


def test_substituting_consumed_samples_fails_before_source_parse(training_source, monkeypatch):
    source, assignments, registry, groups = training_source
    bad = pd.read_csv(assignments)
    bad["partition"] = ["test"] * 4 + ["train"] * 4
    bad.to_csv(assignments, index=False)
    # Even if a caller updates the position/split anchors, the ledger still blocks.
    monkeypatch.setattr(workflow, "ASSIGNMENTS_SHA256", workflow.file_digest(assignments))
    monkeypatch.setattr(
        workflow,
        "TRAIN_POSITIONS_SHA256",
        sha256(np.arange(4, 8, dtype="<i8").tobytes()).hexdigest(),
    )
    read_csv = pd.read_csv

    def guarded(path, **kwargs):
        assert path != source, "Consumed source rows must never be parsed"
        return read_csv(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded)
    with pytest.raises(ValueError, match="consumed holdout"):
        workflow.load_training_only(source, assignments.parent, registry.path)


def test_changed_split_fails_before_parsing(training_source, monkeypatch):
    source, assignments, registry, groups = training_source
    assignments.write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pytest.fail("Changed split parsed"))
    with pytest.raises(ValueError, match="identity changed"):
        workflow.load_training_only(source, assignments.parent, registry.path)
