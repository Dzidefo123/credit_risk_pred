"""Nested isolation, ranking, honest method selection and preserved evidence."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import test_pd_diagnostics as shared_tests
from pandas.testing import assert_frame_equal
from scipy.special import expit

from credit_risk.data.validation import ORIGINATION_FEATURES, ORIGINATION_TARGET
from credit_risk.models.calibration import ProbabilityCalibrator
from credit_risk.utils.config import ModelConfig, XGBoostConfig
from credit_risk.validation import calibration_study as study
from credit_risk.validation import pd_diagnostics as workflow

training_source = shared_tests.training_source


@pytest.fixture
def applicants():
    rng = np.random.default_rng(103)
    n = 240
    frame = pd.DataFrame(
        {name: rng.integers(0, 5, n).astype(float) for name in ORIGINATION_FEATURES}
    )
    frame["age"] = rng.integers(21, 80, n)
    frame["MonthlyIncome"] = rng.lognormal(8, 0.5, n)
    frame["DebtRatio"] = rng.uniform(0, 2, n)
    frame["RevolvingUtilizationOfUnsecuredLines"] = rng.uniform(0, 1.5, n)
    frame[ORIGINATION_TARGET] = rng.binomial(1, expit(-1 + frame.NumberOfTimes90DaysLate / 2))
    frame.loc[::19, "MonthlyIncome"] = np.nan
    return pd.concat([frame, frame.iloc[:6]], ignore_index=True)


@pytest.fixture
def small_study():
    return study.CalibrationStudyConfig(
        outer_folds=3, inner_folds=3, minimum_calibration_events=2, bootstrap_samples=20
    )


@pytest.fixture
def model_config():
    return ModelConfig(xgboost=XGBoostConfig(n_estimators=3, n_jobs=1))


def test_nested_deterministic_no_mutation_and_groups(applicants, small_study, model_config):
    before = applicants.copy(deep=True)
    a = study.nested_calibration_study(applicants, model_config, small_study)
    b = study.nested_calibration_study(applicants, model_config, small_study)
    assert a == b
    assert_frame_equal(applicants, before)
    assert sum(f["evaluation_rows"] for f in a["fold_design"]) == len(applicants)
    for fold in a["fold_design"]:
        assert fold["outer_group_overlap"] == fold["inner"]["inner_group_overlap"] == 0
        assert (
            sum(role["rows"] for role in fold["inner"]["roles"].values()) == fold["training_rows"]
        )
    for name in study.MODELS:
        assert a["recommendations"][name]["recommendation"] in {"RAW", "SIGMOID", "ISOTONIC"}
        for method in (*study.METHODS, "inner_selected"):
            assert len(a["models"][name][method]["folds"]) == small_study.outer_folds
            assert sum(
                b["rows"] for b in a["models"][name][method]["pooled_oof"]["reliability"]["bins"]
            ) == len(applicants)
        for method in study.METHODS[1:]:
            for fold in a["models"][name][method]["folds"]:
                assert fold["ranking"]["rank_inversions"] == 0


def test_inner_role_groups_disjoint_and_cover_every_row(applicants, small_study):
    roles = study.inner_roles(applicants, small_study, 43)
    groups = study.predictor_groups(applicants)
    assert sorted(np.concatenate(list(roles.values()))) == list(range(len(applicants)))
    for a, b in (
        ("base_fit", "calibration"),
        ("base_fit", "selection"),
        ("calibration", "selection"),
    ):
        assert not set(groups[roles[a]]) & set(groups[roles[b]])
    for indices in roles.values():
        assert (0 in indices) == (240 in indices)


class IndexedModel:
    """Unique predictable scores let leakage tests recover calibration row IDs."""

    def fit(self, predictors, labels):
        self.fit_rows = predictors.index.to_numpy()
        return self

    def predict_proba(self, predictors):
        p = 0.1 + predictors.index.to_numpy() / 1000
        return np.column_stack([1 - p, p])


def test_explicit_leakage_regression_calibrator_sees_only_inner_calibration(
    applicants, small_study, model_config, monkeypatch
):
    # This assertion fails if any outer evaluation rows/outcomes reach fit.
    expected = {}
    original_train = study.fit_outer_training
    original_fit = ProbabilityCalibrator.fit
    calls = []

    def guarded_training(training_frame, config, settings, seed):
        roles = study.inner_roles(training_frame, settings, seed)
        expected["cal_rows"] = training_frame.iloc[roles["calibration"]].index.to_numpy()
        expected["cal_labels"] = training_frame.iloc[roles["calibration"]][
            ORIGINATION_TARGET
        ].to_numpy()
        expected["base_rows"] = training_frame.iloc[roles["base_fit"]].index.to_numpy()
        expected["outer_train"] = set(training_frame.index)
        return original_train(training_frame, config, settings, seed)

    class GuardedModel(IndexedModel):
        def fit(self, predictors, labels):
            np.testing.assert_array_equal(predictors.index, expected["base_rows"])
            assert set(predictors.index) <= expected["outer_train"]
            return super().fit(predictors, labels)

    def guarded_calibrator(self, probabilities, labels):
        recovered = np.rint((probabilities - 0.1) * 1000).astype(int)
        np.testing.assert_array_equal(recovered, expected["cal_rows"])
        np.testing.assert_array_equal(labels, expected["cal_labels"])
        assert set(recovered) <= expected["outer_train"]
        calls.append(self.method)
        return original_fit(self, probabilities, labels)

    monkeypatch.setattr(study, "fit_outer_training", guarded_training)
    monkeypatch.setattr(study, "build_pd_model", lambda *args: GuardedModel())
    monkeypatch.setattr(ProbabilityCalibrator, "fit", guarded_calibrator)
    monkeypatch.setattr(
        joblib, "load", lambda *args, **kwargs: pytest.fail("Frozen bundle loading")
    )
    study.nested_calibration_study(applicants, model_config, small_study)
    assert len(calls) == small_study.outer_folds * len(study.MODELS) * 2


def test_outer_outcomes_not_passed_to_prediction_or_selection(
    applicants, small_study, model_config, monkeypatch
):
    original_predict = study.predict_outer
    seen = []

    def guarded_predict(fitted, predictors):
        assert ORIGINATION_TARGET not in predictors.columns
        assert set(predictors.columns) == set(ORIGINATION_FEATURES)
        seen.append(True)
        return original_predict(fitted, predictors)

    monkeypatch.setattr(study, "predict_outer", guarded_predict)
    study.nested_calibration_study(applicants, model_config, small_study)
    assert len(seen) == small_study.outer_folds


def test_outer_label_poisoning_cannot_change_calibrator_fit(applicants, small_study, model_config):
    training = applicants.iloc[:180].copy()
    evaluation = applicants.iloc[180:].copy()
    a, meta_a = study.fit_outer_training(training, model_config, small_study, 43)
    evaluation[ORIGINATION_TARGET] = "OUTER OUTCOMES MUST NOT BE READ"
    b, meta_b = study.fit_outer_training(training, model_config, small_study, 43)
    assert meta_a == meta_b
    predictors = evaluation.loc[:, ORIGINATION_FEATURES]
    pa, pb = study.predict_outer(a, predictors), study.predict_outer(b, predictors)
    for name in study.MODELS:
        for method in (*study.METHODS, "inner_selected"):
            np.testing.assert_array_equal(pa[name][method], pb[name][method])


def test_raw_exact_endpoints_not_transformed(monkeypatch, applicants):
    original = np.array([0.0, 1e-10, 0.2, 0.7, 1 - 1e-10, 1.0])
    monkeypatch.setattr(study, "predict_probability", lambda *args: original)
    calibrators = {
        m: ProbabilityCalibrator(m).fit(np.linspace(0, 1, 30), np.tile([0, 1], 15))
        for m in study.METHODS[1:]
    }
    result = study.predict_outer({"test": (object(), calibrators, "raw")}, applicants.iloc[:6])[
        "test"
    ]
    np.testing.assert_array_equal(result["raw"], original)
    np.testing.assert_array_equal(result["inner_selected"], original)
    assert not np.shares_memory(result["raw"], original)
    for method in study.METHODS[1:]:
        assert np.isfinite(result[method]).all()
        assert (result[method] >= 1e-6).all() and (result[method] <= 1 - 1e-6).all()


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize(
    "probabilities,labels",
    [
        ([0.1, 0.2], [0, 0]),
        ([np.nan, 0.4], [0, 1]),
        ([np.inf, 0.4], [0, 1]),
        ([-0.1, 0.4], [0, 1]),
        ([0.1, 1.4], [0, 1]),
        ([0.1], [0, 1]),
    ],
)
def test_invalid_or_single_class_calibration_fails(method, probabilities, labels):
    with pytest.raises(ValueError):
        ProbabilityCalibrator(method).fit(probabilities, labels)


def test_selection_prefers_simplicity_and_does_not_force_calibration():
    raw = {"brier": 0.05, "log_loss": 0.18}
    same = {m: raw.copy() for m in study.METHODS}
    assert study.choose_on_inner_selection(same, 0.0001) == "raw"
    same["sigmoid"] = {"brier": 0.04999, "log_loss": 0.179}
    assert study.choose_on_inner_selection(same, 0.0001) == "raw"
    same["isotonic"] = {"brier": 0.049, "log_loss": 0.181}
    assert study.choose_on_inner_selection(same, 0.0001) == "raw"
    same["sigmoid"] = same["isotonic"] = {"brier": 0.049, "log_loss": 0.179}
    assert study.choose_on_inner_selection(same, 0.0001) == "sigmoid"


def test_ranking_changes_explained_by_ties_and_nonmonotonic_rejected():
    result = study.ranking_diagnostics(
        [0, 1, 0, 1],
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.1, 0.1, 0.1, 0.9]),
        "isotonic",
        0.001,
    )
    assert result["calibrated_unique_probabilities"] == 2
    assert result["rank_inversions"] == 0
    with pytest.raises(ValueError, match="Non-monotonic"):
        study.ranking_diagnostics(
            [0, 1], np.array([0.1, 0.9]), np.array([0.9, 0.1]), "sigmoid", 0.001
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outer_folds": 1},
        {"inner_folds": 2},
        {"seed": -1},
        {"epsilon": 1e-30},
        {"epsilon": np.nan},
        {"minimum_brier_gain": 0},
        {"bootstrap_samples": 1},
    ],
)
def test_invalid_study_settings(kwargs):
    with pytest.raises(ValueError):
        study.CalibrationStudyConfig(**kwargs)


def test_insufficient_events_and_single_class_fail(applicants, small_study, model_config):
    with pytest.raises(ValueError, match="outcome groups"):
        study.nested_calibration_study(
            applicants.assign(SeriousDlqin2yrs=0), model_config, small_study
        )
    with pytest.raises(ValueError, match="Insufficient events"):
        study.inner_roles(applicants, replace(small_study, minimum_calibration_events=1000), 43)


def test_task5_consumed_holdout_blocked_before_fitting(training_source, monkeypatch):
    source, assignments, registry, groups = training_source
    root = assignments.parent
    experiment = root / "artifacts/phase4-origination-001"
    experiment.mkdir(parents=True)
    bad = pd.read_csv(assignments)
    bad["partition"] = ["test"] * 4 + ["train"] * 4
    bad.to_csv(experiment / "split_assignments.csv", index=False)
    monkeypatch.setattr(
        workflow, "ASSIGNMENTS_SHA256", workflow.file_digest(experiment / "split_assignments.csv")
    )
    monkeypatch.setattr(workflow, "TRAIN_POSITIONS_SHA256", study.positions_digest(np.arange(4, 8)))
    (root / "reports").mkdir()
    (root / "reports/holdout_registry.json").write_bytes(registry.path.read_bytes())
    monkeypatch.setattr(
        study, "nested_calibration_study", lambda *args, **kwargs: pytest.fail("Consumed data fit")
    )
    with pytest.raises(ValueError, match="consumed holdout"):
        study.run_calibration_study(root, source)


def test_committed_task4_evidence_and_frozen_source_hashes_preserved():
    # Read aggregate metadata and source bytes only; never original predictions.
    root = Path(__file__).resolve().parents[1]
    task4 = json.loads((root / "reports/model_validation/pd_diagnostics.json").read_text())
    for name, digest in task4["reused_source_sha256"].items():
        if name == "scripts/pd_diagnostics.py":
            # This non-frozen CLI undergoes Git LF/CRLF conversion on Linux.
            payload = (root / name).read_bytes()
            windows = payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            assert digest in {sha256(payload).hexdigest(), sha256(windows).hexdigest()}
        else:
            # Frozen model/preprocessing/metric source remains byte-exact.
            assert workflow.file_digest(root / name) == digest
    for name, digest in task4["diagnostic_source_sha256"].items():
        # Non-frozen source files undergo Git LF/CRLF conversion on checkout.
        payload = (root / "src/credit_risk/validation" / name).read_bytes()
        windows = payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        assert digest in {sha256(payload).hexdigest(), sha256(windows).hexdigest()}
    marker = json.loads((root / "reports/phase5_validation_summary.json").read_text())
    retained = marker["final_metrics"]["xgboost"]["sigmoid"]
    assert {key: round(retained[key], 6) for key in ("roc_auc", "brier", "log_loss")} == {
        "roc_auc": 0.868152,
        "brier": 0.048545,
        "log_loss": 0.176030,
    }
    assert workflow.file_digest(root / "src/credit_risk/models/calibration.py") == (
        "307cc6f3f5f887392fde6ee06250bb88e16c41c05db456049c079d7a35988344"
    )
    # Hash-only checks of ignored frozen artifacts when available locally.
    phase4 = json.loads((root / "reports/phase4_experiment.json").read_text())["experiment"]
    for folder, metadata in (("phase4-origination-001", phase4), ("phase5-validation-001", marker)):
        for name, digest in metadata["artifacts_sha256"].items():
            path = root / "artifacts" / folder / name
            if path.is_file():
                assert workflow.file_digest(path) == digest, name


def test_recommendation_requires_margin_and_log_loss_guard():
    results = {
        m: {"pooled_oof": {"metrics": {"brier": value}}}
        for m, value in zip(study.METHODS, [0.05, 0.049, 0.048], strict=True)
    }
    comparisons = {
        m: {
            "paired_differences": {
                "intervals": {
                    "brier": {"lower": -0.001, "upper": 0.0},
                    "log_loss": {"lower": -0.01, "upper": 0.0},
                }
            }
        }
        for m in study.METHODS[1:]
    }
    assert study.evidence_recommendation(results, comparisons, 0.0001)["recommendation"] == "RAW"
    bounds = comparisons["isotonic"]["paired_differences"]["intervals"]
    bounds["brier"]["upper"] = -0.0002
    assert (
        study.evidence_recommendation(results, comparisons, 0.0001)["recommendation"] == "ISOTONIC"
    )
    bounds["log_loss"]["upper"] = 0.001
    assert study.evidence_recommendation(results, comparisons, 0.0001)["recommendation"] == "RAW"
    bounds["log_loss"]["upper"] = -0.001
    bounds["brier"]["upper"] = -0.00009
    assert study.evidence_recommendation(results, comparisons, 0.0001)["recommendation"] == "RAW"
