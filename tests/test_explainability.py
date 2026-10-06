"""Feature alignment, log-odds additivity, fold isolation and holdout protection."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import test_calibration_study as shared5
import test_pd_diagnostics as shared4
from pandas.testing import assert_frame_equal

from credit_risk.data.validation import ORIGINATION_FEATURES, ORIGINATION_TARGET
from credit_risk.explainability import study
from credit_risk.explainability.logistic import coefficient_table, local_logistic, transformed_names
from credit_risk.explainability.stability import (
    coefficient_stability,
    feature_ranks,
    rank_agreement,
    sign_stability,
)
from credit_risk.explainability.xgboost import tree_shap
from credit_risk.models.pd import build_pd_model
from credit_risk.validation import pd_diagnostics as workflow

applicants = shared5.applicants
model_config = shared5.model_config
training_source = shared4.training_source


@pytest.fixture
def fitted(applicants, model_config):
    x = applicants.loc[:, ORIGINATION_FEATURES]
    y = applicants[ORIGINATION_TARGET]
    return {
        kind: build_pd_model(kind, model_config).fit(x, y)
        for kind in ("logistic_regression", "xgboost")
    }


def test_coefficients_standardized_effect_and_odds_ratios(fitted, applicants):
    pipeline = fitted["logistic_regression"]
    rows = coefficient_table(pipeline)
    assert [r["feature"] for r in rows] == transformed_names(pipeline)
    for index, row in enumerate(rows):
        beta = pipeline.named_steps["model"].coef_[0, index]
        scale = pipeline.named_steps["scaler"].scale_[index]
        assert row["standardized_coefficient"] == beta
        assert row["transformed_coefficient"] * scale == pytest.approx(beta)
        assert row["odds_ratio_per_training_sd"] == pytest.approx(np.exp(beta))
        assert row["odds_ratio_per_transformed_unit"] == pytest.approx(np.exp(beta / scale))
    values = local_logistic(pipeline, applicants.iloc[:20].loc[:, ORIGINATION_FEATURES])
    np.testing.assert_allclose(
        values["baseline"] + values["contributions"].sum(axis=1), values["margin"]
    )
    np.testing.assert_allclose(
        values["probability"],
        pipeline.predict_proba(applicants.iloc[:20].loc[:, ORIGINATION_FEATURES])[:, 1],
    )


def test_shap_dimensions_alignment_additivity_and_interactions(fitted, applicants):
    pipeline = fitted["xgboost"]
    x = applicants.iloc[:20].loc[:, ORIGINATION_FEATURES]
    before = x.copy(deep=True)
    a = tree_shap(pipeline, x, interactions=True)
    assert a["features"] == transformed_names(pipeline)
    assert a["contributions"].shape == (20, len(a["features"]))
    np.testing.assert_allclose(
        a["baseline"] + a["contributions"].sum(axis=1), a["margin"], atol=1e-5
    )
    assert a["interactions"].shape == (20, len(a["features"]), len(a["features"]))
    np.testing.assert_allclose(a["interactions"].sum(axis=2), a["contributions"], atol=1e-5)
    np.testing.assert_allclose(a["probability"], pipeline.predict_proba(x)[:, 1], atol=1e-6)
    b = tree_shap(pipeline, x.loc[:, list(reversed(ORIGINATION_FEATURES))])
    assert a["features"] == b["features"]
    np.testing.assert_array_equal(a["contributions"], b["contributions"])
    assert_frame_equal(x, before)


def test_invalid_shap_alignment_and_nonfinite_predictors(fitted, applicants, monkeypatch):
    pipeline = fitted["xgboost"]
    x = applicants.iloc[:4].loc[:, ORIGINATION_FEATURES]
    with pytest.raises(ValueError):
        tree_shap(pipeline, x.drop(columns="age"))
    with pytest.raises(ValueError):
        tree_shap(pipeline, x.assign(age=np.inf))
    from credit_risk.explainability import xgboost as module

    monkeypatch.setattr(module, "transformed_names", lambda p: ["wrong"])
    with pytest.raises(ValueError, match="alignment"):
        tree_shap(pipeline, x)


def test_invalid_coefficient_alignment(fitted, monkeypatch):
    from credit_risk.explainability import logistic as module

    monkeypatch.setattr(module, "transformed_names", lambda p: ["wrong"])
    with pytest.raises(ValueError, match="alignment"):
        coefficient_table(fitted["logistic_regression"])


def test_sign_stability_and_deterministic_tied_ranking():
    assert sign_stability([1, 2, 3])["sign_consistency"] == 1
    mixed = sign_stability([-1, 1, -0.5])
    assert mixed["sign_flip"] and mixed["sign_consistency"] == 2 / 3
    assert sign_stability([0, 1e-10])["dominant_sign"] == "near_zero"
    assert feature_ranks({"b": 2, "a": 2, "c": 0}) == {"a": 1.5, "b": 1.5, "c": 3.0}
    assert feature_ranks({"c": 0, "a": 2, "b": 2}) == feature_ranks({"b": 2, "a": 2, "c": 0})


def test_coefficient_aggregation_and_absent_indicator():
    folds = [
        {
            "coefficients": [
                {"feature": "a", "standardized_coefficient": value},
                {"feature": "b", "standardized_coefficient": -value},
            ]
        }
        for value in (1, 2, 3)
    ]
    folds[0]["coefficients"].append({"feature": "indicator", "standardized_coefficient": 0.1})
    a = coefficient_stability(folds)
    assert a == coefficient_stability(folds)
    row = next(r for r in a if r["feature"] == "a")
    assert row["coefficient"] == {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0}
    indicator = next(r for r in a if r["feature"] == "indicator")
    assert indicator["estimated_folds"] == 1 and indicator["fold_coefficients"] == [0.1, None, None]
    assert rank_agreement([{"a": 1, "b": 2}, {"a": 2, "b": 3}], ["a", "b"])[
        "mean_rho"
    ] == pytest.approx(1)
    assert rank_agreement([{"a": 0, "b": 0}] * 2, ["a", "b"])["mean_rho"] is None


@pytest.mark.parametrize("values", [{}, {"a": np.nan}, {"a": -1}])
def test_invalid_ranking(values):
    with pytest.raises(ValueError):
        feature_ranks(values)


def test_deterministic_risk_sample_and_dependence():
    p = np.linspace(0, 1, 100)
    indices, design = study.score_stratified_sample(p, 23, 42)
    again, same = study.score_stratified_sample(p, 23, 42)
    np.testing.assert_array_equal(indices, again)
    assert design == same and len(indices) == len(set(indices)) == 23
    assert sum(design["sample_counts"]) == 23
    assert max(design["sample_counts"]) - min(design["sample_counts"]) <= 1
    bins = study.dependence_summary([0, 0, 1, 1, np.nan], [-1, -1, 1, 1, 0])
    assert bins["missing_rows"] == 1 and [r["rows"] for r in bins["bins"]] == [2, 2]
    assert bins["spearman_rho"] == pytest.approx(1)
    with pytest.raises(ValueError):
        study.dependence_summary([0, 1], [np.nan, 1])
    with pytest.raises(ValueError):
        study.score_stratified_sample([0, np.nan], 1, 42)


def test_fold_study_deterministic_no_mutation_and_systematic_locals(applicants, model_config):
    before = applicants.copy(deep=True)
    a = study.explanation_study(applicants, model_config, folds=3, interaction_rows=12)
    b = study.explanation_study(applicants, model_config, folds=3, interaction_rows=12)
    assert a == b
    assert_frame_equal(applicants, before)
    assert sum(f["evaluation_rows"] for f in a["fold_design"]) == len(applicants)
    assert all(f["group_overlap"] == 0 for f in a["fold_design"])
    assert [e["quantile"] for e in a["local_examples"]] == [0.1, 0.5, 0.9]
    for example in a["local_examples"]:
        for model in example["models"].values():
            assert model["baseline_log_odds"] + sum(
                model["contributions_log_odds"].values()
            ) == pytest.approx(model["margin"], abs=1e-5)
    assert len(a["dependence"]) == 5


def test_explicit_fold_isolation_explanations_never_receive_outcomes(
    applicants, model_config, monkeypatch
):
    real_explain, builder, real_shap = study.explain_fold, study.build_pd_model, study.tree_shap
    expected, visits = {}, []

    def guarded_fold(training_frame, evaluation_predictors, config, seed, interactions):
        expected["train"] = training_frame.index.to_numpy()
        expected["evaluation"] = set(evaluation_predictors.index)
        assert ORIGINATION_TARGET not in evaluation_predictors
        assert not set(expected["train"]) & expected["evaluation"]
        return real_explain(training_frame, evaluation_predictors, config, seed, interactions)

    def guarded_builder(*args):
        pipeline = builder(*args)
        real_fit = pipeline.fit

        def guarded_fit(predictors, labels):
            np.testing.assert_array_equal(predictors.index, expected["train"])
            assert not set(predictors.index) & expected["evaluation"]
            visits.append("fit")
            return real_fit(predictors, labels)

        pipeline.fit = guarded_fit
        return pipeline

    def guarded_shap(pipeline, predictors, **kwargs):
        assert ORIGINATION_TARGET not in predictors
        assert set(predictors.index) <= expected["evaluation"]
        visits.append("explain")
        return real_shap(pipeline, predictors, **kwargs)

    monkeypatch.setattr(study, "explain_fold", guarded_fold)
    monkeypatch.setattr(study, "build_pd_model", guarded_builder)
    monkeypatch.setattr(study, "tree_shap", guarded_shap)
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("Frozen bundle access"))
    study.explanation_study(applicants, model_config, folds=3, interaction_rows=12)
    assert visits.count("fit") == 6 and visits.count("explain") == 6


def test_consumed_samples_block_explanation_before_parse_or_fit(training_source, monkeypatch):
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
    from hashlib import sha256

    monkeypatch.setattr(
        workflow,
        "TRAIN_POSITIONS_SHA256",
        sha256(np.arange(4, 8, dtype="<i8").tobytes()).hexdigest(),
    )
    (root / "reports").mkdir()
    (root / "reports/holdout_registry.json").write_bytes(registry.path.read_bytes())
    original = pd.read_csv

    def guarded_read(path, **kwargs):
        assert Path(path) != source, "Consumed raw samples must not be parsed"
        return original(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read)
    monkeypatch.setattr(
        study, "explanation_study", lambda *args, **kwargs: pytest.fail("Consumed data explained")
    )
    with pytest.raises(ValueError, match="consumed holdout"):
        study.run_explainability_study(root, source)


@pytest.mark.parametrize("kwargs", [{"folds": 1}, {"seed": -1}, {"interaction_rows": 0}])
def test_invalid_study_settings(applicants, model_config, kwargs):
    with pytest.raises(ValueError):
        study.explanation_study(applicants, model_config, **kwargs)


def test_previous_evidence_code_and_registry_hashes_preserved():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (root / "reports/model_validation/calibration_study.json").read_text(encoding="utf-8")
    )
    for name, digest in {
        **evidence["source_code_sha256"],
        **evidence["preserved_task4_and_registry_sha256"],
    }.items():
        payload = (root / name).read_bytes()
        # Git may convert non-frozen text; binaries and frozen code stay exact.
        frozen = {
            "src/credit_risk/models/pd.py",
            "src/credit_risk/features/origination.py",
            "src/credit_risk/utils/config.py",
            "src/credit_risk/validation/metrics.py",
            "src/credit_risk/models/calibration.py",
        }
        if name not in frozen and Path(name).suffix in {".py", ".md", ".json"}:
            from hashlib import sha256

            windows = payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            assert digest in {sha256(payload).hexdigest(), sha256(windows).hexdigest()}
        else:
            assert workflow.file_digest(root / name) == digest
