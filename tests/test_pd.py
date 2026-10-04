"""Training isolation, grouped splits, artifact reproducibility and candidate behavior."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.exceptions import NotFittedError

from credit_risk.data.validation import ORIGINATION_FEATURES, ORIGINATION_TARGET, DataContractError
from credit_risk.features.origination import OriginationFeatures
from credit_risk.models.pd import (
    build_pd_model,
    predict_probability,
    run_origination_experiment,
    split_origination,
)
from credit_risk.utils.config import ModelConfig, XGBoostConfig


@pytest.fixture
def applicants():
    rng = np.random.default_rng(24)
    n = 400
    frame = pd.DataFrame(
        {name: rng.integers(0, 6, n).astype(float) for name in ORIGINATION_FEATURES}
    )
    frame["age"] = rng.integers(21, 80, n)
    frame["MonthlyIncome"] = rng.lognormal(8, 0.5, n)
    frame["DebtRatio"] = rng.uniform(0, 2, n)
    frame["RevolvingUtilizationOfUnsecuredLines"] = rng.uniform(0, 1.5, n)
    frame.loc[frame.index[::13], "MonthlyIncome"] = np.nan
    frame[ORIGINATION_TARGET] = (
        (frame.NumberOfTimes90DaysLate >= 4) & (rng.random(n) > 0.3)
    ).astype(int)
    frame["source_row_id"] = np.arange(1, n + 1)
    return frame


@pytest.fixture
def small_config():
    return ModelConfig(xgboost=XGBoostConfig(n_estimators=12, n_jobs=1))


def test_all_rows_partitioned_once_and_predictor_duplicates_stay_together(applicants, small_config):
    duplicate = applicants.iloc[[0, 1]].copy()
    duplicate["source_row_id"] = [401, 402]
    duplicate[ORIGINATION_TARGET] = 1 - duplicate[ORIGINATION_TARGET]
    frame = pd.concat([applicants, duplicate], ignore_index=True)
    splits = split_origination(frame, small_config)
    assignments = {int(row): name for name, rows in splits.items() for row in rows}
    assert sorted(assignments) == list(range(len(frame)))
    assert assignments[0] == assignments[400]
    assert assignments[1] == assignments[401]
    assert sum(map(len, splits.values())) == len(frame)
    for name, rows in splits.items():
        assert np.array_equal(rows, split_origination(frame, small_config)[name])
        assert set(frame.iloc[rows][ORIGINATION_TARGET].unique()) == {0, 1}
    assert any(
        not np.array_equal(rows, split_origination(frame, small_config, seed=43)[name])
        for name, rows in splits.items()
    )


def test_single_class_and_too_small_population_rejected(applicants, small_config):
    with pytest.raises(ValueError, match="partition|group|outcome"):
        split_origination(applicants.assign(SeriousDlqin2yrs=0), small_config)
    with pytest.raises(ValueError, match="group|outcome"):
        split_origination(applicants.iloc[:4], small_config)


def test_preprocessing_fitted_only_on_train_and_prediction_does_not_refit(applicants, small_config):
    splits = split_origination(applicants, small_config)
    train = applicants.iloc[splits["train"]]
    holdout = applicants.iloc[splits["development"]].loc[:, ORIGINATION_FEATURES].copy()
    holdout["MonthlyIncome"] = 1e12
    pipeline = build_pd_model("logistic_regression", small_config)
    X = train.loc[:, ORIGINATION_FEATURES]
    before = X.copy(deep=True)
    pipeline.fit(X, train[ORIGINATION_TARGET])
    features = pipeline.named_steps["features"]
    imputer = pipeline.named_steps["imputer"]
    expected = X.MonthlyIncome.quantile(small_config.clip_upper_quantile)
    assert features.upper_bounds_["MonthlyIncome"] == pytest.approx(expected)
    statistics = imputer.statistics_.copy()
    probability = predict_probability(pipeline, holdout)
    assert np.isfinite(probability).all()
    assert features.upper_bounds_["MonthlyIncome"] == pytest.approx(expected)
    assert np.array_equal(statistics, imputer.statistics_)
    transformed_train = features.transform(X).to_numpy()
    assert np.allclose(imputer.statistics_, np.nanmedian(transformed_train, axis=0))
    assert_frame_equal(X, before)


@pytest.mark.parametrize("kind", ["logistic_regression", "xgboost"])
def test_model_handles_missing_data_reordered_columns_and_empty_training_feature(
    applicants, small_config, kind
):
    X = applicants.loc[:, ORIGINATION_FEATURES].copy()
    X["MonthlyIncome"] = np.nan
    pipeline = build_pd_model(kind, small_config)
    pipeline.fit(X, applicants[ORIGINATION_TARGET])
    probability = predict_probability(pipeline, X.iloc[:10])
    reordered = predict_probability(pipeline, X.iloc[:10].loc[:, list(reversed(X.columns))])
    assert np.allclose(probability, reordered)
    assert ((probability >= 0) & (probability <= 1)).all()
    assert (
        "missingindicator_MonthlyIncome" in pipeline.named_steps["imputer"].get_feature_names_out()
    )
    with pytest.raises(DataContractError):
        predict_probability(pipeline, X.iloc[:10].assign(future_default=1))
    with pytest.raises(DataContractError):
        predict_probability(pipeline, X.iloc[:10].drop(columns="age"))


def test_unfitted_transformer_rejected():
    with pytest.raises(NotFittedError):
        OriginationFeatures().transform(pd.DataFrame())


def test_experiment_preserves_holdouts_and_serializes_full_pipeline(
    applicants, small_config, tmp_path
):
    source = tmp_path / "source.csv"
    applicants.to_csv(source, index=False)
    run = tmp_path / "run"
    result = run_origination_experiment(source, run, small_config)
    assert result["final_test_scored"] is False and result["calibration_scored"] is False
    assert result["probability_status"] == "raw_uncalibrated"
    assert sum(p["rows"] for p in result["partitions"].values()) == len(applicants)
    assignments = pd.read_csv(run / "split_assignments.csv")
    predictions = pd.read_csv(run / "development_predictions.csv")
    assert set(predictions.row_position) == set(
        assignments.loc[assignments.partition == "development", "row_position"]
    )
    assert "test" not in result["development_metrics"]
    loaded = joblib.load(run / "logistic_regression.joblib")  # Only this test's own new artifact.
    probability = predict_probability(
        loaded["pipeline"], applicants.iloc[predictions.row_position].loc[:, ORIGINATION_FEATURES]
    )
    assert np.allclose(probability, predictions.logistic_regression)
    assert loaded["metadata"]["target_semantics"].startswith("Inherited")
    assert (
        json.loads((run / "experiment.json").read_text())["source_sha256"]
        == result["source_sha256"]
    )
    repeated = run_origination_experiment(source, tmp_path / "repeat", small_config)
    assert repeated["development_metrics"] == result["development_metrics"]
    with pytest.raises(FileExistsError):
        run_origination_experiment(source, run, small_config)


def test_experiment_fit_and_score_access_are_limited_to_declared_partitions(
    applicants, small_config, tmp_path, monkeypatch
):
    from sklearn.pipeline import Pipeline

    source = tmp_path / "source.csv"
    applicants.to_csv(source, index=False)
    expected = split_origination(pd.read_csv(source), small_config)
    fit_rows, score_rows = [], []
    original_fit, original_predict = Pipeline.fit, Pipeline.predict_proba

    def traced_fit(self, X, y, **kwargs):
        fit_rows.append(set(X.index))
        return original_fit(self, X, y, **kwargs)

    def traced_predict(self, X, **kwargs):
        score_rows.append(set(X.index))
        return original_predict(self, X, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", traced_fit)
    monkeypatch.setattr(Pipeline, "predict_proba", traced_predict)
    run_origination_experiment(source, tmp_path / "run", small_config)
    assert fit_rows == [set(expected["train"])] * 2
    assert score_rows == [set(expected["development"])] * 2
