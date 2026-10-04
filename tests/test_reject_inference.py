"""Masked-label isolation, propensity cross-fitting, IPW and simulation failures."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.decisioning import reject_inference as ri
from credit_risk.decisioning.reject_runner import run_reject_experiment
from credit_risk.decisioning.reject_settings import RejectInferenceConfig


@pytest.fixture
def config():
    return RejectInferenceConfig(samples=1500, seeds=[123], propensity_folds=3)


def test_simulation_reproducible_partition_and_masking(config):
    a = ri.generate_selection(config, 123, "mar_overlap")
    b = ri.generate_selection(config, 123, "mar_overlap")
    pd.testing.assert_frame_equal(a.features, b.features)
    np.testing.assert_array_equal(a.oracle_outcome, b.oracle_outcome)
    np.testing.assert_array_equal(a.accepted, b.accepted)
    assert np.isnan(a.observed_outcome[a.accepted == 0]).all()
    np.testing.assert_array_equal(
        a.observed_outcome[a.accepted == 1], a.oracle_outcome[a.accepted == 1]
    )
    assert not set(a.train_positions) & set(a.holdout_positions)
    assert sorted(np.concatenate([a.train_positions, a.holdout_positions])) == list(
        range(config.samples)
    )
    assert a.true_propensity.min() >= config.true_propensity_floor
    assert a.true_propensity.max() <= 1 - config.true_propensity_floor


def test_scenarios_share_population_but_deterministic_has_no_support(config):
    mar = ri.generate_selection(config, 123, "mar_overlap")
    mnar = ri.generate_selection(config, 123, "mnar_hidden")
    deterministic = ri.generate_selection(config, 123, "deterministic_no_overlap")
    np.testing.assert_array_equal(mar.oracle_outcome, mnar.oracle_outcome)
    np.testing.assert_array_equal(mar.train_positions, deterministic.train_positions)
    assert np.any(mar.accepted != mnar.accepted)
    assert set(deterministic.true_propensity) == {0.0, 1.0}
    np.testing.assert_array_equal(deterministic.accepted, deterministic.true_propensity)
    with pytest.raises(ValueError):
        ri.generate_selection(config, 123, "unsupported")


def test_ipw_matches_formula_clipping_and_ess():
    a = np.array([1, 0, 1, 1])
    p = np.array([0.5, 0.1, 0.01, 1.0])
    w, d = ri.inverse_probability_weights(a, p, 0.1, 8.0)
    clipped = np.array([2.0, 8.0, 1.0])
    np.testing.assert_allclose(w, clipped / clipped.mean())
    assert d["accepted_clipped_fraction"] == pytest.approx(1 / 3)
    assert d["effective_sample_size"] == pytest.approx(11**2 / 69)
    assert d["raw_weight_max"] == 100.0
    assert w.mean() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "a,p,floor,cap",
    [
        ([1, 0], [0.5, 0.0], 0.05, 20.0),
        ([1], [np.nan], 0.05, 20.0),
        ([1], [1.1], 0.05, 20.0),
        ([0], [0.5], 0.05, 20.0),
        ([1, 0], [0.5], 0.05, 20.0),
        ([2], [0.5], 0.05, 20.0),
        ([1], [0.5], 0.0, 20.0),
        ([1], [0.5], 0.05, np.inf),
        ([1], [0.5], 0.05, 0.5),
    ],
)
def test_invalid_weights_and_zero_support_rejected(a, p, floor, cap):
    with pytest.raises(ValueError):
        ri.inverse_probability_weights(a, p, floor, cap)


def test_rejected_labels_cannot_enter_training(config):
    sample = ri.generate_selection(config, 123, "mar_overlap")
    with pytest.raises(ValueError, match="masked"):
        ri.fit_observed_models(sample.features, sample.accepted, sample.oracle_outcome, config, 123)
    y = sample.observed_outcome.copy()
    y[sample.accepted == 1] = 0
    with pytest.raises(ValueError, match="both classes"):
        ri.fit_observed_models(sample.features, sample.accepted, y, config, 123)


def test_hidden_truth_not_an_input_and_no_overlap_disables_ipw(config, monkeypatch):
    sample = ri.generate_selection(config, 123, "deterministic_no_overlap")

    def forbidden(*args):
        raise AssertionError("No propensity model may imply recovered zero support")

    monkeypatch.setattr(ri, "crossfit_propensity", forbidden)
    models, diagnostic, propensity = ri.fit_observed_models(
        sample.features,
        sample.accepted,
        sample.observed_outcome,
        config,
        123,
        positivity_supported=False,
    )
    assert set(models) == {"accepted_only"}
    assert diagnostic["ipw_status"] == "disabled_structural_zero_support"
    assert propensity is None


def test_crossfit_never_fits_rows_it_scores(config, monkeypatch):
    sample = ri.generate_selection(config, 123, "mar_overlap")
    calls = []

    class Spy:
        def fit(self, x, y):
            self.seen = set(x.index)
            assert set(y) <= {0, 1}
            calls.append(self)
            return self

        def predict_proba(self, x):
            assert not self.seen & set(x.index)
            self.scored = set(x.index)
            return np.tile([0.4, 0.6], (len(x), 1))

    monkeypatch.setattr(ri, "logistic_model", lambda c: Spy())
    p = ri.crossfit_propensity(sample.features, sample.accepted, 3, 123, 100.0)
    assert len(calls) == 3
    assert set.union(*(c.scored for c in calls)) == set(range(config.samples))
    assert sum(len(c.scored) for c in calls) == config.samples
    np.testing.assert_allclose(p, 0.6)
    with pytest.raises(ValueError):
        ri.crossfit_propensity(sample.features, np.ones(config.samples), 3, 123, 100.0)


def test_outcome_fit_uses_accepted_only_and_weights(config, monkeypatch):
    sample = ri.generate_selection(config, 123, "mar_overlap")
    fits = []

    class Spy:
        def fit(self, x, y, **kwargs):
            np.testing.assert_array_equal(x.index, sample.features.index[sample.accepted == 1])
            np.testing.assert_array_equal(y, sample.observed_outcome[sample.accepted == 1])
            fits.append(kwargs)

    monkeypatch.setattr(ri, "logistic_model", lambda c: Spy())
    monkeypatch.setattr(ri, "crossfit_propensity", lambda *args: np.full(config.samples, 0.5))
    ri.fit_observed_models(sample.features, sample.accepted, sample.observed_outcome, config, 123)
    assert fits[0] == {}
    np.testing.assert_allclose(fits[1]["logisticregression__sample_weight"], 1.0)


def test_sensitivity_is_monotone_and_never_creates_labels():
    a = [1, 1, 0, 0]
    y = np.array([1.0, 0.0, np.nan, np.nan])
    p = [0.1, 0.2, 0.0, 1.0]
    before = y.copy()
    r = ri.reject_sensitivity(a, y, p, [0.5, 1.0, 2.0])
    np.testing.assert_array_equal(y, before)
    assert [x["population_bad_rate_assumption"] for x in r] == [0.5, 0.5, 0.5]
    r = ri.reject_sensitivity(a, y, [0.1, 0.2, 0.2, 0.4], [0.5, 1.0, 2.0])
    assert (
        r[0]["assumed_reject_bad_rate"]
        < r[1]["assumed_reject_bad_rate"]
        < r[2]["assumed_reject_bad_rate"]
    )
    assert r[1]["population_bad_rate_assumption"] == pytest.approx((1 + 0.2 + 0.4) / 4)
    assert ri.reject_sensitivity([1], [0], [0.1], [1.0])[0]["assumed_reject_bad_rate"] is None
    with pytest.raises(ValueError):
        ri.reject_sensitivity(a, [1, 0, 1, 0], p, [1.0])
    with pytest.raises(ValueError):
        ri.reject_sensitivity(a, y, p, [0.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seeds": [1, 1]},
        {"seeds": [-1]},
        {"seeds": [2**32]},
        {"reject_odds_multipliers": [0.0]},
        {"true_propensity_floor": 0.0},
        {"estimated_propensity_floor": 1.0},
        {"maximum_weight": 0.5},
        {"unknown": 1},
    ],
)
def test_config_rejects_invalid_experiments(kwargs):
    with pytest.raises(ValueError):
        RejectInferenceConfig(**kwargs)


def test_runner_persists_masked_training_and_separate_oracle(config, tmp_path):
    m = run_reject_experiment(tmp_path / "run", config)
    assert m["is_synthetic"] and not m["original_final_test_accessed"]
    assert len(m["results"]) == 3 and len(m["aggregate_metrics"]) == 8
    training = pd.read_csv(tmp_path / "run/mar_overlap_123_training.csv")
    assert "synthetic_oracle_outcome" not in training
    assert training.loc[training.accepted == 0, "observed_outcome"].isna().all()
    holdout = pd.read_csv(tmp_path / "run/mar_overlap_123_holdout.csv")
    assert not set(training.row_position) & set(holdout.row_position)
    assert "synthetic_oracle_outcome" in holdout
    deterministic = [r for r in m["results"] if r["scenario"] == "deterministic_no_overlap"][0]
    assert "ipw" not in deterministic["evaluation"]
    assert deterministic["true_zero_support_fraction"] > 0
    with pytest.raises(FileExistsError):
        run_reject_experiment(tmp_path / "run", config)


def test_nonparametric_bounds_do_not_assume_reject_outcomes():
    assert ri.population_risk_bounds([1, 1, 0, 0], [1, 0, np.nan, np.nan]) == {
        "lower": 0.25,
        "upper": 0.75,
    }
    assert ri.population_risk_bounds([0, 0], [np.nan, np.nan]) == {"lower": 0.0, "upper": 1.0}
    assert ri.population_risk_bounds([1, 1], [1, 0]) == {"lower": 0.5, "upper": 0.5}
    with pytest.raises(ValueError):
        ri.population_risk_bounds([1, 0], [1, 0])
