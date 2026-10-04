"""Fixed reference PSI, edge cases, alert policy and frozen scoring isolation."""

import json

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.validation import ORIGINATION_FEATURES
from credit_risk.monitoring import drift, runner
from credit_risk.monitoring.settings import AlertThreshold, MonitoringConfig


def small_config(**kwargs):
    return MonitoringConfig(minimum_population_rows=10, minimum_numeric_rows=2, **kwargs)


def test_psi_formula_and_population_scale_invariance():
    r = np.array([20.0, 80.0])
    c = np.array([50.0, 50.0])
    epsilon = 1e-6
    p = (r / r.sum() + epsilon) / (1 + 2 * epsilon)
    q = (c / c.sum() + epsilon) / (1 + 2 * epsilon)
    result = drift.population_stability_index(r, c, epsilon)
    assert result["psi"] == pytest.approx(((q - p) * np.log(q / p)).sum())
    assert sum(result["contributions"]) == pytest.approx(result["psi"])
    assert sum(result["current_shares"]) == pytest.approx(1.0)
    assert drift.population_stability_index([1, 4], [10, 40])["psi"] == 0.0
    assert drift.population_stability_index([0, 10], [10, 0])["psi"] > 0
    assert drift.population_stability_index(c, r)["psi"] == pytest.approx(result["psi"])


@pytest.mark.parametrize(
    "r,c,e",
    [
        ([], [], 1e-6),
        ([1], [1, 2], 1e-6),
        ([0], [1], 1e-6),
        ([1], [-1], 1e-6),
        ([np.nan], [1], 1e-6),
        ([1], [np.inf], 1e-6),
        ([1], [1], 0.0),
    ],
)
def test_invalid_psi_inputs(r, c, e):
    with pytest.raises(ValueError):
        drift.population_stability_index(r, c, e)


def test_constant_and_missing_reference_buckets_have_frozen_support():
    p = drift.fit_numeric_reference([2.0, 2.0, np.nan])
    assert p["kind"] == "constant" and p["counts"] == [0, 2, 0, 1]
    assert drift.bucket_counts([1.0, 2.0, 3.0, np.nan], p) == [1, 1, 1, 1]
    missing = drift.fit_numeric_reference([np.nan, np.nan])
    assert missing["counts"] == [0, 2]
    assert drift.bucket_counts([2.0, np.nan], missing) == [1, 1]


def test_quantile_ties_endpoints_and_missing_are_conserved():
    values = [0.0] * 20 + [1.0] * 5 + [2.0] * 5 + [np.nan]
    profile = drift.fit_numeric_reference(values)
    assert len(profile["cuts"]) == len(set(profile["cuts"]))
    counts = drift.bucket_counts([-100.0, 0.0, 1.0, 2.0, 100.0, np.nan], profile)
    assert sum(counts) == 6 and counts[-1] == 1
    assert profile["counts"] == drift.bucket_counts(values, profile)


def test_identical_population_zero_drift_and_current_never_refits_bins():
    config = small_config()
    frame = pd.DataFrame({"a": np.arange(100, dtype=float)})
    p = np.linspace(0.01, 0.2, 100)
    reference = drift.fit_reference(frame, p, config)
    before = json.dumps(reference, sort_keys=True)
    result = drift.compare_reference(reference, frame, p, config)
    assert result["status"] == "OK" and not result["alerts"]
    assert all(r["psi"] == 0 for r in result["metrics"].values())
    shifted = drift.compare_reference(reference, frame + 100, p + 0.3, config)
    assert shifted["status"] == "CRITICAL"
    assert shifted["metrics"]["a"]["bin_cuts"] == reference["profiles"]["a"]["cuts"]
    assert json.dumps(reference, sort_keys=True) == before
    assert shifted["outcome_validation"]["status"] == "unavailable"


def test_missingness_including_all_missing_and_constant_range_shift():
    config = small_config()
    frame = pd.DataFrame({"a": [2.0] * 100})
    p = np.full(100, 0.02)
    reference = drift.fit_reference(frame, p, config)
    current = frame.copy()
    current.loc[:29, "a"] = np.nan
    metrics = drift.compare_reference(reference, current, p, config)["metrics"]["a"]
    assert metrics["missing_rate_delta"] == pytest.approx(0.3)
    assert metrics["alerts"]["missingness_delta"] == "CRITICAL"
    assert metrics["wasserstein_reference_scale"] is None
    current["a"] = np.nan
    result = drift.compare_reference(reference, current, p, config)
    assert result["metrics"]["a"]["ks_statistic"] is None
    assert result["metrics"]["a"]["current_counts"][-1] == 100
    missing_reference = drift.fit_reference(current, p, config)
    assert drift.compare_reference(missing_reference, frame, p, config)["status"] == "CRITICAL"
    shifted = drift.compare_reference(reference, frame + 1, p, config)["metrics"]["a"]
    assert shifted["out_of_reference_range_fraction"] == 1.0
    assert shifted["ks_statistic"] == 1.0


def test_score_mapping_monotonic_and_doubles_good_bad_odds():
    config = small_config()
    p = np.array([0.0, 1 / 41, 1 / 21, 0.5, 1.0])
    scores = drift.pd_to_score(p, config)
    assert np.isfinite(scores).all() and np.all(np.diff(scores) < 0)
    assert scores[2] == pytest.approx(600.0)
    assert scores[1] - scores[2] == pytest.approx(20.0)
    for bad in ([np.nan], [-0.1], [1.1]):
        with pytest.raises(ValueError):
            drift.pd_to_score(bad, config)


def test_alert_endpoints_and_small_populations():
    threshold = AlertThreshold(warning=0.1, critical=0.25)
    assert [drift.severity(x, threshold) for x in (None, 0.099, 0.1, 0.25)] == [
        "UNAVAILABLE",
        "OK",
        "WARNING",
        "CRITICAL",
    ]
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    p = [0.02, 0.03]
    config = small_config()
    reference = drift.fit_reference(frame, p, config)
    assert drift.compare_reference(reference, frame, p, config)["status"] == "INSUFFICIENT_DATA"


def test_reference_measurement_and_feature_contract_cannot_silently_change():
    config = small_config()
    frame = pd.DataFrame({"a": np.arange(100, dtype=float)})
    p = np.full(100, 0.02)
    reference = drift.fit_reference(frame, p, config)
    with pytest.raises(ValueError, match="Measurement"):
        drift.compare_reference(reference, frame, p, small_config(bins=5))
    with pytest.raises(ValueError, match="names/order"):
        drift.compare_reference(reference, frame.rename(columns={"a": "b"}), p, config)
    changed = small_config(psi=AlertThreshold(warning=0.2, critical=0.4))
    assert drift.compare_reference(reference, frame, p, changed)["status"] == "OK"
    with pytest.raises(ValueError):
        drift.fit_reference(frame, [0.02], config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bins": 1},
        {"smoothing": 0.0},
        {"score_probability_epsilon": 0.5},
        {"score_points_to_double_odds": 0.0},
        {"minimum_population_rows": 0},
        {"unknown": 1},
    ],
)
def test_invalid_monitoring_config(kwargs):
    with pytest.raises(ValueError):
        MonitoringConfig(**kwargs)
    with pytest.raises(ValueError):
        AlertThreshold(warning=0.25, critical=0.1)


def test_infinite_numeric_values_and_empty_vectors_rejected():
    for values in ([np.inf], []):
        with pytest.raises(ValueError):
            drift.fit_numeric_reference(values)


def test_reference_and_runner_only_score_development_then_label_free_current(tmp_path, monkeypatch):
    frame = pd.DataFrame({name: np.zeros(30) for name in ORIGINATION_FEATURES})
    frame["age"] = np.arange(30, dtype=float) + 20.0
    frame["MonthlyIncome"] = 5000.0
    frame["DebtRatio"] = 0.2
    frame["RevolvingUtilizationOfUnsecuredLines"] = 0.3
    frame["source_row_id"] = np.arange(1, 31)
    source = tmp_path / "source.csv"
    source.write_text("source", encoding="utf-8")
    experiment = {"source_sha256": runner.digest(source)}
    splits = {"development": np.arange(10), "test": np.arange(10, 30)}
    seen = []

    class FrozenModel:
        def predict_proba(self, x):
            seen.append(x.index.tolist())
            assert list(x.columns) == list(ORIGINATION_FEATURES)
            return np.tile([0.98, 0.02], (len(x), 1))

    selection = {"preferred_candidate": "xgboost", "selected_methods": {"xgboost": "sigmoid"}}
    validation = {
        "source_sha256": experiment["source_sha256"],
        "target_semantics": "two-year delinquency",
        "artifacts_sha256": {
            "xgboost_selected.joblib": "modelhash",
            "selection.json": "selectionhash",
        },
    }
    monkeypatch.setattr(
        runner, "verify_experiment", lambda *args: (experiment, None, frame, splits, None)
    )
    monkeypatch.setattr(
        runner, "load_selected_model", lambda *args: (FrozenModel(), selection, validation)
    )
    config = small_config()
    ref = tmp_path / "ref"
    runner.freeze_monitor_reference(source, "run", "validation", ref, config)
    assert seen == [list(range(10))]
    current = tmp_path / "current.csv"
    frame.iloc[:10].to_csv(current, index=False)
    result = runner.run_monitoring(
        source, "run", "validation", ref, current, tmp_path / "out", config
    )
    assert result["status"] == "OK" and not result["final_test_scored"]
    assert len(seen) == 2 and seen[1] == list(range(10))
    assert result["outcome_validation"]["status"] == "unavailable"
    reserved = tmp_path / "reserved.csv"
    frame.iloc[10:20].to_csv(reserved, index=False)
    with pytest.raises(ValueError, match="reserved final-test"):
        runner.run_monitoring(
            source, "run", "validation", ref, reserved, tmp_path / "reserved_out", config
        )
    assert len(seen) == 2
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text(
        json.dumps({"artifacts_sha256": {"current.csv": "bad"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum"):
        runner.run_monitoring(
            source, "run", "validation", ref, current, tmp_path / "badout", config, bad_manifest
        )
    validation["artifacts_sha256"]["xgboost_selected.joblib"] = "different_model"
    with pytest.raises(ValueError, match="Model/source"):
        runner.run_monitoring(
            source, "run", "validation", ref, current, tmp_path / "different", config
        )
    with pytest.raises(FileExistsError):
        runner.freeze_monitor_reference(source, "run", "validation", ref, config)
    (ref / "reference.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        runner.load_reference(ref)


def test_reserved_final_test_rows_cannot_be_reused_for_monitoring():
    frame = pd.DataFrame({name: np.arange(4, dtype=float) for name in ORIGINATION_FEATURES})
    frame["source_row_id"] = [1, 2, 3, 4]
    splits = {"test": np.array([2, 3])}
    runner.guard_reserved_population(frame.iloc[:2], frame, splits)
    with pytest.raises(ValueError, match="identifiers"):
        runner.guard_reserved_population(frame.iloc[[2]], frame, splits)
    with pytest.raises(ValueError, match="predictor groups"):
        runner.guard_reserved_population(
            frame.iloc[[3]].drop(columns="source_row_id"), frame, splits
        )
