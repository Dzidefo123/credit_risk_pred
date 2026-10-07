"""Task10 synthetic regressions; no private risk arrays or locked outcomes."""

import copy
import json

import numpy as np
import pytest

from credit_risk.track_b.macro_hazard.data import DTYPE, EVENT, MACRO, block
from credit_risk.track_b.macro_hazard.ledger import Ledger
from credit_risk.track_b.macro_hazard.models import Encoder, artifact, fit, predict
from credit_risk.track_b.macro_hazard.protocol import PRIMARY, RATE, REDUCED, freeze
from credit_risk.track_b.survival.math import curves


def synthetic():
    data = np.zeros(180, dtype=DTYPE)
    rng = np.random.default_rng(781)
    data["numeric"] = rng.normal(size=(180, 6))
    data["categories"] = "known"
    data["macro"] = rng.normal(size=(180, 8))
    data["duration"] = np.arange(180) % 180
    data["vintage"] = 2006
    data["event"] = np.arange(180) % 3
    data["facility"] = np.arange(180) // 6
    data["month"] = 24000 + np.arange(180)
    return data


def protocol(tmp_path):
    path = tmp_path / "docs/track_b/macro_support_validation_design.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    return freeze(tmp_path)


def test_joint_probabilities_and_frozen_parameter_order(tmp_path):
    data = synthetic()
    model = fit(data, protocol(tmp_path), True, PRIMARY)
    p = predict(model, data)
    assert np.allclose(p.sum(axis=1), 1)
    assert p.shape == (180, 3)
    assert tuple(artifact(model)["macro"]) == PRIMARY
    assert "mortgage_treasury_spread" not in model[0].names


def test_event_encoding_and_no_fourth_censor_class():
    assert EVENT == {"none": 0, "default": 1, "payoff": 2}
    with pytest.raises(KeyError):
        _ = EVENT["administrative"]


@pytest.mark.parametrize(
    "age,index",
    [(0, 0), (12, 0), (13, 1), (24, 1), (25, 2), (60, 3), (61, 4), (120, 5), (121, 6), (241, 8)],
)
def test_duration_encoding_matches_task9a(age, index):
    data = synthetic()[:1]
    data["duration"] = age
    encoder = Encoder(False).fit(data)
    x = encoder.transform(data).toarray()
    assert x[0, :8].sum() == (index != 0)
    if index:
        assert x[0, index - 1] == 1


def test_no_evaluation_preprocessing_fit_or_encoder_mutation():
    data = synthetic()
    encoder = Encoder(True, PRIMARY).fit(data)
    before = copy.deepcopy(encoder.parameters)
    evaluation = data.copy()
    evaluation["role"] = 1
    evaluation["numeric"] = 999
    evaluation["categories"] = "unseen"
    encoder.transform(evaluation)
    assert encoder.parameters == before
    with pytest.raises(ValueError, match="development"):
        Encoder(True).fit(evaluation)
    with pytest.raises(ValueError, match="exactly once"):
        encoder.fit(data)


def test_missingness_is_training_only_and_explicit():
    data = synthetic()
    data["numeric"][0, 0] = np.nan
    encoder = Encoder(True).fit(data)
    x = encoder.transform(data).toarray()
    assert x[0, 6] == 1
    assert np.isfinite(x).all()


def test_unseen_cohort_effect_is_flaggable_reference_contribution():
    data = synthetic()
    encoder = Encoder(False).fit(data)
    encoder.parameters["trained_vintages"] = [2006]
    evaluation = data[:1].copy()
    evaluation["vintage"] = 2022
    assert not encoder.transform(evaluation).toarray()[0, 8:].any()


def test_rate_representation_and_required_macro_firewall():
    data = synthetic()
    assert len(PRIMARY) == 7 and len(RATE) == 7
    assert set(RATE) - set(PRIMARY) == {"mortgage_treasury_spread"}
    with pytest.raises(ValueError, match="dependent"):
        Encoder(True, (*PRIMARY, "mortgage_treasury_spread")).fit(data)
    data["macro"][0, MACRO.index("hpi_yoy")] = np.nan
    with pytest.raises(ValueError, match="PIT"):
        Encoder(True, PRIMARY).fit(data)


def test_reduced_vector_ignores_unavailable_excluded_terms_but_requires_its_inputs():
    data = synthetic()
    assert len(REDUCED) == 5
    for name in ["mortgage_30y_level", "mortgage_treasury_spread", "hpi_yoy"]:
        data["macro"][:, MACRO.index(name)] = np.nan
    encoder = Encoder(True, REDUCED).fit(data)
    assert np.isfinite(encoder.transform(data).data).all()
    data["macro"][0, MACRO.index("unemployment_level")] = np.nan
    with pytest.raises(ValueError, match="PIT"):
        encoder.transform(data)


def test_population_rejects_shared_facility_even_when_support_counts_match(monkeypatch, tmp_path):
    from credit_risk.track_b.macro_hazard import study
    from credit_risk.track_b.macro_support.eligibility import ordinal

    development = synthetic()
    development["month"] = ordinal("2017-01")
    evaluation = development[:6].copy()
    evaluation["month"] = ordinal("2019-01")
    evaluation["role"] = 1
    monkeypatch.setattr(
        study, "load", lambda root: dict(development=development, evaluation=evaluation)
    )
    monkeypatch.setattr(
        study,
        "read_json",
        lambda path: dict(
            validation_counts=dict(PRIMARY=dict(development=study.counts(development))),
            primary_temporal_seen_vintage_counts=study.counts(evaluation),
        ),
    )
    with pytest.raises(ValueError, match="Facility role leakage"):
        study.population(tmp_path, dict(splits=dict(primary_vintages=[2006])))


@pytest.mark.parametrize(
    "month,role,expected",
    [
        ("2017-12", "development", "development"),
        ("2018-01", "development", None),
        ("2018-12", "temporal_evaluation", None),
        ("2019-01", "temporal_evaluation", "evaluation"),
        ("2026-03", "temporal_evaluation", None),
        ("2010-08", "development", None),
    ],
)
def test_frozen_split_boundaries(tmp_path, month, role, expected):
    assert block(month, role, "PRIMARY", protocol(tmp_path)) == expected


def test_ledger_registers_before_fit_binds_inputs_and_consumes_once(tmp_path):
    registration = dict(namespace="TASK10_MACRO_HAZARD", population="hash", specification="fixed")
    ledger = Ledger(tmp_path / "task10_evaluation_ledger.json", registration)
    assert ledger.data["state"] == "REGISTERED_BEFORE_FIT"
    with pytest.raises(ValueError):
        ledger.consume(registration)
    ledger.freeze_models(dict(M2="modelhash"))
    with pytest.raises(ValueError, match="changed"):
        ledger.consume(dict(registration, population="changed"))
    ledger.consume(registration)
    ledger.complete(dict(M2="predhash"))
    with pytest.raises(ValueError):
        ledger.consume(registration)
    assert json.loads(ledger.path.read_text(encoding="utf-8"))["prediction_generation_count"] == 1


@pytest.mark.parametrize("name", ["evaluation_ledger.json", "track_a_ledger.json"])
def test_prior_ledger_names_cannot_be_reused(tmp_path, name):
    with pytest.raises(ValueError):
        Ledger(tmp_path / name, dict(namespace="TASK10_MACRO_HAZARD"))


def test_cif_recursion_competing_payoff_and_conservation():
    q = curves(np.full((1, 12), 0.02), np.full((1, 12), 0.10))
    assert np.allclose(q.sum(axis=2), 1)
    assert q[0, -1, 1] < 1 - 0.98**12
    assert q[0, -1, 0] == pytest.approx(0.88**12)
    with pytest.raises(ValueError):
        curves(np.array([[0.9]]), np.array([[0.9]]))


def test_paired_cluster_bootstrap_identical_models_has_zero_increment():
    from credit_risk.track_b.macro_hazard.metrics import paired

    data = synthetic()
    data["month"] = np.arange(len(data)) % 2 + 24228
    p = np.full((len(data), 3), 1 / 3)
    result = paired(data, p, p, draws=30)
    assert result["clusters"] == 30
    assert result["unit"] == "facility"
    assert all(c["lower"] == c["upper"] == 0 for c in result["intervals"].values())


def test_weighted_auc_handles_ties_like_reference():
    from sklearn.metrics import roc_auc_score

    from credit_risk.track_b.macro_hazard.metrics import auc_cache, weighted_auc

    y = np.tile([0, 1], 30)
    p = np.tile([0.1, 0.1, 0.8, 0.9], 15)
    groups = np.arange(60) % 10
    weights = np.arange(10) + 1
    actual = weighted_auc(auc_cache(y, p, groups), weights)
    assert actual == pytest.approx(roc_auc_score(y, p, sample_weight=weights[groups]))


@pytest.mark.parametrize("tamper", ["future", "current", "unavailable", "wrong_t0"])
def test_macro_join_rejects_future_current_and_unverified_information(tamper):
    from credit_risk.track_b.macro_hazard.data import macro_join

    features = {
        n: dict(
            value=1.0,
            status="AVAILABLE",
            t0="2018-12-31",
            inputs=[
                dict(
                    representation="vintage",
                    archive_start="2018-12-01",
                    archive_end="2026-03-31",
                    reference_period="2018-11-30",
                    publication_upper_bound="2018-12-01",
                    revision_upper_bound="2018-12-01",
                )
            ],
        )
        for n in MACRO
    }
    f = features["unemployment_level"]
    if tamper == "future":
        f["inputs"][0]["revision_upper_bound"] = "2019-01-01"
    if tamper == "current":
        f["inputs"][0]["representation"] = "current"
    if tamper == "unavailable":
        f["status"] = "UNAVAILABLE"
    if tamper == "wrong_t0":
        f["t0"] = "2019-01-31"
    with pytest.raises(ValueError):
        macro_join({"2019-01": dict(features=features)}, "2019-01", PRIMARY)


def test_cif_landmarks_unique_and_future_horizon_truncation():
    from credit_risk.track_b.macro_hazard.cif import landmarks, path_forecast
    from credit_risk.track_b.macro_support.eligibility import ordinal

    data = synthetic()[:12]
    data["month"] = ordinal("2026-01") + np.arange(12) % 6
    entries, subjects = landmarks(data)
    assert len(entries) == len(subjects) == 2
    with pytest.raises(ValueError, match="cutoff"):
        path_forecast((None, None), entries, {}, 12)


def test_prediction_feature_firewall_ignores_outcome_labels(tmp_path):
    data = synthetic()
    bundle = fit(data, protocol(tmp_path), True, PRIMARY)
    before = predict(bundle, data)
    changed = data.copy()
    changed["event"] = (changed["event"] + 1) % 3
    changed["role"] = 1
    assert np.array_equal(predict(bundle, changed), before)


def test_risk_arrays_are_readonly_and_integrity_bound(tmp_path):
    from credit_risk.track_b.data.schemas import digest
    from credit_risk.track_b.macro_hazard.data import load

    private = tmp_path / "data/track_b/models/macro_hazard_v1"
    private.mkdir(parents=True)
    for name in ["development", "evaluation"]:
        np.save(private / (name + ".npy"), synthetic())
    (private / "facility_keys.json").write_text('["synthetic"]', encoding="utf-8")
    audit = dict(
        array_byte_hashes={
            n: digest(private / (n + ".npy")) for n in ["development", "evaluation"]
        },
        facility_keys_sha256=digest(private / "facility_keys.json"),
    )
    (private / "data_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    assert not load(tmp_path)["evaluation"].flags.writeable
    with (private / "evaluation.npy").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        load(tmp_path)
