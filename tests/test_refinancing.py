"""Synthetic Task12 regression checks; no private records or old holdout outcomes."""

import copy
import inspect
import json

import numpy as np
import pytest

from credit_risk.track_b.macro_hazard.data import DTYPE, MACRO
from credit_risk.track_b.macro_hazard.models import Encoder, predict
from credit_risk.track_b.refinancing import study
from credit_risk.track_b.refinancing.audit import evidence_status
from credit_risk.track_b.refinancing.features import (
    checked_market,
    gap,
    proxy_status,
    representation,
)
from credit_risk.track_b.refinancing.models import CONTEXT, RefiEncoder, fit
from credit_risk.track_b.refinancing.protocol import freeze
from credit_risk.track_b.survival.math import curves


def synthetic():
    rng = np.random.default_rng(1201)
    data = np.zeros(150, dtype=DTYPE)
    data["numeric"] = rng.normal(size=(150, 6))
    data["numeric"][:, 4] = rng.uniform(3, 7, 150)
    data["macro"] = rng.normal(size=(150, 8))
    data["macro"][:, 3] = rng.uniform(3, 7, 150)
    data["categories"] = "P"
    data["duration"] = np.arange(150)
    data["vintage"] = 2006
    data["facility"] = np.arange(150) // 5
    data["event"] = np.arange(150) % 3
    return data


def test_subtraction_percentage_points_and_determinism():
    data = synthetic()[:2]
    data["numeric"][:, 4] = [6.5, 4]
    data["macro"][:, 3] = [4, 6.5]
    assert gap(data).tolist() == [2.5, -2.5]
    assert np.array_equal(gap(data), gap(data.copy()))
    assert representation(data).tolist() == [[2.5, 0], [0, -2.5]]
    assert representation(data, True).tolist() == [[2.5], [-2.5]]


@pytest.mark.parametrize("value", [np.nan, np.inf, 0, -1, 100])
def test_missing_or_malformed_contract_rate_stops(value):
    data = synthetic()
    data["numeric"][0, 4] = value
    with pytest.raises(ValueError):
        gap(data)


def market():
    return {
        "2020-01": dict(
            features={
                "mortgage_30y_level": dict(
                    status="AVAILABLE",
                    t0="2019-12-31",
                    value=4,
                    inputs=[
                        dict(
                            representation="vintage",
                            current_revised=False,
                            archive_start="2019-12-20",
                            archive_end="9999-12-31",
                            reference_period="2019-12-19",
                            publication_upper_bound="2019-12-20",
                            revision_upper_bound="2019-12-20",
                        )
                    ],
                )
            }
        )
    }


def test_pit_market_valid_and_no_future_release():
    table = market()
    for name in MACRO:
        table["2020-01"]["features"].setdefault(name, {})
    assert checked_market(table, "2020-01") == 4
    table["2020-01"]["features"]["mortgage_30y_level"]["inputs"][0]["publication_upper_bound"] = (
        "2020-01-01"
    )
    with pytest.raises(ValueError, match="Future"):
        checked_market(table, "2020-01")


@pytest.mark.parametrize(
    "field,value",
    [
        ("representation", "current"),
        ("current_revised", True),
        ("reference_period", "2020-01-02"),
        ("revision_upper_bound", "2020-01-02"),
    ],
)
def test_future_or_revised_market_forbidden(field, value):
    table = market()
    table["2020-01"]["features"]["mortgage_30y_level"]["inputs"][0][field] = value
    with pytest.raises(ValueError):
        checked_market(table, "2020-01")


def test_modified_loan_is_explicit_original_proxy():
    assert proxy_status("4", 6, True) == "MODIFIED_ORIGINAL_COUPON_PROXY"
    assert proxy_status("4", 6, False) == "COUPON_DIFFERS"
    assert proxy_status("", 6, True) == "CURRENT_RATE_MISSING"
    assert proxy_status("bad", 6, False) == "CURRENT_RATE_MALFORMED"


def test_baseline_design_exactly_matches_frozen_encoder():
    data = synthetic()
    a, b = Encoder(True).fit(data), RefiEncoder(True).fit(data)
    assert a.names == b.names
    assert a.parameters == b.parameters
    assert np.array_equal(a.transform(data).toarray(), b.transform(data).toarray())


@pytest.mark.parametrize(
    "name,terms",
    [
        ("P2", ["REFI_POS", "REFI_NEG"]),
        ("LINEAR", ["REFI_GAP"]),
        ("P3", ["REFI_POS", "REFI_NEG", *CONTEXT]),
    ],
)
def test_joint_models_feature_order_and_conservation(name, terms):
    data = synthetic()
    bundle = fit(data, name)
    assert bundle[0].names[6 : 6 + len(terms)] == terms
    assert "mortgage_30y_level" not in bundle[0].names
    assert "treasury_10y_level" not in bundle[0].names
    p = predict(bundle, data)
    assert p.shape == (150, 3)
    assert np.allclose(p.sum(axis=1), 1)
    c = curves(p[:, 1, None].repeat(12, axis=1), p[:, 2, None].repeat(12, axis=1))
    assert np.allclose(c.sum(axis=2), 1)


def test_preprocessing_never_fits_evaluation():
    data = synthetic()
    encoder = RefiEncoder(True, refi=True).fit(data)
    before = copy.deepcopy(encoder.parameters)
    data["role"] = 1
    encoder.transform(data)
    assert encoder.parameters == before
    with pytest.raises(ValueError):
        fit(data, "P2")


@pytest.mark.parametrize("age,index", [(12, 0), (13, 1), (60, 3), (61, 4), (181, 7), (241, 8)])
def test_duration_cohort_encoding_unchanged(age, index):
    data = synthetic()[:1]
    data["duration"] = age
    structural = RefiEncoder(False).fit(data)
    design = structural.transform(data).toarray()
    assert design[0, :8].sum() == (index != 0)
    if index:
        assert design[0, index - 1] == 1


def test_redundant_absolute_rate_block_rejected():
    with pytest.raises(ValueError, match="redundant"):
        RefiEncoder(True, macro=("mortgage_30y_level",), refi=True).fit(synthetic())


@pytest.mark.parametrize(
    "inspected,accessible,sealed,external,expected",
    [
        (True, True, True, False, "EXPLORATORY_ONLY"),
        (False, True, True, False, "INDEPENDENT_CONFIRMATORY_AVAILABLE"),
        (False, True, True, True, "INDEPENDENT_EXTERNAL_AVAILABLE"),
        (False, False, True, True, "INSUFFICIENT_FOR_NEW_VALIDATION"),
    ],
)
def test_evidence_gate(inspected, accessible, sealed, external, expected):
    assert (
        evidence_status(
            [
                dict(
                    inspected=inspected,
                    accessible=accessible,
                    sealed_before_fit=sealed,
                    external=external,
                )
            ]
        )
        == expected
    )


def test_no_old_ledger_api_or_false_holdout_claim():
    source = inspect.getsource(study)
    assert "Ledger(" not in source and ".consume(" not in source
    assert "virgin_holdout=False" in source


def test_prespecification_immutable_and_precedes_fit(tmp_path):
    private = tmp_path / "data/track_b/models/refinancing_v1"
    private.mkdir(parents=True)
    (private / "audit.json").write_text(json.dumps(dict(validation_status="EXPLORATORY_ONLY")))
    old = tmp_path / "data/track_b/models/macro_hazard_v1"
    old.mkdir()
    for n in ["development", "evaluation"]:
        (old / (n + ".npy")).write_bytes(b"synthetic")
    (tmp_path / "docs/track_b").mkdir(parents=True)
    first = freeze(tmp_path)
    assert first == freeze(tmp_path)
    assert first["primary_representation"] == ["REFI_POS", "REFI_NEG"]
    assert first["bootstrap"]["draws"] == 1000
    assert first["virgin_holdout"] is False
    (old / "evaluation.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen"):
        freeze(tmp_path)


def test_future_cif_path_rejected_before_prediction():
    entries = synthetic()[:1]
    entries["month"] = 2026 * 12 + 1
    with pytest.raises(ValueError, match="Future"):
        study.forecast((None, None), entries, {}, 12)


def test_repeated_exploratory_scoring_is_blocked(tmp_path, monkeypatch):
    private = tmp_path / "data/track_b/models/refinancing_v1"
    private.mkdir(parents=True)
    (private / "prediction_P1.npy").write_bytes(b"already scored")
    monkeypatch.setattr(study, "verify", lambda root: None)
    monkeypatch.setattr(study, "freeze", lambda root: {})
    with pytest.raises(ValueError, match="already started"):
        study.explore(tmp_path)


def test_unprespecified_models_and_incomplete_events_rejected():
    data = synthetic()
    with pytest.raises(ValueError, match="Unprespecified"):
        fit(data, "BOOSTED")
    data["event"] = 0
    with pytest.raises(ValueError, match="three competing"):
        fit(data, "P2")


def test_landmark_first_event_exit_and_contiguity():
    from credit_risk.track_b.macro_hazard.cif import landmarks

    data = synthetic()[:5]
    data["facility"] = 0
    data["month"] = 24000 + np.arange(5)
    data["event"] = [0, 0, 0, 0, 2]
    entries, subjects = landmarks(data)
    assert len(entries) == 1 and subjects.event_code.tolist() == [2]
    assert subjects.exit_time.tolist() == [5]
    data["month"][4] += 1
    with pytest.raises(ValueError, match="contiguous"):
        landmarks(data)


def test_preservation_is_line_ending_independent_but_private_bytes_are_frozen(
    tmp_path, monkeypatch
):
    from credit_risk.track_b.data.schemas import digest
    from credit_risk.track_b.multivintage.study import lf_hash
    from credit_risk.track_b.refinancing import audit

    public = tmp_path / "public.md"
    private = tmp_path / "old_prediction.npy"
    public.write_bytes(b"frozen\nreport\n")
    private.write_bytes(b"frozen model probabilities")
    manifest = dict(
        public_lf_hashes={"public.md": lf_hash(public)},
        private_byte_hashes={"old_prediction.npy": digest(private)},
    )
    path = tmp_path / "docs/track_b/refinancing_preservation_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(audit, "earlier", lambda root, source_dir: dict(status="PASSED"))
    public.write_bytes(b"frozen\r\nreport\r\n")
    assert audit.verify(tmp_path)["status"] == "PASSED"
    private.write_bytes(b"different probabilities")
    with pytest.raises(ValueError, match="Frozen private"):
        audit.verify(tmp_path)


def test_independent_status_requires_separate_protocol_and_ledger(tmp_path):
    private = tmp_path / "data/track_b/models/refinancing_v1"
    private.mkdir(parents=True)
    (private / "audit.json").write_text(
        json.dumps(dict(validation_status="INDEPENDENT_CONFIRMATORY_AVAILABLE"))
    )
    with pytest.raises(ValueError, match="amendment"):
        freeze(tmp_path)
