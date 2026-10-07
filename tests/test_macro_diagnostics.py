"""Synthetic post-validation diagnostics; no licensed outcomes or ledger consumption."""

import copy
import json

import numpy as np
import pytest

from credit_risk.track_b.macro_diagnostics.distributions import (
    bin_index,
    frozen_bins,
    regimes,
    support_diagnostics,
)
from credit_risk.track_b.macro_diagnostics.math import (
    ablated,
    components,
    macro_contributions,
    one_at_a_time,
    oracle_intercepts,
    probabilities,
    substitute,
)
from credit_risk.track_b.macro_diagnostics.refits import diagnostic_view
from credit_risk.track_b.macro_diagnostics.verification import verify
from credit_risk.track_b.macro_hazard.data import DTYPE
from credit_risk.track_b.macro_hazard.models import fit, predict
from credit_risk.track_b.macro_hazard.protocol import PRIMARY
from credit_risk.track_b.macro_support.eligibility import ordinal


def data():
    result = np.zeros(300, dtype=DTYPE)
    random = np.random.default_rng(113)
    result["numeric"] = random.normal(size=(300, 6))
    result["categories"] = "known"
    result["macro"] = random.normal(size=(300, 8))
    result["month"] = ordinal("2010-09") + np.arange(300)
    result["duration"] = np.arange(300) % 180
    result["facility"] = np.arange(300) // 6
    result["vintage"] = 2006
    result["event"] = np.arange(300) % 3
    return result


@pytest.fixture(scope="module")
def fitted():
    rows = data()
    spec = dict(estimator=dict(C=1))
    return rows, fit(rows, spec, True, ()), fit(rows, spec, True, PRIMARY)


def test_frozen_logits_contributions_and_probabilities_reconstruct(fitted):
    rows, _, m2 = fitted
    logits, parts = components(m2, rows)
    reconstructed = sum(parts[n] for n in ["intercept", "mortgage", "duration", "cohort"]) + parts[
        "macro"
    ].sum(axis=1)
    assert np.allclose(logits, reconstructed, atol=1e-12)
    assert np.allclose(probabilities(logits), predict(m2, rows), atol=1e-12)


def test_excess_logit_retains_nonmacro_coefficient_changes(fitted):
    rows, m1, m2 = fitted
    first, a = components(m1, rows)
    second, b = components(m2, rows)
    macro = b["macro"].sum(axis=1)
    nonmacro = sum(b[n] - a[n] for n in ["intercept", "mortgage", "duration", "cohort"])
    assert np.allclose(second - first, macro + nonmacro, atol=1e-12)
    assert not np.allclose(second - first, macro, atol=1e-8)


def test_ablation_changes_predictions_without_refit_or_frozen_mutation(fitted):
    rows, _, m2 = fitted
    before = copy.deepcopy(m2)
    frozen = predict(m2, rows)
    logits, _ = components(m2, rows)
    contributions = macro_contributions(m2, rows)
    changed = ablated(logits, contributions, 0)
    assert not np.allclose(changed, frozen)
    assert np.array_equal(m2[1].coef_, before[1].coef_)
    assert np.array_equal(m2[1].intercept_, before[1].intercept_)
    assert m2[0].parameters == before[0].parameters
    assert np.array_equal(predict(m2, rows), frozen)


def test_reference_oat_is_zero_for_reference_and_nonadditive():
    base = np.array([[0.0, 0.0]])
    contributions = np.array([[[0.0, 2.0], [0.0, 2.0]]])
    individual = one_at_a_time(base, contributions)
    joint = probabilities(base + contributions.sum(axis=1)) - probabilities(base)
    assert not np.allclose(individual.sum(axis=1), joint)
    assert not one_at_a_time(base, np.zeros_like(contributions)).any()


def test_development_bins_fixed_under_extreme_evaluation_values():
    edges = frozen_bins(np.arange(100.0))
    before = edges.copy()
    index = bin_index(np.array([-100.0, 10000.0]), edges)
    assert index.tolist() == [0, len(edges) - 2]
    assert np.array_equal(edges, before)


def test_duplicate_development_quantiles_collapse_bins():
    edges = frozen_bins(np.tile([0.0, 1.0], 50))
    assert np.all(np.diff(edges) > 0)
    assert (bin_index([0.0, 1.0], edges) >= 0).all()


def test_macro_support_projection_does_not_use_outcomes():
    development = data()[:100]
    evaluation = data()[100:]
    before = support_diagnostics(development, evaluation)
    evaluation["event"] = (evaluation["event"] + 1) % 3
    after = support_diagnostics(development, evaluation)
    assert before == after
    assert before["multivariate"]["fitted_on"] == "Distinct development months only, no outcomes"


def test_calendar_partitions_exclude_purge_and_preserve_partial_2026():
    rows = data()[:4]
    rows["month"] = [ordinal(s) for s in ["2017-12", "2018-12", "2019-01", "2026-02"]]
    partitions = regimes(rows)
    assert partitions["2016_2017"].tolist() == [True, False, False, False]
    assert partitions["2019"].tolist() == [False, False, True, False]
    assert partitions["2026_partial"].tolist() == [False, False, False, True]


def test_diagnostic_refit_view_is_isolated_and_original_role_immutable():
    rows = data()
    rows["role"] = 1
    rows.flags.writeable = False
    diagnostic = diagnostic_view(rows, 2019, 2021)
    assert (diagnostic["role"] == 0).all()
    assert (rows["role"] == 1).all()
    assert not rows.flags.writeable


def test_component_substitution_cif_conserves_and_changes_competition():
    d = np.full((3, 24), 0.01)
    low = substitute(d, np.full((3, 24), 0.02))
    high = substitute(d, np.full((3, 24), 0.10))
    assert np.allclose(high.sum(axis=2), 1)
    assert (high[:, -1, 1] < low[:, -1, 1]).all()


def test_invalid_component_substitution_is_not_silently_normalized():
    with pytest.raises(ValueError, match="Invalid"):
        substitute(np.array([[0.8]]), np.array([[0.5]]))


def test_oracle_is_labeled_optimistic_and_never_mutates_original_logits():
    logits = np.tile([-0.5, 1.5], (300, 1))
    original = logits.copy()
    y = np.tile([0, 0, 0, 1, 2], 60)
    p, details = oracle_intercepts(logits, y)
    assert details["label"] == "POST_HOC_ORACLE_DIAGNOSTIC"
    assert details["same_sample_optimism"] and not details["independently_validated"]
    assert np.allclose(p[:, 1:].mean(axis=0), [0.2, 0.2], atol=1e-6)
    assert np.array_equal(logits, original)


def preservation_fixture(root):
    from credit_risk.track_b.data.schemas import digest

    folder = root / "data/track_b/models/macro_hazard_v1"
    folder.mkdir(parents=True)
    np.save(folder / "M2_evaluation.npy", np.full((3, 3), 1 / 3))
    (folder / "coefficients.json").write_text("[1,2,3]", encoding="utf-8")
    (folder / "task10_evaluation_ledger.json").write_text(
        json.dumps(dict(state="CONSUMED", prediction_generation_count=1)), encoding="utf-8"
    )
    manifest = dict(
        public_lf_hashes={},
        private_byte_hashes={str(p.relative_to(root)): digest(p) for p in folder.iterdir()},
    )
    docs = root / "docs/track_b"
    docs.mkdir(parents=True)
    (docs / "macro_signal_preservation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return folder


@pytest.mark.parametrize(
    "name", ["M2_evaluation.npy", "coefficients.json", "task10_evaluation_ledger.json"]
)
def test_frozen_predictions_coefficients_and_ledger_tamper_fail(tmp_path, name):
    folder = preservation_fixture(tmp_path)
    assert verify(tmp_path)["status"] == "PASSED"
    with (folder / name).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Frozen private"):
        verify(tmp_path)


def test_no_task10_ledger_consumption_api_for_diagnosis(tmp_path, monkeypatch):
    from credit_risk.track_b.macro_hazard.ledger import Ledger

    preservation_fixture(tmp_path)

    def prohibited(*args):
        raise AssertionError("Old ledger API must not run")

    monkeypatch.setattr(Ledger, "consume", prohibited)
    assert not verify(tmp_path)["task10_ledger_api_called"]


def test_calendar_evidence_serializes_native_partial_flags():
    from credit_risk.track_b.macro_diagnostics.study import calendar_diagnostics

    rows = data()
    rows["month"] = ordinal("2026-02")
    evidence = calendar_diagnostics(rows, dict(M1=np.full((len(rows), 3), 1 / 3)))
    assert evidence["2026"]["partial"] is True
    json.dumps(evidence, allow_nan=False)


def test_exact_brier_variance_covariance_accounting():
    from credit_risk.track_b.macro_diagnostics.math import brier_accounting

    y = np.array([0.0, 0.0, 0.0, 1.0])
    p = np.array([0.05, 0.1, 0.8, 0.9])
    result = brier_accounting(y, p)
    assert result["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert result["mean_bias_squared"] == pytest.approx((p.mean() - y.mean()) ** 2)
