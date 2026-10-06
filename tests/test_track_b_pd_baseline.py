"""Task 3 uses synthetic fixtures; real licensed panel is never a test dependency."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.track_b.data.schemas import AUDIT, OUTCOMES
from credit_risk.track_b.data.schemas import FEATURES as PANEL_FEATURES
from credit_risk.track_b.pd.baseline import (
    FEATURES,
    SALT,
    bootstrap,
    cluster_indices,
    cohort,
    cohort_hash,
    counts,
    features,
    fit_baselines,
    gate,
    metrics,
    split,
)
from credit_risk.track_b.pd.research import frozen_inputs

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "docs/track_b/pd_feature_registry.json").read_text())


def row(loan="a", t0="2012-03", status="negative_survived_horizon", y=0):
    return dict(
        loan_id=loan,
        t0=t0,
        eligible=True,
        delinquency_state="00",
        outcome_status=status,
        binary_default_12m=y,
        orig_credit_score=720.0,
        orig_ltv=80.0,
        loan_age=24.0,
    )


def fixture():
    dev_ids, eval_ids = [], []
    for i in range(200):
        loan = f"synthetic-{i}"
        target = (
            dev_ids
            if int(hashlib.sha256((SALT + loan).encode()).hexdigest(), 16) % 10 < 7
            else eval_ids
        )
        target.append(loan)
    rows = []
    for loans, date in [(dev_ids[:30], "2012-03"), (eval_ids[:20], "2017-03")]:
        for i, loan in enumerate(loans):
            y = int(i < 12)
            r = row(loan, date, "positive_default" if y else "negative_survived_horizon", y)
            r.update(orig_credit_score=600.0 + 5 * i, orig_ltv=60.0 + i, loan_age=float(i + 6))
            rows.append(r)
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "status,y,expected",
    [
        ("positive_default", 1, 1),
        ("negative_survived_horizon", 0, 1),
        ("competing_payoff", 0, 1),
        ("right_censored", np.nan, 0),
        ("ambiguous_event_order", np.nan, 0),
        ("insufficient_followup", np.nan, 0),
    ],
)
def test_known_cohort_semantics(status, y, expected):
    assert len(cohort(pd.DataFrame([row(status=status, y=y)]))) == expected


def test_payoff_exclusion_and_no_physical_requirement():
    f = pd.DataFrame([row(status="competing_payoff")])
    assert len(cohort(f)) == 1
    assert len(cohort(f, payoff=False)) == 0
    assert len(cohort(f, physical=True)) == 0


def test_ineligible_prevalent_default_excluded():
    r = row()
    r.update(eligible=False, delinquency_state="03")
    assert cohort(pd.DataFrame([r])).empty
    r["eligible"] = True
    with pytest.raises(ValueError, match="prevalent"):
        cohort(pd.DataFrame([r]))


@pytest.mark.parametrize("status,y", [("positive_default", 0), ("right_censored", 0)])
def test_reject_contradictory_labels(status, y):
    with pytest.raises(ValueError):
        cohort(pd.DataFrame([row(status=status, y=y)]))


def test_duplicates_rejected_and_quarter_end_rule():
    r = row()
    with pytest.raises(ValueError, match="Duplicate"):
        cohort(pd.DataFrame([r, r]))
    f = pd.DataFrame([row(t0="2012-02"), row(t0="2012-03")])
    assert cohort(f, quarterly=True).t0.tolist() == ["2012-03"]


def test_group_calendar_purge_and_label_blind_assignment():
    f = fixture()
    d, e = split(f)
    assert not set(d.loan_id) & set(e.loan_id)
    assert pd.Period(d.t0.max(), freq="M") + 12 < pd.Period(e.t0.min(), freq="M")
    changed = f.copy()
    changed["binary_default_12m"] = 1 - changed.binary_default_12m
    d2, e2 = split(changed.sample(frac=1, random_state=1))
    assert set(d.loan_id) == set(d2.loan_id) and set(e.loan_id) == set(e2.loan_id)
    middle = f.copy()
    middle["t0"] = "2015-03"
    assert all(x.empty for x in split(middle))


def test_support_gate_stops_before_fit():
    d, e = split(fixture())
    e["binary_default_12m"] = 0
    assert gate(d, e) == "INSUFFICIENT EVENT SUPPORT"
    with pytest.raises(ValueError, match="gate"):
        fit_baselines(d, e, REGISTRY)
    d["binary_default_12m"] = 0
    assert gate(d, fixture()) == "INSUFFICIENT EVENT SUPPORT"


def test_registry_covers_panel_and_timing():
    assert set([*PANEL_FEATURES, *AUDIT, *OUTCOMES]) <= set(REGISTRY["fields"])
    assert REGISTRY["historical_knowledge_time"] == "UNVERIFIED"
    assert REGISTRY["selected"] == list(FEATURES)


@pytest.mark.parametrize(
    "name",
    [
        "binary_default_12m",
        "loan_id",
        "t0",
        "event_offset",
        "observed_followup_months",
        "termination_month",
        "actual_loss",
        "mi_recoveries",
        "future_delinquency",
        "future_balance",
        "future_macro",
        "future_payoff",
    ],
)
def test_feature_firewall(name):
    with pytest.raises(ValueError, match="firewall"):
        features(fixture(), REGISTRY, requested=(*FEATURES, name))


def test_registry_role_tampering_rejected():
    r = copy.deepcopy(REGISTRY)
    r["fields"][FEATURES[0]]["classification"] = "OUTCOME"
    with pytest.raises(ValueError, match="Forbidden"):
        features(fixture(), r)


def test_development_only_preprocessing_and_deterministic_fit():
    d, e = split(fixture())
    d.iloc[0, d.columns.get_loc("orig_credit_score")] = np.nan
    model, p = fit_baselines(d, e, REGISTRY)
    extreme = e.copy()
    extreme["orig_credit_score"] = 9999.0
    second, _ = fit_baselines(d, extreme, REGISTRY)
    assert model[0].statistics_[0] == d.orig_credit_score.median()
    np.testing.assert_allclose(model[1].mean_, second[1].mean_)
    np.testing.assert_allclose(model[-1].coef_, second[-1].coef_)
    _, repeat = fit_baselines(d, e, REGISTRY)
    np.testing.assert_array_equal(p["logistic"], repeat["logistic"])
    assert np.all(p["null"] == d.binary_default_12m.mean())


def test_single_class_metrics_are_explicit():
    m = metrics([0, 0], [0.1, 0.2])
    assert m["roc_auc"] is None and m["average_precision"] is None
    assert m["brier"] == pytest.approx(0.025)
    with pytest.raises(ValueError):
        metrics([0, 1], [np.nan, 0.2])


def test_cluster_bootstrap_retains_whole_loans_and_reports_invalid_draws():
    loans = ["a", "a", "b", "b", "b"]
    for seed in range(10):
        idx = cluster_indices(loans, np.random.default_rng(seed))
        assert sum(idx == 0) == sum(idx == 1)
        assert sum(idx == 2) == sum(idx == 3) == sum(idx == 4)
    f = pd.DataFrame(
        [
            row("a", y=1, status="positive_default"),
            row("a", "2012-04", y=1, status="positive_default"),
            row("b"),
        ]
    )
    a = bootstrap(f, [0.8, 0.7, 0.1], draws=30, seed=4)
    assert a == bootstrap(f, [0.8, 0.7, 0.1], draws=30, seed=4)
    assert a["single_class_draws"] > 0
    assert a["intervals"]["brier"]["valid_draws"] == 30
    assert a["intervals"]["roc_auc"]["valid_draws"] < 30
    assert counts(f)["default_loans"] == 1


def test_canonical_cohort_hash_is_order_independent():
    f = cohort(fixture())
    assert cohort_hash(f) == cohort_hash(f.sample(frac=1, random_state=2))
    changed = f.copy()
    changed.loc[0, "binary_default_12m"] = 1 - changed.loc[0, "binary_default_12m"]
    assert cohort_hash(f) != cohort_hash(changed)


def test_frozen_input_gate_rejects_redraw_before_fitting(tmp_path):
    folder = tmp_path / "data/track_b/manifests/annual_2010_v1"
    folder.mkdir(parents=True)
    (folder / "selected_ids.txt").write_text("synthetic-redraw\n")
    with pytest.raises(ValueError, match="sample mismatch"):
        frozen_inputs(tmp_path, tmp_path / "archive.zip")


def test_empirical_report_anchors_and_sparse_interpretation():
    report = json.loads((ROOT / "reports/track_b/pd_baseline_validation.json").read_text())
    assert (
        report["sample_set_sha256"]
        == "b40de7596b0d4fbb189c1ceec5176f4461f7d290117250c1a7ec15b605ce7b11"
    )
    assert (
        report["source_archive_sha256"]
        == "a82bc0f1efffccfb494b2b33f61877428bb4a6443c1a73d44bc7ee24d77aa45d"
    )
    assert report["decision"] == "BASELINE ESTABLISHED — EXPLORATORY ONLY"
    assert report["temporal_split"]["development"]["default_loans"] == 13
    assert report["temporal_split"]["evaluation"]["default_loans"] == 5
    assert (
        report["baselines"]["logistic"]["calibration"]["support_status"]
        == "UNSTABLE / INSUFFICIENT EVENT SUPPORT"
    )


@pytest.mark.parametrize("bad", ["source", "panel"])
def test_source_and_panel_identity_mismatches_stop(tmp_path, monkeypatch, bad):
    from credit_risk.track_b.pd import research

    folder = tmp_path / "data/track_b/manifests/annual_2010_v1"
    folder.mkdir(parents=True)
    ids = sorted(f"synthetic-{i}" for i in range(1000))
    (folder / "selected_ids.txt").write_text("\n".join(ids) + "\n")
    monkeypatch.setattr(research, "SAMPLE", hashlib.sha256("\n".join(ids).encode()).hexdigest())

    def fake_digest(path):
        if Path(path).name == "archive.zip":
            return "bad" if bad == "source" else research.SOURCE
        return "bad" if bad == "panel" else research.PANEL

    monkeypatch.setattr(research, "digest", fake_digest)
    with pytest.raises(ValueError, match="source/panel mismatch"):
        research.frozen_inputs(tmp_path, tmp_path / "archive.zip")


def test_published_evidence_matches_final_design_registry_and_code():
    report = json.loads((ROOT / "reports/track_b/pd_baseline_validation.json").read_text())
    paths = {
        "design_sha256_lf": ROOT / "docs/track_b/PD_COHORT_DESIGN.md",
        "feature_registry_sha256_lf": ROOT / "docs/track_b/pd_feature_registry.json",
    }
    for key, path in paths.items():
        assert hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == report[key]
    for name, expected in report["implementation_sha256_lf"].items():
        path = ROOT / "src/credit_risk/track_b/pd" / name
        assert hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected
