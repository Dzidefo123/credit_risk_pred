"""Aggregate-only Phase A and outcome-independent expansion safeguards."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.track_b.data.annual import AnnualReader, GateStop
from credit_risk.track_b.data.sampling import SALT, select_loans
from credit_risk.track_b.expansion import empirical as e
from credit_risk.track_b.expansion import planning as p
from credit_risk.track_b.pd import baseline

ROOT = Path(__file__).resolve().parents[1]


def test_calculations_deterministic_and_nested_prediction():
    rows, n = p.calculations()
    assert (rows, n) == p.calculations() and n == 20000
    assert [r["n"] for r in rows] == [5000, 10000, 20000]
    assert rows[1]["joint_probability_lower_bound"] < 0.9
    assert rows[2]["joint_probability_lower_bound"] >= 0.9
    assert rows[2]["support"]["evaluation"]["predictive_95"] == [40, 215]
    assert p.support(5, 1000)["expected"] == 5


def test_planning_only_reads_existing_aggregate_evidence(tmp_path, monkeypatch):
    for name in p.ALLOWED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    (tmp_path / "docs/track_b").mkdir(parents=True)
    reads = []
    original = Path.read_text

    def restricted(path, *args, **kwargs):
        assert str(path.relative_to(tmp_path)).replace("\\", "/") in p.ALLOWED
        reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", restricted)
    result = p.plan(tmp_path)
    assert len(reads) == 2 and result["chosen_n"] == 20000
    assert not (tmp_path / "data").exists()
    assert result["models_permitted"] is False


def test_frozen_amendment_matches_calculations_and_original_split():
    plan = json.loads((ROOT / "docs/track_b/sample_expansion_amendment.json").read_text())
    rows, n = p.calculations()
    assert plan["chosen_n"] == n and plan["candidates"] == rows
    assert plan["phase"] == "A_FROZEN"
    assert plan["cohort"]["development_end"] == baseline.DEV_END == "2014-12"
    assert plan["cohort"]["evaluation_start"] == baseline.EVAL_START == "2016-01"
    assert plan["cohort"]["purged_year"] == "2015"
    assert plan["original_sample_sha256"] == p.SAMPLE
    for name, expected in plan["planning_inputs_sha256_lf"].items():
        assert p.lf_hash(ROOT / name) == expected


def test_phase_b_cannot_access_archive_without_phase_a_tests(tmp_path):
    folder = tmp_path / "docs/track_b"
    folder.mkdir(parents=True)
    (folder / "sample_expansion_amendment.json").write_text("{}")
    with pytest.raises(GateStop, match="Phase A tests"):
        e.run(tmp_path, tmp_path / "nonexistent-archive.zip")
    assert not (tmp_path / "data/track_b/processed").exists()


def test_phase_a_hash_tampering_stops_before_source_access(tmp_path):
    docs = tmp_path / "docs/track_b"
    docs.mkdir(parents=True)
    (docs / "sample_expansion_amendment.json").write_text("{}")
    private = tmp_path / "data/track_b/manifests/expansion_v1"
    private.mkdir(parents=True)
    (private / "phase_a_test_attestation.json").write_text(
        json.dumps(dict(passed=True, amendment_sha256_lf="changed"))
    )
    with pytest.raises(GateStop, match="test/hash"):
        e.run(tmp_path, tmp_path / "nonexistent-archive.zip")


def test_nested_first_n_matches_original_algorithm_and_order_invariant(monkeypatch):
    ids = [f"synthetic-{i}" for i in range(2000)]
    original = select_loans(ids, 1000)
    monkeypatch.setattr(e, "SAMPLE", e.set_hash(original))
    expanded = e.ranking(ids, 1500)
    e.nested(expanded, original)
    random.Random(17).shuffle(ids)
    assert e.ranking(ids, 1500) == expanded
    assert expanded[:1000] == original
    with pytest.raises(GateStop, match="preservation"):
        e.nested(expanded, original[:-1] + ["replacement"])


def test_quarter_order_and_outcome_metadata_irrelevant():
    quarters = {q: [f"Q{q}-{i}" for i in range(20)] for q in range(1, 5)}
    a = [x for q in [1, 2, 3, 4] for x in quarters[q]]
    b = [x for q in [4, 2, 1, 3] for x in reversed(quarters[q])]
    assert e.ranking(a, 40) == e.ranking(b, 40)
    # Mapping values carry distracting metadata; only identifier keys are consumed.
    assert e.ranking(dict.fromkeys(a, {"default": True}), 40) == e.ranking(a, 40)
    expected = sorted(a, key=lambda x: (hashlib.sha256((SALT + ":" + x).encode()).hexdigest(), x))[
        :40
    ]
    assert e.ranking(a, 40) == expected


@pytest.mark.parametrize("n", [0, 20001, True, 1.5])
def test_invalid_expansion_size_rejected(n):
    with pytest.raises(GateStop):
        e.ranking(["a"], n)


def test_same_cohort_and_split_objects_reused():
    assert e.cohort is baseline.cohort and e.split is baseline.split
    rows = [
        dict(
            loan_id="synthetic",
            t0=t,
            eligible=True,
            delinquency_state="00",
            outcome_status="positive_default",
            binary_default_12m=1,
        )
        for t in ["2014-12", "2015-06", "2016-01"]
    ]
    primary = e.cohort(pd.DataFrame(rows))
    d, v = e.split(primary)
    assert not any(f.t0.str.startswith("2015").any() for f in [d, v])
    tally = e.Tally()
    tally.add(primary)
    assert tally.result()["positive_landmarks"] == 3 and tally.result()["default_loans"] == 1


def test_performance_access_requires_freeze_and_disallows_second_scan(monkeypatch):
    reader = e.ExpansionReader.__new__(e.ExpansionReader)
    reader.sample = None
    reader.chosen_n = 2
    reader.scanned_performance = set()
    monkeypatch.setattr(AnnualReader, "lines", lambda self, q, kind: iter(["synthetic-only"]))
    with pytest.raises(GateStop, match="freeze"):
        list(reader.lines(1, "performance"))
    reader.freeze(["a", "b"])
    assert list(reader.lines(1, "performance")) == ["synthetic-only"]
    with pytest.raises(GateStop, match="Second"):
        list(reader.lines(1, "performance"))
    with pytest.raises(GateStop, match="Adaptive"):
        reader.freeze(["a", "c"])
    with pytest.raises(GateStop, match="mismatch"):
        reader.freeze(["a"])


@pytest.mark.parametrize(
    "dev,ev,expected",
    [
        (150, 50, "ADEQUATE"),
        (149, 50, "MARGINAL"),
        (100, 30, "MARGINAL"),
        (200, 29, "INADEQUATE"),
        (99, 50, "INADEQUATE"),
    ],
)
def test_fixed_support_gates(dev, ev, expected):
    assert e.decision(dev, ev) == "EXPANSION SUPPORT " + expected
    assert e.decision(dev, ev, True) == "EXPANSION SUPPORT INADEQUATE"


def test_precision_planning_improves_with_event_support_without_predictions():
    assert p.auc_half_width(0.8, 50, 1250) < p.auc_half_width(0.8, 5, 125)
    assert np.isfinite(p.auc_half_width(0.8, 50, 1250))


def test_completed_empirical_evidence_respects_frozen_boundary():
    r = json.loads((ROOT / "reports/track_b/sample_expansion_feasibility.json").read_text())
    assert r["amendment_sha256_lf"] == p.lf_hash(
        ROOT / "docs/track_b/sample_expansion_amendment.json"
    )
    assert r["sample"]["chosen_n"] == 20000
    assert r["sample"]["original_first_1000_exact"] is True
    assert r["preservation"]["original_panel_equivalent_loans"] == 1000
    assert r["selected_history_missing"] == 0
    assert r["models_fitted"] is False and r["expanded_predictive_metrics_computed"] is False
    assert r["event_support"]["development"]["default_loans"] == 246
    assert r["event_support"]["evaluation"]["default_loans"] == 95
    assert r["any_qualifying_record_default_loans"] == 623
    assert r["event_support"]["primary"]["default_loans"] == 618
    assert r["decision"] == e.decision(246, 95)
    assert r["planning_vs_observed"]["overall"]["observed"] == 623


def test_previous_public_reports_preserved_cross_platform():
    r = json.loads((ROOT / "reports/track_b/sample_expansion_feasibility.json").read_text())
    for name, expected in r["preservation"]["previous_file_sha256"].items():
        if not name.startswith("reports/"):
            continue
        payload = (ROOT / name).read_bytes()
        candidates = [payload]
        if Path(name).suffix in {".md", ".json"}:
            candidates.append(payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        assert expected in [hashlib.sha256(c).hexdigest() for c in candidates], name
