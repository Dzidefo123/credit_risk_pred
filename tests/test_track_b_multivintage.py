"""Pre-access Task8 archive/identity/schema/chronology/APC tests using toy records only."""

import io
import json
import zipfile
from pathlib import Path

import pytest

from credit_risk.track_b.data.annual import GateStop
from credit_risk.track_b.data.schemas import ORIGINATION, PERFORMANCE, digest
from credit_risk.track_b.multivintage.core import (
    apc_diagnostic,
    horizon_support,
    normalize,
    rank_ids,
    rank_key,
    set_hash,
    trajectory,
)
from credit_risk.track_b.multivintage.reader import VintageReader
from credit_risk.track_b.multivintage.schema import parse
from credit_risk.track_b.multivintage.study import checkpoint_valid

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "docs/track_b/multi_vintage_protocol.json").read_text(encoding="utf-8"))
POLICY = json.loads(
    (ROOT / "docs/track_b/mortgage_research_protocol.json").read_text(encoding="utf-8")
)["event"]


def orig(loan="F06Q10000001"):
    d = dict(
        loan_id=loan,
        first_payment_month="200602",
        maturity_month="203601",
        amortization_type="FRM",
        orig_credit_score="700",
        orig_upb="100000",
        orig_ltv="80",
        orig_interest_rate="5",
        original_loan_term="360",
        occupancy_status="P",
        loan_purpose="P",
    )
    return [d.get(k, "") for k in ORIGINATION]


def perf(period="200603", code="", state="00", loan="F06Q10000001"):
    d = dict(
        loan_id=loan,
        reporting_month=period,
        delinquency_state=state,
        current_principal_balance="90000",
        loan_age="2",
        termination_code=code,
        termination_month=period if code else "",
    )
    return [d.get(k, "") for k in PERFORMANCE]


def bundle(path, vintage=2006, defect=None):
    with zipfile.ZipFile(path, "w") as outer:
        for q in range(1, 5):
            if defect == "missing" and q == 4:
                continue
            b = io.BytesIO()
            with zipfile.ZipFile(b, "w", compression=zipfile.ZIP_DEFLATED) as inner:
                loan = f"F{vintage % 100:02d}Q{q}0000001"
                if defect == "quarter_id" and q == 4:
                    loan = f"F{vintage % 100:02d}Q30000001"
                inner.writestr(f"orig_{vintage}Q{q}.txt", "|".join(orig(loan)) + "\n")
                inner.writestr(f"perf_{vintage}Q{q}.txt", "|".join(perf(loan=loan)) + "\n")
                if defect == "extra":
                    inner.writestr("extra.txt", "bad")
                if defect == "nested":
                    inner.writestr("another.zip", b"bad")
                if defect == "unsafe":
                    inner.writestr("../escape.txt", "bad")
            name = f"historical_data_{vintage}Q{q}.zip"
            outer.writestr(name, b"notzip" if defect == "malformed" else b.getvalue())
            if defect == "duplicate" and q == 1:
                outer.writestr(name, b.getvalue())
    return path


def test_valid_nested_annual_archive_and_performance_gate(tmp_path):
    path = bundle(tmp_path / "annual.zip")
    with VintageReader(path, tmp_path, PLAN["limits"], digest(path), 2006) as r:
        assert len(r.inventory) == 4 and len(list(r.lines(1, "origination"))) == 1
        with pytest.raises(GateStop):
            list(r.lines(1, "performance"))
        r.freeze(["F06Q10000001"])
        assert len(list(r.lines(1, "performance"))) == 1
        with pytest.raises(GateStop):
            r.freeze(["different"])


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "extra", "nested", "unsafe", "malformed"]
)
def test_archive_structure_fails_closed(tmp_path, defect):
    path = bundle(tmp_path / "annual.zip", defect=defect)
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        VintageReader(path, tmp_path, PLAN["limits"], digest(path), 2006)


def test_changed_source_hash(tmp_path):
    path = bundle(tmp_path / "annual.zip")
    with pytest.raises(GateStop, match="hash"):
        VintageReader(path, tmp_path, PLAN["limits"], "0" * 64, 2006)


def test_exact_first_hash_ranked_ids_row_and_quarter_order_invariant():
    ids = [f"F06Q{q}{i:07}" for q in range(1, 5) for i in range(5)]
    expected = sorted(ids, key=lambda i: (rank_key(i, 2006), i))[:7]
    assert rank_ids(ids, 2006, 7) == expected
    assert rank_ids(ids[::-1], 2006, 7) == expected
    assert rank_ids(ids[10:] + ids[:10], 2006, 7) == expected
    assert set_hash(expected) == set_hash(expected[::-1])


def test_namespace_outcome_independence_and_cap():
    rows = [dict(id=f"loan{i}", default=i % 2, credit=700) for i in range(20)]
    before = rank_ids([r["id"] for r in rows], 2006, 5)
    for r in rows:
        r["default"] = 1 - r["default"]
        r["credit"] = 0
    assert before == rank_ids([r["id"] for r in rows], 2006, 5)
    assert rank_key("same", 2006) != rank_key("same", 2008)
    assert len(rank_ids([str(i) for i in range(21000)], 2006)) == 20000
    with pytest.raises(ValueError):
        rank_ids(["x"], 2006, 20001)
    with pytest.raises(ValueError):
        rank_ids(rows, 2006)


def test_exact_release_mapping_type_normalization_and_signed_loss():
    r = parse(orig(), "origination", 2006)
    assert r["orig_credit_score"] == 700 and str(r["first_payment_month"]) == "2006-02"
    tokens = perf()
    tokens[PERFORMANCE.index("actual_loss")] = "-12.50"
    r = parse(tokens, "performance", 2006)
    assert r["actual_loss"] == -12.5
    assert r["loan_id"] == "F06Q10000001"
    with pytest.raises(ValueError):
        parse(orig(), "origination", 2008)
    with pytest.raises(ValueError, match="field_count"):
        parse(orig() + ["shift"], "origination", 2006)


@pytest.mark.parametrize(
    "field,token,state",
    [
        ("orig_ltv", "999", "MISSING_IN_SOURCE"),
        ("orig_dti", "", "MISSING_IN_SOURCE"),
        ("orig_upb", "bad", "PARSER_FAILURE"),
        ("net_sale_proceeds", "C", "MISSING_IN_SOURCE"),
    ],
)
def test_missing_and_special_state_preserved(field, token, state):
    assert normalize(token, field)[1] == state


def test_unavailable_not_applicable_and_incompatible_mapping():
    assert normalize("5", "orig_upb", available=False) == (None, "STRUCTURALLY_UNAVAILABLE")
    assert normalize("", "termination_month", applicable=False) == (None, "NOT_APPLICABLE")
    with pytest.raises(ValueError, match="SEMANTICALLY_INCOMPATIBLE"):
        normalize("5", "orig_upb", compatible=False)


def path(*tokens):
    return trajectory(
        [parse(r, "performance", 2006) for r in tokens],
        POLICY,
        PLAN["horizons"],
        PLAN["horizon_support_rule"],
    )


def test_order_duplicates_and_gap_are_audited_without_repair():
    r = path(perf("200605"), perf("200603"), perf("200603"))
    assert r["findings"]["source_out_of_order_facilities"] == 1
    assert r["findings"]["duplicate_facility_months"] == 1
    assert r["findings"]["gap_intervals"] == 1
    assert not r["support"][12]


def test_first_endpoint_and_post_terminal_quarantine():
    r = path(perf("200603"), perf("200604", code="01"), perf("200605", state="03"))
    assert r["endpoint"] == "payoff" and r["risk_event"] == "payoff"
    assert r["annotations"][-1]["post_source_termination"]
    assert not r["annotations"][-1]["analytical_prefix"]
    assert r["support"][12] and not r["at_risk"][12]


@pytest.mark.parametrize(
    "state,code,expected",
    [
        ("03", "", "default"),
        ("RA", "", "default"),
        ("00", "02", "default"),
        ("00", "01", "payoff"),
        ("00", "15", "administrative"),
        ("03", "01", "ambiguous"),
    ],
)
def test_inherited_event_comparability(state, code, expected):
    r = path(perf("200603"), perf("200604", state=state, code=code))
    assert r["endpoint"] == expected


def test_short_followup_and_prevalent_entry_are_not_extrapolated():
    r = path(perf("200603"), perf("200604"))
    assert r["observed_span"] == 1 and not r["support"][12]
    r = path(perf("200603", state="03"))
    assert r["endpoint"] == "default" and not r["support"][12]


def test_gap_never_reenters_analytical_risk():
    r = path(perf("200603"), perf("200605"), perf("200606", state="03"))
    assert r["risk_event"] == "gap" and not r["support"][12]
    assert (
        r["endpoint"] == "default"
    )  # descriptive raw endpoint differs from censored analytic prefix


def test_horizon_thresholds_are_prespecified():
    r = PLAN["horizon_support_rule"]
    assert horizon_support(1000, 10000, r) == "SUPPORTED"
    assert horizon_support(100, 2000, r) == "LIMITED"
    assert horizon_support(0, 20000, r) == "UNSUPPORTED"


def test_apc_identity_rank_deficiency_remains_with_multiple_cohorts():
    r = apc_diagnostic([0, 24, 48], [0, 12, 24, 36])
    assert r["rank"] == 3 and r["columns"] == 4 and r["maximum_identity_residual"] == 0
    # Same age under several periods; same period under several ages.
    cells = {(c + a, a, c) for c in [0, 24, 48] for a in [0, 12, 24, 36]}
    assert len({p for p, a, c in cells if a == 24}) == 3
    assert len({a for p, a, c in cells if p == 48}) == 2


@pytest.mark.parametrize(
    "field", ["source_sha256", "parser_version", "protocol_sha256", "sample_sha256", "code_sha256"]
)
def test_checkpoint_requires_all_frozen_identities(field):
    identity = dict(
        source_sha256="a",
        parser_version="v",
        protocol_sha256="p",
        sample_sha256="s",
        code_sha256={},
    )
    assert checkpoint_valid(identity, identity)
    changed = {**identity, field: "changed"}
    with pytest.raises(GateStop):
        checkpoint_valid(changed, identity)


def test_2010_protocol_and_prior_evidence_are_preserved():
    assert (
        PLAN["existing_2010_sample_sha256"]
        == "e719a6b4d23caca8ac55eeaa6fae54da23c95887dd83b3ac22cd925d63902832"
    )
    assert PLAN["vintages"] == [2006, 2008, 2010, 2014, 2018, 2020, 2022]
    assert PLAN["n_per_vintage"] == 20000 and not PLAN["models_permitted"]


def test_actual_sampling_and_quarter_checkpoint_pipeline(tmp_path):
    from credit_risk.track_b.multivintage.study import retain, sample

    path = bundle(tmp_path / "annual.zip")
    output = tmp_path / "output"
    output.mkdir()
    plan = {**PLAN, "n_per_vintage": 2}
    identity = dict(
        source_sha256=digest(path), parser_version="toy", protocol_sha256="toy", code_sha256={}
    )
    with VintageReader(path, tmp_path, plan["limits"], digest(path), 2006) as reader:
        ids, manifest = sample(tmp_path, reader, 2006, output, plan, identity)
        assert ids == rank_ids([f"F06Q{q}0000001" for q in range(1, 5)], 2006, 2)
        assert manifest["frozen_before_performance_access"]
        cache, counts = retain(reader, output, ids, manifest, plan)
        assert sum(c["retained"] for c in counts) == 2
        _, reused = retain(reader, output, ids, manifest, plan)
        assert reused == counts
        import sqlite3

        with sqlite3.connect(cache) as db:
            db.execute("UPDATE performance SET raw='corrupted' WHERE rowid=1")
        with pytest.raises(GateStop, match="content mismatch"):
            retain(reader, output, ids, manifest, plan)


def test_unrecognized_dot_rate_stops_without_silent_missing_conversion():
    tokens = orig()
    tokens[ORIGINATION.index("orig_interest_rate")] = "."
    with pytest.raises(ValueError, match="invalid_numeric"):
        parse(tokens, "origination", 2006)


def test_cross_vintage_identifier_stops_without_reassignment():
    with pytest.raises(ValueError, match="missing_or_wrong_vintage_loan_id"):
        parse(orig("F09Q10000001"), "origination", 2008)


def test_failed_vintage_blocks_seven_cohort_readiness():
    from credit_risk.track_b.multivintage.reporting import decision

    results = [dict(vintage=y, status="READY_WITH_LIMITATIONS") for y in PLAN["vintages"]]
    assert decision(results) == "MULTI-VINTAGE COHORT READY WITH MATERIAL LIMITATIONS"
    results[0]["status"] = "DATA_QUALITY_STOP"
    assert decision(results) == "STOP — HARMONIZATION INVALID"
    assert decision(results[2:]) == "MULTI-VINTAGE SUPPORT INSUFFICIENT"


def test_incomplete_mapping_is_partial_not_source_unavailability():
    from credit_risk.track_b.multivintage.reporting import field_status, schema_registry

    result = dict(vintage=2006, status="DATA_QUALITY_STOP")
    assert field_status("orig_interest_rate", result)[0] == "PARTIAL"
    registry, matrix = schema_registry([result])
    assert len(registry["fields"]) == len(ORIGINATION) + len(PERFORMANCE)
    assert all(r["available_from"] is None for r in registry["fields"])
    assert all(f["vintages"]["2006"]["status"] == "PARTIAL" for f in matrix["fields"])


def test_missing_rates_do_not_double_count_special_tokens():
    from credit_risk.track_b.multivintage.reporting import missing_rates

    result = dict(
        facilities=10,
        field_missingness={
            "orig_ltv": dict(
                row_states={"OBSERVED": 8, "MISSING_IN_SOURCE": 2, "special_value:999": 2},
                facilities_with_any_nonobserved=2,
                facilities_with_all_nonobserved=2,
                structurally_unavailable=False,
            )
        },
    )
    rates = missing_rates(result)["orig_ltv"]
    assert rates["row_count"] == 10
    assert rates["row_missing_in_source_rate"] == 0.2
    assert not rates["structurally_unavailable"]


def test_supplemental_flags_preserve_canonical_records(tmp_path):
    import csv

    from credit_risk.track_b.multivintage.reporting import supplemental

    folder = tmp_path / "data/track_b/multivintage/processed/v1/2006"
    folder.mkdir(parents=True)
    (folder / "origination.csv").write_text(
        "loan_id,orig_upb,first_payment_month,maturity_month\ntoy,100000,2006-02,2006-12\n",
        encoding="utf-8",
    )
    fields = [
        "loan_id",
        "reporting_month",
        "current_principal_balance",
        "non_interest_upb",
        "analytical_prefix",
        "months_since_first_payment_proxy",
        "loan_age",
    ]
    with (folder / "monthly.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerow(["toy", "2006-01", 100001, 100002, True, -1, 0])
        writer.writerow(["toy", "2007-01", 90000, 0, True, 11, 24])
    before = digest(folder / "monthly.csv")
    report = supplemental(tmp_path, {"vintage": 2006})
    assert report["balance_flags"]["current_upb_above_original_rows"] == 1
    assert report["balance_flags"]["reporting_before_first_payment_rows"] == 1
    assert report["balance_flags"]["reporting_after_scheduled_maturity_rows"] == 1
    assert report["exact_provider_age24_analytical_rows_by_year"] == {"2007": 1}
    assert digest(folder / "monthly.csv") == before


def test_reporting_command_returns_failure_for_closed_gate(tmp_path, monkeypatch):
    import runpy
    import sys

    from credit_risk.track_b.multivintage import reporting

    report = tmp_path / "reports/track_b/MULTI_VINTAGE_DATA_AUDIT.md"
    report.parent.mkdir(parents=True)
    report.write_text("Audit\n", encoding="utf-8")
    monkeypatch.setattr(
        reporting, "build", lambda root: {"decision": "STOP — HARMONIZATION INVALID"}
    )
    monkeypatch.setattr(reporting, "verify_preservation", lambda root, source: {"status": "PASSED"})
    monkeypatch.setattr(reporting, "figure", lambda root, evidence: None)
    monkeypatch.setattr(
        sys, "argv", ["report", "--root", str(tmp_path), "--source-dir", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(ROOT / "scripts/report_track_b_multivintage.py"), run_name="__main__")
    assert stop.value.code == 2
    result = json.loads(
        (report.parent / "multi_vintage_data_audit.json").read_text(encoding="utf-8")
    )
    assert result["decision"] == "STOP — HARMONIZATION INVALID"


@pytest.mark.parametrize("vintage", [2020, 2022])
def test_origination_quarter_conflict_stops_before_sample_freeze(tmp_path, vintage):
    from credit_risk.track_b.multivintage.study import sample

    source = bundle(tmp_path / "annual.zip", vintage=vintage, defect="quarter_id")
    output = tmp_path / "output"
    output.mkdir()
    identity = dict(
        source_sha256=digest(source), parser_version="toy", protocol_sha256="toy", code_sha256={}
    )
    with VintageReader(source, tmp_path, PLAN["limits"], digest(source), vintage) as reader:
        with pytest.raises(GateStop, match="Origination quarter mismatch"):
            sample(tmp_path, reader, vintage, output, PLAN, identity)
        assert reader.sample is None
        assert not (output / "sample_manifest.json").exists()
        assert not (output / "performance.sqlite").exists()


@pytest.mark.parametrize("kind", ["complete", "failed", "incomplete"])
def test_ingestion_command_cannot_report_false_success(tmp_path, monkeypatch, kind):
    import runpy
    import sys

    from credit_risk.track_b.multivintage import study

    results = [dict(status="READY_WITH_LIMITATIONS") for _ in range(7)]
    if kind == "failed":
        results[0]["status"] = "DATA_QUALITY_STOP"
    elif kind == "incomplete":
        results.pop()
    monkeypatch.setattr(study, "run", lambda root, source: results)
    monkeypatch.setattr(sys, "argv", ["ingest", "--source-dir", str(tmp_path)])
    script = ROOT / "scripts/run_track_b_multivintage.py"
    if kind == "complete":
        runpy.run_path(str(script), run_name="__main__")
    else:
        with pytest.raises(SystemExit) as stop:
            runpy.run_path(str(script), run_name="__main__")
        assert stop.value.code == 2


def test_combined_manifest_cannot_be_silently_replaced(tmp_path):
    from credit_risk.track_b.multivintage.reporting import freeze_manifest

    path = tmp_path / "combined.json"
    freeze_manifest(path, {"sample": "a"})
    before = digest(path)
    freeze_manifest(path, {"sample": "a"})
    with pytest.raises(ValueError, match="identity changed"):
        freeze_manifest(path, {"sample": "b"})
    assert digest(path) == before
