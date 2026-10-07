"""Exact authorized-record eligibility and unchanged selected-history semantics."""

import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from credit_risk.track_b.data.annual import GateStop
from credit_risk.track_b.data.schemas import ORIGINATION, PERFORMANCE, digest
from credit_risk.track_b.multivintage.core import rank_key, set_hash, trajectory
from credit_risk.track_b.multivintage.schema import parse
from credit_risk.track_b.recovery.eligibility import StructuralPerformanceReader, classify
from credit_risk.track_b.recovery.study import sample

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads(
    (ROOT / "docs/track_b/multi_vintage_recovery_protocol.json").read_text(encoding="utf-8")
)


def orig(loan="F06Q10000001", rate="5"):
    values = dict(
        loan_id=loan,
        orig_interest_rate=rate,
        first_payment_month="200602",
        maturity_month="203601",
        original_loan_term="360",
        amortization_type="FRM",
    )
    return "|".join(values.get(k, "") for k in ORIGINATION)


def approved(line, vintage=2006, quarter=1, number=1):
    return dict(
        vintage=vintage,
        member=f"{vintage}Q{quarter}",
        line=number,
        loan_id=line.split("|")[19],
        raw=line,
        raw_record_sha256=hashlib.sha256(line.encode()).hexdigest(),
        reason_code="UNRESOLVED_REQUIRED_FIELD_FORMAT",
        key="TOY-A1",
    )


def test_exact_quarantine_preserves_raw_and_excludes_before_source_coercion():
    line = orig(rate=".")
    decision = classify(line, 2006, 1, 1, [approved(line)])
    assert decision["eligible"] is False and decision["raw"] == line
    assert decision["reason"] == "UNRESOLVED_REQUIRED_FIELD_FORMAT"
    with pytest.raises(ValueError, match="invalid_numeric"):
        parse(line.split("|"), "origination", 2006)


@pytest.mark.parametrize("change", ["location", "loan", "content", "columns"])
def test_exact_authorization_does_not_generalize(change):
    line = orig(rate=".")
    registry = [approved(line)]
    number = 1
    if change == "location":
        number = 2
    elif change == "loan":
        line = line.replace("F06Q10000001", "F06Q10000002")
    elif change == "content":
        line = line.replace("200602", "200603")
    else:
        line += "|"
    with pytest.raises(GateStop):
        classify(line, 2006, 1, number, registry)


def test_unexpected_fifth_anomaly_and_unapproved_quarter_fail_closed():
    for line in [orig(rate="."), orig(loan="F06Q20000001"), orig(loan="F07Q10000001")]:
        with pytest.raises(GateStop, match="Unapproved"):
            classify(line, 2006, 1, 12, [])


def test_eligibility_and_rank_use_no_outcome_argument():
    line = orig()
    assert classify(line, 2006, 1, 1, [])["eligible"] is True
    assert (
        rank_key("F06Q10000001", 2006)
        == hashlib.sha256(b"track-b-multivintage-v1:2006:F06Q10000001").hexdigest()
    )
    with pytest.raises(TypeError):
        classify(line, 2006, 1, 1, [], outcome=True)


def perf(**changes):
    values = dict(
        loan_id="F06Q10000001",
        reporting_month="200602",
        delinquency_state="00",
        current_principal_balance="100000",
        loan_age="1",
        current_interest_rate="5",
        remaining_legal_months="359",
        termination_code="",
    )
    values.update(changes)
    return [values.get(k, "") for k in PERFORMANCE]


@pytest.mark.parametrize(
    "changes",
    [
        {"reporting_month": "200613"},
        {"delinquency_state": "ZZ"},
        {"current_interest_rate": "."},
        {"loan_id": "F09Q10000001"},
        {"current_principal_balance": "-1"},
        {"remaining_legal_months": "bad"},
        {"termination_code": "99"},
    ],
)
def test_new_performance_conventions_fail_unchanged_source_parser(changes):
    with pytest.raises(ValueError):
        parse(perf(**changes), "performance", 2006)


def test_expected_performance_and_source_wide_column_gate():
    assert parse(perf(), "performance", 2006)["current_interest_rate"] == 5

    class Reader:
        vintage = 2006

        def lines(self, quarter, kind):
            yield "|".join(perf())
            yield "|".join(perf()) + "|unexpected"

    lines = StructuralPerformanceReader(Reader()).lines(1, "performance")
    assert next(lines).count("|") == 34
    with pytest.raises(GateStop, match="field count"):
        next(lines)


def test_endpoint_ordering_and_post_endpoint_rows_follow_frozen_policy():
    event = json.loads(
        (ROOT / "docs/track_b/mortgage_research_protocol.json").read_text(encoding="utf-8")
    )["event"]
    rows = [
        parse(perf(reporting_month=m, delinquency_state=s), "performance", 2006)
        for m, s in [("200604", "00"), ("200602", "00"), ("200603", "03")]
    ]
    result = trajectory(rows, event, PLAN["horizons"], PLAN["horizon_support_rule"])
    assert result["endpoint"] == "default"
    assert result["annotations"][-1]["post_research_endpoint"] is True
    assert result["annotations"][-1]["analytical_prefix"] is False


@pytest.mark.parametrize("inside", [False, True])
def test_complete_gate_and_zero_impact_required_before_sample_freeze(tmp_path, inside):
    plan = copy.deepcopy(PLAN)
    plan["quarantine"] = [dict(vintage=2006, audit_reference="TOY-A1", eligibility=False)]
    identifiers = [f"F06Q1{i:07}" for i in range(20005)]
    ids_order = sorted(identifiers, key=lambda i: (rank_key(i, 2006), i))
    anomaly = ids_order[0] if inside else ids_order[-1]
    lines = [orig(anomaly, ".")] + [orig(i) for i in identifiers if i != anomaly]
    census_path = tmp_path / "data/track_b/multivintage/reconciliation_v1/identifiers_2006.sqlite"
    census_path.parent.mkdir(parents=True)
    with sqlite3.connect(census_path) as db:
        db.execute("CREATE TABLE identifiers(loan TEXT PRIMARY KEY)")
        db.executemany("INSERT INTO identifiers VALUES (?)", [(i,) for i in identifiers])
    census = dict(
        unique_identifiers=len(identifiers), identifier_database_sha256=digest(census_path)
    )
    output = tmp_path / "output"
    output.mkdir()

    class Reader:
        vintage = 2006
        checked_quarters = []
        sample = None

        def lines(self, quarter, kind):
            assert kind == "origination"
            self.checked_quarters.append(quarter)
            return iter(lines if quarter == 1 else [])

        def freeze(self, ids):
            assert self.checked_quarters == [1, 2, 3, 4]
            assert (output / "sample_manifest.json").exists()
            self.sample = ids

    reader = Reader()
    registry = [approved(lines[0])]
    identity = dict(
        source_sha256="toy", parser_version="toy", protocol_sha256="toy", code_sha256={}
    )
    if inside:
        with pytest.raises(GateStop, match="sample impact differs"):
            sample(tmp_path, reader, output, plan, identity, registry, census)
        assert not (output / "sample_manifest.json").exists()
        assert not (output / "selected_ids.txt").exists()
        assert reader.sample is None
        return
    ids, manifest = sample(tmp_path, reader, output, plan, identity, registry, census)
    assert anomaly not in ids and len(ids) == 20000
    assert manifest["universe"] == len(identifiers) - 1
    assert manifest["sample_symmetric_difference"] == 0
    assert manifest["sample_sha256"] == set_hash(ids_order[:20000])
    assert manifest["frozen_before_performance_access"] is True
    assert (output / "quarantine.json").exists()
    assert registry[0]["raw"] == lines[0]


def test_partial_recovery_refuses_silent_reset(tmp_path):
    (tmp_path / "origination.sqlite").touch()
    with pytest.raises(GateStop, match="explicit review"):
        sample(tmp_path, None, tmp_path, {}, {}, [], {})


def test_prior_audits_and_source_parser_are_unchanged():
    amendment = json.loads(
        (ROOT / "docs/track_b/historical_source_anomaly_amendment.json").read_text(encoding="utf-8")
    )
    assert amendment["parser_changed"] is False
    for name, expected in amendment["prior_evidence_sha256_lf"].items():
        assert (
            hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            == expected
        )


def test_report_distinguishes_active_pandemic_support_from_endpoint_boundary(tmp_path):
    import csv

    from credit_risk.track_b.recovery.reporting import canonical_support

    (tmp_path / "origination.csv").write_text("loan_id,orig_upb\ntoy,100000\n", encoding="utf-8")
    fields = [
        "loan_id",
        "current_principal_balance",
        "non_interest_upb",
        "analytical_prefix",
        "reporting_month",
        "months_since_first_payment_proxy",
        "loan_age",
        "event_category",
    ]
    with (tmp_path / "monthly.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for month, event, prefix, age in [
            ("2020-12", "none", "True", 24),
            ("2021-01", "default", "True", 25),
            ("2021-02", "none", "False", 26),
        ]:
            writer.writerow(
                dict(
                    loan_id="toy",
                    current_principal_balance="90000",
                    non_interest_upb="",
                    analytical_prefix=prefix,
                    reporting_month=month,
                    months_since_first_payment_proxy=str(age + 120),
                    loan_age=str(age),
                    event_category=event,
                )
            )
    support = canonical_support(tmp_path)
    assert support["pandemic_active_facilities"] == {"2020": 1, "2021": 0}
    assert support["active_rows"] == {"2020": 1}
    assert len(support["clocks"]) == 2  # inherited prefix includes the endpoint boundary
    assert support["exact24"] == {"2020": 1}
    assert support["first_payment_proxy_age_calendar_support"] == [
        dict(calendar_year="2020", proxy_age_band="<= 180", rows=1),
        dict(calendar_year="2021", proxy_age_band="<= 180", rows=1),
    ]  # provider age 24 does not imply elapsed first-payment age 24
