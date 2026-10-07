"""Origination-only evidence tests; no empirical missing-value policy is inferred."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from credit_risk.track_b.data.schemas import ORIGINATION, digest
from credit_risk.track_b.multivintage.schema import parse
from credit_risk.track_b.reconciliation.evidence import context, embedded, inspect, rate_class

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "docs/track_b/multi_vintage_protocol.json").read_text(encoding="utf-8"))


def record(loan="F08Q40000001", rate="5.125"):
    values = dict(
        loan_id=loan,
        orig_interest_rate=rate,
        first_payment_month="200903",
        maturity_month="203902",
        original_loan_term="360",
        amortization_type="FRM",
        orig_ltv="80",
        channel="R",
    )
    return [values.get(k, "") for k in ORIGINATION]


@pytest.mark.parametrize("loan", ["", "F08Q50000001", "F08Q4000000", "bad"])
def test_missing_or_malformed_identifier_is_not_a_boundary(loan):
    assert embedded(loan) is None
    with pytest.raises(ValueError, match="loan_id"):
        parse(record(loan), "origination", 2008)


def test_wrong_product_preserved_by_survey_but_rejected_by_standard_parser():
    assert embedded("A08Q40000001")["product"] == "A"
    with pytest.raises(ValueError, match="loan_id"):
        parse(record("A08Q40000001"), "origination", 2008)


def test_equal_quarter_has_no_boundary_flag_and_does_not_rewrite_identifier():
    loan = "F08Q40000001"
    evidence = context(record(loan), embedded(loan), 2008, 4)
    assert evidence["embedded_quarter"] == "2008Q4"
    assert evidence["quarter_delta"] == 0
    assert evidence["adjacent_boundary"] is False
    assert parse(record(loan), "origination", 2008)["loan_id"] == loan


def test_cross_year_boundary_is_evidence_not_documented_permission():
    loan = "F09Q10000001"
    evidence = context(record(loan), embedded(loan), 2008, 4)
    assert evidence["embedded_quarter"] == "2009Q1"
    assert evidence["quarter_delta"] == 1
    assert evidence["adjacent_boundary"] is True
    assert evidence["scheduled_term_coherent"] is True
    assert evidence["first_payment_minus_embedded_quarter_start"] == 2
    with pytest.raises(ValueError, match="loan_id"):
        parse(record(loan), "origination", 2008)


def test_same_year_previous_quarter_audited_separately_from_valid_syntax():
    loan = "F08Q30000001"
    evidence = context(record(loan), embedded(loan), 2008, 4)
    assert evidence["quarter_delta"] == -1 and evidence["adjacent_boundary"] is True
    # The existing study's enclosing-quarter gate still applies after this parser.
    assert parse(record(loan), "origination", 2008)["loan_id"] == loan


@pytest.mark.parametrize(
    "token,classification,value",
    [
        ("5.125", "numeric", 5.125),
        ("", "blank", None),
        (".", "dot", None),
        ("missing", "other_nonnumeric", None),
        ("nan", "other_nonnumeric", None),
        ("inf", "other_nonnumeric", None),
    ],
)
def test_rate_census_categories_do_not_coerce_source_tokens(token, classification, value):
    assert rate_class(token) == (classification, value)


def test_blank_retains_existing_null_behavior_without_inventing_dot_reason():
    assert parse(record(rate=""), "origination", 2008)["orig_interest_rate"] is None
    for token in [".", "missing"]:
        with pytest.raises(ValueError, match="invalid_numeric"):
            parse(record(rate=token), "origination", 2008)
    # No empirical null reason is introduced while the convention is unresolved.
    assert "orig_interest_rate_null_reason" not in parse(record(), "origination", 2008)


def test_column_count_and_delimiter_shift_fail_closed():
    with pytest.raises(ValueError, match="wrong_field_count"):
        parse(record()[:-1], "origination", 2008)
    shifted = record()
    shifted[12], shifted[13] = shifted[13], shifted[12]
    with pytest.raises(ValueError, match="invalid_numeric"):
        parse(shifted, "origination", 2008)


def test_payment_schedule_cannot_establish_exact_origination_date():
    evidence = context(record(), embedded("F08Q40000001"), 2008, 4)
    assert "origination_date" not in evidence
    shifted = record()
    shifted[21] = "359"
    assert context(shifted, embedded("F08Q40000001"), 2008, 4)["scheduled_term_coherent"] is False


def test_census_reads_origination_only_and_counts_duplicates_without_sampling(
    tmp_path, monkeypatch
):
    private = tmp_path / "data/track_b/multivintage/reconciliation_v1"
    private.mkdir(parents=True)
    source = tmp_path / "annual.zip"
    with zipfile.ZipFile(source, "w") as annual:
        for q in range(1, 5):
            inner_bytes = io.BytesIO()
            with zipfile.ZipFile(inner_bytes, "w", compression=zipfile.ZIP_DEFLATED) as inner:
                loan = f"F08Q{q}0000001"
                rows = [record(loan)]
                if q == 4:
                    rows += [record("F09Q10000001", "."), record(loan)]
                inner.writestr(f"orig_2008Q{q}.txt", "\n".join("|".join(r) for r in rows) + "\n")
                inner.writestr(f"perf_2008Q{q}.txt", "FORBIDDEN PERFORMANCE PAYLOAD")
            annual.writestr(f"historical_data_2008Q{q}.zip", inner_bytes.getvalue())
    from credit_risk.track_b.multivintage.reader import VintageReader

    real_lines = VintageReader.lines

    def only_origination(self, quarter, kind):
        assert kind == "origination"
        return real_lines(self, quarter, kind)

    monkeypatch.setattr(VintageReader, "lines", only_origination)
    preflight = {"sha256": digest(source)}
    result = inspect(source, tmp_path, 2008, PLAN, preflight)
    assert result["unique_identifiers"] == 5 and result["duplicate_identifiers"] == 1
    assert result["members"][3]["counts"]["prefix_mismatch"] == 1
    assert result["members"][3]["rate_tokens"]["dot"] == 1
    assert result["dot_contexts"][0]["columns"] == 31
    assert result["dot_contexts"][0]["delimiter_count"] == 30
    assert result["performance_accessed"] is False and result["samples_frozen"] is False
    assert inspect(source, tmp_path, 2008, PLAN, preflight) == result
    (private / "identifiers_2008.sqlite").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identifier evidence changed"):
        inspect(source, tmp_path, 2008, PLAN, preflight)


def test_partial_census_requires_explicit_recovery(tmp_path):
    private = tmp_path / "data/track_b/multivintage/reconciliation_v1"
    private.mkdir(parents=True)
    (private / "identifiers_2008.sqlite").touch()
    with pytest.raises(ValueError, match="Partial census"):
        inspect(tmp_path / "absent.zip", tmp_path, 2008, PLAN, {})


def test_original_task8_reports_protocol_and_parser_remain_frozen():
    review = json.loads(
        (ROOT / "reports/track_b/source_convention_reconciliation.json").read_text(encoding="utf-8")
    )
    assert review["decision"] == "SOURCE CONVENTIONS UNRESOLVED"
    assert review["amendment"] is None and review["parser_changed"] is False
    assert review["new_samples_frozen"] is False and review["new_performance_accessed"] is False
    for name, expected in review["original_task8_evidence_sha256_lf"].items():
        payload = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == expected, name


def test_completed_samples_and_panels_preserved_when_private_checkpoints_available():
    expected = {
        "2010": "e719a6b4d23caca8ac55eeaa6fae54da23c95887dd83b3ac22cd925d63902832",
        "2014": "d26b2ba3e08837f82c940ecefbaa18c4bd201d64a0a114760eafeeb80a8126f5",
        "2018": "b3e2952bb5bc0d0a551105cecfa964b18481b3764442edb8362245f4f17f921e",
    }
    verification = json.loads(
        (ROOT / "reports/track_b/source_convention_preservation.json").read_text(encoding="utf-8")
    )
    assert verification["status"] == "passed"
    assert {y: c["sample_sha256"] for y, c in verification["completed"].items()} == expected
    baseline = ROOT / "data/track_b/multivintage/reconciliation_v1/baseline.json"
    # Hosted CI intentionally has no licensed data; local verification hashes bytes,
    # never parses or evaluates the protected histories.
    if baseline.exists():
        saved = json.loads(baseline.read_text(encoding="utf-8"))["completed"]
        for year, fields in saved.items():
            directory = ROOT / f"data/track_b/multivintage/processed/v1/{year}"
            for name, expected_hash in fields.items():
                if name == "selected_ids_byte_sha256":
                    path = (
                        ROOT / "data/track_b/manifests/expansion_v1/selected_ids.txt"
                        if year == "2010"
                        else directory / "selected_ids.txt"
                    )
                else:
                    path = directory / name
                assert digest(path) == expected_hash, str(path)
