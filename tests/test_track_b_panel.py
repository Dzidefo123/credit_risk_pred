"""Synthetic-fixture engineering regressions; never empirical Freddie performance."""

import io
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from credit_risk.track_b.data.freddie import ParseStats, parse_row, records
from credit_risk.track_b.data.panel import build_panel, outcome
from credit_risk.track_b.data.sampling import select_loans
from credit_risk.track_b.data.schemas import (
    FEATURES,
    LAYOUT,
    LOSS_FIELDS,
    ORIGINATION,
    PERFORMANCE,
    digest,
    feature_frame,
    load_protocol,
)
from credit_risk.track_b.data.workflow import blocked_audit, build

ROOT = Path(__file__).resolve().parents[1]
ID = "F10Q10000001"  # Invented syntactic fixture identifier; no real source record.


@pytest.fixture
def governing_root(tmp_path):
    for name in [
        "mortgage_research_protocol.json",
        "field_dictionary_comparison.json",
        "LONGITUDINAL_DATA_CONTRACT.md",
    ]:
        p = tmp_path / "docs/track_b" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "docs/track_b" / name, p)
    return tmp_path


@pytest.fixture
def policy():
    return load_protocol(ROOT)[0]


def origin(loan=ID):
    values = dict.fromkeys(ORIGINATION, "")
    values.update(
        loan_id=loan,
        first_payment_month="201001",
        maturity_month="204001",
        amortization_type="FRM",
        orig_credit_score="700",
        orig_upb="100000",
    )
    return [values[k] for k in ORIGINATION]


def performance(offset=0, state="00", code="", loan=ID, **overrides):
    period = pd.Period("2010-01", freq="M") + offset
    values = dict.fromkeys(PERFORMANCE, "")
    values.update(
        loan_id=loan,
        reporting_month=period.strftime("%Y%m"),
        current_principal_balance="99000",
        delinquency_state=state,
        termination_code=code,
        termination_month=period.strftime("%Y%m") if code else "",
    )
    values.update(overrides)
    return [values[k] for k in PERFORMANCE]


def row(offset=0, **kwargs):
    return parse_row(performance(offset, **kwargs), "performance")


def history(length=20):
    return {r["reporting_month"]: r for r in [row(i) for i in range(length)]}


def archive(root, rows=None, orig=None):
    p = root / "sample_2010.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("sample_orig_2010.txt", "|".join(origin() if orig is None else orig) + "\n")
        z.writestr(
            "sample_perf_2010.txt",
            "\n".join("|".join(r) for r in (rows or [performance(i) for i in range(20)])) + "\n",
        )
    auth = {
        "source_kind": "synthetic_fixture",
        "source_organization": "synthetic fixture",
        "vintage": 2010,
        "source_release": "47",
        "layout": LAYOUT,
        "source_sha256": digest(p),
        "acquired_at": "2026-01-01T00:00:00+00:00",
        "official_source_reference": "https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset",
        "terms_accepted": True,
        "license_redistribution_status": "prohibited",
        "aggregate_publication_permitted": False,
    }
    a = root / "fixture_authorization.json"
    a.write_text(json.dumps(auth), encoding="utf-8")
    return p, a


def test_layout_counts_source_types_masks_and_signed_loss():
    assert len(ORIGINATION) == 31 and len(PERFORMANCE) == 35
    data = origin()
    data[0] = "9999"
    assert parse_row(data, "origination")["orig_credit_score"] is None
    value = row(mi_recoveries="-100", net_sale_proceeds="U", total_expenses="50")
    assert value["mi_recoveries"] == -100 and value["net_sale_proceeds"] is None
    assert value["net_sale_proceeds_disclosure_code"] == "U"
    assert value["total_expenses"] == 50


@pytest.mark.parametrize("change", ["count", "id", "date", "numeric", "state", "terminal"])
def test_malformed_rows_are_counted_without_guessed_alignment(change):
    values = performance()
    if change == "count":
        values.pop()
    elif change == "id":
        values[0] = ""
    elif change == "date":
        values[1] = "201013"
    elif change == "numeric":
        values[2] = "NaN"
    elif change == "state":
        values[3] = "garbage"
    elif change == "terminal":
        values[8] = "77"
    stats = ParseStats()
    assert list(records(io.StringIO("|".join(values)), "performance", stats)) == []
    assert (
        stats.source_rows == stats.malformed_rows == 1 and stats.valid_rows == 0 and stats.rejects
    )


def test_unsupported_layout_stops_before_parsing():
    with pytest.raises(ValueError, match="Unsupported"):
        list(records(io.StringIO("anything"), "performance", ParseStats(), layout="old-unverified"))


def test_id_selection_is_bounded_order_invariant_and_outcome_independent():
    ids = [f"F10Q1{i:07d}" for i in range(1200)]
    assert select_loans(ids) == select_loans(reversed(ids)) and len(select_loans(ids)) == 1000
    assert select_loans(ids + ids) == select_loans(ids)
    with pytest.raises(ValueError):
        select_loans(ids, 1001)
    with pytest.raises(ValueError):
        select_loans(ids, 0)


@pytest.mark.parametrize(
    "offset,expected",
    [
        (0, "not_incident_risk_eligible"),
        (12, "positive_default"),
        (13, "negative_survived_horizon"),
    ],
)
def test_twelve_month_boundary_and_t0_default(policy, offset, expected):
    h = history(20)
    t0 = pd.Period("2010-06", freq="M")
    h[t0 + offset] = row(5 + offset, state="03")
    if offset == 0:
        frame, _ = build_panel({ID: parse_row(origin(), "origination")}, list(h.values()), policy)
        got = frame.loc[frame.t0.eq(str(t0)), "outcome_status"].iloc[0]
    else:
        got = outcome(t0, h, policy["event"])[0]
    assert got == expected


@pytest.mark.parametrize(
    "state,code,expected",
    [
        ("00", "01", "competing_payoff"),
        ("03", "01", "ambiguous_event_order"),
        ("00", "15", "right_censored"),
        ("00", "16", "right_censored"),
        ("00", "96", "right_censored"),
        ("00", "02", "positive_default"),
        ("RA", "", "positive_default"),
        ("XX", "", "insufficient_followup"),
    ],
)
def test_source_events_are_distinct_and_not_silently_negative(policy, state, code, expected):
    h = history(20)
    t0 = pd.Period("2010-06", freq="M")
    h[t0 + 1] = row(6, state=state, code=code)
    result = outcome(t0, h, policy["event"])
    assert result[0] == expected
    if expected in {"right_censored", "ambiguous_event_order", "insufficient_followup"}:
        assert result[1] is None


def test_payoff_before_later_default_does_not_become_default(policy):
    h = history(20)
    t0 = pd.Period("2010-06", freq="M")
    h[t0 + 2] = row(7, code="01")
    h[t0 + 4] = row(9, state="03")
    assert outcome(t0, h, policy["event"])[:3] == ("competing_payoff", 0, 2)


def test_gap_cutoff_unknown_and_date_ambiguity(policy):
    t0 = pd.Period("2010-06", freq="M")
    h = history(9)
    assert outcome(t0, h, policy["event"])[0] == "right_censored"
    h = history(20)
    del h[t0 + 3]
    assert outcome(t0, h, policy["event"])[3] == 2
    h = history(20)
    h[t0 + 2] = row(7, code="03", termination_month="201009")
    assert outcome(t0, h, policy["event"])[0] == "ambiguous_event_order"


def test_panel_keeps_censoring_ineligible_and_future_does_not_change_features(policy):
    orig = {ID: parse_row(origin(), "origination")}
    base = [row(i) for i in range(20)]
    a, _ = build_panel(orig, base, policy)
    future = [
        row(i, state="03" if i == 10 else "00", mi_recoveries="-999" if i > 5 else "")
        for i in range(20)
    ]
    b, _ = build_panel(orig, future, policy)
    assert_frame_equal(
        feature_frame(a[a.t0.eq("2010-06")]).reset_index(drop=True),
        feature_frame(b[b.t0.eq("2010-06")]).reset_index(drop=True),
    )
    assert len(a) == 20 and "right_censored" in set(a.outcome_status)
    assert "not_incident_risk_eligible" in set(a.outcome_status)
    assert a.loc[a.outcome_status.eq("right_censored"), "binary_default_12m"].isna().all()


def test_sort_duplicates_gaps_and_terminal_integrity(policy):
    orig = {ID: parse_row(origin(), "origination")}
    source = [row(i) for i in range(20) if i != 9]
    source.append(row(6, state="01"))
    source[7] = row(7, code="01")
    frame, integrity = build_panel(orig, list(reversed(source)), policy)
    assert list(frame.t0) == sorted(frame.t0)
    assert integrity["findings"]["gap_intervals"] == 1
    assert integrity["findings"]["conflicting_duplicate_months"] == 1
    assert integrity["findings"]["unexpected_post_terminal_rows"] > 0
    assert not frame.loc[frame.t0.eq("2010-07"), "eligible"].iloc[0]


@pytest.mark.parametrize(
    "column",
    [
        "loan_id",
        "t0",
        "binary_default_12m",
        "outcome_status",
        *LOSS_FIELDS,
        "future_macro",
        "future_balance",
    ],
)
def test_feature_firewall_rejects_audit_outcome_recovery_and_future_fields(policy, column):
    frame, _ = build_panel(
        {ID: parse_row(origin(), "origination")}, [row(i) for i in range(20)], policy
    )
    frame[column] = None if column not in frame else frame[column]
    with pytest.raises(ValueError, match="firewall"):
        feature_frame(frame, [column])
    assert list(feature_frame(frame).columns) == list(FEATURES)


def test_fixture_workflow_manifests_hashes_private_outputs_and_no_models(
    governing_root, monkeypatch
):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    def forbidden(*a, **k):
        pytest.fail("No model loading or fitting in Task 2")

    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(XGBClassifier, "fit", forbidden)
    source, auth = archive(governing_root)
    result, manifest = build(governing_root, source, auth)
    assert result["status"] == "FIXTURE_ONLY" and result["cohort"]["selected_loans"] == 1
    assert manifest["output_sha256"] == digest(governing_root / "data/track_b/processed/panel.csv")
    assert result["provenance"]["source_sha256"] == digest(source)
    assert len(result["provenance"]["source_files"]) == 2
    assert result["cohort"]["followup"]["denominator_eligible_landmarks"] == 15
    assert not (governing_root / "reports/track_b").exists()
    with pytest.raises(ValueError, match="overwrite"):
        build(governing_root, source, auth)


def test_source_authorization_and_publication_gate(governing_root):
    source, auth = archive(governing_root)
    with pytest.raises(ValueError, match="Public aggregates"):
        build(governing_root, source, auth, publish_aggregates=True)
    data = json.loads(auth.read_text())
    data["source_sha256"] = "0" * 64
    auth.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        build(governing_root, source, auth)


def test_parser_rejects_block_workflow_without_writing_panel(governing_root):
    values = performance()
    values.pop()
    source, auth = archive(governing_root, rows=[values])
    with pytest.raises(ValueError, match="schema/linkage rejects"):
        build(governing_root, source, auth)
    assert not (governing_root / "data/track_b/processed/panel.csv").exists()


def test_protocol_tampering_stops_before_source_access(governing_root):
    p = governing_root / "docs/track_b/mortgage_research_protocol.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="protocol changed"):
        build(governing_root, governing_root / "absent.zip", governing_root / "absent-auth.json")


def test_not_acquired_report_has_no_fabricated_empirical_values(governing_root):
    result = blocked_audit(governing_root)
    assert result["status"] == "NOT_ACQUIRED" and result["cohort"] is None
    assert result["followup"] is None and set(result["empirical_questions"].values()) == {
        "not_assessable"
    }
    assert (
        "empirical feasibility"
        in (governing_root / "reports/track_b/FREDDIE_2010_DATA_AUDIT.md").read_text().lower()
    )


def test_selected_ids_unchanged_by_performance_outcomes(governing_root):
    ids = ["F10Q10000001", "F10Q10000002"]
    selected_hashes = []
    for run_index, severe in enumerate([False, True]):
        root = governing_root / f"run-{run_index}"
        for name in [
            "mortgage_research_protocol.json",
            "field_dictionary_comparison.json",
            "LONGITUDINAL_DATA_CONTRACT.md",
        ]:
            p = root / "docs/track_b" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(governing_root / "docs/track_b" / name, p)
        source, auth = archive(root)
        with zipfile.ZipFile(source, "w") as z:
            z.writestr(
                "sample_orig_2010.txt",
                "\n".join("|".join(origin(loan=i)) for i in reversed(ids)) + "\n",
            )
            z.writestr(
                "sample_perf_2010.txt",
                "\n".join(
                    "|".join(performance(k, loan=i, state="03" if severe and k == 10 else "00"))
                    for i in ids
                    for k in range(20)
                )
                + "\n",
            )
        a = json.loads(auth.read_text())
        a["source_sha256"] = digest(source)
        auth.write_text(json.dumps(a))
        result, _ = build(root, source, auth, n=1)
        selected_hashes.append(result["provenance"]["selected_id_set_sha256"])
    assert selected_hashes[0] == selected_hashes[1]


def test_unknown_outcome_never_controls_landmark_inclusion(policy):
    orig = {ID: parse_row(origin(), "origination")}
    complete, _ = build_panel(orig, [row(i) for i in range(20)], policy)
    short, _ = build_panel(orig, [row(i) for i in range(8)], policy)
    early = complete[complete.t0.isin(short.t0)]
    assert list(early.t0) == list(short.t0)
    assert list(early.eligible) == list(short.eligible)
    assert len(short) == 8
    assert short.loc[short.eligible, "binary_default_12m"].isna().all()


def test_duplicate_placeholders_do_not_choose_a_conflicting_source_row(policy):
    source = [row(i) for i in range(20)] + [row(6, state="03")]
    panel, integrity = build_panel({ID: parse_row(origin(), "origination")}, source, policy)
    duplicate = panel.loc[panel.t0.eq("2010-07")].iloc[0]
    assert duplicate["current_principal_balance"] is None or pd.isna(
        duplicate["current_principal_balance"]
    )
    assert integrity["findings"]["conflicting_duplicate_months"] == 1
    assert not duplicate.eligible


def test_missing_selected_history_is_reported_not_resampled(policy):
    orig = {
        ID: parse_row(origin(), "origination"),
        "F10Q10000002": parse_row(origin("F10Q10000002"), "origination"),
    }
    panel, integrity = build_panel(orig, [row(i) for i in range(20)], policy)
    assert len(orig) == 2 and integrity["findings"]["selected_origination_without_performance"] == 1
    assert panel.loan_id.nunique() == 1


def test_wrong_archive_members_rejected_without_extraction(governing_root):
    source, auth = archive(governing_root)
    with zipfile.ZipFile(source, "w") as z:
        z.writestr("../../sample_orig_2010.txt", "unsafe")
    data = json.loads(auth.read_text())
    data["source_sha256"] = digest(source)
    auth.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="sample pair"):
        build(governing_root, source, auth)
    assert not (governing_root / "sample_orig_2010.txt").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("terms_accepted", False),
        ("vintage", 2011),
        ("source_release", "46"),
        ("layout", "unknown-layout"),
        ("official_source_reference", "https://example.org/file.zip"),
    ],
)
def test_unapproved_or_wrong_source_attestation_fails(governing_root, key, value):
    source, auth = archive(governing_root)
    data = json.loads(auth.read_text())
    data[key] = value
    auth.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        build(governing_root, source, auth)
    assert not (governing_root / "data/track_b/processed/panel.csv").exists()


def test_rejection_carries_structured_counts_without_row_values(governing_root):
    from credit_risk.track_b.data.workflow import IngestionRejected

    bad = performance()
    bad.pop()
    source, auth = archive(governing_root, rows=[bad])
    with pytest.raises(IngestionRejected) as exc:
        build(governing_root, source, auth)
    result = exc.value.audit
    assert result["parse_counts"]["performance"]["malformed_rows"] == 1
    assert result["cohort"] is None and ID not in json.dumps(result)


def test_metadata_cannot_contain_credentials_or_malformed_types(governing_root):
    source, auth = archive(governing_root)
    original = json.loads(auth.read_text())
    for malformed in [
        [original],
        {**original, "cookie": "not-a-real-secret"},
        {**original, "acquired_at": None},
        {**original, "official_source_reference": "https://:synthetic@www.freddiemac.com/path"},
        {
            **original,
            "official_source_reference": "https://www.freddiemac.com/path?token=synthetic",
        },
    ]:
        auth.write_text(json.dumps(malformed))
        with pytest.raises(ValueError):
            build(governing_root, source, auth)


def test_private_data_zones_are_git_ignored():
    import subprocess

    names = [
        "data/track_b/raw/sample_2010.zip",
        "data/track_b/interim/selected_performance.csv",
        "data/track_b/processed/panel.csv",
        "data/track_b/manifests/authorization.json",
    ]
    result = subprocess.run(
        ["git", "check-ignore", *names], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0 and set(result.stdout.splitlines()) == set(names)


def test_cli_no_source_stops_nonzero_with_honest_report(governing_root):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_track_b_panel.py"),
            "--root",
            str(governing_root),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "NOT_ACQUIRED" in result.stdout
    assert not (governing_root / "data/track_b/processed/panel.csv").exists()
    audit = json.loads(
        (governing_root / "reports/track_b/freddie_2010_data_audit.json").read_text()
    )
    assert audit["cohort"] is None


def test_cli_source_rejection_writes_private_counts_and_stops(governing_root):
    import subprocess
    import sys

    bad = performance()
    bad.pop()
    source, auth = archive(governing_root, rows=[bad])
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_track_b_panel.py"),
            "--root",
            str(governing_root),
            "--source",
            str(source),
            "--authorization",
            str(auth),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "ingestion stopped" in result.stderr
    audit = json.loads(
        (governing_root / "data/track_b/manifests/ingestion_rejection.json").read_text()
    )
    assert audit["parse_counts"]["performance"]["malformed_rows"] == 1
    assert not (governing_root / "reports/track_b").exists()
