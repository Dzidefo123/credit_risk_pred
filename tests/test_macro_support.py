"""Synthetic eligibility regressions; no licensed data or model outcomes loaded."""

import copy
import json
from pathlib import Path

import pytest

from credit_risk.track_b.macro_support.eligibility import (
    REDUCED,
    complete_months,
    duration_band,
    eligible_intervals,
    macro_reasons,
    monthly_text,
    ordinal,
    support_windows,
    validation_role,
)
from credit_risk.track_b.macro_support.study import immutable_json
from credit_risk.track_b.multivintage.study import lf_hash


def history():
    return [
        dict(
            reporting_month=monthly_text(ordinal("2010-01") + i),
            analytical_prefix=True,
            event_category="none",
            loan_age=48 + i,
            months_since_first_payment_proxy=48 + i,
        )
        for i in range(12)
    ]


def mapping(months=None):
    return {
        month: dict(features={name: dict(status="AVAILABLE") for name in REDUCED})
        for month in (months or [r["reporting_month"] for r in history()])
    }


def test_surviving_old_vintage_contributes_later_intervals_without_deletion():
    rows = history()
    before = copy.deepcopy(rows)
    risk = eligible_intervals(
        rows, mapping(["2010-09", "2010-10", "2010-11", "2010-12"]), REDUCED, "2026-03"
    )
    assert len(risk) == 4
    assert risk[0]["duration"] == 55
    assert rows == before


@pytest.mark.parametrize("event", ["default", "payoff"])
def test_one_endpoint_per_design_and_no_reentry(event):
    rows = history()
    rows[10]["event_category"] = event
    rows[11]["analytical_prefix"] = False
    risk = eligible_intervals(rows, mapping(), REDUCED, "2026-03")
    assert sum(r["event"] == event for r in risk) == 1
    assert risk[-1]["target_month"] == "2010-11"


def test_event_before_support_blocks_later_exposure():
    rows = history()
    rows[6]["event_category"] = "default"
    for row in rows[7:]:
        row["analytical_prefix"] = False
    assert not eligible_intervals(rows, mapping(["2010-09"]), REDUCED, "2026-03")


@pytest.mark.parametrize("status", ["UNAVAILABLE", "STALE", "MISSING_OPERAND", "UNVERIFIED"])
def test_partial_or_unverified_macro_cannot_enter(status):
    table = mapping()
    table["2010-09"]["features"]["cpi_yoy"]["status"] = status
    assert macro_reasons("2010-09", table, REDUCED, "2026-03")
    assert "2010-09" not in [
        r["target_month"] for r in eligible_intervals(history(), table, REDUCED, "2026-03")
    ]


def test_current_revised_input_is_not_pit():
    table = mapping()
    table["2010-09"]["features"]["cpi_yoy"]["inputs"] = [dict(current_revised=True)]
    assert "MACRO_PROVENANCE_INSUFFICIENT" in macro_reasons("2010-09", table, REDUCED, "2026-03")


def test_missing_change_operand_and_future_cutoff():
    table = {"2026-03": mapping()["2010-09"]}
    table["2026-03"]["features"]["unemployment_change_3m"]["status"] = "MISSING_OPERAND"
    assert "NO_UNEMPLOYMENT_CHANGE_OPERAND" in macro_reasons("2026-03", table, REDUCED, "2026-03")
    assert macro_reasons("2026-04", table, REDUCED, "2026-03") == ["STUDY_CUTOFF_EXCEEDED"]


def test_reduced_design_is_frozen_and_not_promoted():
    assert REDUCED == (
        "unemployment_level",
        "unemployment_change_3m",
        "treasury_10y_level",
        "cpi_yoy",
        "gdp_qoq",
    )
    table = mapping()
    table["2010-09"]["features"]["hpi_yoy"] = dict(status="UNAVAILABLE")
    assert not macro_reasons("2010-09", table, REDUCED, "2026-03")
    assert macro_reasons("2010-09", table, [*REDUCED, "hpi_yoy"], "2026-03")


def test_windows_do_not_bridge_missing_month():
    assert len(support_windows(["2010-09", "2010-11"])) == 2
    table = [dict(reporting_month=m, **v) for m, v in mapping(["2010-09"]).items()]
    assert complete_months(table, REDUCED) == ["2010-09"]
    with pytest.raises(ValueError, match="Duplicate"):
        complete_months(table * 2, REDUCED)


@pytest.mark.parametrize("event", ["unknown", "administrative", "ambiguous"])
def test_unascertainable_or_admin_state_is_censored(event):
    rows = history()
    rows[8]["event_category"] = event
    for row in rows[9:]:
        row["analytical_prefix"] = False
    risk = eligible_intervals(rows, mapping(), REDUCED, "2026-03")
    assert risk[-1]["target_month"] == "2010-08"
    assert all(r["event"] == "none" for r in risk)


def test_lookback_negative_proxy_and_duplicate_month():
    rows = history()
    risk = eligible_intervals(rows, mapping(), REDUCED, "2026-03")
    assert risk[0]["t0"] == "2010-06"
    rows[5]["months_since_first_payment_proxy"] = -1
    assert eligible_intervals(rows, mapping(), REDUCED, "2026-03")[0]["t0"] == "2010-07"
    with pytest.raises(ValueError, match="Duplicate"):
        eligible_intervals([*rows, rows[-1]], mapping(), REDUCED, "2026-03")


@pytest.mark.parametrize(
    "age,band",
    [
        (0, "0-12"),
        (12, "0-12"),
        (13, "13-24"),
        (121, "121-180"),
        (241, "241+"),
        (None, "UNAVAILABLE"),
    ],
)
def test_duration_bands(age, band):
    assert duration_band(age) == band


def test_facility_role_is_deterministic():
    assert validation_role(2006, "synthetic", "salt") == validation_role(2006, "synthetic", "salt")


def test_specification_requires_amendment_if_changed(tmp_path):
    path = tmp_path / "spec.json"
    immutable_json(path, dict(version=1))
    immutable_json(path, dict(version=1))
    with pytest.raises(ValueError, match="amendment"):
        immutable_json(path, dict(version=2))


def test_prior_public_evidence_is_preserved():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "docs/track_b/macro_support_preservation_manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in manifest["prior_public_lf_hashes"].items():
        assert lf_hash(root / name) == expected, name


def test_full_support_frozen_boundaries_and_retrieval_cutoff():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (root / "docs/track_b/macro_support_eligibility_spec.json").read_text(encoding="utf-8")
    )
    primary = spec["designs"]["PRIMARY"]
    reduced = spec["designs"]["REDUCED"]
    assert primary["support_windows"] == [dict(first="2010-09", last="2026-02", months=186)]
    assert "2010-08" not in primary["support_months"]
    assert "2008-12" not in primary["support_months"]
    assert reduced["support_windows"] == [dict(first="2006-02", last="2026-02", months=241)]
    assert "2026-03" not in primary["support_months"]
    assert spec["study_end"]["archive_cutoff"] == "2026-03-31"
    assert not spec["study_end"]["evidence_updates_permitted"]
    assert spec["study_end"]["frozen_retrieval_times"]
    assert not spec["transformations_and_selector"]["current_revised_backfill"]


def test_public_support_reproduction_fingerprint():
    root = Path(__file__).resolve().parents[1]
    from credit_risk.track_b.macro.information import feature_hash

    report = json.loads(
        (root / "reports/track_b/macro_support_eligibility.json").read_text(encoding="utf-8")
    )
    assert feature_hash(report["mortgage_eligibility"]) == report["reproduction"]["counts_sha256"]
    assert feature_hash(report["macro_support"]["matrix"]) == report["macro_support_matrix_sha256"]
    assert report["reproduction"]["actual_second_full_canonical_pass"]
    assert len(report["reproduction"]["facility_files_sha256"]) == 14


def test_unseen_vintages_are_separated_from_primary_temporal_validation():
    root = Path(__file__).resolve().parents[1]
    d = json.loads(
        (root / "docs/track_b/macro_support_validation_design.json").read_text(encoding="utf-8")
    )
    assert d["seen_development_vintages"] == ["2006", "2008", "2010", "2014"]
    assert d["unseen_temporal_vintages"] == ["2018", "2020", "2022"]
    assert not set(d["seen_development_vintages"]) & set(d["unseen_temporal_vintages"])
    assert d["temporal_seen_vintage_feasibility_passed"]
    assert d["candidate_dates_and_facility_roles_unchanged"]
    assert "UNSEEN_VINTAGE" in d["unseen_temporal_handling"]
    assert not d["model_fitting_performed"]


def test_source_hashes_and_final_design_are_consistent():
    root = Path(__file__).resolve().parents[1]
    d = json.loads(
        (root / "reports/track_b/macro_support_eligibility.json").read_text(encoding="utf-8")
    )
    for name, expected in d["source_evidence_sha256_lf"].items():
        assert lf_hash(root / name) == expected, name
    assert (
        lf_hash(root / "docs/track_b/macro_support_eligibility_spec.json")
        == (d["task10_specification_sha256_lf"])
    )
    assert (
        lf_hash(root / "docs/track_b/macro_support_validation_design.json")
        == (d["finalized_validation_design_sha256_lf"])
    )


def test_historical_fragment_does_not_admit_current_history_backfill():
    root = Path(__file__).resolve().parents[1]
    d = json.loads(
        (root / "docs/track_b/macro_historical_coverage_resolution.json").read_text(
            encoding="utf-8"
        )
    )
    assert not d["admitted_to_frozen_features"]
    assert not d["api_reacquisition"]
    assert d["complete_pre2010_extension_status"] == "UNAVAILABLE"
    assert d["march_2026"]["classification"] == "SOURCE_OBSERVATION_MISSING"
