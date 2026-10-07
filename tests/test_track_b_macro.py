"""Task 7 synthetic clocks/scenarios and immutable prior-evidence checks; no models."""

import copy
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from credit_risk.track_b.macro.contracts import (
    AcquisitionManifest,
    Scenario,
    validate_scenario_set,
    verify_raw,
    write_raw,
)
from credit_risk.track_b.macro.information import (
    Observation,
    feature_hash,
    national_geography,
    release_lags,
    select_asof,
    spread_asof,
    transform_asof,
)

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/track_b"


def obs(**changes):
    values = dict(
        series_id="toy",
        geography="US",
        reference_period="2014-03-31",
        release_date="2014-04-04",
        vintage_date="2014-04-04",
        revision_date="2014-04-04",
        revision_sequence=0,
        value=100,
        source="official toy fixture",
        frequency="M",
        units="index",
        seasonal_adjustment="SA",
        retrieved_at="2026-10-06T10:00:00+00:00",
        source_hash="a" * 64,
    )
    values.update(changes)
    return Observation(**values)


def versions():
    return [
        obs(),
        obs(value=120, vintage_date="2014-06-10", revision_date="2014-06-10", revision_sequence=1),
    ]


@pytest.mark.parametrize(
    "t0,value",
    [("2014-03-31", None), ("2014-04-30", 100), ("2014-05-31", 100), ("2014-07-31", 120)],
)
def test_required_examples_A_B_C_D(t0, value):
    r = select_asof(versions(), "toy", date.fromisoformat(t0))
    assert (r.value if r else None) == value


def test_first_release_does_not_become_latest_revision():
    r = select_asof(versions(), "toy", date(2014, 7, 31), rule="first_release")
    assert r.value == 100
    assert select_asof(versions()[1:], "toy", date(2014, 7, 31), rule="first_release") is None


def test_archive_delay_blocks_agency_released_version():
    r = obs(vintage_date="2014-08-01")
    assert select_asof([r], "toy", date(2014, 7, 31)) is None


def test_future_reference_rejected_even_with_matching_month_label():
    r = obs(
        reference_period="2014-04-30",
        release_date="2014-05-01",
        revision_date="2014-05-01",
        vintage_date="2014-05-01",
    )
    assert select_asof([r], "toy", date(2014, 4, 1)) is None


@pytest.mark.parametrize(
    "change",
    [
        dict(release_date=None),
        dict(value=float("nan")),
        dict(revision_date="2014-04-01"),
        dict(vintage_date="2014-04-01"),
        dict(release_date="2014-03-01"),
        dict(source_hash="fake"),
        dict(retrieved_at="2026-10-06T10:00:00"),
        dict(geography="CA"),
    ],
)
def test_observation_contract_rejects_invalid_provenance(change):
    with pytest.raises(ValueError):
        obs(**change)


def test_current_history_and_descriptive_rule_fail_closed():
    with pytest.raises(ValueError, match="Current-revised"):
        select_asof([obs(representation="current_revised")], "toy", date(2014, 7, 31))
    with pytest.raises(ValueError):
        select_asof(versions(), "toy", date(2014, 7, 31), rule="current")


def test_duplicate_and_inconsistent_units_are_not_arbitrarily_selected():
    with pytest.raises(ValueError, match="Duplicate"):
        select_asof([obs(), obs()], "toy", date(2014, 5, 31))
    rows = versions()
    rows[1] = obs(
        value=120,
        vintage_date="2014-06-10",
        revision_date="2014-06-10",
        revision_sequence=1,
        units="other_base",
    )
    with pytest.raises(ValueError, match="Metadata"):
        select_asof(rows, "toy", date(2014, 7, 31))


def test_newest_available_period_and_freshness():
    r = obs(
        reference_period="2014-04-30",
        release_date="2014-05-02",
        revision_date="2014-05-02",
        vintage_date="2014-05-02",
        value=140,
    )
    assert select_asof(versions() + [r], "toy", date(2014, 7, 31)).value == 140
    assert select_asof([obs()], "toy", date(2014, 7, 31), max_age_days=62) is None


def test_yoy_uses_same_historic_information_set_and_propagates_lineage():
    prior = obs(
        reference_period="2013-03-31",
        release_date="2013-04-05",
        vintage_date="2013-04-05",
        revision_date="2013-04-05",
        value=80,
    )
    later = obs(
        reference_period="2013-03-31",
        release_date="2013-04-05",
        vintage_date="2014-06-10",
        revision_date="2014-06-10",
        value=90,
        revision_sequence=1,
    )
    rows = versions() + [prior, later]
    early = transform_asof(rows, "toy", date(2014, 5, 31), months=12, kind="growth_pct")
    late = transform_asof(rows, "toy", date(2014, 7, 31), months=12, kind="growth_pct")
    assert early["value"] == pytest.approx(25)
    assert late["value"] == pytest.approx(100 * (120 / 90 - 1))
    assert all(r["vintage_date"] <= early["t0"] for r in early["inputs"])
    assert all(r["source_hash"] == "a" * 64 for r in early["inputs"])
    assert feature_hash(early) != feature_hash(late)


def test_lag_missing_or_unreleased_is_not_backfilled():
    prior = obs(
        reference_period="2013-03-31",
        release_date="2014-07-01",
        vintage_date="2014-07-01",
        revision_date="2014-07-01",
        value=80,
    )
    assert (
        transform_asof([obs(), prior], "toy", date(2014, 5, 31), months=12, kind="growth_pct")
        is None
    )


def test_current_revised_denominator_is_prohibited():
    prior = obs(
        reference_period="2013-03-31",
        release_date="2013-04-05",
        vintage_date="2013-04-05",
        revision_date="2013-04-05",
        representation="current_revised",
    )
    with pytest.raises(ValueError):
        transform_asof([obs(), prior], "toy", date(2014, 5, 31), months=12, kind="growth_pct")


def test_quarter_growth_alignment_and_difference():
    a = obs(frequency="Q")
    b = obs(
        frequency="Q",
        reference_period="2013-12-31",
        release_date="2014-01-30",
        vintage_date="2014-01-30",
        revision_date="2014-01-30",
        value=80,
    )
    assert (
        transform_asof([a, b], "toy", date(2014, 5, 31), months=3, kind="difference")["value"] == 20
    )
    with pytest.raises(ValueError):
        transform_asof([a, b], "toy", date(2014, 5, 31), months=1, kind="difference")


def test_rate_spread_keeps_both_dates_and_excludes_future_rate():
    a = obs(series_id="mortgage", units="Percent", value=5)
    b = obs(series_id="treasury", units="Percent", value=3)
    r = spread_asof([a, b], "mortgage", "treasury", date(2014, 5, 31))
    assert r["value"] == 2 and len(r["inputs"]) == 2
    assert spread_asof([a, b], "mortgage", "treasury", date(2014, 3, 31)) is None


@pytest.mark.parametrize("value", [None, "", "US"])
def test_national_mapping_handles_missing_loan_geography(value):
    assert national_geography("national", value) == "US"


@pytest.mark.parametrize(
    "level,value", [("state", "CA"), ("msa", "12345"), ("postal", "123"), ("national", "CA")]
)
def test_overgranularity_is_prohibited(level, value):
    with pytest.raises(ValueError):
        national_geography(level, value)


def scenario(**changes):
    values = dict(
        scenario_name="baseline",
        kind="forecast",
        as_of="2014-03-31",
        forecast_origin="2014-03-15",
        publication_date="2014-03-20",
        provider="toy provider",
        provenance="synthetic issued forecast; not official",
        consistency_review="toy review",
        conversion_rule="monthly native",
        horizon=2,
        paths=[
            dict(
                series_id="toy",
                geography="US",
                period_end=d,
                horizon=h,
                value=4,
                units="percent",
                frequency="M",
                source_kind="forecast",
                source_hash="a" * 64,
            )
            for h, d in [(1, "2014-04-30"), (2, "2014-05-31")]
        ],
    )
    values.update(changes)
    return values


def test_valid_forecast_and_optional_weights():
    a = Scenario(**scenario())
    b = Scenario(**scenario(scenario_name="adverse"))
    assert validate_scenario_set([a, b])
    assert a.scenario_probability is None


@pytest.mark.parametrize(
    "defect",
    [
        "order",
        "frequency",
        "observed",
        "date",
        "duplicate",
        "gap",
        "units",
        "future_publication",
        "origin",
    ],
)
def test_invalid_scenario_paths(defect):
    v = scenario()
    if defect == "order":
        v["paths"].reverse()
    if defect == "frequency":
        v["paths"][0]["frequency"] = "Q"
    if defect == "observed":
        v["paths"][0]["source_kind"] = "observed"
    if defect == "date":
        v["paths"][0]["period_end"] = "2014-03-31"
    if defect == "duplicate":
        v["paths"][1]["horizon"] = 1
    if defect == "gap":
        v["paths"].pop()
    if defect == "units":
        v["paths"][1]["units"] = "other"
    if defect == "future_publication":
        v["publication_date"] = "2014-04-01"
    if defect == "origin":
        v["forecast_origin"] = "2014-04-01"
    with pytest.raises(ValueError):
        Scenario(**v)


def test_replay_preserves_original_dates_and_cannot_masquerade_as_forecast():
    v = scenario(kind="historical_replay")
    for p in v["paths"]:
        p.update(
            source_kind="observed",
            original_reference_period="2010-01-31",
            observed_release_date="2010-02-05",
        )
    assert Scenario(**v).kind == "historical_replay"
    v["kind"] = "forecast"
    with pytest.raises(ValueError):
        Scenario(**v)


def test_unissued_observed_replay_rejected():
    v = scenario(kind="historical_replay")
    for p in v["paths"]:
        p.update(
            source_kind="observed",
            original_reference_period="2014-04-30",
            observed_release_date="2014-05-02",
        )
    with pytest.raises(ValueError):
        Scenario(**v)


def test_weights_require_evidence_and_complete_normalization():
    with pytest.raises(ValueError):
        Scenario(**scenario(scenario_probability=0.6))
    a = Scenario(**scenario(scenario_probability=0.6, probability_evidence="toy method"))
    b = Scenario(
        **scenario(
            scenario_name="adverse", scenario_probability=0.4, probability_evidence="toy method"
        )
    )
    assert validate_scenario_set([a, b])
    with pytest.raises(ValueError):
        validate_scenario_set([a])
    with pytest.raises(ValueError):
        validate_scenario_set([a, Scenario(**scenario(scenario_name="other"))])
    with pytest.raises(ValueError):
        validate_scenario_set([])
    with pytest.raises(ValueError):
        Scenario(**scenario(scenario_probability=float("nan")))


def manifest():
    return dict(
        source="official toy",
        series=["toy"],
        geography="US",
        acquisition_time="2026-10-06T10:00:00+00:00",
        source_url="https://example.gov/toy",
        source_hash="a" * 64,
        vintage_coverage=dict(start="2014-01-01", end="2014-06-30"),
        reference_coverage=dict(start="2013-01-01", end="2014-03-31"),
        release_coverage=dict(start="2013-02-01", end="2014-06-30"),
        units="index",
        seasonal_adjustment="SA",
        license="toy public-domain fixture",
        parser_version="1.0.0",
        vintage_status="VINTAGE-AWARE AVAILABLE",
        date_evidence="toy dated release archive",
        raw_path="data/track_b/macro/raw/toy.json",
        complete_revision_history=True,
    )


def test_manifest_required_fields_and_schema():
    v = manifest()
    assert AcquisitionManifest(**v).geography == "US"
    for key in v:
        broken = copy.deepcopy(v)
        broken.pop(key)
        with pytest.raises(ValueError):
            AcquisitionManifest(**broken)
    for name, cls in [
        ("macro_acquisition_manifest.schema.json", AcquisitionManifest),
        ("macro_scenario.schema.json", Scenario),
        ("macro_observation.schema.json", Observation),
    ]:
        assert json.loads((D / name).read_text(encoding="utf-8")) == cls.model_json_schema()


def test_raw_is_exclusive_and_hash_detects_mutation(tmp_path):
    p = tmp_path / "raw/toy.json"
    h = write_raw(p, b"official fixture")
    assert verify_raw(p, h) == h
    with pytest.raises(FileExistsError):
        write_raw(p, b"replacement")
    p.write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        verify_raw(p, h)


def test_feature_hash_deterministic_and_content_sensitive():
    assert feature_hash(dict(a=1, b=2)) == feature_hash(dict(b=2, a=1))
    assert feature_hash(dict(a=1)) != feature_hash(dict(a=2))
    with pytest.raises(ValueError):
        feature_hash(dict(a=float("nan")))


def test_release_lag_probe_calculations_are_documented_not_full_history():
    r = json.loads(
        (ROOT / "reports/track_b/macro_release_lag_probe.json").read_text(encoding="utf-8")
    )
    for summary in r["series"]:
        rows = [x for x in r["observations"] if x["series_id"] == summary["series_id"]]
        toy = [
            obs(
                reference_period=x["reference_period_end"],
                release_date=x["release_date"],
                revision_date=x["release_date"],
                vintage_date=x["release_date"],
            )
            for x in rows
        ]
        result = release_lags(toy)
        assert result == {k: summary[k] for k in result}


def test_previous_track_b_and_track_a_evidence_unchanged():
    r = json.loads((D / "macro_preservation_manifest.json").read_text(encoding="utf-8"))
    for name, spec in r["evidence"].items():
        payload = (ROOT / name).read_bytes()
        if spec["mode"] == "LF":
            payload = payload.replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == spec["sha256"], name
    assert r["locked_predictions_and_consumed_ledgers_accessed"] is False
    assert r["track_a_tag"][0] == "20be4bf1bd291afb5dfad5b0befbe761651f1de1"


def test_protocol_freezes_scope_and_preferred_design():
    p = json.loads((D / "macro_research_protocol.json").read_text(encoding="utf-8"))
    r = json.loads((D / "macro_feature_registry.json").read_text(encoding="utf-8"))
    assert p["decision"] == "MACRO DESIGN READY WITH MATERIAL IDENTIFICATION LIMITATIONS"
    assert len(r["series"]) == 8 and p["geography"] == "national_US_only"
    assert (
        "model_fit" in p["prohibited_actions"]
        and "consumed_ledger_access" in p["prohibited_actions"]
    )
    assert p["identification_strategy"].startswith("A:")
    assert p["proposed_vintages"] == [2006, 2008, 2010, 2014, 2018, 2020, 2022]


def test_freshness_applies_to_current_feature_not_intentionally_old_denominator():
    prior = obs(
        reference_period="2013-03-31",
        release_date="2013-04-05",
        revision_date="2013-04-05",
        vintage_date="2013-04-05",
        value=80,
    )
    r = transform_asof(
        [obs(), prior], "toy", date(2014, 5, 31), months=12, kind="growth_pct", max_age_days=62
    )
    assert r["value"] == pytest.approx(25)
    assert (
        transform_asof(
            [obs(), prior], "toy", date(2014, 8, 31), months=12, kind="growth_pct", max_age_days=62
        )
        is None
    )


def test_local_private_byte_preservation_when_evidence_is_present():
    r = json.loads((D / "macro_preservation_manifest.json").read_text(encoding="utf-8"))
    for name, expected in r["private_byte_hashes"].items():
        path = ROOT / name
        if path.exists():  # Hosted CI does not carry ignored licensed data/frozen bundles.
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name


def test_web_probe_hash_when_private_capture_present():
    r = json.loads((ROOT / "reports/track_b/macro_source_probe.json").read_text(encoding="utf-8"))
    cap = r["web_capture"]
    assert "NOT original" in cap["transport"]
    path = ROOT / cap["raw_path"]
    if path.exists():
        assert verify_raw(path, cap["source_hash"]) == cap["source_hash"]


@pytest.mark.parametrize(
    "raw_path",
    [
        "other/raw/test.json",
        "data/track_b/macro/raw/../test",
        "data/track_b/macro/raw",
        "C:/raw/test.json",
    ],
)
def test_manifest_prohibits_outside_raw_zone(raw_path):
    v = manifest()
    v["raw_path"] = raw_path
    with pytest.raises(ValueError):
        AcquisitionManifest(**v)


def test_new_documents_have_valid_utf8_and_local_links():
    import re
    from urllib.parse import urlsplit

    for name in ["MACRO_DATA_PROVENANCE.md", "MACRO_IDENTIFICATION_DESIGN.md"]:
        body = (D / name).read_text(encoding="utf-8")
        assert "\u00e2\u20ac" not in body
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", body):
            parts = urlsplit(target)
            if not parts.scheme:
                assert (D / parts.path).resolve().is_file(), target


@pytest.mark.parametrize(
    "period,frequency", [("2014-03-01", "M"), ("2014-01-01", "Q"), ("2014-02-28", "Q")]
)
def test_provider_start_labels_cannot_masquerade_as_period_end(period, frequency):
    with pytest.raises(ValueError, match="period-end"):
        obs(reference_period=period, frequency=frequency)
