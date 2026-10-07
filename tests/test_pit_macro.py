"""Task9 information-time, transformation, acquisition and normalized-join regression tests."""

import hashlib
import json
import urllib.error
from datetime import date, datetime
from pathlib import Path

import pytest

from credit_risk.track_b.pit_macro.acquisition import (
    Requests,
    acquire,
    metadata_matches,
    normalize_api,
    observation_windows,
    validate_run,
    vintage_dates,
)
from credit_risk.track_b.pit_macro.engine import (
    VintageValue,
    engineer,
    left_join,
    month_table,
    period_end,
    previous_month_end,
    select,
    validate_versions,
)
from credit_risk.track_b.pit_macro.reporting import vintage_diagnostics

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "docs/track_b/pit_macro_series_registry.json").read_text())
FEATURES = json.loads((ROOT / "docs/track_b/pit_macro_feature_registry.json").read_text())[
    "features"
]
CAPS = {s["series_id"]: s["freshness_days"] for s in REGISTRY["series"]}


def value(**changes):
    v = dict(
        series_id="UNRATE",
        reference_period="2014-03-31",
        value=100,
        archive_start="2014-04-04",
        archive_end="9999-12-31",
        publication_upper_bound="2014-04-04",
        revision_upper_bound="2014-04-04",
        release_date="2014-04-04",
        revision_date="2014-04-04",
        certified_initial=True,
        initial_evidence="Synthetic agency fixture",
        provenance="VINTAGE_AWARE_AVAILABLE",
        date_evidence="exact",
        source="Synthetic official fixture",
        source_hash="a" * 64,
        retrieved_at="2026-10-07T10:00:00+00:00",
        frequency="M",
        units="Percent",
        seasonal_adjustment="SA",
        measurement_regime="toy_native",
    )
    v.update(changes)
    return VintageValue(**v)


def versions():
    return [
        value(archive_end="2014-06-09"),
        value(
            value=120,
            archive_start="2014-06-10",
            revision_date="2014-06-10",
            revision_upper_bound="2014-06-10",
            certified_initial=False,
        ),
    ]


@pytest.mark.parametrize(
    "t0,expected",
    [
        ("2014-03-31", None),
        ("2014-04-04", 100),
        ("2014-04-30", 100),
        ("2014-05-31", 100),
        ("2014-07-31", 120),
    ],
)
def test_release_and_revision_knowledge_time(t0, expected):
    r = select(versions(), "UNRATE", date.fromisoformat(t0))
    assert (r.value if r else None) == expected


def test_certified_initial_survives_revision_but_unverified_initial_is_not_invented():
    assert select(versions(), "UNRATE", date(2014, 7, 31), rule="initial_release").value == 100
    unknown = value(
        release_date=None,
        revision_date=None,
        certified_initial=False,
        initial_evidence=None,
        date_evidence="conservative_archive_upper_bound",
    )
    assert select([unknown], "UNRATE", date(2014, 7, 31), rule="initial_release") is None
    assert select([unknown], "UNRATE", date(2014, 4, 4)).release_date is None


def test_order_invariance():
    assert select(versions(), "UNRATE", date(2014, 7, 31)) == select(
        versions()[::-1], "UNRATE", date(2014, 7, 31)
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(representation="current_revised"),
        dict(archive_start="2014-06-01", archive_end="2014-05-01"),
        dict(date_evidence="exact", release_date=None),
        dict(release_date="2014-03-01"),
        dict(value=float("nan")),
        dict(source_hash="bad"),
        dict(reference_period="2014-03-01"),
        dict(certified_initial=True, initial_evidence=None),
    ],
)
def test_invalid_provenance_or_current_history_fails_closed(change):
    with pytest.raises(ValueError):
        validate_versions([value(**change)])


def test_duplicate_real_time_ranges_fail():
    with pytest.raises(ValueError, match="overlapping"):
        select([value(), value()], "UNRATE", date(2014, 5, 31))


def test_intraday_not_silently_coerced_to_end_of_day():
    with pytest.raises(ValueError, match="intraday"):
        select([value()], "UNRATE", datetime(2014, 4, 4, 8))


@pytest.mark.parametrize(
    "label,freq,end",
    [
        ("2014-03-01", "M", "2014-03-31"),
        ("2014-01-01", "Q", "2014-03-31"),
        ("2014-04-03", "W", "2014-04-03"),
    ],
)
def test_native_period_mapping(label, freq, end):
    assert period_end(label, freq).isoformat() == end


def feature(name):
    return next(f for f in FEATURES if f["name"] == name)


def test_yoy_same_asof_revised_denominator_and_missing_future_operand():
    prior = value(
        series_id="CPIAUCSL",
        reference_period="2013-03-31",
        value=80,
        archive_start="2013-04-15",
        publication_upper_bound="2013-04-15",
        revision_upper_bound="2013-04-15",
        release_date="2013-04-15",
        revision_date="2013-04-15",
    )
    current = value(series_id="CPIAUCSL")
    r = engineer([current, prior], feature("cpi_yoy"), date(2014, 5, 31), CAPS)
    assert r["value"] == pytest.approx(25)
    future = prior.model_copy(update=dict(archive_start=date(2014, 6, 1)))
    assert (
        engineer([current, future], feature("cpi_yoy"), date(2014, 5, 31), CAPS)["status"]
        == "MISSING_OPERAND"
    )


@pytest.mark.parametrize(
    "kind,name,prior_value,expected",
    [("M", "unemployment_change_3m", 80, 20), ("Q", "gdp_qoq", 80, 25)],
)
def test_short_change_and_nonannualized_quarter_growth(kind, name, prior_value, expected):
    sid = feature(name)["series_id"]
    current = value(series_id=sid, frequency=kind)
    prior = value(
        series_id=sid,
        frequency=kind,
        reference_period="2013-12-31",
        value=prior_value,
        archive_start="2014-01-30",
        publication_upper_bound="2014-01-30",
        revision_upper_bound="2014-01-30",
        release_date="2014-01-30",
        revision_date="2014-01-30",
    )
    assert engineer([current, prior], feature(name), date(2014, 5, 31), CAPS)["value"] == expected
    mismatch = prior.model_copy(update=dict(measurement_regime="different_base"))
    assert (
        engineer([current, mismatch], feature(name), date(2014, 5, 31), CAPS)["status"]
        == "METADATA_REGIME_MISMATCH"
    )


def test_spread_freshness_and_separate_reference_dates():
    a = value(
        series_id="MORTGAGE30US",
        frequency="W",
        reference_period="2014-04-24",
        value=5,
        archive_start="2014-04-24",
        publication_upper_bound="2014-04-24",
        revision_upper_bound="2014-04-24",
        release_date="2014-04-24",
        revision_date="2014-04-24",
    )
    b = value(
        series_id="DGS10",
        frequency="D",
        reference_period="2014-04-29",
        value=3,
        archive_start="2014-04-30",
        publication_upper_bound="2014-04-30",
        revision_upper_bound="2014-04-30",
        release_date="2014-04-30",
        revision_date="2014-04-30",
    )
    result = engineer([a, b], feature("mortgage_treasury_spread"), date(2014, 4, 30), CAPS)
    assert result["value"] == 2
    assert len({r["reference_period"] for r in result["inputs"]}) == 2
    assert (
        engineer([a, b], feature("mortgage_treasury_spread"), date(2014, 5, 31), CAPS)["status"]
        == "STALE"
    )


def test_month_table_join_preserves_rows_missing_months_and_previous_month_knowledge():
    assert previous_month_end("2014-04") == date(2014, 3, 31)
    table = month_table([value()], ["2014-05", "2014-04"], [feature("unemployment_level")], CAPS)
    assert table[0]["features"]["unemployment_level"]["value"] is None
    assert table[1]["features"]["unemployment_level"]["value"] == 100
    records = [dict(reporting_month="2014-05", loan="a"), dict(reporting_month="2014-06", loan="b")]
    joined = list(left_join(records, table))
    assert len(joined) == 2 and joined[1]["macro"] is None and records[0].get("macro") is None
    with pytest.raises(ValueError):
        month_table([], ["2014-05", "2014-05"], FEATURES, CAPS)
    with pytest.raises(ValueError):
        list(left_join(records, table + table))


def test_quarterly_carry_forward_only_after_release():
    g = value(series_id="GDPC1", frequency="Q")
    assert select([g], "GDPC1", date(2014, 3, 31)) is None
    assert select([g], "GDPC1", date(2014, 5, 31), cap=183) == g
    assert select([g], "GDPC1", date(2014, 12, 31), cap=183) is None


def test_api_parser_requires_complete_real_time_payload_and_preserves_unknown_dates():
    spec = next(s for s in REGISTRY["series"] if s["series_id"] == "UNRATE")
    payload = json.dumps(
        dict(
            count=1,
            offset=0,
            observations=[
                dict(
                    date="2014-03-01",
                    value="6.7",
                    realtime_start="2014-04-04",
                    realtime_end="9999-12-31",
                )
            ],
        )
    ).encode()
    entry = dict(
        parameters=dict(output_type=1),
        content_sha256=hashlib.sha256(payload).hexdigest(),
        retrieved_at="2026-10-07T10:00:00+00:00",
    )
    rows, missing = normalize_api(payload, spec, entry)
    assert missing == 0 and rows[0].release_date is None and not rows[0].certified_initial
    data = json.loads(payload)
    data["count"] = 2
    modified = json.dumps(data).encode()
    with pytest.raises(ValueError, match="pagination"):
        normalize_api(
            modified, spec, dict(entry, content_sha256=hashlib.sha256(modified).hexdigest())
        )
    with pytest.raises(ValueError, match="hash"):
        normalize_api(modified, spec, entry)
    with pytest.raises(ValueError):
        normalize_api(payload, spec, dict(entry, parameters=dict(output_type=2)))


def test_metadata_mismatch_rejected():
    spec = next(s for s in REGISTRY["series"] if s["series_id"] == "UNRATE")
    payload = json.dumps(
        dict(
            seriess=[
                dict(
                    id="UNRATE", frequency_short="M", units="wrong", seasonal_adjustment_short="SA"
                )
            ]
        )
    ).encode()
    with pytest.raises(ValueError, match="metadata"):
        metadata_matches(payload, spec)


def test_network_failure_bounded_redacted_and_no_payload_fabricated(tmp_path):
    calls = []

    def broken(request, timeout):
        calls.append((request, timeout))
        raise urllib.error.HTTPError(request.full_url, 403, "contains_secret", {}, None)

    client = Requests(
        tmp_path, dict(max_requests=1, timeout_seconds=1, max_response_bytes=1000), broken
    )
    payload, entry = client.get("series", dict(series_id="UNRATE"), key="secret")
    assert payload is None and entry["response_status"] == 403 and "secret" not in json.dumps(entry)
    assert "api_key" not in entry["parameters"] and len(calls) == 1
    with pytest.raises(ValueError, match="budget"):
        client.get("series", dict(series_id="UNRATE"))


def test_raw_exclusive_response_hash_and_budget(tmp_path):
    class Response:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            return b"{}"

    limits = dict(max_requests=2, timeout_seconds=1, max_response_bytes=100)
    client = Requests(tmp_path, limits, lambda *a, **k: Response())
    _, entry = client.get("series", dict(series_id="UNRATE"))
    assert (tmp_path / entry["raw_path"]).read_bytes() == b"{}"
    client2 = Requests(tmp_path, limits, lambda *a, **k: Response())
    with pytest.raises(FileExistsError):
        client2.get("series", dict(series_id="UNRATE"))


def test_frozen_prior_evidence_and_sample_hashes():
    manifest = json.loads((ROOT / "docs/track_b/pit_macro_preservation_manifest.json").read_text())
    for name, spec in manifest["evidence"].items():
        payload = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == spec["sha256"], name
    assert manifest["base_commit"] == "7e0f6d0b9fdfde67af0a7da3204b76a40aa8b8c2"
    assert len(REGISTRY["series"]) == 6 and len(FEATURES) == 8


def test_vintage_date_pages_complete_hashed_and_ascending(tmp_path):
    class Response:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __init__(self, offset):
            self.offset = offset

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            return json.dumps(
                dict(
                    count=2,
                    offset=self.offset,
                    realtime_start="1776-07-04",
                    realtime_end="2026-03-31",
                    vintage_dates=[["2014-04-04"], ["2014-06-10"]][self.offset],
                )
            ).encode()

    def opener(request, timeout):
        from urllib.parse import parse_qs, urlsplit

        return Response(int(parse_qs(urlsplit(request.full_url).query)["offset"][0]))

    client = Requests(
        tmp_path,
        dict(max_requests=2, timeout_seconds=1, max_response_bytes=1000),
        opener,
        run="task9_api_v2",
    )
    dates, entries = vintage_dates(
        client, "UNRATE", dict(real_time_request=["1776-07-04", "2026-03-31"]), "secret"
    )
    assert dates == ["2014-04-04", "2014-06-10"] and len(entries) == 2
    for entry in entries:
        assert "secret" not in json.dumps(entry)
        assert "task9_api_v2" in entry["raw_path"]
        assert (
            hashlib.sha256((tmp_path / entry["raw_path"]).read_bytes()).hexdigest()
            == entry["content_sha256"]
        )


@pytest.mark.parametrize("name", ["../unsafe", "task9_../unsafe", "task9_bad/name"])
def test_acquisition_namespace_rejects_path_traversal(name):
    with pytest.raises(ValueError, match="run name"):
        validate_run(name)


def test_successor_without_key_preserves_failed_run_and_makes_no_request(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    called = []
    with pytest.raises(ValueError, match="FRED_API_KEY unavailable"):
        acquire(ROOT, run="task9_test_no_key", opener=lambda *a, **k: called.append(a))
    assert not called
    assert not (ROOT / "data/track_b/macro/manifests/task9_test_no_key").exists()


def test_real_time_partition_has_no_overlap_gap_or_excess_vintage_dates():
    dates = ["2014-01-01", "2014-02-01", "2014-03-01", "2014-04-01", "2014-05-01"]
    windows = observation_windows(dates, ["1776-07-04", "2026-03-31"], maximum=2)
    assert windows == [
        ("1776-07-04", "2014-02-28"),
        ("2014-03-01", "2014-04-30"),
        ("2014-05-01", "2026-03-31"),
    ]
    assert all(sum(start <= d <= end for d in dates) <= 2 for start, end in windows)
    with pytest.raises(ValueError):
        observation_windows(dates[::-1], ["1776-07-04", "2026-03-31"])


def test_revision_diagnostics_do_not_mix_units_or_certify_archive_initial():
    rows = versions()
    stats = vintage_diagnostics(rows)
    assert stats["within_regime_retained_revision_comparisons"][0]["maximum_absolute_change"] == 20
    assert not stats["within_regime_retained_revision_comparisons"][0]["initial_release_certified"]
    assert stats["earliest_retained_availability_bound_lags"]["UNRATE"]["minimum"] == 4
    mixed = [rows[0], rows[1].model_copy(update={"units": "different base"})]
    assert vintage_diagnostics(mixed)["within_regime_retained_revision_comparisons"] == []
