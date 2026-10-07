"""Bounded official API acquisition; raw responses immutable and credentials redacted."""

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from credit_risk.track_b.macro.contracts import verify_raw, write_raw
from credit_risk.track_b.macro.information import feature_hash
from credit_risk.track_b.multivintage.study import lf_hash
from credit_risk.track_b.pit_macro.engine import VintageValue, period_end, validate_versions

API = "https://api.stlouisfed.org/fred/"


def validate_run(run):
    if not re.fullmatch(r"task9_[a-z0-9_]+", run):
        raise ValueError("Invalid acquisition run name")
    return run


class Requests:
    def __init__(self, root, limits, opener=urllib.request.urlopen, *, run="task9_v1"):
        self.root, self.limits, self.opener = Path(root), limits, opener
        self.count = 0
        self.entries = []
        self.run = validate_run(run)

    def get(self, endpoint, parameters, *, key=None, absolute=False):
        self.count += 1
        if self.count > self.limits["max_requests"]:
            raise ValueError("Frozen acquisition request budget exhausted")
        if "api_key" in parameters:
            raise ValueError("Secrets must not enter recorded parameters")
        url = endpoint if absolute else API + endpoint
        if not url.startswith((API, "https://alfred.stlouisfed.org/")):
            raise ValueError("Unapproved acquisition provider")
        safe = dict(parameters)
        query = dict(safe)
        if key:
            query["api_key"] = key
        started = datetime.now(UTC).isoformat()
        entry = dict(
            provider="Federal Reserve Bank of St. Louis / ALFRED",
            endpoint=url,
            parameters=safe,
            retrieved_at=started,
            series=safe.get("series_id", safe.get("seid")),
            license_access_notes="Official source; registry retains terms; raw stays ignored",
        )
        payload = None
        try:
            request = urllib.request.Request(
                url + "?" + urllib.parse.urlencode(query),
                headers={"User-Agent": "credit-risk-lab-pit-research/1.0"},
            )
            with self.opener(request, timeout=self.limits["timeout_seconds"]) as response:
                entry["response_status"] = response.status
                entry["content_type"] = response.headers.get("Content-Type")
                payload = response.read(self.limits["max_response_bytes"] + 1)
                if len(payload) > self.limits["max_response_bytes"]:
                    raise ValueError("Response exceeds frozen byte budget")
        except urllib.error.HTTPError as exc:
            entry["response_status"] = exc.code
            entry["error_type"] = "HTTPError"
            # Provider error text can echo URLs. No error body or exception string
            # containing a credential-bearing request is persisted.
        except (OSError, ValueError) as exc:
            entry["response_status"] = None
            entry["error_type"] = type(exc).__name__
        if payload is not None and "error_type" not in entry:
            relative = f"data/track_b/macro/raw/{self.run}/request_{self.count:03d}.bin"
            entry["content_sha256"] = write_raw(self.root / relative, payload)
            entry["raw_path"] = relative
            entry["bytes"] = len(payload)
        self.entries.append(entry)
        return payload if "error_type" not in entry else None, entry


def regime(spec, available):
    if spec["series_id"] == "GDPC1":
        matches = [
            r
            for r in spec["units_by_availability"]
            if date.fromisoformat(r["start"]) <= available <= date.fromisoformat(r["end"])
        ]
        if len(matches) != 1:
            raise ValueError("GDP vintage units lack authoritative mapping")
        return matches[0]["units"], "GDP_" + matches[0]["start"]
    if spec["series_id"] == "MORTGAGE30US":
        return spec["units"], "PMMS_application" if available >= date(
            2022, 11, 17
        ) else "PMMS_survey"
    return spec["units"], spec["series_id"] + "_native"


def normalize_api(payload, spec, entry):
    """Real-time-period rows only; no guessed first-release rank or exact dates."""
    if entry.get("parameters", {}).get("output_type") != 1:
        raise ValueError("Real-time-period output contract required")
    import hashlib

    if hashlib.sha256(payload).hexdigest() != entry["content_sha256"]:
        raise ValueError("Raw payload hash differs from manifest")
    data = json.loads(payload)
    for key in ["realtime_start", "realtime_end"]:
        if key in entry["parameters"] and data.get(key) != entry["parameters"][key]:
            raise ValueError("Provider real-time window differs from frozen request")
    observations = data["observations"]
    if data["count"] != len(observations) or data.get("offset", 0) != 0:
        raise ValueError("Incomplete API pagination; no partial series admitted")
    rows, missing = [], 0
    for raw in observations:
        if raw["value"] == ".":
            missing += 1
            continue
        start = date.fromisoformat(raw["realtime_start"])
        units, measurement = regime(spec, start)
        rows.append(
            VintageValue(
                series_id=spec["series_id"],
                reference_period=period_end(raw["date"], spec["frequency"]),
                value=float(raw["value"]),
                archive_start=start,
                archive_end=date.fromisoformat(raw["realtime_end"]),
                publication_upper_bound=start,
                revision_upper_bound=start,
                release_date=None,
                revision_date=None,
                certified_initial=False,
                provenance="VINTAGE_AWARE_AVAILABLE",
                date_evidence="conservative_archive_upper_bound",
                source=spec["provider"],
                source_hash=entry["content_sha256"],
                retrieved_at=entry["retrieved_at"],
                frequency=spec["frequency"],
                units=units,
                seasonal_adjustment=spec["seasonal_adjustment"],
                measurement_regime=measurement,
            )
        )
    validate_versions(rows)
    return rows, missing


def metadata_matches(payload, spec):
    series = json.loads(payload)["seriess"]
    if len(series) != 1:
        raise ValueError("Ambiguous provider series metadata")
    s = series[0]
    expected_adjustment = "SAAR" if spec["series_id"] == "GDPC1" else spec["seasonal_adjustment"]
    if (s["id"], s["frequency_short"], s["units"], s["seasonal_adjustment_short"]) != (
        spec["series_id"],
        spec["frequency"],
        spec["units"],
        expected_adjustment,
    ):
        raise ValueError("Acquired metadata differs from frozen registry")
    return s


def vintage_dates(requests, sid, plan, key):
    """Acquire every page of the official series revision/release-date index."""
    dates, entries, expected = [], [], None
    while expected is None or len(dates) < expected:
        payload, entry = requests.get(
            "series/vintagedates",
            dict(
                series_id=sid,
                file_type="json",
                realtime_start=plan["real_time_request"][0],
                realtime_end=plan["real_time_request"][1],
                sort_order="asc",
                offset=len(dates),
                limit=10000,
            ),
            key=key,
        )
        if payload is None:
            return None, entries + [entry]
        verify_raw(requests.root / entry["raw_path"], entry["content_sha256"])
        data = json.loads(payload)
        count = data["count"]
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Empty or invalid provider vintage index")
        if expected is not None and count != expected:
            raise ValueError("Vintage index changed during pagination")
        expected = count
        if data["offset"] != len(dates):
            raise ValueError("Vintage index offset differs from request")
        for bound in ["realtime_start", "realtime_end"]:
            if data[bound] != entry["parameters"][bound]:
                raise ValueError("Vintage index real-time window mismatch")
        page = data["vintage_dates"]
        if not page or len(dates) + len(page) > count:
            raise ValueError("Incomplete or excessive vintage index page")
        dates.extend(page)
        entries.append(entry)
    parsed = [date.fromisoformat(d) for d in dates]
    if parsed != sorted(set(parsed)) or any(
        not date.fromisoformat(plan["real_time_request"][0])
        <= d
        <= date.fromisoformat(plan["real_time_request"][1])
        for d in parsed
    ):
        raise ValueError("Unordered, duplicate or out-of-window vintage dates")
    return dates, entries


def observation_windows(dates, bounds, maximum=1999):
    """Partition inclusive real-time bounds under FRED's JSON vintage-date limit.

    Reserve one slot below the JSON limit of 2000: the provider can count the
    carried initial snapshot as an additional vintage at a clipped window start.
    Boundaries begin on actual vintage dates. The prior interval ends the day
    before, so repeated historical observations have disjoint validity segments.
    These transport segments are not asserted to be distinct economic revisions.
    """
    if maximum < 1 or dates != sorted(set(dates)):
        raise ValueError("Invalid vintage partition")
    starts = [bounds[0]] + dates[maximum::maximum]
    ends = [(date.fromisoformat(d) - timedelta(days=1)).isoformat() for d in starts[1:]] + [
        bounds[1]
    ]
    return list(zip(starts, ends, strict=True))


def acquire(root, *, opener=urllib.request.urlopen, run="task9_v1"):
    root = Path(root)
    run = validate_run(run)
    base = root / f"data/track_b/macro/manifests/{run}"
    frozen = json.loads(
        (root / "data/track_b/macro/manifests/task9_v1/registry_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    registry_path = root / "docs/track_b/pit_macro_series_registry.json"
    protocol_path = root / "docs/track_b/pit_macro_research_protocol.json"
    if (
        lf_hash(registry_path) != frozen["registry_sha256_lf"]
        or lf_hash(protocol_path) != frozen["protocol_sha256_lf"]
    ):
        raise ValueError("Frozen pre-acquisition registry/protocol changed")
    if (base / "acquisition.json").exists():
        raise FileExistsError(
            "Acquisition exists; separate version required, no silent retry/reset"
        )
    plan = json.loads(protocol_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    requests = Requests(root, plan["network_limits"], opener, run=run)
    key = os.environ.get("FRED_API_KEY")
    if run != "task9_v1" and not key:
        raise ValueError("FRED_API_KEY unavailable; authenticated successor run not started")
    rows, summaries = [], []
    for spec in registry["series"]:
        sid = spec["series_id"]
        result = dict(
            series_id=sid,
            provenance="INSUFFICIENT_PROVENANCE",
            normalized_versions=0,
            exact_provider_dates_certified=False,
            status="NO_PREDICTOR_OBSERVATIONS",
        )
        if not key:
            _, attempt = requests.get(
                "https://alfred.stlouisfed.org/series/downloaddata", dict(seid=sid), absolute=True
            )
            result.update(
                reason="API key absent; public ALFRED form supplied no dated observation export",
                export_form_response_status=attempt["response_status"],
                form_error=attempt.get("error_type"),
                form_content_hash=attempt.get("content_sha256"),
                metadata_html_is_predictor_data=False,
            )
            summaries.append(result)
            print(sid + ": no validated vintage payload; feature unavailable", flush=True)
            continue
        metadata, meta_entry = requests.get(
            "series", dict(series_id=sid, file_type="json"), key=key
        )
        if metadata is None:
            result["reason"] = "Provider metadata request failed"
            summaries.append(result)
            continue
        try:
            metadata_matches(metadata, spec)
            dates, date_entries = vintage_dates(requests, sid, plan, key)
            if dates is None:
                result["reason"] = "Provider vintage-date request failed"
                summaries.append(result)
                continue
            result.update(
                vintage_dates=dates,
                vintage_date_content_hashes=[e["content_sha256"] for e in date_entries],
                vintage_date_count=len(dates),
                vintage_dates_are_exact_agency_dates=False,
            )
            parameters = dict(
                series_id=sid,
                file_type="json",
                units="lin",
                output_type=1,
                realtime_start=plan["real_time_request"][0],
                realtime_end=plan["real_time_request"][1],
                observation_start=plan["reference_acquisition_window"][0],
                observation_end=plan["reference_acquisition_window"][1],
                offset=0,
                limit=plan["network_limits"]["max_observation_rows_per_series"],
            )
            normalized, missing, observation_entries = [], 0, []
            partitions = observation_windows(dates, plan["real_time_request"])
            for start, end in partitions:
                payload, entry = requests.get(
                    "series/observations",
                    dict(parameters, realtime_start=start, realtime_end=end),
                    key=key,
                )
                if payload is None:
                    break
                segment, omitted = normalize_api(payload, spec, entry)
                normalized.extend(segment)
                missing += omitted
                observation_entries.append(entry)
            if len(observation_entries) != len(partitions):
                result["reason"] = "Provider observations request failed"
            else:
                if not normalized:
                    raise ValueError("No dated numeric observations acquired")
                validate_versions(normalized)
                if any(r.archive_start.isoformat() not in dates for r in normalized):
                    raise ValueError("Observation version absent from provider vintage index")
                rows.extend(normalized)
                result.update(
                    provenance="VINTAGE_AWARE_AVAILABLE",
                    status="DATED_VERSIONS_ACQUIRED",
                    normalized_versions=len(normalized),
                    missing_source_values=missing,
                    observation_content_hashes=[e["content_sha256"] for e in observation_entries],
                    real_time_partitions=partitions,
                    partition_boundary_is_economic_revision=False,
                    metadata_hash=meta_entry["content_sha256"],
                    availability_rule="Archive start bounds publication/revision conservatively",
                    reference_first=min(r.reference_period for r in normalized).isoformat(),
                    reference_last=max(r.reference_period for r in normalized).isoformat(),
                    archive_first=min(r.archive_start for r in normalized).isoformat(),
                    archive_last=max(r.archive_start for r in normalized).isoformat(),
                )
        except (ValueError, KeyError, TypeError) as exc:
            result.update(
                status="PROVENANCE_GATE_STOP", reason=type(exc).__name__ + ": " + str(exc)
            )
            summaries.append(result)
            break
        summaries.append(result)
    value = dict(
        acquisition_run=run,
        registry_sha256_lf=frozen["registry_sha256_lf"],
        protocol_sha256_lf=frozen["protocol_sha256_lf"],
        api_key_configured=bool(key),
        requests=requests.entries,
        series=summaries,
        requested_series=[s["series_id"] for s in registry["series"]],
        models_fitted=False,
        outcome_selection=False,
        normalized_versions_sha256=feature_hash([row.model_dump(mode="json") for row in rows]),
    )
    for path, content in [
        (base / "acquisition.json", value),
        (
            root / f"data/track_b/macro/interim/{run}/versions.json",
            [row.model_dump(mode="json") for row in rows],
        ),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(content, stream, indent=2)
            stream.write("\n")
    return value, rows


def load_versions(root, run="task9_v1"):
    run = validate_run(run)
    path = Path(root) / f"data/track_b/macro/interim/{run}/versions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [VintageValue(**r) for r in data]
    manifest = json.loads(
        (Path(root) / f"data/track_b/macro/manifests/{run}/acquisition.json").read_text(
            encoding="utf-8"
        )
    )
    if feature_hash(data) != manifest["normalized_versions_sha256"]:
        raise ValueError("Normalized version rows changed")
    hashes = {
        entry["content_sha256"] for entry in manifest["requests"] if "content_sha256" in entry
    }
    for entry in manifest["requests"]:
        if "content_sha256" in entry:
            verify_raw(Path(root) / entry["raw_path"], entry["content_sha256"])
    if any(r.source_hash not in hashes for r in rows):
        raise ValueError("Observation lineage absent from acquisition manifest")
    validate_versions(rows)
    return manifest, rows
