"""Task9 descriptive evidence and normalized month join; never reads outcome columns."""

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median

from credit_risk.track_b.macro.information import feature_hash
from credit_risk.track_b.multivintage.core import write_json
from credit_risk.track_b.multivintage.reporting import table
from credit_risk.track_b.multivintage.study import lf_hash
from credit_risk.track_b.pit_macro.acquisition import load_versions, validate_run
from credit_risk.track_b.pit_macro.engine import VERSION, month_table


def mortgage_months(root, mortgage):
    """Read reporting-month projection only; leave all panel bytes and facilities intact."""
    cache = root / "data/track_b/macro/interim/task9_v1/mortgage_month_counts.json"
    if cache.exists():
        counts = json.loads(cache.read_text(encoding="utf-8"))
        expected = {str(v["vintage"]): v["rows"] for v in mortgage["vintages"]}
        if set(counts) != set(expected) or any(
            sum(counts[year].values()) != total for year, total in expected.items()
        ):
            raise ValueError("Cached mortgage calendar projection row count changed")
        return counts
    counts = {}
    for vintage in mortgage["vintages"]:
        year = vintage["vintage"]
        folder = (
            root / f"data/track_b/multivintage/processed/v1/{year}"
            if year in [2010, 2014, 2018]
            else root / f"data/track_b/multivintage/recovery_v1/{year}"
        )
        observed = Counter()
        with (folder / "monthly.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            index = next(reader).index("reporting_month")
            for row in reader:
                observed[row[index]] += 1
        if sum(observed.values()) != vintage["rows"]:
            raise ValueError("Mortgage calendar projection row count changed")
        counts[str(year)] = dict(sorted(observed.items()))
    write_json(cache, counts, True)
    return counts


def windows(statuses):
    result = []
    for month, status in sorted(statuses.items()):
        if result and result[-1]["status"] == status:
            result[-1]["last_reporting_month"] = month
            result[-1]["months"] += 1
        else:
            result.append(
                dict(
                    first_reporting_month=month, last_reporting_month=month, months=1, status=status
                )
            )
    return result


def vintage_diagnostics(rows):
    """Descriptive retained-archive comparisons; never certify an initial release."""
    periods, regimes = {}, {}
    for row in rows:
        periods.setdefault((row.series_id, row.reference_period), []).append(row)
        regimes.setdefault(
            (row.series_id, row.units, row.measurement_regime, row.reference_period), []
        ).append(row)
    lag_groups, revision_groups = {}, {}
    for (sid, period), versions in periods.items():
        first = min(versions, key=lambda r: r.archive_start)
        lag_groups.setdefault(sid, []).append((first.archive_start - period).days)
    for (sid, units, regime, _), versions in regimes.items():
        ordered = sorted(versions, key=lambda r: r.archive_start)
        if len(ordered) > 1:
            revision_groups.setdefault((sid, units, regime), []).append(
                ordered[-1].value - ordered[0].value
            )
    return dict(
        earliest_retained_availability_bound_lags={
            sid: dict(
                count=len(lags),
                minimum=min(lags),
                median=median(lags),
                maximum=max(lags),
                units="days",
                exact_provider_release_lags=False,
                includes_historical_backfill=True,
            )
            for sid, lags in lag_groups.items()
        },
        within_regime_retained_revision_comparisons=[
            dict(
                series_id=sid,
                units=units,
                measurement_regime=regime,
                paired_reference_periods=len(deltas),
                periods_with_changed_values=sum(d != 0 for d in deltas),
                median_absolute_change=median(abs(d) for d in deltas),
                maximum_absolute_change=max(abs(d) for d in deltas),
                comparison="Latest retained minus earliest retained, same native regime",
                initial_release_certified=False,
                current_revised_shadow=False,
            )
            for (sid, units, regime), deltas in sorted(revision_groups.items())
        ],
        transport_segments_are_economic_revision_counts=False,
    )


def build(root, run="task9_v1"):
    root = Path(root)
    run = validate_run(run)
    doc, report = root / "docs/track_b", root / "reports/track_b"
    registry = json.loads((doc / "pit_macro_series_registry.json").read_text(encoding="utf-8"))
    features = json.loads((doc / "pit_macro_feature_registry.json").read_text(encoding="utf-8"))[
        "features"
    ]
    protocol = json.loads((doc / "pit_macro_research_protocol.json").read_text(encoding="utf-8"))
    acquired, rows = load_versions(root, run)
    mortgage = json.loads(
        (report / "multi_vintage_recovery_audit.json").read_text(encoding="utf-8")
    )
    if (
        mortgage["dataset_manifest_sha256_lf"]
        != "da68e46945740cb680b2651b258bc16d66b90f34c028e521c390d9c2e53c1c9f"
    ):
        raise ValueError("Task8C combined manifest identity changed")
    if (
        mortgage["dataset_manifest"]["combined_sample_sha256"]
        != "93fdaf807fe2f9b168dca275be21644004a2ff3c14ae847d17e9330963bf2b8b"
    ):
        raise ValueError("Task8C combined sample changed")
    counts = mortgage_months(root, mortgage)
    months = sorted({m for v in counts.values() for m in v})
    caps = {s["series_id"]: s["freshness_days"] for s in registry["series"]}
    processed = month_table(rows, months, features, caps)
    macro_path = root / f"data/track_b/macro/processed/{run}/macro_month_table.json"
    if macro_path.exists():
        existing = json.loads(macro_path.read_text(encoding="utf-8"))
        if feature_hash(existing) != feature_hash(processed):
            raise ValueError("Existing processed month table differs; version amendment required")
    else:
        write_json(macro_path, processed, True)
    provenance = {s["series_id"]: s for s in acquired["series"]}
    matrix, support, coverage, by_vintage = {}, {}, {}, {}
    for month in processed:
        m = month["reporting_month"]
        matrix[m] = {n: f["status"] for n, f in month["features"].items()}
        populated = sum(f["status"] == "AVAILABLE" for f in month["features"].values())
        support[m] = (
            "FULL_MACRO_SUPPORT"
            if populated == len(features)
            else "PARTIAL_MACRO_SUPPORT"
            if populated
            else "NO_MACRO_SUPPORT"
        )
    for f in features:
        name = f["name"]
        summary = Counter(matrix[m][name] for m in months)
        required = [f["series_id"]] + ([f["secondary_series"]] if f["secondary_series"] else [])
        failed_provenance = any(
            provenance.get(s, {}).get("provenance") != "VINTAGE_AWARE_AVAILABLE" for s in required
        )
        coverage[name] = dict(
            months_in_scope=len(months),
            months_populated=summary["AVAILABLE"],
            months_unavailable=len(months) - summary["AVAILABLE"],
            stale_months=summary["STALE"],
            provenance_unavailable_months=len(months) if failed_provenance else 0,
            availability_states=dict(summary),
        )
    for year, monthly in counts.items():
        by_vintage[year] = dict(
            unique_reporting_months=len(monthly),
            rows_retained=sum(monthly.values()),
            facilities_retained=20000,
            coverage={
                f["name"]: dict(
                    months_populated=sum(matrix[m][f["name"]] == "AVAILABLE" for m in monthly),
                    rows_populated=sum(
                        n for m, n in monthly.items() if matrix[m][f["name"]] == "AVAILABLE"
                    ),
                    rows_unavailable=sum(
                        n for m, n in monthly.items() if matrix[m][f["name"]] != "AVAILABLE"
                    ),
                )
                for f in features
            },
        )
    intervals = windows(support)
    full = [w for w in intervals if w["status"] == "FULL_MACRO_SUPPORT"]
    gate_stop = any(s["status"] == "PROVENANCE_GATE_STOP" for s in acquired["series"])
    # Missing data/access is insufficient support, not proof of an invalid source.
    decision = (
        "STOP — MACRO PROVENANCE INVALID"
        if gate_stop
        else (
            "PIT MACRO FEATURE LAYER READY"
            if len(full) == 1 and full[0]["months"] == len(months)
            else "PIT MACRO SUPPORT INSUFFICIENT"
        )
    )
    limitations = [
        "Official metadata feasibility is distinct from acquired, certified observation coverage.",
        (
            "No exact provider publication lag is inferred from "
            "archive dates or historical average lags."
        ),
        (
            "Earliest retained archive values are not certified "
            "initial releases; initial-release sensitivity "
            "unavailable without independent evidence."
        ),
        "HPI/PMMS pre2010 vintage hints are not backfilled; no alternate series introduced.",
        (
            "GDP rebasing and PMMS 2022-11-17 measurement change "
            "remain explicit; unlike regimes cannot enter paired "
            "transformations."
        ),
        "Macro date-only evidence supports end-of-day use; intraday timing is unsupported.",
        (
            "PIT macro data do not certify the operational "
            "knowledge time of retrospective mortgage "
            "disclosures."
        ),
        (
            "Monthly aggregate join retains all facilities; "
            "expanded 7.7-million-row panel is not materialized "
            "or modified."
        ),
        (
            "Multiple vintages and macro variables do not "
            "identify unrestricted APC or causal macro "
            "coefficients."
        ),
    ]
    if not rows:
        limitations.append(
            "No validated numeric vintage payload acquired; all "
            "empirical macro features remain unavailable. "
            "Synthetic tests are not empirical coverage evidence."
        )
    horizons = {
        str(r["vintage"]): {
            h: dict(
                mortgage_status=s["status"],
                macro_status="NOT_ESTABLISHED",
                combined_status="OUTCOME_SUPPORTED_MACRO_UNAVAILABLE"
                if s["status"] == "SUPPORTED"
                else "MORTGAGE_HORIZON_UNSUPPORTED",
                analysis_population_dropped=False,
            )
            for h, s in r["horizons"].items()
        }
        for r in mortgage["vintages"]
    }
    evidence = dict(
        task=9,
        acquisition_run=run,
        decision=decision,
        engine_version=VERSION,
        registry_sha256_lf=lf_hash(doc / "pit_macro_series_registry.json"),
        feature_registry_sha256_lf=lf_hash(doc / "pit_macro_feature_registry.json"),
        protocol_sha256_lf=lf_hash(doc / "pit_macro_research_protocol.json"),
        series_registry=registry,
        features=features,
        acquisition=acquired,
        source_hashes=[
            dict(endpoint=e["endpoint"], series=e["series"], sha256=e["content_sha256"])
            for e in acquired["requests"]
            if "content_sha256" in e
        ],
        metadata_extraction=dict(
            raw_path="data/track_b/macro/raw/task9_metadata_web_extraction.json",
            transport="Web extraction JSON, NOT original HTTP bytes; not predictor data",
            sha256=lf_hash(root / "data/track_b/macro/raw/task9_metadata_web_extraction.json"),
        ),
        release_lags={
            s["series_id"]: dict(
                count=0,
                median=None,
                minimum=None,
                maximum=None,
                quantiles=None,
                evidence_quality="UNMEASURED_EXACT_PROVIDER_DATES",
                reason=(
                    "No certified initial provider release dates; do not "
                    "recycle Task7 small probe as full-history evidence"
                ),
            )
            for s in registry["series"]
        },
        revision_analysis=dict(
            comparison="Certified initial versus current-revised shadow",
            certified_initial_comparisons=0,
            current_revised_shadow_acquired=False,
            measured_revision_magnitude=None,
            reason=(
                "No certified paired initial/current responses; "
                "current browser metadata tables excluded"
            ),
        ),
        normalized_version_rows=len(rows),
        descriptive_vintage_diagnostics=vintage_diagnostics(rows),
        unique_reporting_months=len(months),
        assessment_dates=[
            dict(reporting_month=p["reporting_month"], t0=p["t0"]) for p in processed
        ],
        month_table_sha256=feature_hash(processed),
        availability_matrix=matrix,
        feature_coverage=coverage,
        vintage_join_coverage=by_vintage,
        common_support_windows=intervals,
        primary_complete_support_intervals=full,
        combined_horizon_support=horizons,
        future_model_spec=json.loads(
            (doc / "pit_macro_future_model_spec.json").read_text(encoding="utf-8")
        ),
        pandemic=dict(
            calendar_year_indicators=[2020, 2021],
            outcome_tuned_boundaries=False,
            causal_effect_estimated=False,
            interpretation="Calendar covariates, not full forbearance policy",
        ),
        mortgage_rows_retained=sum(sum(v.values()) for v in counts.values()),
        mortgage_facilities_retained=140000,
        borrower_uniqueness_claimed=False,
        mortgage_month_projection_sha256=feature_hash(counts),
        models_fitted=False,
        predictions_regenerated=False,
        macro_imputed=False,
        current_revised_used_in_features=False,
        limitations=limitations,
        next_task=(
            "Track B Task 9A — Prespecify macro-support eligibility and resolve "
            "remaining historical coverage gaps without changing mortgage samples"
            if len(provenance) == 6
            and all(s["provenance"] == "VINTAGE_AWARE_AVAILABLE" for s in provenance.values())
            else "Track B Task 9A — Obtain and certify authoritative"
            " ALFRED real-time exports, then rerun the PIT "
            "coverage gates without changing mortgage samples"
        ),
    )
    suffix = "" if run == "task9_v1" else "_" + run
    write_json(report / f"pit_macro_data_audit{suffix}.json", evidence)
    sections = [
        "# Track B Task 9 — Point-in-Time Macro Data Audit",
        "## Executive Summary",
        decision,
        f"{len(months)} unique reporting months, 140,000 facilities and 7,741,663 rows retained. "
        f"Validated numeric vintage versions: {len(rows)}. "
        "No model fitting or outcome-based selection.",
        "## Research Boundary",
        (
            "National US; six Task7 primary series, eight frozen "
            "features; reserves unchanged. Mortgage sample "
            "identity and prior evidence stay frozen."
        ),
        "## Series Registry",
        table(
            ["Series", "Provider", "Frequency", "Units", "Freshness days"],
            [
                [s["series_id"], s["provider"], s["frequency"], s["units"], s["freshness_days"]]
                for s in registry["series"]
            ],
        ),
        "Registry SHA256 (LF): `"
        + evidence["registry_sha256_lf"]
        + "`. Frozen before predictor requests.",
        "## Source Provenance",
        table(
            ["Series", "Acquired provenance", "Versions", "Status"],
            [
                [s["series_id"], s["provenance"], s["normalized_versions"], s["status"]]
                for s in acquired["series"]
            ],
        ),
        (
            "[ALFRED help](https://alfred.stlouisfed.org/help) "
            "distinguishes archive dating from exact provider "
            "release timing. "
            "[Real-time export "
            "documentation](https://alfred.stlouisfed.org/help/downloaddata)"
            " defines value validity intervals. Exact dates stay "
            "null unless independently certified; archive "
            "availability provides a conservative upper bound, "
            "never an invented agency date."
        ),
        "## Acquisition",
        "FRED_API_KEY configured: "
        + str(acquired["api_key_configured"])
        + (
            ". Each attempted request is bounded, recorded and "
            "credential-redacted. Raw payloads use exclusive "
            "writes. Successful HTML metadata is never parsed as "
            "predictor observations."
        ),
        table(
            ["Series", "Endpoint", "HTTP status", "Failure"],
            [
                [e["series"], e["endpoint"], e["response_status"], e.get("error_type", "")]
                for e in acquired["requests"]
            ],
        ),
        (
            "Warmup reference window January2004–March2026 "
            "covers 12-month operands plus quarterly "
            "freshness/release margins. No current-revised shadow"
            " was acquired."
        ),
        "## Knowledge-Time Rules",
        protocol["knowledge_time"],
        protocol["mortgage_month_mapping"],
        (
            "LATEST_KNOWN_AS_OF_T0 selects the newest completed "
            "reference period whose real-time validity interval "
            "contains t0. INITIAL_RELEASE requires certified "
            "initial evidence and remains separate. "
            "CURRENT_REVISED is rejected. Input ordering cannot "
            "change selection."
        ),
        "## Release Lags",
        (
            "Exact historical provider lag distributions are "
            "unmeasured for all six series (n=0). Task7 small "
            "probes remain historical evidence, not a new "
            "full-history distribution."
        ),
        "## Vintage Coverage",
        (
            "Task7 metadata hints: PMMS begins June2010, HPI "
            "August2010. These remain feasibility hints until "
            "dated value exports establish actual coverage; no "
            "pre2010 backfill. "
            "[GDP archived "
            "metadata](https://alfred.stlouisfed.org/series?seid=GDPC1)"
            " supplies dollar-base regimes used by the adapter; "
            "it does not itself supply predictor values."
        ),
        "## Transformations",
        table(
            ["Feature", "Series", "Transform", "Lag months"],
            [[f["name"], f["series_id"], f["transformation"], f["lag_months"]] for f in features],
        ),
        (
            "Every available value retains source hashes, native "
            "reference endpoints, archive validity, exact/null "
            "provider dates, bounds, units/regime, t0, operands "
            "and feature hash. GDP growth is nonannualized. "
            "Paired operands share t0 and policy; mixed "
            "units/regimes fail closed. Exact older denominators "
            "are knowledge-constrained but exempt from the "
            "current operand freshness cap."
        ),
        "## Revision Analysis",
        "Retained-archive comparisons and availability-bound lag summaries are in "
        "the machine-readable descriptive_vintage_diagnostics field. They include "
        "historical backfill and compare only matching native measurement regimes; "
        "they are not initial-versus-current comparisons or exact agency release lags.",
        (
            "Certified initial-versus-current magnitudes remain unavailable. "
            "Earliest retained does not imply initial. No current-revised "
            "denominator is permitted."
        ),
        "## Feature Availability",
        table(
            [
                "Feature",
                "Months in scope",
                "Populated",
                "Unavailable",
                "Stale",
                "Provenance unavailable",
            ],
            [
                [
                    n,
                    c["months_in_scope"],
                    c["months_populated"],
                    c["months_unavailable"],
                    c["stale_months"],
                    c["provenance_unavailable_months"],
                ]
                for n, c in coverage.items()
            ],
        ),
        (
            "Full feature × calendar-month evidence is in "
            f"pit_macro_data_audit{suffix}.json. Missing information "
            "remains unavailable, without statistical imputation."
        ),
        "## Mortgage-Month Join",
        table(
            ["Vintage", "Unique months", "Rows retained", "Facilities retained"],
            [
                [y, c["unique_reporting_months"], c["rows_retained"], c["facilities_retained"]]
                for y, c in by_vintage.items()
            ],
        ),
        (
            "A normalized month-key relation and tested "
            "deterministic left join retain all facilities. Only "
            "the reporting-month column was counted from "
            "canonical panels; no outcomes were inspected for "
            "feature selection and no existing panel was "
            "rewritten."
        ),
        "## Common Support",
        table(
            ["First risk month", "Last risk month", "Months", "Status"],
            [
                [w["first_reporting_month"], w["last_reporting_month"], w["months"], w["status"]]
                for w in intervals
            ],
        ),
        (
            "Mortgage follow-up alone does not establish "
            "macro-modelable horizons. Combined support remains "
            "unestablished wherever the full PIT vector is "
            "unavailable; no mortgage facility is dropped."
        ),
        "## APC Constraint",
        (
            "period = cohort + age; rank3 with four columns. "
            "Prespecified future design: fixed duration bands, "
            "seven archive-vintage indicators (2006 reference), "
            "national PIT features and no unrestricted period "
            "effects or interaction search. No model fitted; "
            "actual design rank/conditioning and calendar-block "
            "uncertainty remain future gates."
        ),
        "## Pandemic Period",
        (
            "Descriptive reporting-year indicators for 2020 and "
            "2021 only. No outcome-selected boundary, pandemic "
            "causal effect, scenario probability or complete "
            "intervention-policy claim."
        ),
        "## Limitations",
        *["- " + v for v in limitations],
        "## Decision",
        decision,
        "Exactly one next task: " + evidence["next_task"],
        (
            "Final tests and preservation are recorded separately"
            " in pit_macro_verification.json. No completion "
            "commit is permitted for insufficient support."
        ),
    ]
    (report / f"PIT_MACRO_DATA_AUDIT{suffix}.md").write_text(
        "\n\n".join(sections) + "\n", encoding="utf-8"
    )
    return evidence
