"""Reproduce frozen macro support, then count existing mortgage intervals without fitting."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np

from credit_risk.track_b.data.schemas import digest
from credit_risk.track_b.macro.information import feature_hash
from credit_risk.track_b.multivintage.core import horizon_support, write_json
from credit_risk.track_b.multivintage.study import lf_hash
from credit_risk.track_b.pit_macro.acquisition import load_versions
from credit_risk.track_b.pit_macro.engine import (
    VERSION as PIT_VERSION,
)
from credit_risk.track_b.pit_macro.engine import (
    lag_period,
    month_table,
    period_end,
    previous_month_end,
    select,
)

from .eligibility import (
    REDUCED,
    VERSION,
    Reason,
    complete_months,
    duration_band,
    eligible_intervals,
    exit_reason,
    macro_reasons,
    ordinal,
    support_windows,
    validation_role,
)

FIELDS = [
    "loan_id",
    "reporting_month",
    "event_category",
    "analytical_prefix",
    "loan_age",
    "months_since_first_payment_proxy",
]
HORIZONS = [12, 24, 36, 60, 84, 120]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def immutable_json(path, value):
    if Path(path).exists():
        if feature_hash(read_json(path)) != feature_hash(value):
            raise ValueError("Frozen output changed; amendment required: " + str(path))
    else:
        write_json(path, value, True)


def macro_support(root):
    root = Path(root)
    acquisition, versions = load_versions(root, "task9_api_v5")
    features = read_json(root / "docs/track_b/pit_macro_feature_registry.json")["features"]
    registry = read_json(root / "docs/track_b/pit_macro_series_registry.json")
    protocol = read_json(root / "docs/track_b/pit_macro_research_protocol.json")
    retained = read_json(root / "data/track_b/macro/processed/task9_api_v5/macro_month_table.json")
    caps = {s["series_id"]: s["freshness_days"] for s in registry["series"]}
    reproduced = month_table(versions, [r["reporting_month"] for r in retained], features, caps)
    if feature_hash(retained) != feature_hash(reproduced):
        raise ValueError("Frozen Task9 month table did not reproduce")
    cutoff = protocol["reporting_month_window"][1]
    primary = [f["name"] for f in features]
    mapping = {r["reporting_month"]: r for r in retained}
    missing = defaultdict(list)
    for entry in acquisition["requests"]:
        if entry["endpoint"].endswith("/observations"):
            spec = next(s for s in registry["series"] if s["series_id"] == entry["series"])
            for row in read_json(root / entry["raw_path"])["observations"]:
                if row["value"] == ".":
                    missing[(entry["series"], period_end(row["date"], spec["frequency"]))].append(
                        dict(row, source_sha256=entry["content_sha256"])
                    )
    matrix = {}
    for month, row in mapping.items():
        t0 = previous_month_end(month)
        cell = {}
        for f in features:
            feature = row["features"][f["name"]]
            details = dict(
                status="AVAILABLE" if feature["status"] == "AVAILABLE" else "UNAVAILABLE",
                original_status=feature["status"],
                reasons=macro_reasons(month, mapping, [f["name"]], cutoff),
                t0=str(t0),
            )
            if feature["status"] == "MISSING_OPERAND" and f["lag_months"]:
                current = select(versions, f["series_id"], t0, cap=caps[f["series_id"]])
                if current:
                    required = lag_period(current.reference_period, f["lag_months"])
                    details.update(
                        current_reference_period=str(current.reference_period),
                        missing_operand_reference_period=str(required),
                    )
                    raw_missing = [
                        r
                        for r in missing[(f["series_id"], required)]
                        if r["realtime_start"] <= str(t0) <= r["realtime_end"]
                    ]
                    if raw_missing:
                        details.update(
                            missingness_class="SOURCE_OBSERVATION_MISSING",
                            source_missing_rows=raw_missing,
                        )
                        details["reasons"].append(Reason.MACRO_SOURCE_OBSERVATION_MISSING.value)
                    else:
                        details["missingness_class"] = "OPERAND_NOT_AVAILABLE_AT_T0"
            elif feature["status"] == "STALE":
                details["missingness_class"] = "FROZEN_FRESHNESS_CAP_EXCEEDED"
            elif feature["status"] == "UNAVAILABLE":
                details["missingness_class"] = "NO_DATED_NUMERIC_VERSION_KNOWN_AT_T0"
            cell[f["name"]] = details
        matrix[month] = cell
    definitions = []
    for f in features:
        available = [m for m, r in matrix.items() if r[f["name"]]["status"] == "AVAILABLE"]
        source = next(s for s in registry["series"] if s["series_id"] == f["series_id"])
        definitions.append(
            dict(
                **f,
                frequency=source["frequency"],
                provider=source["provider"],
                required_operands=(
                    [f["series_id"], f["secondary_series"]]
                    if f["secondary_series"]
                    else [f["series_id"] + ":current", f["series_id"] + f":lag{f['lag_months']}"]
                    if f["lag_months"]
                    else [f["series_id"] + ":current"]
                ),
                earliest_reporting_month=min(available) if available else None,
                latest_reporting_month=max(available) if available else None,
                earliest_t0=str(previous_month_end(min(available))) if available else None,
                latest_t0=str(previous_month_end(max(available))) if available else None,
                available_months=len(available),
            )
        )
    designs = {
        "PRIMARY": dict(features=primary, support_months=complete_months(retained, primary)),
        "REDUCED": dict(features=list(REDUCED), support_months=complete_months(retained, REDUCED)),
    }
    for design in designs.values():
        design["support_windows"] = support_windows(design["support_months"])
    return dict(
        source_run="task9_api_v5",
        versions=len(versions),
        matrix=matrix,
        definitions=definitions,
        designs=designs,
        reproduced_month_table_sha256=feature_hash(retained),
        archive_cutoff=protocol["real_time_request"][1],
        reporting_cutoff=cutoff,
        retrieval_times=sorted({r["retrieved_at"] for r in acquisition["requests"]}),
        bottleneck_counts={
            f["name"]: sum(r[f["name"]]["status"] != "AVAILABLE" for r in matrix.values())
            for f in features
        },
    ), mapping


def freeze_spec(root, support):
    root = Path(root)
    task9 = read_json(root / "docs/track_b/pit_macro_future_model_spec.json")
    value = dict(
        version=VERSION,
        task="9A",
        source_run="task9_api_v5",
        geography="US national",
        source_evidence_manifest="macro_support_preservation_manifest.json",
        task9_decision_preserved="PIT MACRO SUPPORT INSUFFICIENT for full frozen window",
        designs=support["designs"],
        reduced_rationale=(
            "Labor level/change, Treasury cost of funds, price inflation and real activity. "
            "Exclude only the three housing/mortgage-rate features with pre2010 archive gaps; "
            "retain all remaining prespecified transformations. Not performance-selected."
        ),
        mortgage_interval=dict(
            current_month="m-1",
            target_month="m",
            minimum_known_pre_t0_months=6,
            current_state="none in unchanged analytical_prefix",
            target_states=["none", "default", "payoff"],
            consecutive_month_required=True,
            no_reentry_after_mortgage_exit=True,
            negative_first_payment_proxy_intervals_eligible=False,
            event_definitions_source="mortgage_research_protocol.json",
            event_protocol_sha256_lf=lf_hash(root / "docs/track_b/mortgage_research_protocol.json"),
            event_count_unit="unique facility per design, no overlapping-landmark count",
            eligibility_unit="monthly risk interval; sample membership never changed",
        ),
        duration=dict(
            clock=(
                "Completed months since scheduled first-payment proxy at t0; "
                "not exact origination age"
            ),
            delayed_entry=(
                "Enter only while active; condition on survival into support, no pre-entry exposure"
            ),
            bounds=task9["duration_bands_months"],
            top_band="241+",
            descriptive_provider_age_reported_separately=True,
        ),
        cohort=dict(
            vintages=task9["vintages"], reference=2006, representation="vintage indicators"
        ),
        period=dict(unrestricted_fixed_effects=False, structural_trend=False, interactions=[]),
        study_end=dict(
            archive_cutoff=support["archive_cutoff"],
            reporting_cutoff=support["reporting_cutoff"],
            frozen_retrieval_times=support["retrieval_times"],
            evidence_updates_permitted=False,
            latest_primary_month=support["designs"]["PRIMARY"]["support_months"][-1],
            future_macro_path_required_for_observed_interval=False,
            prospective_forecast_requires_separate_scenario_paths=True,
        ),
        reason_codes=[r.value for r in Reason],
        transformations_and_selector=dict(
            version=PIT_VERSION,
            source="src/credit_risk/track_b/pit_macro/engine.py",
            sha256_lf=lf_hash(root / "src/credit_risk/track_b/pit_macro/engine.py"),
            current_revised_backfill=False,
            imputation=False,
            relaxed_staleness=False,
        ),
        candidate_validation=dict(
            salt="track-b-macro-eligibility-v1",
            assignment="SHA256(salt:vintage:ID) modulo10; 0..6 development,7..9 evaluation",
            development=[support["designs"]["PRIMARY"]["support_months"][0], "2017-12"],
            purge=["2018-01", "2018-12"],
            temporal_evaluation=["2019-01", "2026-02"],
            reduced_development=[support["designs"]["REDUCED"]["support_months"][0], "2017-12"],
            minimum_unique_defaults=100,
            minimum_unique_payoffs=500,
            dates_chosen_before_task9a_event_counts=True,
            no_row_random_split=True,
            leave_vintage_out=(
                "Prespecified Task10 sensitivity across seven frozen vintages; "
                "event feasibility by vintage reported here"
            ),
            uncertainty="Facility and calendar-block uncertainty; borrower clustering unavailable",
            finalization=(
                "Dates and hash roles frozen here; Task10 must seal a new ledger "
                "and exact modeling protocol before fitting"
            ),
            prior_outcomes_inspected=True,
            virgin_holdout_claim=False,
        ),
        causal_interpretation=False,
        models_fitted=False,
        primary_vs_reduced=(
            "At most one of each; reduced is sensitivity, never promoted by performance"
        ),
    )
    path = root / "docs/track_b/macro_support_eligibility_spec.json"
    immutable_json(path, value)
    return value


def histories(path):
    """Read only six projected fields; licensed rows stay in their existing files."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = [header.index(n) for n in FIELDS]
        projected = (dict(zip(FIELDS, (row[i] for i in indices), strict=True)) for row in reader)
        previous = None
        for loan, records in groupby(projected, key=lambda r: r["loan_id"]):
            if previous is not None and loan <= previous:
                raise ValueError("Canonical facility ordering changed")
            rows = []
            for row in records:
                if row["analytical_prefix"] not in {"True", "False"}:
                    raise ValueError("Unknown prefix flag")
                row["analytical_prefix"] = row["analytical_prefix"] == "True"
                row["loan_age"] = float(row["loan_age"]) if row["loan_age"] else None
                row["months_since_first_payment_proxy"] = int(
                    row["months_since_first_payment_proxy"]
                )
                rows.append(row)
            previous = loan
            yield loan, rows


def aggregate(root, support, mapping, spec, directory="v1"):
    root = Path(root)
    mortgage = read_json(root / "reports/track_b/multi_vintage_recovery_audit.json")
    aggregate_results = {name: {} for name in support["designs"]}
    calendars = {name: defaultdict(lambda: Counter()) for name in support["designs"]}
    facility_sets = {name: defaultdict(set) for name in support["designs"]}
    vintage_sets = {name: defaultdict(set) for name in support["designs"]}
    apc_clocks = {name: set() for name in support["designs"]}
    folds = {
        name: {"development": Counter(), "temporal_evaluation": Counter()}
        for name in support["designs"]
    }
    fingerprints = {name: hashlib.sha256() for name in support["designs"]}
    source_hashes = {}
    for vintage in mortgage["vintages"]:
        year = vintage["vintage"]
        zone = "processed/v1" if year in [2010, 2014, 2018] else "recovery_v1"
        path = root / f"data/track_b/multivintage/{zone}/{year}/monthly.csv"
        source_hashes[str(year)] = digest(path)
        counts = {name: Counter() for name in support["designs"]}
        bands = {name: Counter() for name in support["designs"]}
        provider_bands = {name: Counter() for name in support["designs"]}
        exits = {name: Counter() for name in support["designs"]}
        horizons = {name: {str(h): Counter() for h in HORIZONS} for name in support["designs"]}
        eligible_dates = {name: [] for name in support["designs"]}
        files = {}
        try:
            for name in support["designs"]:
                private = (
                    root / f"data/track_b/macro_support/{directory}/"
                    f"eligibility_{name.lower()}_{year}.jsonl"
                )
                private.parent.mkdir(parents=True, exist_ok=True)
                files[name] = private.open("x", encoding="utf-8", newline="\n")
            source_rows = source_facilities = 0
            for loan, history in histories(path):
                source_facilities += 1
                source_rows += len(history)
                for name, design in support["designs"].items():
                    risk = eligible_intervals(
                        history, mapping, design["features"], support["reporting_cutoff"]
                    )
                    role = validation_role(year, loan, spec["candidate_validation"]["salt"])
                    subject = dict(
                        vintage=year,
                        facility=loan,
                        contributes=bool(risk),
                        role=role,
                        intervals=len(risk),
                        first_target=risk[0]["target_month"] if risk else None,
                        last_target=risk[-1]["target_month"] if risk else None,
                        exit_reason=exit_reason(
                            history, risk, design["features"], mapping, support["reporting_cutoff"]
                        ),
                    )
                    files[name].write(json.dumps(subject, sort_keys=True) + "\n")
                    counts[name]["sample_facilities_retained"] += 1
                    if not risk:
                        counts[name]["no_eligible_intervals"] += 1
                        continue
                    counts[name]["contributing_facilities"] += 1
                    counts[name]["eligible_intervals"] += len(risk)
                    exits[name][subject["exit_reason"]] += 1
                    counts[name]["unique_default_facilities"] += risk[-1]["event"] == "default"
                    counts[name]["unique_payoff_facilities"] += risk[-1]["event"] == "payoff"
                    counts[name]["censored_facilities"] += risk[-1]["event"] == "none"
                    eligible_dates[name].extend([risk[0]["target_month"], risk[-1]["target_month"]])
                    entry = ordinal(risk[0]["t0"])
                    followup = ordinal(risk[-1]["target_month"]) - entry
                    for h in HORIZONS:
                        horizons[name][str(h)]["at_risk"] += followup >= h
                        horizons[name][str(h)]["known_status"] += followup >= h or (
                            risk[-1]["event"] != "none" and followup <= h
                        )
                    validation = spec["candidate_validation"]
                    lower, upper = (
                        (
                            validation["development"]
                            if name == "PRIMARY"
                            else validation["reduced_development"]
                        )
                        if role == "development"
                        else validation["temporal_evaluation"]
                    )
                    fold_rows = [r for r in risk if lower <= r["target_month"] <= upper]
                    folds[name][role]["facilities"] += bool(fold_rows)
                    folds[name][role]["intervals"] += len(fold_rows)
                    folds[name][role]["defaults"] += any(r["event"] == "default" for r in fold_rows)
                    folds[name][role]["payoffs"] += any(r["event"] == "payoff" for r in fold_rows)
                    for row in risk:
                        month = row["target_month"]
                        yr = month[:4]
                        quarter = f"{yr}Q{(int(month[5:]) - 1) // 3 + 1}"
                        for period in [yr, quarter]:
                            calendars[name][period]["intervals"] += 1
                            calendars[name][period]["defaults"] += row["event"] == "default"
                            calendars[name][period]["payoffs"] += row["event"] == "payoff"
                            facility_sets[name][period].add((year, loan))
                            vintage_sets[name][period].add(year)
                        bands[name][duration_band(row["duration"])] += 1
                        provider_bands[name][duration_band(row["provider_age"])] += 1
                        period_clock = ordinal(row["t0"])
                        apc_clocks[name].add(
                            (period_clock, period_clock - row["duration"], row["duration"])
                        )
                        fingerprints[name].update(f"{year}:{loan}:{row['t0']}:{month}\n".encode())
            if source_facilities != 20000 or source_rows != vintage["rows"]:
                raise ValueError("Frozen source cardinality changed")
            for name in support["designs"]:
                observed = horizons[name]
                for counter in observed.values():
                    counter["status"] = horizon_support(
                        counter["at_risk"],
                        counter["known_status"],
                        mortgage["protocol"]["horizon_support_rule"],
                    )
                aggregate_results[name][str(year)] = dict(
                    counts=dict(counts[name]),
                    exits=dict(exits[name]),
                    first_month=min(eligible_dates[name]) if eligible_dates[name] else None,
                    last_month=max(eligible_dates[name]) if eligible_dates[name] else None,
                    duration_bands=dict(bands[name]),
                    provider_age_bands=dict(provider_bands[name]),
                    horizons={
                        h: dict(
                            **counter,
                            mortgage_status=vintage["horizons"][h]["status"],
                            combined_status="MORTGAGE_UNSUPPORTED"
                            if vintage["horizons"][h]["status"] != "SUPPORTED"
                            else "MORTGAGE_SUPPORTED_MACRO_SUPPORTED"
                            if counter["status"] == "SUPPORTED"
                            else "MORTGAGE_SUPPORTED_MACRO_UNSUPPORTED",
                        )
                        for h, counter in observed.items()
                    },
                )
            print(
                f"{year}: descriptive eligibility audited; {source_rows:,} rows unchanged",
                flush=True,
            )
        finally:
            for stream in files.values():
                stream.close()
    calendar_results, apc = {}, {}
    for name in support["designs"]:
        calendar_results[name] = {
            period: dict(
                **counts,
                facilities=len(facility_sets[name][period]),
                contributing_vintages=sorted(vintage_sets[name][period]),
                vintage_count=len(vintage_sets[name][period]),
            )
            for period, counts in sorted(calendars[name].items())
        }
        clocks = np.asarray(sorted(apc_clocks[name]), dtype=float)
        residual = float(np.max(np.abs(clocks[:, 0] - clocks[:, 1] - clocks[:, 2])))
        clocks -= clocks.min(axis=0)
        # Centering each column by its own minimum can shift the identity's intercept.
        design = np.column_stack([np.ones(len(clocks)), clocks])
        apc[name] = dict(
            unique_clock_rows=len(clocks),
            columns=4,
            rank=int(np.linalg.matrix_rank(design)),
            max_identity_residual=residual,
            exact_identity="t0 month = scheduled first-payment month + proxy duration",
            unrestricted_apc_identified=False,
            causal_interpretation=False,
        )
    return dict(
        by_vintage=aggregate_results,
        calendar=calendar_results,
        apc=apc,
        validation_counts={n: {r: dict(c) for r, c in roles.items()} for n, roles in folds.items()},
        interval_key_sha256={n: h.hexdigest() for n, h in fingerprints.items()},
        mortgage_source_byte_hashes=source_hashes,
        horizons_origin=(
            "One conditional active entry per facility and design; "
            "NOT Task8 first-observation origin"
        ),
        horizon_interpretation=(
            "Observed monthly risk sets use macro known each interval. "
            "No future macro path required at entry. "
            "Tail support does not authorize prospective forecasts."
        ),
        no_model_fitting=True,
        no_predictions_created=True,
    )
