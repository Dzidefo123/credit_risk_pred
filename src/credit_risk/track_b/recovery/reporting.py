"""Separate recovery evidence; never rewrites original STOP or policy-design artifacts."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from credit_risk.track_b.multivintage.core import age_band, apc_diagnostic, set_hash, write_json
from credit_risk.track_b.multivintage.reporting import (
    LIMITATIONS,
    missing_rates,
    ready,
    schema_registry,
    table,
)
from credit_risk.track_b.multivintage.study import lf_hash


def canonical_support(path):
    """Necessary new support summaries from canonical CSVs, never source archive rescans."""
    clocks = set()
    active_facilities = defaultdict(set)
    exact24 = Counter()
    active_rows = Counter()
    flags = Counter()
    proxy_support = Counter()
    original = {}
    with (path / "origination.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            original[row["loan_id"]] = float(row["orig_upb"]) if row["orig_upb"] else None
    with (path / "monthly.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            principal = (
                float(row["current_principal_balance"])
                if row["current_principal_balance"]
                else None
            )
            upb = original[row["loan_id"]]
            if principal is not None and upb is not None and principal > upb:
                flags["current_upb_above_original_rows"] += 1
            if (
                principal is not None
                and row["non_interest_upb"]
                and float(row["non_interest_upb"]) > principal
            ):
                flags["non_interest_upb_above_current_rows"] += 1
            if row["analytical_prefix"] != "True":
                continue
            year, month = map(int, row["reporting_month"].split("-"))
            period = year * 12 + month - 1
            age_proxy = int(row["months_since_first_payment_proxy"])
            clocks.add((period, period - age_proxy, age_proxy))
            if age_proxy >= 0:
                proxy_support[(str(year), age_band(age_proxy))] += 1
            if row["loan_age"] and float(row["loan_age"]) == 24:
                exact24[str(year)] += 1
            if row["event_category"] == "none":
                active_rows[str(year)] += 1
                if year in {2020, 2021}:
                    active_facilities[str(year)].add(row["loan_id"])
    return dict(
        clocks=clocks,
        exact24=dict(sorted(exact24.items())),
        active_rows=dict(sorted(active_rows.items())),
        pandemic_active_facilities={str(y): len(active_facilities[str(y)]) for y in [2020, 2021]},
        balance_flags=dict(flags),
        first_payment_proxy_age_calendar_support=[
            dict(calendar_year=y, proxy_age_band=a, rows=n)
            for (y, a), n in sorted(proxy_support.items())
        ],
    )


def build(root):
    root = Path(root)
    private = root / "data/track_b/multivintage/recovery_v1"
    out, doc = root / "reports/track_b", root / "docs/track_b"
    plan = json.loads((doc / "multi_vintage_recovery_protocol.json").read_text(encoding="utf-8"))
    results = json.loads((private / "results.json").read_text(encoding="utf-8"))
    mapping = {r["vintage"]: r for r in results}
    vintages, manifests, quarantine = [], [], []
    combined_ids = []
    for y in plan["vintages"]:
        path = (
            root / f"data/track_b/multivintage/processed/v1/{y}"
            if y in plan["reused_vintages"]
            else private / str(y)
        )
        result = mapping.get(y, dict(vintage=y, status="NOT_EXECUTED_AFTER_STOP"))
        frozen = path / "sample_manifest.json"
        manifest = json.loads(frozen.read_text(encoding="utf-8")) if frozen.exists() else None
        if manifest:
            ids_path = (
                root / "data/track_b/manifests/expansion_v1/selected_ids.txt"
                if y == 2010
                else path / "selected_ids.txt"
            )
            ids = ids_path.read_text(encoding="utf-8").splitlines()
            if (
                len(ids) != 20000
                or len(set(ids)) != 20000
                or set_hash(ids) != manifest["sample_sha256"]
            ):
                raise ValueError("Recovery combined sample identity mismatch")
            combined_ids.extend(f"{y}:{loan}" for loan in ids)
            manifests.append(
                dict(
                    vintage=y,
                    source_sha256=manifest["source_sha256"],
                    eligible_universe_count=manifest.get("universe"),
                    eligible_universe_status="full gate certified"
                    if manifest.get("universe")
                    else "inherited 2010 sample; full universe not reconstructed",
                    quarantine_count=manifest.get("quarantine_count", 0),
                    sample_size=20000,
                    sample_sha256=manifest["sample_sha256"],
                    parser_version=manifest["parser_version"],
                    harmonization_version=plan["eligibility_version"],
                    source_protocol_sha256=manifest["protocol_sha256"],
                    performance_row_count=result.get("rows"),
                    calendar=result.get("calendar"),
                    followup=result.get("observed_month_span"),
                    event_counts=result.get("first_endpoints"),
                    harmonization_status=result["status"],
                )
            )
        if (path / "quarantine.json").exists():
            quarantine.extend(json.loads((path / "quarantine.json").read_text(encoding="utf-8")))
        result["sample_manifest"] = manifest
        vintages.append(result)
    complete = [r for r in vintages if ready(r)]
    all_complete = len(complete) == 7
    stopped = [
        r for r in vintages if r.get("status") in {"DATA_QUALITY_STOP", "SCHEMA_INCOMPATIBLE"}
    ]
    decision = (
        "STOP — HARMONIZATION INVALID"
        if stopped
        else "MULTI-VINTAGE COHORT READY WITH MATERIAL LIMITATIONS"
        if all_complete
        else "MULTI-VINTAGE SUPPORT INSUFFICIENT"
    )
    registry, comparable = schema_registry(vintages)
    registry["eligibility_version"] = plan["eligibility_version"]
    write_json(doc / "multi_vintage_recovery_schema_registry.json", registry)
    write_json(doc / "multi_vintage_recovery_field_comparability.json", comparable)
    (doc / "MULTI_VINTAGE_RECOVERY_FIELD_COMPARABILITY.md").write_text(
        "# Task 8C field comparability — historical matrices preserved\n\n"
        + table(
            ["Field", *plan["vintages"]],
            [
                [f["field"], *[f["vintages"][str(y)]["status"] for y in plan["vintages"]]]
                for f in comparable["fields"]
            ],
        )
        + (
            "\nCurrent-release positional compatibility is not a historical-release at"
            "testation. PARTIAL on incomplete vintages means validation unfinished; n"
            "o equivalence or structural absence is inferred. Source-level quarantine"
            " does not change selected-field definitions.\n"
        ),
        encoding="utf-8",
    )
    write_json(
        out / "multi_vintage_quarantine_ledger.json",
        dict(
            policy_version="1.1.0",
            records=quarantine,
            source_semantics="UNRESOLVED",
            raw_sources_preserved=True,
            licensed_identifiers_public=False,
        ),
    )
    combined = dict(
        complete=all_complete,
        components=manifests,
        frozen_facilities=sum(r["sample_size"] for r in manifests),
        combined_sample_sha256=set_hash(combined_ids) if combined_ids else None,
        key_definition="archive_vintage:immutable source facility ID; no borrower uniqueness claim",
        quarantine_policy_sha256_lf=lf_hash(doc / "historical_source_anomaly_amendment.json"),
        protocol_sha256_lf=lf_hash(doc / "multi_vintage_recovery_protocol.json"),
        schema_sha256_lf=lf_hash(doc / "multi_vintage_recovery_schema_registry.json"),
    )
    manifest_path = out / "multi_vintage_recovery_dataset_manifest.json"
    write_json(manifest_path, combined)
    matrix = defaultdict(lambda: defaultdict(dict))
    proxy_matrix = defaultdict(lambda: defaultdict(dict))
    clocks = set()
    pandemic = {str(y): {} for y in [2020, 2021]}
    for r in complete:
        y = r["vintage"]
        r["missingness_rates"] = missing_rates(r)
        for field in r["missingness_rates"].values():
            field["row_state_counts"]["SOURCE_QUARANTINED"] = 0
        r["source_quarantined_records"] = sum(q["vintage"] == y for q in quarantine)
        for item in r["age_calendar_support"]:
            matrix[item["calendar_year"]][item["provider_age_band"]][str(y)] = item["rows"]
        path = (
            root / f"data/track_b/multivintage/processed/v1/{y}"
            if y in plan["reused_vintages"]
            else private / str(y)
        )
        supplement = canonical_support(path)
        clocks.update(supplement.pop("clocks"))
        for item in supplement["first_payment_proxy_age_calendar_support"]:
            proxy_matrix[item["calendar_year"]][item["proxy_age_band"]][str(y)] = item["rows"]
        r["supplemental_support"] = supplement
        for year in pandemic:
            pandemic[year][str(y)] = dict(
                active_facilities=supplement["pandemic_active_facilities"][year],
                active_rows=supplement["active_rows"].get(year, 0),
            )
    proxy = np.asarray(sorted(clocks), float)
    apc = dict(
        synthetic=apc_diagnostic([y * 12 for y in plan["vintages"]], [0, 12, 24, 60, 120]),
        empirical_proxy=dict(
            unique_clock_rows=len(clocks),
            columns=4,
            rank=int(np.linalg.matrix_rank(np.column_stack([np.ones(len(proxy)), proxy])))
            if len(proxy)
            else None,
            maximum_identity_residual=float(np.abs(proxy[:, 0] - proxy[:, 1] - proxy[:, 2]).max())
            if len(proxy)
            else None,
            scope="First-payment proxy clocks; not exact origination date or provider age",
        ),
        structural_identity="period = cohort + age",
        unrestricted_apc_identified=False,
        causal_macro_effects_identified=False,
        future_constraints=plan["future_constraints"],
    )
    bands = [f"<= {n}" for n in [12, 24, 36, 60, 84, 120, 180, 240]] + [">240"]
    occupied = sum(len(v) for ages in matrix.values() for v in ages.values())
    grid = len(matrix) * len(bands) * len(complete)
    overlap = dict(
        matrix=dict(matrix),
        scope=(
            "Inherited analytical-prefix support includes terminal/censor boundary ro"
            "ws; not pure active exposure"
        ),
        calendar_year_vintage_counts={
            year: len({y for v in ages.values() for y in v}) for year, ages in matrix.items()
        },
        calendar_years_by_age_band={
            a: sorted(year for year, ages in matrix.items() if a in ages) for a in bands
        },
        occupied_age_year_vintage_cells=occupied,
        possible_rectangular_cells=grid,
        rectangular_density=occupied / grid if grid else None,
        density_interpretation=(
            "Descriptive; includes structurally infeasible grid cells, not a scientif"
            "ic support threshold"
        ),
    )
    overlap["calendar_years_with_multiple_vintages"] = [
        year for year, count in overlap["calendar_year_vintage_counts"].items() if count > 1
    ]
    overlap["assessment"] = (
        "Completed seven-vintage support can inform the preferred constrained predictive design; "
        "regime coverage, macro availability, censoring and evaluation gates remain required. "
        "Multiple occupied cells do not establish unrestricted APC identification."
        if all_complete
        else "Partial support only; seven-vintage feasibility awaits completed recovery."
    )
    for year, values in pandemic.items():
        pandemic[year] = dict(
            vintages=values,
            contributing_vintages=sum(v["active_facilities"] > 0 for v in values.values()),
            definition=(
                "analytical_prefix=True and event_category=none; distinct facilities with"
                "in vintage/year"
            ),
        )
    horizons = {
        str(r["vintage"]): r.get("horizons", {str(h): dict(status=None) for h in plan["horizons"]})
        for r in vintages
    }
    write_json(out / "multi_vintage_recovery_horizon_support.json", horizons)
    limitations = [
        *LIMITATIONS,
        (
            "Source semantics for A1-A4 remain unresolved; exact-record eligibility e"
            "xclusion does not establish provider error or harmlessness."
        ),
        (
            "Inherited 2010 frozen sample has no reconstructed full eligible-universe"
            " count; do not present identifier census as full eligibility certificati"
            "on."
        ),
        (
            "Source-wide structural performance column checks; semantic typing covers"
            " selected histories, not every unselected performance record."
        ),
        (
            "Provider-reported age diverges from the immutable first-payment elapsed clock "
            "on some histories. Both support matrices are retained separately; no reset "
            "mechanism is inferred and no provider age is rewritten. Broad provider-age "
            "coverage does not establish original-duration overlap."
        ),
    ]
    if not all_complete:
        limitations.append(
            "Prespecified seven-vintage cohort is incomplete; frozen IDs do not certi"
            "fy missing performance evidence."
        )
    evidence = dict(
        decision=decision,
        chronology=[
            "Original Task8 STOP",
            "Task8A SOURCE CONVENTIONS UNRESOLVED",
            "Task8B robustness design",
            "Task8C Policy B authorized, followed by gated recovery",
        ],
        protocol=plan,
        quarantine=quarantine,
        vintages=vintages,
        dataset_manifest=combined,
        dataset_manifest_sha256_lf=lf_hash(manifest_path),
        completed_facilities=sum(r["facilities"] for r in complete),
        completed_performance_rows=sum(r["rows"] for r in complete),
        horizon_support=horizons,
        age_period_support=overlap,
        first_payment_proxy_age_period_support=dict(
            matrix=dict(proxy_matrix),
            scope="Nonnegative elapsed months since immutable first payment; analytical-prefix "
            "rows including endpoint/censor boundaries, not exact origination age",
            calendar_years_by_age_band={
                a: sorted(year for year, ages in proxy_matrix.items() if a in ages) for a in bands
            },
        ),
        apc=apc,
        pandemic_support=pandemic,
        limitations=limitations,
        all_seven_complete=all_complete,
        next_task=(
            "Point-in-time macro acquisition and feature engineering, constrained to "
            "demonstrated periods/horizons and APC restrictions"
        )
        if all_complete
        else (
            "Investigate and resolve the newly observed source blocker through a sepa"
            "rately versioned amendment before further recovery"
        ),
        models_fitted=False,
        macro_acquired=False,
        new_performance_access_after_sample_freeze=True,
        performance_accessed_vintages=[
            y
            for y in plan["recovered_vintages"]
            if (private / str(y) / "performance.sqlite").exists()
        ],
        original_reports_rewritten=False,
    )
    write_json(out / "multi_vintage_recovery_audit.json", evidence)
    sections = [
        "# Track B Task 8C — multi-vintage cohort recovery",
        decision,
        (
            "Original Task 8 STOP, Task 8A unresolved audit and Task 8B design remain"
            " unchanged. Policy B changes research eligibility only. Four exact sourc"
            "e observations remain raw and unresolved; no token is coerced or identif"
            "ier/cohort rewritten."
        ),
        "## Gate and sample evidence",
        table(
            ["Vintage", "Status", "Eligible universe", "Quarantine", "Sample SHA256"],
            [
                [
                    r["vintage"],
                    r["status"],
                    (r["sample_manifest"] or {}).get(
                        "universe",
                        "Not reconstructed; inherited frozen sample"
                        if r["sample_manifest"]
                        else "Not certified / sample not frozen",
                    ),
                    (r["sample_manifest"] or {}).get("quarantine_count", 0),
                    (r["sample_manifest"] or {}).get("sample_sha256", "Not frozen"),
                ]
                for r in vintages
            ],
        ),
        "Frozen samples are not completed cohorts. Combined frozen facilities: "
        + str(combined["frozen_facilities"])
        + "; completed audited facilities: "
        + str(evidence["completed_facilities"])
        + "; completed monthly rows: "
        + str(evidence["completed_performance_rows"])
        + ". No borrower uniqueness claimed.",
        "## Fail-closed findings",
        *[json.dumps(r, ensure_ascii=False) for r in stopped],
        "## Events",
        table(
            [
                "Vintage",
                "Default",
                "Payoff/maturity",
                "Administrative",
                "Ambiguous",
                "Active/unknown",
            ],
            [
                [
                    r["vintage"],
                    *[
                        r["first_endpoints"].get(k, 0)
                        for k in [
                            "default",
                            "payoff",
                            "administrative",
                            "ambiguous",
                            "active_or_unknown",
                        ]
                    ],
                ]
                for r in complete
            ],
        ),
        (
            "Descriptive first observed raw endpoints under frozen Task2 definitions;"
            " not estimated incidence or model performance."
        ),
        "## Follow-up and calendar",
        table(
            [
                "Vintage",
                "Raw span min/median/max",
                "Analytical follow-up min/median/max",
                "Calendar first/last",
            ],
            [
                [
                    r["vintage"],
                    *[
                        "/".join(str(r[k][q]) for q in ["minimum", "median", "maximum"])
                        for k in ["observed_month_span", "contiguous_analytical_followup"]
                    ],
                    str(r["calendar"]["first"]) + " / " + str(r["calendar"]["last"]),
                ]
                for r in complete
            ],
        ),
        (
            "Natural intervals from first observation, with right censoring; no commo"
            "n truncation or inferred daily origination date."
        ),
        "## Horizon support",
        table(
            ["Vintage", *plan["horizons"]],
            [
                [
                    r["vintage"],
                    *[
                        horizons[str(r["vintage"])][str(h)]["status"] or "Unassessed"
                        for h in plan["horizons"]
                    ],
                ]
                for r in vintages
            ],
        ),
        (
            "Unchanged thresholds: SUPPORTED requires >=1000 at risk and >=10000 know"
            "n status; LIMITED >=100 and >=2000; otherwise UNSUPPORTED. Earlier defau"
            "lt/payoff outcomes can be known; unknown/gap/administrative censoring is"
            " not filled."
        ),
        "## Missingness, distributions and release mapping",
        (
            "Machine evidence includes original score/LTV/DTI/UPB/rate/term quantiles"
            ", purpose/occupancy distributions and field missingness for completed co"
            "horts only. SOURCE_QUARANTINED is counted separately at the source-recor"
            "d level; selected-field SOURCE_QUARANTINED counts are zero because exclu"
            "sions precede selection. Entirely blank fields remain MISSING_IN_SOURCE,"
            " never presumed STRUCTURALLY_UNAVAILABLE. No feature selection or modeli"
            "ng."
        ),
        "## Age-period and pandemic support",
        table(
            ["Calendar year", "Contributing vintages"],
            [[year, count] for year, count in overlap["calendar_year_vintage_counts"].items()],
        ),
        table(
            ["Provider age band", "Calendar years represented", "First/last year"],
            [
                [band, len(years), " / ".join([years[0], years[-1]]) if years else "None"]
                for band, years in overlap["calendar_years_by_age_band"].items()
            ],
        ),
        "Full age × year × vintage counts and overlap density are in the companion JSON. "
        "These inherited prefix counts include terminal/censor boundaries. "
        + overlap["assessment"],
        "Provider age and the first-payment elapsed clock are not interchangeable. "
        "A separate full proxy-clock matrix and exact provider-age-24 counts appear in JSON. "
        "No age reset mechanism is inferred or source age corrected.",
        table(
            ["Vintage", "2020 active facilities", "2021 active facilities"],
            [
                [
                    r["vintage"],
                    *[
                        pandemic[str(year)]["vintages"][str(r["vintage"])]["active_facilities"]
                        for year in [2020, 2021]
                    ],
                ]
                for r in complete
            ],
        ),
        "Active pandemic counts require a known no-event state inside the analytical prefix; "
        "they exclude endpoints, unknowns and later post-endpoint rows.",
        (
            "Old source performance archives were not rescanned. Canonical completed "
            "CSVs were read for newly required active-pandemic facility counts and cl"
            "ock diagnostics; their bytes remain unchanged."
        ),
        "## APC diagnostics",
        json.dumps(apc, indent=2),
        (
            "Overlap supports assessment of a future constrained predictive design, n"
            "ot unrestricted APC or causal macro identification."
        ),
        "## Resources",
        table(
            ["Vintage", "Performance/audit seconds", "Peak process MiB", "Output MiB"],
            [
                [
                    r["vintage"],
                    round(r.get("seconds", 0), 1),
                    round((r.get("peak_memory_bytes") or 0) / 2**20, 1),
                    round(r.get("output_bytes", 0) / 2**20, 1),
                ]
                for r in complete
            ],
        ),
        (
            "Peak memory is process high-water. Stage timing excludes origination whe"
            "re documented; reused rows retain historical measurements. ZIP materiali"
            "zation is measured; SQLite temporary sort spill is not instrumented."
        ),
        "## Limitations",
        *["- " + v for v in limitations],
        "## Verification",
        (
            "Final tests and preservation are recorded separately in multi_vintage_re"
            "covery_verification.json. Historical AUC0.868152, Brier0.048545 and log "
            "loss0.176030 remain retained results; no holdout evaluation or artifact "
            "regeneration."
        ),
        "## Exactly one next task",
        evidence["next_task"],
    ]
    (out / "MULTI_VINTAGE_RECOVERY_AUDIT.md").write_text(
        "\n\n".join(sections) + "\n", encoding="utf-8"
    )
    return evidence
