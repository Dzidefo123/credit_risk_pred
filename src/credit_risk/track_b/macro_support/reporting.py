"""Aggregate Task9A decision evidence; no licensed identifiers or estimates."""

from collections import Counter
from pathlib import Path

from credit_risk.track_b.macro.information import feature_hash
from credit_risk.track_b.multivintage.study import lf_hash

from .eligibility import VERSION
from .study import immutable_json, read_json
from .validation import finalize

READY = "MACRO ELIGIBILITY DESIGN READY WITH MATERIAL LIMITATIONS"
NEXT = (
    "Track B Task 10 — Prespecified Macro-Conditioned Competing-Risk Modeling "
    "and Temporal/Vintage Validation"
)
LIMITATIONS = [
    "Primary cannot evaluate 2007–2009 macro regimes; reduced is one fixed sensitivity only.",
    "Mortgage records are current-release retrospective disclosures; macro PIT does not certify "
    "historical mortgage operational knowledge time.",
    "Survival into the support window creates delayed entry and a conditional population; "
    "old-vintage survivors are not the full origination cohorts.",
    "Default is a monthly research proxy; payoff includes maturity. Unknown and administrative "
    "states censor before unascertainable exposure; censoring independence is not established.",
    "Scheduled first-payment proxy is not exact origination age; provider age remains separate.",
    "Facility disjointness is not borrower disjointness. Monthly rows and macro periods are "
    "dependent; prospective validation must account for facility/calendar clustering.",
    "Unrestricted age-period-cohort effects remain unidentified. Future coefficients would be "
    "predictive conditional associations, not causal economic effects.",
    "Prior outcomes have been inspected. Future temporal evaluation is prespecified reused "
    "research evidence, not a virgin holdout; Task10 needs a new sealed ledger before fitting.",
    "No complete authoritative pre2010 HPI/PMMS release chain admitted. Isolated historical "
    "HPI evidence does not extend frozen feature support.",
    "Observed interval support does not provide future macro scenario paths or authorize "
    "unsupported tail forecasts, IFRS9 staging or ECL claims.",
    "Earlier development cannot estimate later-vintage fixed effects. Primary temporal "
    "metrics must use seen vintages; unseen and held-out vintages require explicitly flagged "
    "reference-effect extrapolation sensitivity, never learned unseen-cohort effects.",
    "Mortgage rate, Treasury yield and their spread are algebraically redundant. All eight "
    "features remain required for eligibility; Task10 must prespecify an identifiable "
    "coefficient constraint/basis before fitting, with no unrestricted three-rate effects.",
]


def totals(counts, design):
    result = Counter()
    for vintage in counts["by_vintage"][design].values():
        result.update(vintage["counts"])
    return dict(result)


def make_report(root):
    root = Path(root)
    private = root / "data/track_b/macro_support"
    support = read_json(private / "support.json")
    counts = read_json(private / "counts.json")
    spec = read_json(root / "docs/track_b/macro_support_eligibility_spec.json")
    history = read_json(root / "docs/track_b/macro_historical_coverage_resolution.json")
    reproduction = read_json(private / "reproduction.json")
    preservation = read_json(private / "postcheck.json")
    before = read_json(private / "before_counts.json")
    for name, expected in before["public_lf_hashes"].items():
        if lf_hash(root / name) != expected:
            raise ValueError("Pre-count specification/source changed: " + name)
    if reproduction["counts_sha256"] != feature_hash(counts):
        raise ValueError("Reproduction does not match audit")
    validation = spec["candidate_validation"]
    final_validation = finalize(root)
    frozen_table = read_json(
        root / "data/track_b/macro/processed/task9_api_v5/macro_month_table.json"
    )
    rate_names = ["treasury_10y_level", "mortgage_30y_level", "mortgage_treasury_spread"]
    residuals = [
        abs(f[rate_names[2]]["value"] - f[rate_names[1]]["value"] + f[rate_names[0]]["value"])
        for row in frozen_table
        if all((f := row["features"])[n]["status"] == "AVAILABLE" for n in rate_names)
    ]
    macro_identification = dict(
        relation="mortgage_treasury_spread = mortgage_30y_level - treasury_10y_level",
        complete_rate_months=len(residuals),
        max_absolute_relation_residual=max(residuals),
        unrestricted_three_rate_coefficients_identified=False,
        all_eight_eligibility_features_retained=True,
        task10_identifiable_parameter_constraint_required_before_fit=True,
        no_model_fit=True,
    )
    feasible = all(
        c.get("defaults", 0) >= validation["minimum_unique_defaults"]
        and c.get("payoffs", 0) >= validation["minimum_unique_payoffs"]
        for c in counts["validation_counts"]["PRIMARY"].values()
    )
    all_vintages = all(
        v["counts"].get("contributing_facilities", 0) > 0
        for v in counts["by_vintage"]["PRIMARY"].values()
    )
    clocks_valid = all(
        a["rank"] == 3 and a["max_identity_residual"] == 0 for a in counts["apc"].values()
    )
    feasible = feasible and final_validation["temporal_seen_vintage_feasibility_passed"]
    decision = (
        READY
        if feasible and all_vintages and clocks_valid
        else "HISTORICAL MACRO COVERAGE REQUIRES FURTHER RESOLUTION"
    )
    if not clocks_valid:
        decision = "STOP — SUPPORT DESIGN INVALID"
    value = dict(
        version=VERSION,
        decision=decision,
        task9_boundary="PIT MACRO SUPPORT INSUFFICIENT for original full window; unchanged",
        source_evidence_sha256_lf={
            n: lf_hash(root / n)
            for n in [
                "reports/track_b/pit_macro_data_audit_task9_api_v5.json",
                "docs/track_b/pit_macro_feature_registry.json",
                "docs/track_b/pit_macro_series_registry.json",
                "docs/track_b/pit_macro_research_protocol.json",
                "docs/track_b/macro_historical_coverage_resolution.json",
                "docs/track_b/macro_support_preservation_manifest.json",
            ]
        },
        macro_support_matrix_sha256=feature_hash(support["matrix"]),
        macro_support=support,
        macro_coefficient_identification=macro_identification,
        mortgage_eligibility=counts,
        totals={n: totals(counts, n) for n in support["designs"]},
        historical_resolution=history,
        task10_specification_sha256_lf=lf_hash(
            root / "docs/track_b/macro_support_eligibility_spec.json"
        ),
        before_counts=before,
        finalized_validation_design=final_validation,
        finalized_validation_design_sha256_lf=lf_hash(
            root / "docs/track_b/macro_support_validation_design.json"
        ),
        validation_feasibility=dict(
            primary_pooled_blocks_pass=feasible,
            adopted_candidate_blocks_after_counts=feasible,
            candidate_blocks_changed=False,
            minimum_unique_defaults=validation["minimum_unique_defaults"],
            minimum_unique_payoffs=validation["minimum_unique_payoffs"],
            counts=counts["validation_counts"],
            all_seven_vintages_contribute=all_vintages,
            leave_vintage_out_cause_feasibility={
                n: {
                    y: dict(
                        default_count=v["counts"]["unique_default_facilities"],
                        payoff_count=v["counts"]["unique_payoff_facilities"],
                        minimum_default_count_met=v["counts"]["unique_default_facilities"] >= 100,
                        minimum_payoff_count_met=v["counts"]["unique_payoff_facilities"] >= 500,
                    )
                    for y, v in counts["by_vintage"][n].items()
                }
                for n in support["designs"]
            },
            future_ledger_required=True,
            models_fitted=False,
        ),
        reproduction=reproduction,
        preservation=preservation,
        limitations=LIMITATIONS,
        source_table_unaltered=True,
        future_eligible_interval_key_hashes=counts["interval_key_sha256"],
        next_task=NEXT
        if decision == READY
        else "Track B Task9A amendment — Resolve validation event support without model fitting",
        task10_implemented=False,
        predictions_created=False,
    )
    immutable_json(root / "reports/track_b/macro_support_eligibility.json", value)
    document = markdown(value, spec)
    path = root / "reports/track_b/MACRO_SUPPORT_ELIGIBILITY.md"
    if path.exists() and path.read_text(encoding="utf-8") != document:
        raise ValueError("Report amendment required")
    if not path.exists():
        path.write_text(document, encoding="utf-8", newline="\n")
    print(decision, flush=True)
    return value


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def design_summary(design):
    return "; ".join(
        f"{w['first']} through {w['last']} ({w['months']} months)"
        for w in design["support_windows"]
    )


def markdown(value, spec):
    support = value["macro_support"]
    counts = value["mortgage_eligibility"]
    parts = ["# Track B Task9A — Macro-Support Eligibility\n"]

    def section(title, content):
        parts.extend(["\n## " + title + "\n", "\n" + content + "\n"])

    section(
        "Executive Summary",
        value["decision"]
        + ".\n\n"
        + table(
            [
                "Design",
                "Calendar window",
                "Months",
                "Facilities",
                "Intervals",
                "Defaults",
                "Payoffs",
            ],
            [
                [
                    n,
                    design_summary(d),
                    len(d["support_months"]),
                    value["totals"][n]["contributing_facilities"],
                    value["totals"][n]["eligible_intervals"],
                    value["totals"][n]["unique_default_facilities"],
                    value["totals"][n]["unique_payoff_facilities"],
                ]
                for n, d in support["designs"].items()
            ],
        ),
    )
    section(
        "Task 9 Boundary",
        value["task9_boundary"] + ". Successful frozen acquisition "
        "and all earlier failed/partial audits remain unchanged. This task intersects "
        "support with unchanged mortgage risk intervals; it does not declare Task9 complete. "
        "No API reacquisition or credential access.\n\n"
        + f"Frozen source run: `{support['source_run']}`; "
        f"{support['versions']:,} numeric version/validity segments, not revision-event counts. "
        f"Archive cutoff `{support['archive_cutoff']}`; "
        f"reporting boundary `{support['reporting_cutoff']}`. "
        "Retrieval times are frozen in the specification and JSON evidence.",
    )
    section(
        "Full Feature Support",
        table(
            [
                "Feature",
                "Source",
                "Native frequency",
                "First eligible risk month",
                "Last",
                "Months",
            ],
            [
                [
                    d["name"],
                    d["series_id"],
                    d["frequency"],
                    d["earliest_reporting_month"],
                    d["latest_reporting_month"],
                    d["available_months"],
                ]
                for d in support["definitions"]
            ],
        )
        + "\nExact transformations, operands, units and assessment-date limits are in the JSON "
        "definitions and unchanged [feature registry](../../docs/track_b/"
        "pit_macro_feature_registry.json). Risk month m uses end-of-month m−1 information; "
        "all required operands must be known then, with completed reference periods. "
        "Archive validity is a conservative knowledge bound, not an independently "
        "certified exact original release timestamp. Frozen freshness caps: UNRATE/CPI "
        "62 days; Treasury 7; PMMS 14; HPI/GDP 183. Exact lag operands are not subject "
        "to current-value freshness, but must share the t0 information/metadata regime.",
    )
    section(
        "Coverage Bottlenecks",
        table(
            ["Feature", "Unavailable risk months, out of 243"],
            [[n, c] for n, c in support["bottleneck_counts"].items()],
        )
        + "\nThese counts overlap. The JSON contains all 243 × 8 cells, original selector "
        "status and deterministic reasons. Pre2010 PMMS/HPI archive support prevents full "
        "coverage. January2006 Treasury exceeds the frozen freshness cap. March2026 "
        "requires October2025 unemployment as the exact three-month-change operand "
        "of January2026; the raw API has a missing token. [BLS confirms October2025 "
        "CPS observations were not collected](https://www.bls.gov/cps/methods/"
        "2025-federal-government-shutdown-impact-cps.htm). No interpolation or "
        "future revision is admitted. Missing source observation, missing operand, "
        "unreleased/unarchived version, insufficient provenance and stale current "
        "value remain separate machine-readable reasons.",
    )
    section(
        "Primary Common-Support Design",
        design_summary(support["designs"]["PRIMARY"])
        + "\n\nAll eight features must be available for every eligible interval. "
        "All 140,000 sampled facilities remain in the registry. Early-vintage loans "
        "still active later may enter with delayed entry; no synthetic pre-entry exposure.",
    )
    section(
        "Reduced Historical Sensitivity",
        design_summary(support["designs"]["REDUCED"])
        + ". Features: "
        + ", ".join(support["designs"]["REDUCED"]["features"])
        + "\n\nOne fixed sensitivity excludes PMMS level, mortgage/Treasury spread and "
        "HPI YoY because their dated housing archives restrict early coverage. It "
        "retains labor-market, yield, inflation and output information. Economic "
        "and provenance rationale was frozen before counts; no performance search "
        "or promotion to primary is permitted.",
    )
    section(
        "Historical Archive Investigation",
        "The [FHFA 2008Q2 release](https://www.fhfa.gov/"
        "reports/house-price-index/2008/Q2), published 26August2008, contains a "
        "same-publication national all-transactions pair: 2007Q2=387.45 and "
        "2008Q2=380.82 (1980Q1=100), printed page52. This isolated pair is "
        "PIT_RECONSTRUCTED_AUTHORITATIVELY; it is not admitted into frozen features. "
        "It does not establish a complete release chain or semantic bridge to "
        "the frozen USSTHPI versions. Original PDF bytes are retained privately "
        "with SHA256 `"
        + value["historical_resolution"]["fhfa"]["raw_document_sha256"]
        + "`. The [official PMMS archive](https://www.freddiemac.com/pmms/archive) "
        "offers compiled history, but the bounded investigation did not establish "
        "complete pre2010 publication/revision provenance. Its classification is "
        "HISTORICAL_VALUE_KNOWN_BUT_RELEASE_PROVENANCE_INSUFFICIENT. This is not "
        "proof no earlier releases exist. Complete extension: UNAVAILABLE; no "
        "current revised backfill or fixed-lag assumptions.",
    )
    eligibility_rows = []
    event_rows = []
    for name, vintages in counts["by_vintage"].items():
        for year, v in vintages.items():
            c = v["counts"]
            eligibility_rows.append(
                [
                    name,
                    year,
                    c["sample_facilities_retained"],
                    c["contributing_facilities"],
                    c["eligible_intervals"],
                    v["first_month"],
                    v["last_month"],
                ]
            )
            event_rows.append(
                [
                    name,
                    year,
                    c["unique_default_facilities"],
                    c["unique_payoff_facilities"],
                    c["censored_facilities"],
                    str(v["exits"]),
                ]
            )
    section(
        "Mortgage Eligibility",
        table(
            ["Design", "Vintage", "Retained", "Contributing", "Intervals", "First", "Last"],
            eligibility_rows,
        )
        + "\nIntervals are (currentmonth, nextmonth], both in the unchanged contiguous "
        "analytical prefix. Six known pre-t0 months, current active none state, "
        "consecutive target and nonnegative scheduled-first-payment proxy age are "
        "required. No reentry after an endpoint, gap or unknown prefix. The original "
        "7,741,663 rows are read only; no samples redrawn or facilities deleted.",
    )
    section(
        "Event Support",
        table(
            ["Design", "Vintage", "Unique defaults", "Unique payoffs", "Censored", "Exit reasons"],
            event_rows,
        )
        + "\nCount one first endpoint per facility and design; no overlapping-landmark "
        "event multiplication. Default uses unchanged delinquency≥3/REO or credit "
        "termination02/03/09 research semantics; payoff01 includes maturity. "
        "Administrative15/16/96, ambiguous termination timing and unknown states "
        "censor before unascertainable target intervals. Administrative and other "
        "censor reasons are reported separately; count totals need not match raw "
        "Task8 endpoints because earlier failures/lookback are excluded.",
    )
    section(
        "Duration Support",
        "Both proxy duration and provider age are reported "
        "separately per vintage in JSON. Primary future clock: completed scheduled "
        "first-payment proxy months at t0; bands0–12,13–24,25–36,37–60,61–84,"
        "85–120,121–180,181–240,241+. No outcome-selected knots.\n\n"
        + table(
            ["Design", "Vintage", "Proxy age: eligible interval counts"],
            [
                [n, y, str(v["duration_bands"])]
                for n, vs in counts["by_vintage"].items()
                for y, v in vs.items()
            ],
        ),
    )
    section(
        "Vintage Overlap",
        "Calendar-year counts below; every quarter and its "
        "contributing vintage set are included in JSON. Counts refer to risk "
        "target months, not origination years.\n\n"
        + table(
            ["Design", "Year", "Intervals", "Facilities", "Contributing vintages"],
            [
                [n, period, c["intervals"], c["facilities"], str(c["contributing_vintages"])]
                for n, ps in counts["calendar"].items()
                for period, c in ps.items()
                if len(period) == 4
            ],
        ),
    )
    section(
        "APC Implications",
        table(
            ["Design", "Clock rows", "Columns", "Rank", "Identity residual"],
            [
                [n, a["unique_clock_rows"], a["columns"], a["rank"], a["max_identity_residual"]]
                for n, a in counts["apc"].items()
            ],
        )
        + "\nCalendar t0 = scheduled first-payment cohort + proxy age exactly. "
        "Overlap does not identify unrestricted age, period and cohort effects. "
        "Freeze six vintage indicators with2006 reference, constrained duration "
        "bands and eight primary macro features. No unrestricted period fixed "
        "effects, trend or interactions; no causal coefficients. Archive vintage "
        "is also not exact origination/first-payment cohort.\n\n"
        "Separately, mortgage rate − Treasury yield equals their spread: maximum "
        "absolute residual is "
        + str(value["macro_coefficient_identification"]["max_absolute_relation_residual"])
        + " over 189 complete rate months. These three unrestricted coefficients are "
        "not separately identifiable. All eight features remain required for eligibility. "
        "Before fitting, Task10 must seal an identifiable coefficient constraint/basis; "
        "no performance-driven feature deletion or claim of three independent rate effects.",
    )
    section(
        "Horizon Support",
        "Conditional elapsed months from one active entry per "
        "facility/design differ from Task8's first-observation origin. Known status "
        "means observed to horizon or an earlier first event. Combined support "
        "requires original mortgage horizon SUPPORTED and conditional macro "
        "risk≥1000/known≥10000 (LIMITED≥100/≥2000); it cannot upgrade originally "
        "unsupported mortgage tails. The compact table gives combined labels; "
        "denominators and separate statuses are in JSON.\n\n"
        + table(
            ["Design", "Vintage", "12", "24", "36", "60", "84", "120"],
            [
                [
                    n,
                    y,
                    *[v["horizons"][str(h)]["combined_status"] for h in [12, 24, 36, 60, 84, 120]],
                ]
                for n, vs in counts["by_vintage"].items()
                for y, v in vs.items()
            ],
        )
        + "\nObserved monthly hazards do not require future macro paths at entry. "
        "Scenario forecasting and cumulative prospective risks require separately "
        "governed future macro paths; none are generated here.",
    )
    for title, years in [
        ("Pandemic Coverage", ["2020", "2021"]),
        ("Financial-Crisis Coverage", ["2007", "2008", "2009"]),
    ]:
        section(
            title,
            table(
                ["Design", "Year", "Intervals", "Facilities", "Vintages"],
                [
                    [
                        n,
                        y,
                        counts["calendar"][n].get(y, {}).get("intervals", 0),
                        counts["calendar"][n].get(y, {}).get("facilities", 0),
                        str(counts["calendar"][n].get(y, {}).get("contributing_vintages", [])),
                    ]
                    for n in support["designs"]
                    for y in years
                ],
            )
            + "\nHistorical economic-regime coverage only; no pandemic or GFC effect estimated.",
        )
    section(
        "Future Task 10 Population",
        "[Frozen specification](../../docs/track_b/"
        "macro_support_eligibility_spec.json), LF SHA256 `"
        + value["task10_specification_sha256_lf"]
        + "`. The pre-count freeze and "
        "interval-key hashes are in JSON. Validation candidates were chosen before "
        "counts and are adopted only after feasibility checks; no block changed "
        "in response to outcomes. Development uses70% facility hash roles through "
        "2017December, excludes2018, and evaluates the disjoint30% roles in "
        "2019January–2026February. Both pooled primary blocks must have≥100 "
        "unique defaults and≥500 payoffs.\n\n"
        + table(
            ["Design", "Block", "Facilities", "Intervals", "Defaults", "Payoffs"],
            [
                [
                    n,
                    r,
                    c.get("facilities", 0),
                    c.get("intervals", 0),
                    c.get("defaults", 0),
                    c.get("payoffs", 0),
                ]
                for n, roles in counts["validation_counts"].items()
                for r, c in roles.items()
            ],
        )
        + "\nThe pooled candidate temporal block contains unseen 2018/2020/2022 "
        "vintages. Development only estimates 2006/2008/2010/2014 cohort effects. "
        "The [final validation design](../../docs/track_b/"
        "macro_support_validation_design.json) therefore separates primary seen-vintage "
        "temporal metrics from explicitly restricted unseen-vintage extrapolation. "
        "Dates and hash roles are unchanged.\n\n"
        + table(
            ["Temporal population", "Facilities", "Intervals", "Defaults", "Payoffs"],
            [
                [label, c["facilities"], c["intervals"], c["defaults"], c["payoffs"]]
                for label, c in [
                    (
                        "Primary seen vintages",
                        value["finalized_validation_design"][
                            "primary_temporal_seen_vintage_counts"
                        ],
                    ),
                    (
                        "Separate unseen-vintage sensitivity",
                        value["finalized_validation_design"]["external_unseen_vintage_counts"],
                    ),
                ]
            ],
        )
        + "\nThe seen-vintage block must itself meet the frozen event minimums. "
        "For unseen-vintage sensitivity, its unestimated coefficient is explicitly "
        "fixed to zero relative to the 2006 training reference and flagged "
        "UNSEEN_VINTAGE; never pool its metrics with primary temporal results. "
        "Leave-vintage-out applies the same explicit reference-effect restriction; "
        "if 2006 is held out, training reference becomes 2008. This tests conditional "
        "extrapolation, not learned unseen-cohort effects.\n\n"
        "Leave-vintage-out is prespecified across seven vintages; sparse "
        "causes must be reported as infeasible where applicable. Vintage endpoint "
        "counts are feasibility, not measured model performance. Task10 must "
        "seal a new consumption ledger and complete modeling protocol before "
        "fitting. Previously inspected outcomes cannot become a virgin holdout.",
    )
    section("Limitations", "\n".join("- " + item for item in value["limitations"]))
    section(
        "Decision",
        value["decision"]
        + ".\n\nNext: "
        + value["next_task"]
        + ". Not implemented. Actual second full canonical pass reproduced "
        "all aggregate counts, interval fingerprints and 14 private facility "
        "eligibility files. Preservation PASSED: 395 prior public LF hashes, "
        "94 Task9 private byte hashes, 51 earlier private hashes and 371 "
        "prior tracked byte hashes; frozen tag, samples, source ZIPs and "
        "consumed ledgers remain unchanged. Complete details are in JSON."
        + "\n\nRetained Track A AUC 0.868152 / Brier 0.048545 / log loss 0.176030 "
        "are historical results only, not newly evaluated. Test execution "
        "evidence is recorded separately in `macro_support_verification.json`.",
    )
    return "\n".join(parts)
