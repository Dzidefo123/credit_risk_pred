"""Aggregate empirical reporting from the immutable private panel; no label changes."""

from collections import Counter
from pathlib import Path

import pandas as pd

from .schemas import digest
from .workflow import write_json


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, r)) + " |" for r in rows],
        ]
    )


def enrich(root, audit):
    root = Path(root)
    path = root / "data/track_b/processed/annual_2010_v1/panel.csv"
    before = digest(path)
    dtype = {
        "loan_id": "string",
        "t0": "string",
        "delinquency_state": "string",
        "payment_deferral_flag": "string",
        "modification_flag": "string",
        "assistance_plan": "string",
    }
    panel = pd.read_csv(path, dtype=dtype, low_memory=False)
    eligible = panel[panel.eligible.eq(True)].copy()
    periods = pd.PeriodIndex(panel.t0, freq="M").asi8
    loan_months = {}
    for loan, period in zip(panel.loan_id, periods, strict=True):
        loan_months.setdefault(loan, set()).add(int(period))
    eligible["physical_full12"] = [
        all(int(pd.Period(month, freq="M").ordinal) + k in loan_months[loan] for k in range(1, 13))
        for loan, month in zip(eligible.loan_id, eligible.t0, strict=True)
    ]
    groups = {
        "all_eligible": eligible,
        "physical_full12": eligible[eligible.physical_full12],
        "not_physical_full12": eligible[~eligible.physical_full12],
    }
    contrast = {}
    for name, group in groups.items():
        contrast[name] = {
            "landmarks": len(group),
            "represented_loans": int(group.loan_id.nunique()),
            "statuses": dict(Counter(group.outcome_status)),
            "median_loan_age": float(group.loan_age.median()),
            "median_principal": float(group.current_principal_balance.median()),
            "median_orig_score": float(group.orig_credit_score.median()),
        }
    audit["followup_selection_contrast"] = contrast
    audit["history_length_quantiles"] = (
        panel.groupby("loan_id").size().quantile([0, 0.25, 0.5, 0.75, 1]).to_dict()
    )
    audit["integrity_rates"] = {
        "duplicate_loan_month_rate": 0.0
        if not audit["cohort"]["temporal_integrity"].get("duplicate_loan_months", 0)
        else None,
        "gap_interval_count": audit["cohort"]["temporal_integrity"].get("gap_intervals", 0),
        "post_terminal_unexplained_rows": audit["cohort"]["temporal_integrity"].get(
            "unexpected_post_terminal_rows", 0
        ),
        "selected_loans_with_history": int(panel.loan_id.nunique()),
        "selected_loans_without_history": audit["sample"]["selected_count"]
        - int(panel.loan_id.nunique()),
        "eligible_principal_missing": int(eligible.current_principal_balance.isna().sum()),
        "eligible_principal_zero": int(eligible.current_principal_balance.eq(0).sum()),
        "all_observation_principal_zero": int(panel.current_principal_balance.eq(0).sum()),
    }
    assessments = [
        ("annual_frame_and_identifier_sampling", "CONFIRMED", "No outcome-based selection"),
        (
            "pinned_31_35_column_structure",
            "CONFIRMED WITH QUALIFICATION",
            "Typed selected rows and structural probes; not all unselected attributes",
        ),
        ("facility_identity_linkage", "CONFIRMED WITH QUALIFICATION", "Not borrower identity"),
        (
            "monthly_time_and_delinquency_bands",
            "CONFIRMED WITH QUALIFICATION",
            "No precise daily or knowledge time",
        ),
        ("historical_feature_availability", "NOT TESTABLE", "Revised release only"),
        ("first_payment_equals_origination", "NOT TESTABLE", "Inference explicitly prohibited"),
        (
            "all_termination_dates_align_with_report_month",
            "CONTRADICTED",
            "Ambiguous records quarantined, no date reconciliation",
        ),
        (
            "payoff_maturity_combined_cause",
            "CONFIRMED WITH QUALIFICATION",
            "Cannot isolate voluntary prepayment",
        ),
        ("monthly_principal_exposure_proxy", "CONFIRMED WITH QUALIFICATION", "Not accounting EAD"),
        (
            "recovery_gain_signs",
            "CONFIRMED WITH QUALIFICATION",
            "Signed values retained; expense credit requires review",
        ),
        ("timed_workout_ledger", "NOT TESTABLE", "No dated transaction-level recoveries"),
    ]
    audit["assumptions"] = {
        key: {"classification": status, "qualification": note} for key, status, note in assessments
    }
    audit["empirical_gates"] = {
        "A": "PASS with release-vintage and acquisition-time qualifications",
        "B": "PASS pinned mapping, no column shifts",
        "C": "PASS source ID linkage; no borrower independence",
        "D": "PASS monthly chronology; knowledge time unverified",
        "E": "CONDITIONAL: ambiguous windows retained as unknown",
        "F": "PASS statuses distinguish censored/competing/event-free",
        "G": "PASS nominal-time firewall; historical PIT unverified",
        "H": "PASS monthly principal proxy only",
        "LGD": "LIMITED: sparse actual-loss observations, no timed workouts",
    }
    audit["followup_selection_warning"] = (
        "Physical-full12 filtering drops event/censored cases; not an approved modeling filter."
    )
    if digest(path) != before:
        raise ValueError("Reporting mutated private panel")
    write_json(root / "data/track_b/manifests/annual_2010_v1/run_audit.json", audit)
    write_json(root / "reports/track_b/freddie_2010_data_audit.json", audit)
    render(root, audit)
    return audit


def render(root, a):
    c, sample = a["cohort"], a["sample"]
    denominator = c["eligible_landmarks"]
    scanned = sum(q["performance_rows_scanned"] for q in a["quarter_counts"].values())
    status_rows = [
        [name, count, f"{100 * count / denominator:.4f}%"]
        for name, count in c["followup"]["status_counts"].items()
    ]
    contrast = a["followup_selection_contrast"]
    all_rows = contrast["all_eligible"]
    full = contrast["physical_full12"]
    excluded = contrast["not_physical_full12"]
    excluded_percent = 100 * excluded["landmarks"] / all_rows["landmarks"]
    chunks = [
        "# Freddie 2010 empirical longitudinal audit",
        "## Decision and evidence boundaries",
        "**PROCEED WITH CONDITIONS** for mortgage PD/survival cohort design. "
        "No models, calibration, LGD/EAD estimates or ECL were calculated. "
        "Historical feature availability remains unverified in this revised release.",
        "The [initial stop](FREDDIE_2010_INITIAL_PREFLIGHT_STOP.json) is preserved. "
        "The [annual amendment](../../docs/track_b/ANNUAL_BUNDLE_ACQUISITION_AMENDMENT.md) "
        "changes only input frame/resources. The original scientific rules remain fixed.",
        "## Source, sampling and scan",
        f"Original ZIP SHA-256: `{a['source_sha256']}`. No extraction or source copy.",
        f"Annual universe: **{a['annual_universe']:,} IDs**; selected "
        f"**{sample['selected_count']}** before performance access. No quotas or replacements.",
        f"Salt: `{sample['salt']}`; sample-set SHA-256: `{sample['sample_set_sha256']}`.",
        table(
            ["Quarter", "Orig IDs", "Selected", "Perf scanned", "Retained"],
            [
                [
                    q,
                    values["valid_ids"],
                    sample["selected_by_quarter"][q],
                    values["performance_rows_scanned"],
                    values["selected_rows_retained"],
                ]
                for q, values in a["quarter_counts"].items()
            ],
        ),
        f"Total performance scanned: **{scanned:,}**; retained: "
        f"**{c['monthly_performance_records']:,}** "
        f"({100 * c['monthly_performance_records'] / scanned:.5f}% hit rate). "
        "Nonselected attributes discarded after ID. Selected records passed typed parsing; "
        "nonselected attributes were not fully validated. Zero unmatched IDs.",
        "## Longitudinal integrity",
        f"Panel landmarks: **{c['panel_observations']:,}**; date range "
        f"{c['reporting_period_range'][0]} to {c['reporting_period_range'][1]}.",
        table(["Integrity/availability measure", "Value"], list(a["integrity_rates"].items())),
        table(["History-length quantile", "Months"], list(a["history_length_quantiles"].items())),
        "No duplicates, gaps or unexpected post-terminal rows were observed in selected histories. "
        "This is not a guarantee of borrower independence or full-source cleanliness.",
        "## Twelve-month follow-up",
        f"Eligible landmarks: **{denominator:,}**; ineligible: "
        f"**{c['ineligible_landmarks']:,}**. All are retained. "
        "Overlapping landmarks are not independent defaults; denominator is all eligible t0 rows.",
        table(["Protocol status", "Count", "Eligible fraction"], status_rows),
        table(
            ["First observed loan endpoint", "Loans"], list(a["loan_first_observed_events"].items())
        ),
        table(["Qualifying-record loan count", "Loans"], list(a["default_record_loans"].items())),
        "Payoff/maturity cannot be separated into voluntary prepayment alone. "
        "Ambiguous windows remain unknown. Censoring was not converted to a negative.",
        "## Complete-follow-up selection effect",
        f"Physical 12-month record coverage retains **{full['landmarks']:,}** of "
        f"**{all_rows['landmarks']:,}** landmarks and excludes "
        f"**{excluded['landmarks']:,} ({excluded_percent:.2f}%)**.",
        table(
            ["Group", "Landmarks", "Loans", "Median age", "Median principal"],
            [
                [
                    name,
                    data["landmarks"],
                    data["represented_loans"],
                    data["median_loan_age"],
                    data["median_principal"],
                ]
                for name, data in contrast.items()
            ],
        ),
        table(["Excluded protocol status", "Count"], list(excluded["statuses"].items())),
        "This restriction removes early payoff/default/censoring cases and changes the population. "
        "It is not an approved modeling filter. Physical presence is not event-free survival "
        "or proof of genuine information availability.",
        table(
            ["Year", "Eligible", "Protocol status counts"],
            [
                [
                    year,
                    values["eligible"],
                    "; ".join(f"{k}: {v}" for k, v in values["statuses"].items()),
                ]
                for year, values in a["followup_by_year"].items()
            ],
        ),
        "## Exposure, missingness and losses",
        table(["Eligible principal statistic", "Value"], list(a["exposure_eligible"].items())),
        "Monthly principal is supported only as the approved proxy, not accounting/regulatory EAD. "
        "Terminal zero balances do not establish zero EAD at first default.",
        table(
            ["Origination field", "Missing selected loans"],
            list(c["origination_missingness"].items()),
        ),
        table(
            ["Aggregate loss field", "Available observations", "Min", "Median", "Max"],
            [
                [name, values["available"], values["min"], values["median"], values["max"]]
                for name, values in a["loss_availability"].items()
            ],
        ),
        "Very sparse actual-loss disclosures support inspection, not an LGD model. "
        "Signed gains and net expense credits are retained; no clipping or silent normalization. "
        "No transaction timing or ultimate-workout completion is recovered from aggregates. "
        "Blank/no-plan conventions in assistance/modification flags need explicit encoding review.",
        "## Protocol verification and empirical gates",
        table(
            ["Assumption", "Assessment", "Qualification"],
            [[k, v["classification"], v["qualification"]] for k, v in a["assumptions"].items()],
        ),
        table(["Gate", "Assessment"], list(a["empirical_gates"].items())),
        "## Resources and manifests",
        table(["Engineering measure", "Value"], list(a["resources"].items())),
        "Private panel/IDs/manifests remain Git-ignored. Source, sample, protocol, amendment "
        "and transformation hashes permit reproduction. Reporter did not mutate the panel.",
        "## Next task",
        "**Track B Task 3 - Cohort Design and 12-Month PD Baseline.** "
        "Freeze calendar/entity splits and censoring/competing-risk treatment first. "
        "The same 1,000 IDs remain fixed despite sparse defaults/loss observations. "
        "No modeling implemented here; Track A remains unchanged.",
    ]
    (Path(root) / "reports/track_b/FREDDIE_2010_DATA_AUDIT.md").write_text(
        "\n\n".join(chunks) + "\n", encoding="utf-8"
    )
