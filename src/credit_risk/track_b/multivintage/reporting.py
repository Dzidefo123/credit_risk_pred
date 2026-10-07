"""Aggregate-only Task8 evidence; never fits an outcome model or releases identifiers."""

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from credit_risk.track_b.data.freddie import INTEGER, MONTHS, NUMERIC
from credit_risk.track_b.data.schemas import ORIGINATION, PERFORMANCE, digest

from .core import apc_diagnostic, set_hash, write_json
from .study import lf_hash

STATES = {
    "OBSERVED",
    "MISSING_IN_SOURCE",
    "STRUCTURALLY_UNAVAILABLE",
    "NOT_APPLICABLE",
    "PARSER_FAILURE",
}
STATUSES = {"EXACT", "TRANSFORMABLE", "PARTIAL", "UNAVAILABLE", "SEMANTICALLY_INCOMPATIBLE"}
SCHEMA = "track-b-multivintage-r47-canonical-v1"
LIMITATIONS = [
    (
        "Current-release retrospective disclosure: operational "
        "knowledge time and historical revisions remain unverified."
    ),
    (
        "ZIP timestamps and matching field counts do not "
        "independently attest the exact source release."
    ),
    (
        "Facility identity is not borrower identity; cross-facility "
        "or cross-vintage borrower independence is not established."
    ),
    (
        "First payment month is not exact origination or accounting "
        "recognition; no fabricated daily dates or exact DPD."
    ),
    (
        "Monthly research default is a proxy; payoff includes "
        "maturity, not uniquely voluntary prepayment."
    ),
    (
        "Loss amounts are signed aggregate disclosures, not timed "
        "recovery cash flows, accounting LGD or ECL."
    ),
    (
        "Modification and assistance disclosures do not identify "
        "every intervention or pandemic forbearance."
    ),
    (
        "Natural follow-up is unequal; no outcome extrapolation, "
        "adaptive replacement or global truncation."
    ),
    (
        "Overlap does not identify unrestricted age, period and "
        "cohort effects or causal macro coefficients."
    ),
]
NOTES = {
    "loan_id": (
        "Source mortgage facility identifier; combine with vintage. Never a borrower identifier."
    ),
    "orig_credit_score": (
        "Provider origination credit score; do not equate to VantageScore or time-current score."
    ),
    "vantage_score": "Separate disclosed VantageScore field; absence is not a substitute score.",
    "first_payment_month": (
        "First scheduled payment month; not exact origination, recognition or observation entry."
    ),
    "maturity_month": "Disclosed scheduled maturity month; not observed payoff date.",
    "loan_age": (
        "Provider mortgage age in monthly intervals; audited "
        "against first-payment proxy, not overwritten."
    ),
    "delinquency_state": (
        "Monthly categorical state; XX unknown, RA REO, numeric bands. Not exact daily DPD."
    ),
    "termination_code": (
        "Provider zero-balance code; 01 combines payoff/maturity, "
        "02/03/09 research-default causes, 15/16/96 administrative."
    ),
    "termination_month": (
        "Reported zero-balance month; missing without termination "
        "is NOT_APPLICABLE. Mismatch with reporting month is "
        "ambiguous."
    ),
    "net_sale_proceeds": (
        "Aggregate sale proceeds; alpha disclosure codes preserved "
        "in raw cache, not guessed as zero."
    ),
    "actual_loss": (
        "Signed provider actual-loss disclosure; aggregate audit "
        "only, not discounted workout cash flow."
    ),
    "modification_flag": (
        "Disclosed modification indicator; absence does not prove no historical intervention."
    ),
    "assistance_plan": "Disclosed assistance plan; no inference of complete forbearance coverage.",
    "payment_deferral_flag": (
        "Disclosed deferral flag; reporting scope may depend on servicing regime."
    ),
    "pre_harp_loan_id": "Source linkage field; not general borrower linkage.",
    "postal_prefix": (
        "Coarsened source postal geography; private only, no public facility geography."
    ),
}
PARTIAL = {
    "orig_credit_score",
    "vantage_score",
    "first_payment_month",
    "maturity_month",
    "loan_age",
    "delinquency_state",
    "termination_code",
    "termination_month",
    "modification_flag",
    "assistance_plan",
    "payment_deferral_flag",
    "net_sale_proceeds",
    "actual_loss",
    "pre_harp_loan_id",
}
LOSS = set(PERFORMANCE[13:23]) | {
    "period_modification_costs",
    "bankruptcy_cramdown_costs",
    "delinquent_accrued_interest",
}
DERIVED = {
    "vintage": ("EXACT", "Prespecified archive vintage; not inferred borrower cohort."),
    "research_key": (
        "TRANSFORMABLE",
        "Composite (vintage, source_facility_id); private identity only.",
    ),
    "event_category": (
        "PARTIAL",
        "Unchanged Task2 default/payoff/administrative/ambiguous/unknown categories.",
    ),
    "analytical_prefix": (
        "TRANSFORMABLE",
        (
            "First contiguous known active history; stops at gap, "
            "duplicate, unknown or endpoint; no re-entry."
        ),
    ),
    "post_research_endpoint": (
        "TRANSFORMABLE",
        "Quarantine flag after first observed research endpoint; raw observations retained.",
    ),
    "post_source_termination": (
        "TRANSFORMABLE",
        "Quarantine flag after first source termination; raw observations retained.",
    ),
    "pandemic_regime": (
        "TRANSFORMABLE",
        "Calendar flag for 2020/2021; not actual forbearance or causal treatment.",
    ),
    "months_since_first_payment_proxy": (
        "PARTIAL",
        "Reporting month minus scheduled first-payment month; not exact mortgage age.",
    ),
    "borrower_id": (
        "UNAVAILABLE",
        "No usable borrower linkage supplied; do not assume independence.",
    ),
    "exact_origination_date": ("UNAVAILABLE", "Not supplied by first-payment month."),
    "timed_recovery_cashflows": (
        "UNAVAILABLE",
        "Aggregate loss disclosures cannot supply timed recovery cash flows.",
    ),
    "exact_days_past_due": (
        "SEMANTICALLY_INCOMPATIBLE",
        "Monthly delinquency bands cannot be harmonized to exact daily DPD.",
    ),
}


def ready(result):
    return result.get("status") in {"READY", "READY_WITH_LIMITATIONS"}


def decision(results):
    if any(r.get("status") in {"DATA_QUALITY_STOP", "SCHEMA_INCOMPATIBLE"} for r in results):
        return "STOP — HARMONIZATION INVALID"
    if len(results) != 7 or not all(ready(r) for r in results):
        return "MULTI-VINTAGE SUPPORT INSUFFICIENT"
    return "MULTI-VINTAGE COHORT READY WITH MATERIAL LIMITATIONS"


def field_status(field, result):
    if not ready(result):
        return (
            "PARTIAL",
            "Validation incomplete; does not establish source unavailability or equivalence.",
        )
    missing = result["field_missingness"].get(field, {})
    states = missing.get("row_states", {})
    if states.get("OBSERVED", 0) == 0 or field in PARTIAL or field in LOSS:
        return (
            "PARTIAL",
            (
                "Disclosure/semantic limitation or entirely nonobserved "
                "field; see missingness and definition."
            ),
        )
    return (
        "TRANSFORMABLE" if field in NUMERIC or field in MONTHS else "EXACT"
    ), "Pinned field positions; explicit type normalization only."


def schema_registry(results):
    records = []
    matrix = []
    for kind, fields in [("origination", ORIGINATION), ("performance", PERFORMANCE)]:
        for pos, field in enumerate(fields, 1):
            cells = {}
            for result in results:
                status, reason = field_status(field, result)
                cells[str(result["vintage"])] = dict(status=status, reason=reason)
                records.append(
                    dict(
                        schema_version=SCHEMA,
                        vintage=result["vintage"],
                        source_release="R47 adapter; exact archive release unverified",
                        source_kind=kind,
                        field_name=field,
                        source_position=pos,
                        source_type="YYYYMM"
                        if field in MONTHS
                        else "numeric or missing token"
                        if field in NUMERIC
                        else "categorical/text",
                        canonical_name=field,
                        canonical_type="monthly period or null"
                        if field in MONTHS
                        else "integer-valued number or null"
                        if field in INTEGER
                        else "number or null"
                        if field in NUMERIC
                        else "text",
                        semantic_definition=NOTES.get(
                            field,
                            "Pinned official R47 field: "
                            + field.replace("_", " ")
                            + "; no alternate concept inferred.",
                        ),
                        available_from=None,
                        available_to=None,
                        availability_dates_status="Unverified; not inferred from vintage",
                        compatibility_status=status,
                        verification="Complete selected-history mapping"
                        if ready(result)
                        else "Incomplete origination gate; performance not certified",
                    )
                )
            matrix.append(
                dict(
                    field=field,
                    kind=kind,
                    definition=NOTES.get(field, "Official pinned R47 " + field.replace("_", " ")),
                    vintages=cells,
                )
            )
    for field, (status, definition) in DERIVED.items():
        matrix.append(
            dict(
                field=field,
                kind="derived_or_prohibited",
                definition=definition,
                vintages={
                    str(r["vintage"]): dict(
                        status=status if ready(r) else "PARTIAL",
                        reason="Applied/unsupported as explicitly defined"
                        if ready(r)
                        else "Not certified for stopped vintage",
                    )
                    for r in results
                },
            )
        )
    return dict(
        schema_version=SCHEMA,
        fields=records,
        derived_and_prohibited=[f for f in matrix if f["kind"] == "derived_or_prohibited"],
    ), dict(schema_version=SCHEMA, allowed_statuses=sorted(STATUSES), fields=matrix)


def missing_rates(result):
    answer = {}
    for field, item in result["field_missingness"].items():
        row_states = {s: int(item["row_states"].get(s, 0)) for s in sorted(STATES)}
        total = sum(row_states.values())
        answer[field] = {
            **item,
            "row_state_counts": row_states,
            "row_count": total,
            "unknown_delinquency_code_rows": item["row_states"].get("special_value:XX", 0)
            if field == "delinquency_state"
            else 0,
            "observed_definition": "Decoded token; XX remains unknown and censors follow-up",
            "row_missing_in_source_rate": row_states["MISSING_IN_SOURCE"] / total
            if total
            else None,
            "facilities_any_nonobserved_rate": item["facilities_with_any_nonobserved"]
            / result["facilities"],
            "facilities_all_nonobserved_rate": item["facilities_with_all_nonobserved"]
            / result["facilities"],
        }
    return answer


def supplemental(root, result):
    """Read new canonical outputs only for clock/rank and balance integrity, no outcomes fit."""
    path = root / f"data/track_b/multivintage/processed/v1/{result['vintage']}"
    original = {}
    orig_dates = {}
    findings = Counter()
    with (path / "origination.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            original[row["loan_id"]] = float(row["orig_upb"]) if row["orig_upb"] else None
            orig_dates[row["loan_id"]] = (row["first_payment_month"], row["maturity_month"])
            if row["maturity_month"] and row["maturity_month"] < row["first_payment_month"]:
                findings["maturity_before_first_payment_facilities"] += 1
    clocks = set()
    exact24 = Counter()
    with (path / "monthly.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            first_payment, maturity = orig_dates[row["loan_id"]]
            if row["reporting_month"] < first_payment:
                findings["reporting_before_first_payment_rows"] += 1
            if maturity and row["reporting_month"] > maturity:
                findings["reporting_after_scheduled_maturity_rows"] += 1
            principal = (
                float(row["current_principal_balance"])
                if row["current_principal_balance"]
                else None
            )
            upb = original.get(row["loan_id"])
            if principal is not None and upb is not None and principal > upb:
                findings["current_upb_above_original_rows"] += 1
            if (
                row["non_interest_upb"]
                and principal is not None
                and float(row["non_interest_upb"]) > principal
            ):
                findings["non_interest_upb_above_current_rows"] += 1
            if row["analytical_prefix"] != "True":
                continue
            year, month = map(int, row["reporting_month"].split("-"))
            period = year * 12 + month - 1
            age = int(row["months_since_first_payment_proxy"])
            clocks.add((period, period - age, age))
            if row["loan_age"] and float(row["loan_age"]) == 24:
                exact24[str(year)] += 1
    return dict(
        balance_flags=dict(findings),
        exact_provider_age24_analytical_rows_by_year=dict(sorted(exact24.items())),
        empirical_first_payment_proxy_clocks=clocks,
    )


def freeze_manifest(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Combined manifest identity changed; explicit recovery required")
        return
    write_json(path, value, exclusive=True)


def table(headers, rows):
    return (
        "| "
        + " | ".join(map(str, headers))
        + " |\n| "
        + " | ".join("---" for _ in headers)
        + " |\n"
        + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
        + "\n"
    )


def build(root):
    root = Path(root)
    private = root / "data/track_b/multivintage/manifests"
    results = json.loads((private / "results.json").read_text(encoding="utf-8"))
    plan = json.loads(
        (root / "docs/track_b/multi_vintage_protocol.json").read_text(encoding="utf-8")
    )
    for r in results:
        r["temporary_disk_measurement_scope"] = (
            "ZIP materialization only; SQLite temporary sort spill not instrumented"
        )
        r["acquisition"] = json.loads(
            (private / f"acquisition_{r['vintage']}.json").read_text(encoding="utf-8")
        )
    complete = [r for r in results if ready(r)]
    registry, comparable = schema_registry(results)
    doc = root / "docs/track_b"
    out = root / "reports/track_b"
    write_json(doc / "multi_vintage_schema_registry.json", registry)
    write_json(doc / "multi_vintage_field_comparability.json", comparable)
    report = [
        "# Multi-vintage field comparability",
        (
            "Classification covers source concepts, transformations and "
            "observed availability separately. PARTIAL on a stopped "
            "vintage means incomplete validation, not permission to use "
            "it. Null availability dates mean unverified, never "
            "inferred from vintage. Positions are pinned to the R47 "
            "layout; a matching column count alone does not certify "
            "release."
        ),
        table(
            ["Field", *plan["vintages"]],
            [
                [f["field"], *[f["vintages"][str(y)]["status"] for y in plan["vintages"]]]
                for f in comparable["fields"]
            ],
        ),
        "## Field definitions and boundaries",
    ]
    report.extend("- **" + f["field"] + "**: " + f["definition"] for f in comparable["fields"])
    report.extend(
        [
            "",
            (
                "No historical semantic change is declared absent merely "
                "because current-release columns match. Entirely blank "
                "fields are MISSING_IN_SOURCE; this is not evidence of "
                "STRUCTURALLY_UNAVAILABLE."
            ),
            (
                "Official layout: "
                "https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx"
            ),
        ]
    )
    (doc / "MULTI_VINTAGE_FIELD_COMPARABILITY.md").write_text(
        "\n\n".join(report) + "\n", encoding="utf-8"
    )
    combined = []
    components = []
    clocks = set()
    for r in complete:
        y = r["vintage"]
        ids_path = root / (
            "data/track_b/manifests/expansion_v1/selected_ids.txt"
            if y == 2010
            else f"data/track_b/multivintage/processed/v1/{y}/selected_ids.txt"
        )
        ids = ids_path.read_text(encoding="utf-8").splitlines()
        if (
            len(ids) != r["facilities"]
            or len(set(ids)) != len(ids)
            or set_hash(ids) != r["sample_sha256"]
        ):
            raise ValueError("Combined sample identity mismatch")
        combined.extend(f"{y}:{loan}" for loan in ids)
        components.append(dict(vintage=y, facilities=len(ids), sample_sha256=r["sample_sha256"]))
        r["missingness_rates"] = missing_rates(r)
        extra = supplemental(root, r)
        clocks.update(extra.pop("empirical_first_payment_proxy_clocks"))
        r["supplemental_integrity"] = extra
    combined_hash = set_hash(combined)
    freeze_manifest(
        private / "combined_sample_manifest.json",
        dict(
            key="(vintage, source_facility_id)",
            included=components,
            excluded=[
                dict(vintage=r["vintage"], status=r["status"], error=r.get("error"))
                for r in results
                if not ready(r)
            ],
            combined_sample_sha256=combined_hash,
            schema_version=SCHEMA,
            complete_seven_vintage_population=False if len(complete) != 7 else True,
        ),
    )
    empirical = np.array(sorted(clocks), dtype=float)
    if len(empirical):
        centered = empirical - empirical.mean(axis=0)
        scale = np.maximum(np.std(centered, axis=0), 1)
        design = np.column_stack([np.ones(len(empirical)), centered / scale])
        empirical_apc = dict(
            unique_clock_rows=len(empirical),
            columns=4,
            rank=int(np.linalg.matrix_rank(design)),
            maximum_identity_residual=float(
                np.abs(empirical[:, 0] - empirical[:, 1] - empirical[:, 2]).max()
            ),
            scope=(
                "Observed analytical-prefix clock support using "
                "first-payment proxy, not exact origination or provider age"
            ),
        )
    else:
        empirical_apc = dict(unique_clock_rows=0, rank=None)
    support = defaultdict(lambda: defaultdict(dict))
    for r in complete:
        for cell in r["age_calendar_support"]:
            support[cell["calendar_year"]][cell["provider_age_band"]][str(r["vintage"])] = cell[
                "rows"
            ]
    age_period = {year: dict(ages) for year, ages in sorted(support.items())}
    apc = dict(
        synthetic=apc_diagnostic(plan["vintages"], [0, 12, 24, 60, 120]),
        empirical_proxy=empirical_apc,
        unrestricted_parameterization=(
            "For exact clocks, a drift added to period and subtracted "
            "from cohort and age leaves the linear predictor unchanged. "
            "Unrestricted categorical APC retains this alias plus "
            "ordinary dummy/intercept constraints."
        ),
        future_constraints=plan["future_constraints"],
        constraints_selected_from_outcomes=False,
        causal_identification_established=False,
    )
    horizons = {
        str(r["vintage"]): r.get(
            "horizons",
            {
                str(h): dict(
                    status=None,
                    reason="Vintage stopped before usable follow-up; not inferred as unsupported",
                )
                for h in plan["horizons"]
            },
        )
        for r in results
    }
    write_json(
        out / "multi_vintage_horizon_support.json",
        dict(origin=plan["horizon_origin"], rule=plan["horizon_support_rule"], matrix=horizons),
    )
    manifest = dict(
        protocol_sha256_lf=lf_hash(doc / "multi_vintage_protocol.json"),
        protocol_version=plan["version"],
        sources=[
            dict(
                vintage=r["vintage"],
                sha256=r["source"]["sha256"],
                bytes=r["source"]["byte_size"],
                status=r["status"],
            )
            for r in results
        ],
        samples=components,
        combined_sample_sha256=combined_hash,
        combined_manifest_sha256=lf_hash(private / "combined_sample_manifest.json"),
        manifest_hash_mode="LF-normalized UTF-8; raw archives and canonical CSVs use byte hashes",
        harmonization_version="multivintage-r47-v1.0.0",
        per_vintage=[
            dict(
                vintage=r["vintage"],
                facilities=r["facilities"],
                rows=r["rows"],
                calendar=r["calendar"],
                followup=r["observed_month_span"],
                events=r["first_endpoints"],
                canonical_output_sha256=r["output_hashes"],
            )
            for r in complete
        ],
        harmonization_code_sha256=complete[0]["code_sha256"] if complete else {},
        schema_version=SCHEMA,
        schema_registry_sha256=lf_hash(doc / "multi_vintage_schema_registry.json"),
        harmonization_sha256=lf_hash(doc / "multi_vintage_field_comparability.json"),
        canonical_rows=sum(r["rows"] for r in complete),
        facilities=sum(r["facilities"] for r in complete),
        incomplete_vintages=[r["vintage"] for r in results if not ready(r)],
        limitations=LIMITATIONS,
        decision=decision(results),
    )
    write_json(out / "multi_vintage_dataset_manifest.json", manifest)
    evidence = dict(
        decision=decision(results),
        protocol=plan,
        dataset_manifest=manifest,
        vintages=results,
        horizon_support=horizons,
        age_period_support=age_period,
        apc=apc,
        limitations=LIMITATIONS,
        failure_diagnostics=json.loads(
            (private / "failure_details.json").read_text(encoding="utf-8")
        ),
    )
    write_json(out / "multi_vintage_data_audit.json", evidence)
    sections = [
        "# Multi-Vintage Data Audit",
        "## Executive Summary",
        evidence["decision"]
        + (
            ". All seven authorized local archives are present. Stopped "
            "vintages remain explicitly represented; the surviving "
            "subset is not certified as the prescribed seven-vintage "
            "cohort. No model, macro join, holdout evaluation or sample "
            "replacement was performed."
        ),
        "## Acquisition",
        (
            "User supplied the seven local Standard archives following "
            "the official acquisition request. No authentication "
            "automated or alternate mirror used. Exact download dates "
            "are unknown; filesystem/ZIP timestamps are not "
            "substituted. Private immutable acquisition/preflight "
            "manifests retain directory inventory and access "
            "provenance."
        ),
        "## Source Integrity",
        table(
            ["Vintage", "SHA-256", "Status"],
            [[r["vintage"], r["source"]["sha256"], r["status"]] for r in results],
        ),
        "## Vintage Status",
        table(
            ["Vintage", "Facilities frozen", "Rows retained", "Error"],
            [
                [
                    r["vintage"],
                    r.get("facilities", "Not frozen"),
                    r.get("rows", "Not scanned"),
                    r.get("error", "—"),
                ]
                for r in results
            ],
        ),
        (
            "2006 Q3 line 107274: original rate token `.` is not "
            "recognized by the pinned numeric adapter. 2008 Q4 line "
            "206261: source identifier starts `F09Q1`, contrary to the "
            "vintage/quarter gate. Neither failure is fixed by shifting "
            "columns, deleting records or redrawing a sample."
        ),
        (
            "2020 Q4 line 17103 contains prefix F20Q3; 2022 Q4 line "
            "98552 contains prefix F22Q3. Both stop the quarter-consistency "
            "gate before sample freeze. These may reflect legitimate "
            "source conventions; wrong archives are not established."
        ),
        "## Release-Aware Schemas",
        (
            "Completed vintages validate 31 origination and 35 "
            "performance positions against the pinned July2026 R47 "
            "mapping, pipe delimiter and UTF-8 compatible decoding. "
            "Failed vintages have 31 columns at the failed origination "
            "records; their performance layouts are not certified. "
            "Exact archive release attestation remains unverified. No "
            "heuristic field shifting; signed loss amounts remain "
            "signed. Official layout SHA-256:"
        )
        + plan["layout_sha256"],
        "## Canonical Schema",
        "Version "
        + SCHEMA
        + (
            ". All 66 supplied positions are registered with explicit "
            "numeric/monthly/text types. Private canonical origination "
            "and monthly outputs retain facility keys and audit flags; "
            "no records or IDs are public. Unavailable borrower "
            "identity and timed cash flows are not fabricated."
        ),
        "## Field Comparability",
        (
            "See "
            "../../docs/track_b/MULTI_VINTAGE_FIELD_COMPARABILITY.md "
            "and the machine registry. EXACT refers to a supplied "
            "concept in the pinned mapping, not causal comparability or "
            "verified historical reporting. TRANSFORMABLE uses explicit "
            "normalization; PARTIAL preserves disclosure/semantic "
            "limits; UNAVAILABLE and SEMANTICALLY_INCOMPATIBLE block "
            "invented concepts."
        ),
        "## Sampling",
        table(
            ["Vintage", "N", "Frozen set SHA-256"],
            [[r["vintage"], r["facilities"], r["sample_sha256"]] for r in complete],
        ),
        (
            "New samples use "
            "SHA256(track-b-multivintage-v1:vintage:loan_id), digest "
            "then ID, first 20,000 of the complete valid universe. No "
            "outcomes, credit score or geography used. Failed "
            "origination universes do not produce samples. Existing "
            "2010 exact set reused:"
        )
        + plan["existing_2010_sample_sha256"]
        + ". Combined partial-set hash: "
        + combined_hash,
        "## Longitudinal Integrity",
        (
            "Raw records retained; analytical follow-up never re-enters "
            "after a gap, unknown state, duplicate or endpoint. "
            "Post-terminal records are flagged rather than silently "
            "deleted. Missing linkage, malformed selected rows or "
            "unexplained numeric/schema failures stop a vintage. "
            "Balance-above-original flags are descriptive, not "
            "automatically impossible after modification."
        ),
    ]
    sections.extend(
        "- **"
        + str(r["vintage"])
        + "**: "
        + json.dumps(r["integrity"], sort_keys=True)
        + "; supplemental: "
        + json.dumps(r["supplemental_integrity"]["balance_flags"], sort_keys=True)
        for r in complete
    )
    sections.extend(
        [
            "## Missingness",
            (
                "Machine evidence reports separate OBSERVED, "
                "MISSING_IN_SOURCE, STRUCTURALLY_UNAVAILABLE, "
                "NOT_APPLICABLE and PARSER_FAILURE counts, row rates, "
                "any/all nonobserved facility rates and special tokens. "
                "All-empty current-release columns remain "
                "MISSING_IN_SOURCE, not inferred historical structural "
                "unavailability. Failed vintages have no certified monthly "
                "missingness estimates."
            ),
        ]
    )
    for r in complete:
        absent = [
            k for k, v in r["missingness_rates"].items() if v["row_state_counts"]["OBSERVED"] == 0
        ]
        sections.append(
            "- **" + str(r["vintage"]) + "**, entirely nonobserved fields: " + ", ".join(absent)
        )
    sections.extend(
        [
            "## Event Support",
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
                "First observed raw endpoint counts preserve Task2/4/6 "
                "definitions. They are descriptive feasibility counts, not "
                "cumulative incidence or validated 12-month targets. "
                "Contiguous analytical support is separately censored at "
                "unknowns/gaps."
            ),
            "## Follow-Up",
            table(
                ["Vintage", "Raw span min/median/max", "Analytical prefix min/median/max"],
                [
                    [
                        r["vintage"],
                        "/".join(
                            str(r["observed_month_span"][k])
                            for k in ["minimum", "median", "maximum"]
                        ),
                        "/".join(
                            str(r["contiguous_analytical_followup"][k])
                            for k in ["minimum", "median", "maximum"]
                        ),
                    ]
                    for r in complete
                ],
            ),
            (
                "Intervals from first observation, not exact origination. "
                "Unequal follow-up preserved; no common truncation."
            ),
            "## Horizon Support",
            table(
                ["Vintage", *plan["horizons"]],
                [
                    [
                        r["vintage"],
                        *[
                            horizons[str(r["vintage"])][str(h)]["status"] or "Not assessed"
                            for h in plan["horizons"]
                        ],
                    ]
                    for r in results
                ],
            ),
            (
                "Both horizon risk set and known status must meet frozen "
                "thresholds: SUPPORTED >=1,000 and >=10,000; LIMITED >=100 "
                "and >=2,000; otherwise UNSUPPORTED. Known status includes "
                "earlier observed default/payoff; early "
                "administrative/unknown/gap censoring is not presumed "
                "known. No extrapolation."
            ),
            "## Calendar Coverage",
            table(
                [
                    "Vintage",
                    "First reporting month",
                    "Last reporting month",
                    "Provider age min/max",
                ],
                [
                    [
                        r["vintage"],
                        r["calendar"]["first"],
                        r["calendar"]["last"],
                        str(r["provider_age"]["minimum"]) + "/" + str(r["provider_age"]["maximum"]),
                    ]
                    for r in complete
                ],
            ),
            "## Age-Period Overlap",
            (
                "The machine age × year × vintage matrix counts only first "
                "analytical-prefix records with known nonnegative provider "
                "age. Exact provider age 24 months is separately tabulated "
                "by calendar year. Failed cohorts are not silently counted "
                "as absent economic support."
            ),
            "2018 overlap: " + json.dumps(age_period.get("2018", {}), sort_keys=True),
            "## APC Identification Diagnostics",
            json.dumps(apc, indent=2),
            (
                "Rank diagnostics show an alias, not an identified model. "
                "First-payment proxy clocks are constructed explicitly; "
                "provider age residuals remain separately audited. Smooth "
                "duration, parsimonious cohort terms and national macro "
                "variables may define a constrained predictive design "
                "later; they cannot establish causal macro identification."
            ),
            "## Resource Behavior",
            table(
                ["Vintage", "Seconds", "Peak process MiB", "ZIP temp MiB", "Output MiB"],
                [
                    [
                        r["vintage"],
                        round(r.get("seconds", 0), 1),
                        round((r.get("peak_memory_bytes") or 0) / 2**20, 1)
                        if ready(r)
                        else "Not recorded",
                        round(r.get("temporary_disk_peak", 0) / 2**20, 1)
                        if ready(r)
                        else "Not recorded",
                        round(r.get("output_bytes", 0) / 2**20, 1)
                        if ready(r)
                        else "Partial origination DB",
                    ]
                    for r in results
                ],
            ),
            (
                "Vintages processed sequentially. Stored nested ZIP members "
                "streamed with bounded seek views; no full performance "
                "extraction. ZIP temporary materialization is measured; "
                "SQLite temporary sort spill is not instrumented. "
                "Peak memory is process high-water, not "
                "additive per-vintage memory. Per-quarter retained "
                "checksums and source/parser/protocol/sample/code "
                "identities gate reuse. Partial uncompleted stages require "
                "explicit recovery review."
            ),
            "## Limitations",
            *["- " + v for v in LIMITATIONS],
            (
                "- Four origination gates failed; prescribed "
                "seven-vintage harmonization is incomplete. Dataset "
                "selection cannot be silently changed to the successful "
                "subset."
            ),
            "## Decision",
            evidence["decision"],
            "## Next Task",
            (
                "One corrective task: resolve the 2006 missing-rate "
                "convention and the 2008/2020/2022 archive/identifier "
                "vintage/quarter conflicts against official Freddie documentation, with a "
                "versioned adapter/protocol amendment and explicit "
                "checkpoint recovery before rerunning those origination "
                "gates. Do not acquire/join macro data yet."
            ),
        ]
    )
    (out / "MULTI_VINTAGE_DATA_AUDIT.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return evidence


def verify_preservation(root, source_dir):
    baseline = json.loads(
        (root / "data/track_b/multivintage/manifests/preservation_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    for group in ["tracked", "private"]:
        for name, expected in baseline[group].items():
            if digest(root / name) != expected:
                raise ValueError("Preservation failed: " + name)
    tag = subprocess.check_output(
        ["git", "rev-parse", "track-a-v1.0", "track-a-v1.0^{}"], cwd=root, text=True
    ).splitlines()
    if tag != baseline["tag"]:
        raise ValueError("Track A tag changed")
    source_hashes = {}
    for year in [2006, 2008, 2010, 2014, 2018, 2020, 2022]:
        pre = json.loads(
            (root / f"data/track_b/multivintage/manifests/preflight_{year}.json").read_text(
                encoding="utf-8"
            )
        )
        actual = digest(source_dir / f"historical_data_{year}.zip")
        if actual != pre["sha256"]:
            raise ValueError("Raw source changed")
        source_hashes[str(year)] = actual
    original = json.loads(
        (root / "data/track_b/manifests/freddie_2010_manifest.json").read_text(encoding="utf-8")
    )
    if original["sha256"] != source_hashes["2010"]:
        raise ValueError("Original 2010 source identity changed")
    old_path = Path(original["original_local_source"])
    if old_path.exists() and digest(old_path) != source_hashes["2010"]:
        raise ValueError("Original source at retained path changed")
    location_status = (
        "Original path and bytes verified"
        if old_path.exists()
        else "Old path absent; byte-identical source verified in user-supplied directory"
    )
    ledger_hashes = {}
    for report, folder in [
        ("expanded_pd_validation", "expanded_pd_v1"),
        ("survival_competing_risk_validation", "survival_v1"),
    ]:
        public = json.loads((root / f"reports/track_b/{report}.json").read_text(encoding="utf-8"))
        frozen_public = json.dumps(public["evaluation_ledger"], indent=2) + "\n"
        expected = {
            hashlib.sha256(frozen_public.encode()).hexdigest(),
            hashlib.sha256(frozen_public.replace("\n", "\r\n").encode()).hexdigest(),
        }
        path = root / f"data/track_b/models/{folder}/evaluation_ledger.json"
        actual = digest(path)
        if actual not in expected:
            raise ValueError("Consumed ledger differs from retained public ledger")
        ledger_hashes[folder] = actual
    return dict(
        status="PASSED",
        preexisting_tracked_files=len(baseline["tracked"]),
        retained_private_byte_hashes=len(baseline["private"]),
        track_a_artifacts=12,
        track_a_tag=tag,
        source_hashes=source_hashes,
        original_2010_source_sha256=source_hashes["2010"],
        original_2010_location_status=location_status,
        consumed_ledger_byte_hashes=ledger_hashes,
        ledger_check="Byte hashing against retained public serialization; no consumption API",
        locked_holdout_scored=False,
        model_artifacts_regenerated=False,
        retained_historical_metrics=dict(auc=0.868152, brier=0.048545, log_loss=0.176030),
        metrics_status="Retained historical evidence; not newly evaluated",
    )


def figure(root, evidence):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = [r for r in evidence["vintages"] if ready(r)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    fig.suptitle("Task 8: audited subset — seven-vintage harmonization incomplete", fontsize=15)
    years = [str(r["vintage"]) for r in results]
    calendar, follow, heat, events = axes.ravel()
    for i, r in enumerate(results):
        start, end = [
            float(v[:4]) + (int(v[5:]) - 1) / 12
            for v in [r["calendar"]["first"], r["calendar"]["last"]]
        ]
        calendar.plot([start, end], [i, i], linewidth=8, solid_capstyle="butt", color="#2878a6")
    calendar.set(
        yticks=range(len(years)),
        yticklabels=years,
        xlabel="Calendar year",
        title="Natural reporting coverage",
    )
    calendar.grid(axis="x", alpha=0.2)
    for i, r in enumerate(results):
        q = r["contiguous_analytical_followup"]
        follow.plot([q["minimum"], q["maximum"]], [i, i], color="#9aa7af", linewidth=2)
        follow.plot([q["p25"], q["p75"]], [i, i], color="#2878a6", linewidth=8)
        follow.scatter(q["median"], i, c="#cf513d", s=35, zorder=3)
    follow.set(
        yticks=range(len(years)),
        yticklabels=years,
        xlabel="Monthly intervals from first observation",
        title="Analytical follow-up: range, IQR, median",
    )
    age_bands = [f"<= {n}" for n in [12, 24, 36, 60, 84, 120, 180, 240]] + [">240"]
    periods = sorted(evidence["age_period_support"])
    matrix = np.array(
        [
            [sum(evidence["age_period_support"][p].get(a, {}).values()) for p in periods]
            for a in age_bands
        ]
    )
    im = heat.imshow(
        np.ma.masked_equal(np.log10(1 + matrix), 0), aspect="auto", origin="lower", cmap="Blues"
    )
    heat.set(
        yticks=range(len(age_bands)),
        yticklabels=age_bands,
        xticks=range(0, len(periods), 2),
        xticklabels=periods[::2],
        title="Analytical rows by provider age band × year",
        xlabel="Calendar year",
        ylabel="Disjoint upper-bound age bands (months)",
    )
    fig.colorbar(im, ax=heat, label="log10(1 + rows)", shrink=0.8)
    bottom = np.zeros(len(results))
    for name, color in [
        ("payoff", "#2878a6"),
        ("default", "#cf513d"),
        ("administrative", "#a782bc"),
        ("ambiguous", "#df9f38"),
        ("active_or_unknown", "#b9c5cc"),
    ]:
        values = np.array(
            [100 * r["first_endpoints"].get(name, 0) / r["facilities"] for r in results]
        )
        events.bar(years, values, bottom=bottom, label=name.replace("_", " "), color=color)
        bottom += values
    events.set(
        ylabel="Facilities (%)",
        title="First observed raw endpoints; not incidence estimates",
        ylim=(0, 100),
    )
    events.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    target = root / "reports/track_b/multi_vintage_diagnostics.png"
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return target
