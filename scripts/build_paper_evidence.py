"""Build Task 14 planning artifacts by copying frozen public evidence, never research data."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "aa6eb3a001429d362c7fd76a45b9affc4995846f"
PAPER = ROOT / "docs/paper"
FILES = {
    "pd": "reports/track_b/expanded_pd_validation.json",
    "pd_design": "docs/track_b/expanded_pd_design.json",
    "small": "reports/track_b/pd_baseline_validation.json",
    "expansion": "reports/track_b/sample_expansion_feasibility.json",
    "survival": "reports/track_b/survival_competing_risk_validation.json",
    "event": "docs/track_b/mortgage_research_protocol.json",
    "macro_design": "reports/track_b/macro_design_validation.json",
    "macro": "reports/track_b/pit_macro_data_audit_task9_api_v5.json",
    "support": "reports/track_b/macro_support_eligibility.json",
    "population": "reports/track_b/multi_vintage_recovery_audit.json",
    "manifest": "reports/track_b/multi_vintage_recovery_dataset_manifest.json",
    "old_manifest": "reports/track_b/multi_vintage_dataset_manifest.json",
    "m10": "reports/track_b/macro_competing_risk_validation.json",
    "m11": "reports/track_b/macro_signal_attribution_stability.json",
    "p12": "reports/track_b/refinancing_incentive_payoff_research.json",
    "fannie": "reports/track_b/fannie_source_provenance_closure.json",
    "fannie_protocol": "docs/track_b/fannie_external_replication_protocol.json",
}


def sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def pointer(value, address):
    for token in address.strip("/").split("/") if address else []:
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def write(name, value):
    path = PAPER / name
    if path.exists():
        raise ValueError("Evidence already exists; use a versioned amendment")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(name, text):
    path = PAPER / name
    if path.exists():
        raise ValueError("Planning artifact already exists")
    path.write_text(text, encoding="utf-8")


def build():
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != BASE:
        raise ValueError("Task 14 must start at the authorized base commit")
    PAPER.mkdir(parents=True, exist_ok=True)
    data = {key: read(name) for key, name in FILES.items()}
    commits = {
        key: subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", name], cwd=ROOT, text=True
        ).strip()
        for key, name in FILES.items()
    }

    def ref(key, address="", role="evidence"):
        return {
            "path": FILES[key],
            "json_pointer": address,
            "sha256_lf": sha(ROOT / FILES[key]),
            "source_commit": commits[key],
            "role": role,
            "evidence_level": 3 if key == "macro_design" else 1,
        }

    def value(key, address):
        return pointer(data[key], address)

    events = data["event"]
    experiments = {
        "A": {
            "placement": "BACKGROUND_ONLY",
            "description": (
                "Earlier static public-data benchmark; no Track A result promoted "
                "to longitudinal evidence"
            ),
            "source_keys": ["event"],
            "population": None,
        },
        "B3": {
            "placement": "SUPPLEMENT",
            "description": "Small-sample feasibility only",
            "source_keys": ["small"],
            "population": value("small", "/primary"),
        },
        "B5": {
            "placement": "MAIN_TEXT",
            "description": "Expanded 12-month default-before-payoff landmark research",
            "source_keys": ["pd", "pd_design", "expansion"],
            "population": value("pd", "/cohort_reproduction/cohorts"),
            "vintages": [2010],
            "development_window": {"end": value("pd_design", "/development_end")},
            "purge_window": value("pd_design", "/purged"),
            "evaluation_window": {"start": value("pd_design", "/evaluation_start")},
            "unit": "monthly 12-month landmark",
            "bootstrap": value("pd", "/clustered_uncertainty"),
            "prior_exposure": value("pd", "/evaluation_ledger/prior_access"),
        },
        "B6": {
            "placement": "MAIN_TEXT",
            "description": (
                "Conditional-entry competing-risk survival; no lifetime-from-origination claim"
            ),
            "source_keys": ["survival"],
            "population": value("survival", "/survival_audit/partitions"),
            "vintages": [2010],
            "development_window": {"end": value("survival", "/protocol/split/development_end")},
            "purge_window": value("survival", "/protocol/split/purged_year"),
            "evaluation_window": {"start": value("survival", "/protocol/split/evaluation_start")},
            "unit": "monthly risk interval or facility-entry horizon, specified by metric",
            "bootstrap": value("survival", "/protocol/uncertainty"),
            "prior_exposure": value("survival", "/protocol/evaluation"),
        },
        "B7_9A": {
            "placement": "MAIN_TEXT",
            "description": "PIT macro provenance, seven-vintage recovery and support constraints",
            "source_keys": ["macro_design", "macro", "support", "population", "manifest"],
            "population": value("support", "/totals"),
            "vintages": [2006, 2008, 2010, 2014, 2018, 2020, 2022],
            "unit": "facility risk intervals and distinct macro calendar months",
        },
        "B10": {
            "placement": "MAIN_TEXT",
            "description": (
                "Primary macro temporal validation; previously inspected support outcomes"
            ),
            "source_keys": ["m10"],
            "population": value("m10", "/split_counts"),
            "vintages": value("m10", "/protocol/splits/primary_vintages"),
            "development_window": value("m10", "/protocol/splits/development"),
            "purge_window": value("m10", "/protocol/splits/purge"),
            "evaluation_window": value("m10", "/protocol/splits/evaluation"),
            "unit": "monthly risk interval; first eligible entry per facility for CIF",
            "bootstrap": value("m10", "/protocol/bootstrap"),
            "prior_exposure": value("m10", "/ledger"),
            "future_macro_path": value("m10", "/protocol/cif/future_macro"),
        },
        "B11": {
            "placement": "MAIN_TEXT",
            "description": "POST_VALIDATION_DIAGNOSTIC; no causal attribution",
            "source_keys": ["m11"],
            "population": value("m11", "/counts"),
            "vintages": value("m10", "/protocol/splits/primary_vintages"),
            "development_window": value("m10", "/protocol/splits/development"),
            "purge_window": value("m10", "/protocol/splits/purge"),
            "evaluation_window": value("m10", "/protocol/splits/evaluation"),
            "unit": "risk intervals, calendar months, and diagnostic refit windows",
            "prior_exposure": (
                "Task 10 inspected outcomes; same-sample oracle and post-hoc "
                "frozen-score diagnostics"
            ),
        },
        "B12": {
            "placement": "MAIN_TEXT",
            "description": "EXPLORATORY_ONLY refinancing hypothesis; no independent validation",
            "source_keys": ["p12"],
            "population": value("p12", "/counts"),
            "vintages": value("p12", "/prespecification/split/primary_vintages"),
            "development_window": value("p12", "/prespecification/split/development"),
            "purge_window": value("p12", "/prespecification/split/purge"),
            "evaluation_window": value("p12", "/prespecification/split/exploratory"),
            "unit": "monthly risk interval; facility entry for CIF",
            "bootstrap": value("p12", "/prespecification/bootstrap"),
            "prior_exposure": value("p12", "/limitations/0"),
            "future_macro_path": value("p12", "/prespecification/cif/path"),
        },
        "B13_13S": {
            "placement": "SUPPLEMENT",
            "description": (
                "Proposed external replication awaiting provenance closure and "
                "protocol freeze; no results"
            ),
            "source_keys": ["fannie", "fannie_protocol"],
            "population": None,
            "unit": None,
            "protocol_status": "DRAFT_NOT_YET_AUTHORIZED",
            "task13a_authorized": False,
        },
    }
    # Large uncertainty arrays stay in original artifacts; design carries their exact references.
    experiments["B5"]["bootstrap"] = value("pd_design", "/bootstrap")
    claims = []

    def add(
        cid,
        name,
        exp,
        key,
        address,
        text,
        cls="SUPPORTED_EMPIRICAL",
        strength="SECONDARY",
        metric=None,
        uncertainty=None,
        section="6",
        figure=None,
        table=None,
        qualifier=None,
        quantitative=True,
    ):
        design = experiments[exp]
        exploratory = cls in {"EXPLORATORY_DIAGNOSTIC", "EXPLORATORY_HYPOTHESIS"}
        if exploratory and strength == "PRIMARY":
            raise ValueError("Exploratory finding cannot be PRIMARY")
        refs = [ref(key, address, "point_estimate" if quantitative else "method_or_status")]
        if uncertainty:
            refs.append(ref(key, uncertainty, "uncertainty"))
        # Scope evidence binds each claim's population, clock and restrictions.
        scope_address = {
            "pd": "/cohort_reproduction/cohorts",
            "survival": "/protocol",
            "m10": "/protocol",
            "m11": "/protocol",
            "p12": "/prespecification",
            "support": "/finalized_validation_design",
            "fannie": "/scope",
        }.get(key)
        if scope_address:
            refs.append(ref(key, scope_address, "scope"))
        if exp != "B13_13S":
            refs.append(ref("event", "/event", "event_definition"))
            refs.append(ref("event", "/labels", "censoring_definition"))
        limitations = [
            "Facility clustering does not establish borrower independence",
            "Results are conditional predictive associations, not causal or regulatory validation",
        ]
        if exp in {"B5", "B6", "B10", "B11", "B12"}:
            limitations += [
                "Retrospective mortgage operational knowledge time remains unverified",
                "Prior outcome exposure is documented; no virgin-holdout claim",
            ]
        if qualifier:
            limitations.append(qualifier)
        claims.append(
            {
                "claim_id": cid,
                "short_name": name,
                "experiment_id": exp,
                "claim_class": cls,
                "claim_strength": strength,
                "candidate_claim": text,
                "allowed_language": text,
                "prohibited_language": [
                    "Industry-wide non-transportability",
                    "Causal macro effect",
                    "Regulator validated model",
                    "Cross-provider transport failure",
                ]
                + (
                    ["Confirmatory refinancing success", "Independent validation"]
                    if exp == "B12"
                    else []
                ),
                "population": (
                    "National US macro series and monthly assessment-date information sets"
                )
                if key == "macro"
                else design.get("population"),
                "dataset": "FRED/ALFRED vintage-aware candidate macro series"
                if key == "macro"
                else (
                    "Fannie Mae Primary SF Loan Performance (schema only)"
                    if exp == "B13_13S"
                    else "Freddie Mac Single-Family Loan-Level Standard Dataset"
                ),
                "provider": "FRED/ALFRED with original agency metadata retained"
                if key == "macro"
                else ("Fannie Mae" if exp == "B13_13S" else "Freddie Mac"),
                "origination_vintage_scope": design.get("vintages"),
                "observation_window": [
                    {
                        "vintage": component["vintage"],
                        "first": component["calendar"]["first"],
                        "last": component["calendar"]["last"],
                    }
                    for component in value("manifest", "/components")
                    if component["vintage"] in design.get("vintages", [])
                ]
                if design.get("vintages")
                else None,
                "development_window": design.get("development_window"),
                "purge_window": design.get("purge_window"),
                "evaluation_window": design.get("evaluation_window"),
                "unit_of_analysis": design.get("unit"),
                "independence_unit": "facility; borrower dependence unresolved"
                if exp != "B13_13S"
                else None,
                "event_definition": events["event"] if exp != "B13_13S" else None,
                "competing_event_definition": (
                    "Verified code 01 payoff/maturity, not isolated voluntary refinancing"
                )
                if exp != "B13_13S"
                else None,
                "censoring_definition": events["labels"]["right_censored"]
                if exp != "B13_13S"
                else None,
                "model_or_comparison": name,
                "metric": metric if quantitative else None,
                "point_estimate": value(key, address) if quantitative else None,
                "uncertainty": value(key, uncertainty) if uncertainty else None,
                "statistical_method": (
                    "Copied frozen aggregate; uncertainty only as previously recorded"
                )
                if quantitative
                else "Frozen design/status audit",
                "source_artifacts": refs,
                "source_commit": commits[key],
                "frozen_status": "HASH_PINNED_AT_TASK14_NOT_NEW_EVALUATION",
                "exploratory_status": "EXPLORATORY_ONLY"
                if exp == "B12"
                else ("POST_VALIDATION_DIAGNOSTIC" if exp == "B11" else "NOT_NEW_RESEARCH"),
                "known_limitations": limitations,
                "figure_candidate": figure,
                "table_candidate": table,
                "manuscript_sections": [section],
                "verification_status": "VERIFIED_WITH_QUALIFICATION"
                if qualifier or exploratory
                else "VERIFIED_EXACT",
                "quantitative": quantitative,
                "task15_admission": "EXPLORATORY_LABEL_REQUIRED"
                if exploratory
                else "ELIGIBLE_WITH_RECORDED_SCOPE",
            }
        )

    def aggregate(
        cid,
        name,
        exp,
        key,
        address,
        cls="SUPPORTED_METHODOLOGICAL",
        section="4",
        qualifier=None,
        figure=None,
        table=None,
    ):
        add(
            cid,
            name,
            exp,
            key,
            address,
            "Frozen " + name + "; use only within " + experiments[exp]["description"],
            cls=cls,
            strength="CONTEXTUAL",
            metric="frozen_aggregate_record",
            section=section,
            qualifier=qualifier,
            figure=figure,
            table=table,
        )

    for model in ["logistic", "xgboost"]:
        for metric in [
            "roc_auc",
            "average_precision",
            "brier",
            "log_loss",
            "mean_probability",
            "observed_rate",
        ]:
            add(
                "PD_" + model.upper() + "_" + metric.upper(),
                model + " temporal " + metric,
                "B5",
                "pd",
                f"/temporal_results/{model}/metrics/{metric}",
                (
                    f"The frozen {model} achieved the recorded {metric} on the Task 5 "
                    f"Freddie 2010-vintage temporal landmarks."
                ),
                uncertainty=f"/clustered_uncertainty/intervals/{model}/{metric}",
                metric=metric,
                section="6.1",
                table="T3",
            )
    for metric in ["calibration_intercept", "calibration_slope"]:
        add(
            "PD_" + metric.upper(),
            "logistic " + metric,
            "B5",
            "pd",
            "/temporal_results/logistic/calibration/" + metric,
            "The Task 5 logistic temporal calibration diagnostic has the recorded "
            + metric
            + "; raw probabilities underpredict the observed landmark event rate.",
            metric=metric,
            section="6.1",
            qualifier="Calibration point diagnostics have no frozen confidence intervals",
            table="T3",
        )
    for cid, name, address in [
        ("PD_COUNTS", "expanded cohort counts", "/cohort_reproduction"),
        ("PD_CHAMPION", "champion decision", "/champion"),
        ("PD_FEATURES", "feature and calibration design", "/model_specifications/design"),
        ("PD_HAZARD", "hazard diagnostic and net-risk projection", "/hazard"),
        ("PD_PSI", "stability and PSI diagnostics", "/stability"),
        (
            "PD_DISTRESS",
            "distress-state dependence supplement",
            "/post_evaluation_diagnostic_supplement",
        ),
        ("PD_PAIRED", "paired champion uncertainty", "/clustered_uncertainty/paired"),
        ("PD_LIMITS", "expanded PD limitations", "/limitations"),
    ]:
        aggregate(
            cid,
            name,
            "B5",
            "pd",
            address,
            cls="LIMITATION"
            if cid == "PD_LIMITS"
            else (
                "EXPLORATORY_DIAGNOSTIC"
                if cid in {"PD_DISTRESS", "PD_PSI", "PD_HAZARD"}
                else "SUPPORTED_METHODOLOGICAL"
            ),
            section="6.1",
            qualifier=(
                "Monthly hazard boundary recognition and net-risk projections are "
                "not validated annual competing-risk forecasts"
            )
            if cid == "PD_HAZARD"
            else None,
        )
    aggregate(
        "PD_SAMPLE",
        "deterministic nested expansion sample",
        "B5",
        "expansion",
        "/sample",
        table="T1",
    )
    aggregate(
        "B3_FEASIBILITY",
        "small-sample feasibility limitations",
        "B3",
        "small",
        "/limitations",
        cls="LIMITATION",
        section="Appendix",
    )
    aggregate(
        "SURV_COUNTS",
        "conditional-entry cohort and risk-set hashes",
        "B6",
        "survival",
        "/survival_audit",
        table="T1",
    )
    aggregate(
        "SURV_DESIGN",
        "conditional-entry competing-risk design",
        "B6",
        "survival",
        "/protocol",
        section="5.2",
        figure="F2",
        table="T2",
    )
    for name in ["naive_net_default", "default_cif"]:
        add(
            "SURV_60_" + name.upper(),
            "60-month observed " + name,
            "B6",
            "survival",
            "/horizon_results/60/observed/" + name,
            "At 60 months after conditional evaluation entry, the frozen descriptive "
            + name
            + (
                " has the recorded value; net default with payoff censored differs"
                " from competing-risk incidence."
            ),
            strength="PRIMARY",
            metric=name,
            uncertainty="/uncertainty/intervals/60/default_cif" if name == "default_cif" else None,
            section="6.2",
            qualifier=(
                "Descriptive nonparametric comparison only; 60-month model "
                "forecast lacks training-time support"
            ),
            figure="F2",
            table="T4",
        )
    for horizon in [12, 24, 36]:
        for metric in ["ipcw_brier", "cumulative_dynamic_auc"]:
            add(
                f"SURV_{horizon}_{metric.upper()}",
                f"conditional {horizon}-month {metric}",
                "B6",
                "survival",
                f"/horizon_results/{horizon}/metrics/{metric}",
                (
                    f"The frozen conditional-entry structural model has the recorded "
                    f"{horizon}-month {metric} under pooled censoring assumptions."
                ),
                metric=metric,
                uncertainty=f"/uncertainty/intervals/{horizon}/"
                + ("brier" if metric == "ipcw_brier" else "auc"),
                section="6.2",
                table="T4",
                qualifier="Horizon event support varies; 12-month AUC uses only 14 event facilities"
                if horizon == 12
                else None,
            )
    add(
        "SURV_IBS",
        "integrated Brier 1 through 36",
        "B6",
        "survival",
        "/integrated_brier_1_36",
        (
            "The frozen integrated default Brier score covers months 1 through"
            " 36 after conditional entry."
        ),
        metric="integrated_brier_1_36",
        uncertainty="/integrated_brier_interval",
        section="6.2",
        table="T4",
    )
    aggregate(
        "SURV_DURATION",
        "duration support limitation",
        "B6",
        "survival",
        "/time_support_audit",
        cls="EXPLORATORY_DIAGNOSTIC",
        section="11",
    )
    for cid, key, address, name in [
        (
            "COHORT_RECOVERY",
            "population",
            "/completed_facilities",
            "seven-vintage retained facility count",
        ),
        (
            "COHORT_ROWS",
            "population",
            "/completed_performance_rows",
            "seven-vintage canonical row count",
        ),
        ("COHORT_HASH", "manifest", "/combined_sample_sha256", "combined identifier-sample hash"),
        ("COHORT_COMPONENTS", "manifest", "/components", "source and per-vintage sample manifests"),
        ("PIT_FEATURES", "macro", "/features", "eight eligibility feature transformations"),
        ("PIT_SERIES", "macro", "/series_registry", "six candidate source series"),
        ("PIT_PROVENANCE", "macro", "/source_hashes", "vintage-aware source response hashes"),
        ("PIT_RELEASES", "macro", "/release_lags", "release lag evidence"),
        ("PIT_GAPS", "macro", "/limitations", "macro vintage and support limitations"),
        ("PIT_SUPPORT", "support", "/macro_support/designs", "primary and reduced support windows"),
        ("PIT_COUNTS", "support", "/totals", "eligible support populations"),
        ("PIT_APC", "support", "/mortgage_eligibility/apc", "age-period-cohort rank deficiency"),
        ("PIT_RATES", "support", "/macro_coefficient_identification", "algebraic rate redundancy"),
    ]:
        aggregate(
            cid,
            name,
            "B7_9A",
            key,
            address,
            cls="LIMITATION"
            if cid in {"PIT_GAPS", "PIT_APC", "PIT_RATES"}
            else "SUPPORTED_METHODOLOGICAL",
            section="4.5",
            figure="F1" if cid == "PIT_SUPPORT" else None,
            table="T1",
        )
    aggregate(
        "MACRO_ACTUAL_FEATURES",
        "seven actual M2 macro predictors and sensitivity sets",
        "B10",
        "m10",
        "/protocol",
        section="5.3",
        table="T2",
    )
    for model in ["M0", "M1", "M2", "RATE", "REDUCED_M1", "REDUCED_M2"]:
        for split in ["development", "primary"]:
            for metric in [
                "joint_log_loss",
                "default_brier",
                "payoff_brier",
                "default_auc",
                "payoff_auc",
            ]:
                primary = model in {"M1", "M2"} and metric in {
                    "joint_log_loss",
                    "payoff_auc",
                    "payoff_brier",
                }
                add(
                    f"MACRO_{split.upper()}_{model}_{metric.upper()}",
                    f"{model} {split} {metric}",
                    "B10",
                    "m10",
                    f"/{split}/{model}/scores/{metric}",
                    (
                        f"Within the frozen Task 10 Freddie design, {model} has the "
                        f"recorded {split} {metric}; development fit and temporal results "
                        f"must be reported together."
                    ),
                    strength="PRIMARY" if primary else "SECONDARY",
                    metric=metric,
                    section="6.3" if split == "development" else "6.4",
                    figure="F3",
                    table="T5",
                    qualifier="Development scores are fit performance, not independent validation"
                    if split == "development"
                    else None,
                )
    for unit in ["facility", "calendar"]:
        for metric in [
            "joint_log_loss",
            "default_brier",
            "payoff_brier",
            "default_auc",
            "payoff_auc",
        ]:
            if metric not in value("m10", f"/paired_{unit}/intervals"):
                continue  # No frozen calendar AUC interval exists; never fabricate it.
            add(
                f"MACRO_DELTA_{unit.upper()}_{metric.upper()}",
                "M2 minus M1 " + metric,
                "B10",
                "m10",
                f"/paired_{unit}/intervals/{metric}/delta",
                (
                    "The frozen paired M2-minus-M1 temporal difference and interval "
                    "have the recorded orientation; positive proper-score differences "
                    "are deterioration."
                ),
                metric="paired_delta_" + metric,
                uncertainty=f"/paired_{unit}/intervals/{metric}",
                strength="PRIMARY" if metric == "joint_log_loss" else "SECONDARY",
                section="6.4",
                figure="F3",
                table="T5",
                qualifier=(
                    "Fixed-model uncertainty; facility draws condition on the macro "
                    "path; calendar sensitivity has only eight annual blocks"
                ),
            )
    for cause in ["default", "payoff"]:
        for model in ["M1", "M2"]:
            aggregate(
                f"MACRO_CAL_{model}_{cause.upper()}",
                f"{model} {cause} temporal calibration",
                "B10",
                "m10",
                f"/primary/{model}/calibration/{cause}",
                cls="SUPPORTED_EMPIRICAL",
                section="6.5",
                figure="F5",
                table="T6",
            )
    for name, address in [
        ("CALENDAR", "/stability/calendar"),
        ("VINTAGE", "/stability/vintage"),
        ("PANDEMIC", "/period_sensitivity"),
        ("UNSEEN", "/unseen_vintage"),
        ("RATE_SENS", "/rate_representation_sensitivity"),
        ("REDUCED_SENS", "/reduced_historical_sensitivity"),
        ("DESIGN_COUNTS", "/split_counts"),
        ("LEDGER", "/ledger"),
        ("LIMITS", "/limitations"),
    ]:
        aggregate(
            "MACRO_" + name,
            name.lower() + " frozen evidence",
            "B10",
            "m10",
            address,
            cls="LIMITATION" if name == "LIMITS" else "SUPPORTED_EMPIRICAL",
            section="6.4",
            table="T6",
        )
    for horizon in [12, 24, 36, 60]:
        aggregate(
            f"MACRO_CIF_{horizon}_SUPPORT",
            f"{horizon}-month CIF support and path design",
            "B10",
            "m10",
            f"/cif/horizons/{horizon}/status",
            section="6.6",
            qualifier="Historical rolling PIT path; not an entry-time prospective forecast",
        )
        for cause in ["default", "payoff"]:
            for model in ["M1", "M2"]:
                add(
                    f"MACRO_CIF_{horizon}_{model}_{cause.upper()}",
                    f"{horizon}-month {model} {cause} CIF",
                    "B10",
                    "m10",
                    f"/cif/horizons/{horizon}/models/{model}/{cause}/mean_predicted_cif",
                    (
                        f"The frozen {horizon}-month rolling historical PIT projection for "
                        f"{model} has the recorded mean {cause} CIF."
                    ),
                    metric="mean_predicted_cif",
                    strength="PRIMARY" if horizon == 24 and cause == "payoff" else "SECONDARY",
                    section="6.6",
                    figure="F6",
                    table="T5",
                    qualifier=(
                        "No prospective macro forecast claim; horizon-specific calendar "
                        "eligibility and support must accompany each value"
                    ),
                )
            add(
                f"MACRO_CIF_{horizon}_OBSERVED_{cause.upper()}",
                f"{horizon}-month observed {cause} CIF",
                "B10",
                "m10",
                f"/cif/horizons/{horizon}/models/M1/{cause}/observed_default_cif",
                (
                    f"The recorded horizon-specific Aalen-Johansen {cause} CIF is the "
                    f"observed reference for the {horizon}-month rolling PIT "
                    f"comparison."
                ),
                metric="observed_cif",
                strength="PRIMARY" if horizon == 24 and cause == "payoff" else "SECONDARY",
                section="6.6",
                figure="F6",
                table="T5",
                qualifier=(
                    "The inherited payoff metric key observed_default_cif denotes the "
                    "selected cause; do not mislabel payoff as default"
                ),
            )
    # Post-hoc records and their existing diagnostic refits remain exploratory.
    for cid, address, name in [
        (
            "DIAG_SUPPORT",
            "/macro_shift/multivariate/evaluation_outside_reference_months",
            "out-of-support distinct months",
        ),
        ("DIAG_COMPOSITION", "/composition", "mortgage composition"),
        ("DIAG_SURVIVORS", "/survivor_composition_by_vintage", "survivor composition"),
        ("DIAG_PAYOFF_VARIANCE", "/score_accounting/brier/payoff", "payoff variance accounting"),
        (
            "DIAG_NO_EVENT",
            "/score_accounting/joint_loss_by_realized_event",
            "no-event excess loss accounting",
        ),
        ("DIAG_ANNUAL", "/annual_calibration", "annual calibration and 2020 versus 2023"),
        ("DIAG_COEFFICIENTS", "/coefficient_stability", "coefficient window instability"),
        ("DIAG_ORACLE", "/oracle", "same-sample intercept oracle"),
        ("DIAG_CIF_COMPONENTS", "/cif/horizons", "default/payoff component substitutions"),
        ("DIAG_ASSESSMENT", "/assessment/hypothesis_register", "frozen hypothesis assessment"),
    ]:
        aggregate(
            cid,
            name,
            "B11",
            "m11",
            address,
            cls="EXPLORATORY_DIAGNOSTIC",
            section="7",
            figure="F4" if cid == "DIAG_SUPPORT" else "F6",
            table="T6",
            qualifier=(
                "Post-hoc diagnostic association or error accounting; no causal "
                "feature importance or independently validated repair"
            ),
        )
    for feature in ["unemployment_level", "mortgage_30y_level", "hpi_yoy"]:
        aggregate(
            "DIAG_RANGE_" + feature.upper(),
            feature + " range-frequency evidence",
            "B11",
            "m11",
            "/macro_shift/distinct_month_weighted/" + feature,
            cls="EXPLORATORY_DIAGNOSTIC",
            section="7.1",
            figure="F4",
            table="T6",
        )
    for feature in ["unemployment_level", "unemployment_change_3m"]:
        aggregate(
            "DIAG_ZERO_" + feature.upper(),
            "zeroing " + feature,
            "B11",
            "m11",
            "/ablation/" + feature,
            cls="EXPLORATORY_DIAGNOSTIC",
            section="7.3",
            table="T6",
        )
    for window in value("m11", "/diagnostic_refits"):
        aggregate(
            "DIAG_GEOMETRY_" + window.upper(),
            "correlation geometry " + window,
            "B11",
            "m11",
            f"/diagnostic_refits/{window}/macro_correlation_geometry",
            cls="EXPLORATORY_DIAGNOSTIC",
            section="7.1",
            table="T6",
        )
    for model in ["P0", "P1", "P2", "LINEAR", "P3"]:
        for split, address in [
            ("development", f"/development/{model}/development"),
            ("exploratory", f"/metrics/{model}"),
        ]:
            for metric in [
                "joint_log_loss",
                "payoff_brier",
                "payoff_auc",
                "default_brier",
                "default_auc",
            ]:
                add(
                    f"REFI_{split.upper()}_{model}_{metric.upper()}",
                    f"{model} {split} {metric}",
                    "B12",
                    "p12",
                    address + "/" + metric,
                    (
                        f"In the EXPLORATORY_ONLY Task 12 Freddie study, {model} has the "
                        f"recorded {split} {metric}; improved payoff ranking does not "
                        f"establish restored temporal probability performance."
                    ),
                    cls="EXPLORATORY_DIAGNOSTIC",
                    metric=metric,
                    section="8",
                    figure="F7",
                    table="T7",
                    qualifier=(
                        "Hypothesis followed inspected Tasks 10 and 11 outcomes; original "
                        "coupon is a proxy, not current borrower contract or actual "
                        "refinancing behavior"
                    ),
                )
    for cid, address, name in [
        (
            "REFI_PRESPEC",
            "/prespecification",
            "refinancing formulas, units, models and original contract-rate proxy",
        ),
        ("REFI_PRESPEC_HASH", "/prespecification_sha256", "refinancing prespecification hash"),
        ("REFI_GAP", "/gap_shift", "refinancing gap shift"),
        ("REFI_PAIRED", "/paired", "paired facility and calendar uncertainty"),
        ("REFI_CAL", "/calibration", "refinancing calibration"),
        ("REFI_CALENDAR", "/calendar", "calendar results"),
        ("REFI_GROUPS", "/groups", "vintage, duration and regime results"),
        ("REFI_CIF", "/cif", "refinancing CIF and default spillover"),
        ("REFI_DECISION", "/decision", "mixed exploratory decision"),
        ("REFI_LIMITS", "/limitations", "refinancing limitations"),
    ]:
        aggregate(
            cid,
            name,
            "B12",
            "p12",
            address,
            cls="LIMITATION" if cid == "REFI_LIMITS" else "EXPLORATORY_DIAGNOSTIC",
            section="8",
            figure="F7",
            table="T7",
        )
    add(
        "FANNIE_PROPOSED",
        "proposed external extension",
        "B13_13S",
        "fannie",
        "/protocol_status",
        (
            "A Fannie Mae replication is proposed to test transport to a "
            "second GSE population, awaiting source-provenance closure and "
            "protocol freeze; it is draft, unauthorized and unexecuted."
        ),
        cls="PROPOSED_EXTERNAL_REPLICATION",
        strength="CONTEXTUAL",
        quantitative=False,
        section="9",
        figure="F8",
        table="T9",
        qualifier="DRAFT_NOT_YET_AUTHORIZED; Task 13A unauthorized; no external outcome analysis",
    )
    add(
        "FANNIE_PROVENANCE",
        "Fannie provenance gate",
        "B13_13S",
        "fannie",
        "/decision",
        (
            "Fannie source provenance requires provider confirmation; the "
            "accepted terms version remains unresolved."
        ),
        cls="LIMITATION",
        strength="CONTEXTUAL",
        quantitative=False,
        section="9",
        table="T9",
    )
    # Candidate mechanisms: explanation status is an audit interpretation, not a causal result.
    hypothesis_map = [
        ("support extrapolation", "H1", "SUPPORTED_DIAGNOSTICALLY"),
        ("coefficient instability", "H3", "SUPPORTED_DIAGNOSTICALLY"),
        ("composition/survivor shift", "H6", "PARTIALLY_SUPPORTED"),
        ("pandemic regime change", "H7", "PARTIALLY_SUPPORTED"),
        ("macro collinearity", "H5", "PARTIALLY_SUPPORTED"),
        ("intercept drift", "H2", "PARTIALLY_SUPPORTED"),
        ("payoff hazard amplification", "H1", "SUPPORTED_DIAGNOSTICALLY"),
        ("default hazard instability", "H4", "PARTIALLY_SUPPORTED"),
    ]
    mechanisms = []
    for index, (name, hid, status) in enumerate(hypothesis_map, 1):
        cid = f"MECHANISM_{index}"
        add(
            cid,
            name,
            "B11",
            "m11",
            f"/assessment/hypothesis_register/{hid}",
            name
            + (
                " is a possible explanatory hypothesis with the recorded "
                "diagnostic support; no causal mechanism has been established."
            ),
            cls="EXPLORATORY_HYPOTHESIS",
            strength="CONTEXTUAL",
            quantitative=False,
            section="7.5",
            qualifier=(
                "Pandemic alone and intercept drift dominance were NOT_SUPPORTED; "
                "evidence of some drift/regime association is weaker than "
                "dominance"
            ),
        )
        mechanisms.append(
            {
                "mechanism": name,
                "explanation_status": status,
                "claim_id": cid,
                "source_artifacts": [ref("m11", f"/assessment/hypothesis_register/{hid}")],
                "causal_finding": False,
                "qualification": (
                    "PARTIALLY_SUPPORTED for presence/contribution; pandemic alone and"
                    " intercept dominance NOT_SUPPORTED"
                )
                if hid in {"H2", "H7"}
                else "Separate causal contribution unidentified",
            }
        )
    rejected = [
        ("R_CAUSAL", "Macroeconomic variables caused mortgage models to fail", "PROHIBITED"),
        ("R_EXTERNAL", "Cross-provider mortgage transport failure was demonstrated", "PROHIBITED"),
        ("R_FANNIE_SEALED", "The Fannie protocol is sealed or pre-registered", "PROHIBITED"),
        ("R_FANNIE_RESULT", "Fannie independently replicated the Freddie result", "PROHIBITED"),
        (
            "R_REGULATORY",
            "This is a regulator validated IFRS 9 or IRB PD/LGD/EAD model",
            "PROHIBITED",
        ),
        ("R_REFI_SOLVES", "Refinancing incentive solves temporal payoff transport", "PROHIBITED"),
        ("R_BEHAVIOR", "Borrowers refinance at a demonstrated gap threshold", "UNSUPPORTED"),
        (
            "R_LIFETIME",
            "Conditional-entry estimates are lifetime risk from origination",
            "PROHIBITED",
        ),
        (
            "R_GENERAL",
            "National macro variables universally worsen consumer credit prediction",
            "UNSUPPORTED",
        ),
        ("R_INDEPENDENCE", "Monthly intervals are independent borrowers", "PROHIBITED"),
    ]
    for cid, text, cls in rejected:
        add(
            cid,
            "rejected candidate",
            "B13_13S" if "FANNIE" in cid else "B10",
            "fannie" if "FANNIE" in cid else "m10",
            "/scope" if "FANNIE" in cid else "/limitations",
            text,
            cls=cls,
            strength="CONTEXTUAL",
            quantitative=False,
            section="11",
        )
        claims[-1]["allowed_language"] = (
            "Do not advance this candidate; retain the narrower qualified claims"
        )
        claims[-1]["prohibited_language"].append(text)
        claims[-1]["verification_status"] = (
            "INSUFFICIENT_EVIDENCE" if cls == "UNSUPPORTED" else "NOT_APPLICABLE"
        )
        claims[-1]["task15_admission"] = "BLOCKED"
    write(
        "track_b_claim_evidence_registry.json",
        {
            "version": "task14-v1",
            "base_commit": BASE,
            "source_of_truth_hierarchy": [
                {"level": n, "description": t}
                for n, t in enumerate(
                    [
                        "Frozen machine-readable experiment artifacts/manifests",
                        "Frozen generated validation reports",
                        "Task verification evidence",
                        "Experiment documentation",
                        "Git history/preservation",
                        "Human summaries",
                    ],
                    1,
                )
            ],
            "conflict_policy": (
                "EVIDENCE_CONFLICT; retain both sources and a documented "
                "resolution or carry-forward limitation"
            ),
            "claims": claims,
        },
    )
    write("mechanism_claim_gate.json", {"causal_findings": False, "mechanisms": mechanisms})
    for exp, design in experiments.items():
        design["experiment_id"] = exp
        design["claim_ids"] = [c["claim_id"] for c in claims if c["experiment_id"] == exp]
        design["source_artifacts"] = [ref(k) for k in design.pop("source_keys")]
    write(
        "experiment_design_freeze.json",
        {
            "version": "task14-v1",
            "new_experiments": False,
            "experiments": list(experiments.values()),
        },
    )
    boundary_specs = [
        ("mortgage_vs_consumer", "Selected mortgage facilities", "General consumer credit"),
        (
            "freddie_vs_us",
            "Deterministic Freddie samples",
            "All US mortgages or industry-wide transport",
        ),
        (
            "facility_vs_borrower",
            "Facility-level clustered uncertainty",
            "Independent or uniquely linked borrowers",
        ),
        (
            "mortgage_vs_revolving_ead",
            "No EAD empirical finding in this paper",
            "Revolving commitments or CCF",
        ),
        (
            "severity_vs_lgd",
            "Aggregate disposition-loss proxy as background only",
            "Timed workout or regulatory LGD",
        ),
        ("sicr", "Research considerations only", "Validated IFRS 9 SICR or staging"),
        (
            "research_vs_regulatory_pd",
            "Composite monthly adverse-event proxy",
            "Regulatory default equivalence",
        ),
        (
            "conditional_vs_lifetime",
            "Conditional-entry risk within support",
            "Lifetime-from-origination risk",
        ),
        (
            "national_vs_geographic",
            "National PIT macro series",
            "Geographic macro or household conditions",
        ),
        (
            "temporal_vs_external",
            "Temporal Freddie evaluation",
            "Completed external Fannie validation",
        ),
        (
            "association_vs_causality",
            "Predictive associations and diagnostics",
            "Identified causal macro mechanisms",
        ),
        (
            "pit_macro_vs_mortgage",
            "Vintage-aware macro information",
            "Historically verified operational mortgage information",
        ),
        (
            "historical_vs_forecast",
            "Rolling historical PIT CIF paths",
            "Prospective entry-time macro forecasts",
        ),
    ]
    write(
        "generalization_boundaries.json",
        {
            "boundaries": [
                {
                    "boundary_id": bid,
                    "allowed_scope": yes,
                    "prohibited_scope": no,
                    "source_artifacts": [
                        ref("event", "/prohibited_claims"),
                        ref("m10", "/limitations"),
                    ],
                }
                for bid, yes, no in boundary_specs
            ],
            "regulatory_prohibited_claims": [
                "IFRS 9 model",
                "IRB model",
                "regulatory PD model",
                "regulatory LGD model",
                "regulatory EAD model",
                "production bank model",
                "regulator validated",
                "audit approved",
            ],
            "allowed_relevance": (
                "Credit-risk research, temporal validation and model-risk/IFRS "
                "9/IRB research considerations only"
            ),
        },
    )
    facts = []
    for key, addresses in {
        "pd": [
            "/source_sha256",
            "/sample_sha256",
            "/panel_sha256",
            "/cohort_reproduction/cohorts",
            "/cohort_reproduction/hashes",
            "/model_specifications/design",
            "/limitations",
        ],
        "survival": [
            "/survival_audit",
            "/protocol/origin",
            "/protocol/events",
            "/protocol/risk_set",
            "/protocol/knowledge_time",
        ],
        "manifest": [
            "/components",
            "/frozen_facilities",
            "/combined_sample_sha256",
            "/key_definition",
        ],
        "population": ["/completed_performance_rows", "/chronology", "/apc", "/limitations"],
        "support": ["/totals", "/macro_support/designs"],
        "event": ["/event", "/labels", "/time"],
        "m10": ["/split_counts", "/protocol/splits", "/limitations"],
        "p12": ["/counts", "/prespecification/population", "/limitations"],
    }.items():
        for address in addresses:
            facts.append(
                {
                    "fact_id": key + ":" + address,
                    "value": value(key, address),
                    "source_artifacts": [ref(key, address)],
                }
            )
    write(
        "dataset_facts_registry.json",
        {
            "facts": facts,
            "missing_information": {
                "borrower_identity": "UNAVAILABLE",
                "mortgage_operational_knowledge_time": "UNVERIFIED",
                "exact_origination_date_from_first_payment": "PROHIBITED",
                "release_specific_generalization": (
                    "Retrospective Release 47 research; source metadata do not "
                    "establish historic operational availability"
                ),
            },
        },
    )
    units = []
    for exp, key, address, n in [
        (
            "B5",
            "pd",
            "/clustered_uncertainty",
            value("pd", "/cohort_reproduction/cohorts/evaluation/loans"),
        ),
        (
            "B6",
            "survival",
            "/uncertainty",
            value("survival", "/survival_audit/partitions/evaluation/facilities"),
        ),
        (
            "B6",
            "survival",
            "/monthly_uncertainty",
            value("survival", "/survival_audit/partitions/evaluation/facilities"),
        ),
        ("B10", "m10", "/paired_facility", value("m10", "/paired_facility/clusters")),
        ("B10", "m10", "/paired_calendar", value("m10", "/paired_calendar/clusters")),
        ("B12", "p12", "/paired/facility", value("p12", "/paired/facility/clusters")),
        ("B12", "p12", "/paired/calendar_year", value("p12", "/paired/calendar_year/clusters")),
    ]:
        record = value(key, address)
        units.append(
            {
                "experiment_id": exp,
                "observation_unit": experiments[exp]["unit"],
                "cluster_unit": record["unit"],
                "resampling_unit": record["unit"],
                "number_of_clusters": n,
                "number_of_bootstrap_draws": record.get("draws", record.get("requested_draws")),
                "interval_type": "Percentile 95; fixed-model"
                if exp in {"B10", "B12"}
                else "Frozen percentile intervals; method recorded in protocol/report",
                "uncertainty_record": record,
                "source_artifacts": [ref(key, address)],
                "limitations": (
                    "No training, borrower dependence or future macro-path "
                    "uncertainty; calendar has eight year blocks including partial "
                    "2026"
                ),
            }
        )
    write(
        "statistical_unit_audit.json",
        {
            "units": units,
            "intervals_are_not_independent_observations": True,
            "point_calibration_CI": None,
            "diagnostic_refit_coefficient_CI": None,
        },
    )
    conflicts = [
        {
            "conflict_id": "C_MANIFEST",
            "status": "EVIDENCE_CONFLICT",
            "sources": [
                ref("old_manifest", "/decision"),
                ref("manifest", "/complete"),
                ref("population", "/decision"),
            ],
            "resolution": "RESOLVED_DOCUMENTED_SUPERSESSION",
            "reason": (
                "Earlier 60000-facility STOP remains historical; Task 8C recovery "
                "completes seven vintages and 140000 facilities. Use recovery "
                "manifest for paper population; never rewrite old evidence."
            ),
        },
        {
            "conflict_id": "C_BRIER_PRECISION",
            "status": "EVIDENCE_CONFLICT",
            "sources": [
                ref("m10", "/primary/M1/scores/payoff_brier"),
                ref("m11", "/score_accounting/brier/payoff/M1/brier"),
            ],
            "resolution": "RESOLVED_FLOATING_POINT_ACCOUNTING_PRECISION",
            "values": [
                value("m10", "/primary/M1/scores/payoff_brier"),
                value("m11", "/score_accounting/brier/payoff/M1/brier"),
            ],
            "reason": (
                "Different frozen numerical accounting paths; preserve both exact "
                "values. Task 10 proper-score artifact governs headline, Task 11 "
                "decomposition remains diagnostic."
            ),
        },
        {
            "conflict_id": "C_FANNIE_TERMINOLOGY",
            "status": "EVIDENCE_CONFLICT",
            "sources": [
                ref("fannie", "/protocol_status"),
                ref("fannie", "/scope/task13a_authorized"),
            ],
            "lower_priority_source": (
                "Human proposal describing a sealed/pre-registered Fannie extension"
            ),
            "resolution": "RESOLVED_HIGHER_PRIORITY_STATUS",
            "reason": (
                "Draft, unauthorized and unexecuted; no sealed or pre-registered "
                "Fannie assertion admitted."
            ),
        },
    ]
    write("evidence_conflicts.json", {"conflicts": conflicts, "unresolved_primary_conflicts": []})
    headlines = [
        (
            "H1",
            "QUALIFY",
            [
                "PD_LOGISTIC_ROC_AUC",
                "PD_LOGISTIC_MEAN_PROBABILITY",
                "PD_LOGISTIC_OBSERVED_RATE",
                "PD_CALIBRATION_INTERCEPT",
                "PD_CALIBRATION_SLOPE",
            ],
            "Meaningful ranking and material underprediction in Task 5, not deployable calibration",
        ),
        (
            "H2",
            "QUALIFY",
            ["SURV_60_NAIVE_NET_DEFAULT", "SURV_60_DEFAULT_CIF"],
            (
                "Different estimands: descriptive net risk versus observed "
                "competing CIF, conditional entry"
            ),
        ),
        (
            "H3",
            "VERIFY",
            [
                "MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS",
                "MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS",
                "MACRO_PRIMARY_M1_JOINT_LOG_LOSS",
                "MACRO_PRIMARY_M2_JOINT_LOG_LOSS",
                "MACRO_DELTA_FACILITY_JOINT_LOG_LOSS",
                "MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS",
            ],
            "Task 10 design-specific development gain and temporal deterioration",
        ),
        (
            "H4",
            "VERIFY",
            [
                "MACRO_PRIMARY_M1_PAYOFF_AUC",
                "MACRO_PRIMARY_M2_PAYOFF_AUC",
                "MACRO_PRIMARY_M1_PAYOFF_BRIER",
                "MACRO_PRIMARY_M2_PAYOFF_BRIER",
                "MACRO_CAL_M1_PAYOFF",
                "MACRO_CAL_M2_PAYOFF",
            ],
            "Payoff ranking gain accompanies worse proper scores/calibration in this design",
        ),
        (
            "H5",
            "QUALIFY",
            [
                "MACRO_CIF_24_OBSERVED_PAYOFF",
                "MACRO_CIF_24_M1_PAYOFF",
                "MACRO_CIF_24_M2_PAYOFF",
                "DIAG_CIF_COMPONENTS",
            ],
            (
                "Large payoff CIF discrepancy; component substitution traces "
                "numerical propagation, not a causal mechanism"
            ),
        ),
        (
            "H6",
            "QUALIFY",
            ["DIAG_SUPPORT", "DIAG_COEFFICIENTS", "DIAG_ASSESSMENT"],
            "Post-hoc diagnostic support only",
        ),
        (
            "H7",
            "QUALIFY",
            [
                "REFI_EXPLORATORY_P1_JOINT_LOG_LOSS",
                "REFI_EXPLORATORY_P2_JOINT_LOG_LOSS",
                "REFI_EXPLORATORY_P1_PAYOFF_AUC",
                "REFI_EXPLORATORY_P2_PAYOFF_AUC",
                "REFI_PAIRED",
                "REFI_DECISION",
            ],
            "EXPLORATORY_ONLY; small payoff Brier gain is not robust to calendar block uncertainty",
        ),
        (
            "H8",
            "VERIFY",
            ["FANNIE_PROPOSED", "FANNIE_PROVENANCE"],
            "Cross-provider transport has not been tested",
        ),
    ]
    write(
        "headline_claim_audit.json",
        {
            "headlines": [
                {"headline_id": h, "decision": verdict, "claim_ids": ids, "qualification": why}
                for h, verdict, ids, why in headlines
            ]
        },
    )
    questions = [
        (
            "Does the frozen PIT macro feature set improve temporal "
            "competing-risk probability quality relative to the mortgage "
            "baseline in the evaluated Freddie population?"
        ),
        (
            "How do payoff ranking and calibration diverge under temporal "
            "shift in the frozen mortgage competing-risk design?"
        ),
        (
            "How do competing payoff and historical rolling PIT hazard "
            "mappings alter conditional cumulative incidence?"
        ),
        (
            "Does the exploratory asymmetric original-coupon versus PIT "
            "market-rate gap restore temporal probability performance?"
        ),
    ]
    contributions = [
        ("EMPIRICAL_FINDING", "H3"),
        ("NEGATIVE_RESULT", "H4"),
        ("METHOD", "H2"),
        ("DIAGNOSTIC_INSIGHT", "H6"),
        ("VALIDATION_DESIGN", "H1"),
        ("PROPOSED_EXTERNAL_EXTENSION", "H8"),
        ("REPRODUCIBILITY/GOVERNANCE", "H8"),
    ]
    titles = [
        (
            "Temporal Transport of Mortgage Competing-Risk Probabilities with "
            "Point-in-Time Macroeconomic Information"
        ),
        "Ranking Gains and Calibration Losses in Temporally Shifted Mortgage Risk Models",
        (
            "When Mortgage Probabilities Fail to Travel Across Time: Default, "
            "Payoff, and Macroeconomic Information"
        ),
        (
            "Point-in-Time Macroeconomic Features and Temporal Deterioration "
            "in Freddie Mortgage Risk Modeling"
        ),
        "Competing Payoff and Probability Calibration under Mortgage Temporal Shift",
        "Mortgage Default and Payoff under Temporal Shift: A Frozen-Evidence Validation Study",
        "Temporal Mortgage Probability Transport and an Exploratory Refinancing-Incentive Study",
    ]
    write(
        "paper_contribution_candidates.json",
        {
            "research_questions": [
                {
                    "rank": i,
                    "formulation": q,
                    "causal": False,
                    "headline_ids": {1: ["H3", "H4"], 2: ["H4"], 3: ["H2", "H5"], 4: ["H7"]}[i],
                }
                for i, q in enumerate(questions, 1)
            ],
            "contributions": [
                {
                    "category": kind,
                    "headline_id": h,
                    "claim_ids": next(ids for hid, _, ids, _ in headlines if hid == h),
                    "original_method_novelty_claimed": False,
                }
                for kind, h in contributions
            ],
            "titles": [{"title": t, "status": "PROVISIONAL"} for t in titles],
        },
    )
    figure_specs = [
        (
            "F1",
            "Research/data/validation architecture",
            ["PIT_SUPPORT", "PD_SAMPLE", "MACRO_LEDGER"],
            "MAIN_TEXT",
        ),
        (
            "F2",
            "Competing-risk state diagram and descriptive net-risk/CIF contrast",
            ["SURV_DESIGN", "SURV_60_NAIVE_NET_DEFAULT", "SURV_60_DEFAULT_CIF"],
            "MAIN_TEXT",
        ),
        (
            "F3",
            "Development versus temporal model scores",
            ["MACRO_DEVELOPMENT_M0_JOINT_LOG_LOSS"] + headlines[2][2],
            "MAIN_TEXT",
        ),
        (
            "F4",
            "Temporal macro support/extrapolation",
            [
                "DIAG_SUPPORT",
                "DIAG_RANGE_UNEMPLOYMENT_LEVEL",
                "DIAG_RANGE_MORTGAGE_30Y_LEVEL",
                "DIAG_RANGE_HPI_YOY",
            ],
            "MAIN_TEXT",
        ),
        (
            "F5",
            "Payoff discrimination versus calibration; annual panels",
            headlines[3][2] + ["DIAG_ANNUAL"],
            "MAIN_TEXT",
        ),
        (
            "F6",
            "Observed versus modeled conditional CIF, rolling historical PIT",
            headlines[4][2],
            "MAIN_TEXT",
        ),
        (
            "F7",
            "Exploratory refinancing-gap shift and P1/P2 comparison",
            headlines[6][2] + ["REFI_GAP"],
            "MAIN_TEXT",
        ),
        (
            "F8",
            "Freddie evidence to proposed Fannie extension",
            ["FANNIE_PROPOSED", "FANNIE_PROVENANCE"],
            "SUPPLEMENT",
        ),
    ]
    table_specs = [
        (
            "T1",
            "Dataset/cohort facts",
            [
                "PD_COUNTS",
                "SURV_COUNTS",
                "COHORT_RECOVERY",
                "COHORT_ROWS",
                "COHORT_COMPONENTS",
                "PIT_COUNTS",
            ],
            "MAIN_TEXT",
        ),
        (
            "T2",
            "Frozen model specifications",
            ["PD_FEATURES", "SURV_DESIGN", "MACRO_ACTUAL_FEATURES", "REFI_PRESPEC"],
            "MAIN_TEXT",
        ),
        (
            "T3",
            "Expanded 12-month PD",
            [c["claim_id"] for c in claims if c["table_candidate"] == "T3"],
            "MAIN_TEXT",
        ),
        (
            "T4",
            "Conditional competing-risk validation",
            [c["claim_id"] for c in claims if c["table_candidate"] == "T4"],
            "MAIN_TEXT",
        ),
        (
            "T5",
            "Macro ladder and cumulative incidence",
            [c["claim_id"] for c in claims if c["table_candidate"] == "T5"],
            "MAIN_TEXT",
        ),
        (
            "T6",
            "Temporal and calibration diagnostics",
            [c["claim_id"] for c in claims if c["table_candidate"] == "T6"],
            "SUPPLEMENT",
        ),
        (
            "T7",
            "Exploratory refinancing results",
            [c["claim_id"] for c in claims if c["table_candidate"] == "T7"],
            "MAIN_TEXT",
        ),
        (
            "T8",
            "Generalization and limitation boundaries",
            ["MACRO_LIMITS", "PD_LIMITS", "REFI_LIMITS"],
            "MAIN_TEXT",
        ),
        (
            "T9",
            "Proposed Fannie status and mapping",
            ["FANNIE_PROPOSED", "FANNIE_PROVENANCE"],
            "SUPPLEMENT",
        ),
    ]
    lookup = {c["claim_id"]: c for c in claims}

    def presentation(spec, kind):
        ident, purpose, ids, placement = spec
        refs = [
            r
            for cid in ids
            for r in lookup[cid]["source_artifacts"]
            if r["role"] in {"point_estimate", "method_or_status"}
        ]
        return {
            kind + "_id": ident,
            "purpose": purpose,
            "claim_ids": ids,
            "source_artifacts": refs,
            "data_needed": (
                "Only pinned aggregate values and documented design; each "
                "quantitative cell must use a claim JSON pointer"
            ),
            "can_generate_from_existing_frozen_artifacts": True,
            "requires_raw_data": False,
            "requires_new_calculation": False,
            "main_or_supplement": placement,
            "status": "PLAN_ONLY_NOT_GENERATED",
            "scope": (
                "Redrawing/formatting frozen values permitted in Task 15; new "
                "curves, bins, bootstrap or outcome metrics prohibited"
            ),
            "result_cells": [
                {"claim_id": cid, "source_artifacts": lookup[cid]["source_artifacts"]}
                for cid in ids
                if lookup[cid]["quantitative"]
            ],
        }

    figures = [presentation(s, "figure") for s in figure_specs]
    tables = [presentation(s, "table") for s in table_specs]
    write(
        "figure_registry.json",
        {
            "figures": figures,
            "raw_only_candidates": [
                {
                    "purpose": (
                        "New borrower-level curves, new calibration bins or new uncertainty bands"
                    ),
                    "requires_raw_data": True,
                    "requires_new_calculation": True,
                    "status": "DO_NOT_GENERATE",
                }
            ],
        },
    )
    write("table_registry.json", {"tables": tables})
    reproductions = [
        {
            "artifact_id": x.get("figure_id", x.get("table_id")),
            "status": "REPRODUCIBLE_FROM_FROZEN_AGGREGATES",
            "claim_ids": x["claim_ids"],
            "recompute_empirical_experiment": "REPRODUCIBLE_WITH_PROVIDER_DATA_ACCESS",
            "publication_status": "PUBLICATION_REVIEW_REQUIRED",
            "limitations": (
                "Formatting is reproducible; no claim of independent reproduction "
                "of underlying licensed experiment"
            ),
        }
        for x in figures + tables
    ]
    reproductions += [
        {
            "artifact_id": c["claim_id"],
            "status": "NOT_CURRENTLY_REPRODUCIBLE"
            if c["experiment_id"] == "B13_13S"
            else "REPRODUCIBLE_FROM_FROZEN_AGGREGATES",
            "scope": "Proposed experiment has no empirical result"
            if c["experiment_id"] == "B13_13S"
            else "Exact paper-facing evidence lookup only",
            "source_artifacts": c["source_artifacts"],
        }
        for c in claims
        if c["claim_class"] not in {"UNSUPPORTED", "PROHIBITED"}
    ]
    write(
        "reproducibility_matrix.json",
        {
            "entries": reproductions,
            "code": "PUBLICLY_REPRODUCIBLE",
            "underlying_loan_panel": "PRIVATE_DATA_REQUIRED",
            "proposed_fannie_experiment": "NOT_CURRENTLY_REPRODUCIBLE",
        },
    )
    write(
        "publication_data_boundary.json",
        {
            "legal_permission_determined": False,
            "publication_status": "PUBLICATION_REVIEW_REQUIRED",
            "license_sources": [
                ref("fannie", "/license"),
                {
                    "path": "docs/track_b/LONGITUDINAL_DATA_CONTRACT.md",
                    "sha256_lf": sha(ROOT / "docs/track_b/LONGITUDINAL_DATA_CONTRACT.md"),
                    "evidence_level": 4,
                },
            ],
            "classes": [
                {"object": name, "status": status, "restriction": rule}
                for name, status, rule in [
                    ("raw records", "PRIVATE_NON_COMMITTABLE", "Exclude from preprint/repository"),
                    (
                        "loan identifiers",
                        "PRIVATE_NON_COMMITTABLE",
                        "No direct or reconstructible identifiers",
                    ),
                    ("archive names", "PUBLIC_METADATA_ONLY", "No personal paths/account labels"),
                    ("archive hashes", "PUBLIC_METADATA_ONLY", "Integrity metadata, not records"),
                    (
                        "aggregated counts",
                        "PUBLICATION_REVIEW_REQUIRED",
                        "No sparse reconstructible cells; existing suppression applies",
                    ),
                    (
                        "model metrics",
                        "PUBLICATION_REVIEW_REQUIRED",
                        "Use frozen aggregate references",
                    ),
                    (
                        "coefficient tables",
                        "PUBLICATION_REVIEW_REQUIRED",
                        "Predictive associations; no causal or household reconstruction",
                    ),
                    (
                        "figures",
                        "PUBLICATION_REVIEW_REQUIRED",
                        "Aggregate-only; no raw scatter or IDs",
                    ),
                    (
                        "schemas",
                        "PUBLIC_DOCUMENTATION_REFERENCES",
                        "Cite provider; do not redistribute documents without review",
                    ),
                    (
                        "code",
                        "PUBLICLY_REPRODUCIBLE",
                        "No credentials, identifiers or bundled licensed data",
                    ),
                    (
                        "provider documentation",
                        "REFERENCE_ONLY",
                        "Link official sources; avoid reproducing full licensed documents",
                    ),
                ]
            ],
        },
    )
    sections = [
        ("1", "Introduction — motivation slots only", ["H3", "H4", "H8"]),
        ("2", "Related Work — search plan only", []),
        ("3.1", "Longitudinal mortgage risk", ["H1"]),
        ("3.2", "Default and payoff as competing events", ["H2"]),
        ("3.3", "Temporal transport", ["H3", "H8"]),
        ("3.4", "Point-in-time information constraint", ["H3"]),
        ("4.1", "Freddie source", ["COHORT_COMPONENTS"]),
        (
            "4.2",
            "Cohort construction",
            ["PD_COUNTS", "SURV_COUNTS", "COHORT_RECOVERY", "COHORT_ROWS"],
        ),
        ("4.3", "Event definitions", ["SURV_DESIGN"]),
        ("4.4", "Censoring", ["SURV_DESIGN"]),
        ("4.5", "Macro data", ["PIT_SERIES", "PIT_FEATURES", "PIT_PROVENANCE", "PIT_SUPPORT"]),
        ("4.6", "Data limitations", ["PIT_GAPS", "PD_LIMITS"]),
        ("5.1", "12-month PD baseline", ["PD_FEATURES"]),
        ("5.2", "Discrete-time competing risks", ["SURV_DESIGN"]),
        ("5.3", "M0/M1/M2 ladder", ["MACRO_ACTUAL_FEATURES"]),
        ("5.4", "Temporal validation", ["MACRO_DESIGN_COUNTS", "MACRO_LEDGER"]),
        (
            "5.5",
            "Calibration",
            ["PD_CALIBRATION_INTERCEPT", "MACRO_CAL_M1_PAYOFF", "MACRO_CAL_M2_PAYOFF"],
        ),
        ("5.6", "CIF estimation", ["MACRO_CIF_24_SUPPORT"]),
        (
            "5.7",
            "Cluster bootstrap",
            ["MACRO_DELTA_FACILITY_JOINT_LOG_LOSS", "MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS"],
        ),
        ("5.8", "Support diagnostics", ["DIAG_SUPPORT"]),
        ("6.1", "Expanded PD results", ["H1"]),
        ("6.2", "Competing-risk foundation", ["H2"]),
        ("6.3", "Macro development performance", ["H3"]),
        ("6.4", "Temporal deterioration", ["H3"]),
        ("6.5", "Discrimination versus calibration", ["H4"]),
        ("6.6", "Cumulative incidence", ["H5"]),
        ("7.1", "Covariate support", ["DIAG_SUPPORT", "DIAG_RANGE_UNEMPLOYMENT_LEVEL"]),
        ("7.2", "Composition shift", ["DIAG_COMPOSITION", "DIAG_SURVIVORS"]),
        ("7.3", "Payoff instability", ["DIAG_PAYOFF_VARIANCE", "DIAG_ZERO_UNEMPLOYMENT_LEVEL"]),
        ("7.4", "Calendar regimes", ["DIAG_ANNUAL"]),
        ("7.5", "Limits of attribution", ["H6"]),
        ("8", "Exploratory refinancing incentive", ["H7", "REFI_GAP", "REFI_PAIRED"]),
        ("9", "Proposed external replication — no results", ["H8"]),
        ("10", "Discussion — bounded implications slots", ["H3", "H4", "H6", "H7"]),
        (
            "11",
            "Limitations",
            ["MACRO_LIMITS", "PD_LIMITS", "PIT_APC", "REFI_LIMITS", "FANNIE_PROVENANCE"],
        ),
        ("12", "Reproducibility and governance", ["MACRO_LEDGER", "FANNIE_PROVENANCE"]),
        ("13", "Conclusion — verified claim slots only", ["H2", "H3", "H4"]),
        (
            "Appendix",
            "Supplement — sensitivities and provenance",
            ["B3_FEASIBILITY", "SURV_DURATION", "MACRO_REDUCED_SENS", "MACRO_RATE_SENS"],
        ),
    ]
    parents = {
        "3": "Problem Formulation",
        "4": "Data",
        "5": "Methods",
        "6": "Results",
        "7": "Diagnostic Analysis",
    }
    expanded_sections = []
    for number, title, ids in sections:
        if number.endswith(".1") and number.split(".")[0] in parents:
            parent = number.split(".")[0]
            expanded_sections.append(
                (
                    parent,
                    parents[parent],
                    [c["claim_id"] for c in claims if parent in c["manuscript_sections"]],
                )
            )
        expanded_sections.append((number, title, ids))
    sections = expanded_sections
    hlookup = {h: ids for h, _, ids, _ in headlines}
    architecture = (
        "# Task 14 arXiv manuscript architecture\n\nPlanning outline only. "
        "No abstract, manuscript introduction or conclusions written.\n\n"
    )
    section_map = []
    for number, title, ids in sections:
        ids = [c for ident in ids for c in hlookup.get(ident, [ident])]
        section_map.append(
            {"section_id": number, "title": title, "claim_ids": list(dict.fromkeys(ids))}
        )
        architecture += (
            f"## {number}. {title}\n\n- Claim IDs: "
            + (", ".join(ids) or "None; literature search pending, no empirical claim")
            + (
                "\n- Gate: retain source population, clock, estimand, uncertainty "
                "and exploratory status.\n\n"
            )
        )
    architecture += (
        "## Task 15 drafting constraints\n\n- Fannie: proposed external "
        "replication awaiting provenance closure and protocol freeze; "
        "DRAFT_NOT_YET_AUTHORIZED.\n- Temporal Freddie deterioration only; "
        "no demonstrated cross-provider failure.\n- Task 6 60-month "
        "evidence: descriptive net-risk/CIF contrast, not supported model "
        "forecast.\n- Task 10/12 CIF: historical rolling PIT paths, not "
        "known-at-entry prospective paths.\n- Task 11 diagnostics and Task "
        "12 exploration stay labeled; no causal explanation.\n- Literature "
        "novelty and publication permission remain review items.\n"
    )
    write_text("arxiv_manuscript_architecture.md", architecture)
    write("manuscript_section_map.json", {"sections": section_map})
    buckets = [
        (
            "Mortgage default prediction",
            "mortgage default; delinquency landmarks; temporal validation",
            "Loan-level longitudinal mortgage designs; exact target and cohort description",
        ),
        (
            "Mortgage competing risks",
            "mortgage default prepayment competing risks cumulative incidence",
            "Explicit default/payoff/censoring estimands and conditional entry",
        ),
        (
            "Prepayment modeling",
            "mortgage payoff maturity prepayment hazard",
            "Distinguish observed payoff from identified voluntary refinancing",
        ),
        (
            "Macro-conditioned credit risk",
            "real-time vintage macro mortgage hazard ALFRED",
            "Vintage-aware information sets; disclose revised-data use",
        ),
        (
            "Temporal/domain shift",
            "mortgage credit temporal transport out-of-time validation",
            "Separate temporal from cross-provider validation",
        ),
        (
            "Calibration under shift",
            "probability calibration distribution shift proper scoring",
            "Evaluate ranking alongside Brier/log loss; uncertainty under dependence",
        ),
        (
            "Survival/CIF evaluation",
            "Aalen Johansen IPCW cumulative dynamic AUC integrated Brier competing risks",
            "Specify controls, censoring assumptions and horizon support",
        ),
        (
            "Refinancing incentive",
            "mortgage original coupon market rate incentive burnout refinancing",
            "Identify proxy rates, transaction costs and behavioral identification limits",
        ),
        (
            "Regime/model risk",
            "mortgage model validation pandemic regime change coefficient stability",
            "Predictive diagnostics without unsupported causal attribution",
        ),
    ]
    literature = (
        "# Related-work search plan\n\nNo literature review or citations "
        "fabricated. Search execution reserved for Task 15; novelty not "
        "established.\n\n| Bucket | Search concepts | Inclusion criteria |\n|"
        " --- | --- | --- |\n"
    ) + "\n".join("| " + " | ".join(b) + " |" for b in buckets)
    literature += (
        "\n\n- Prefer original papers and authoritative methodological "
        "sources; verify DOI, version and publication status.\n- Record "
        "population, event definition, information clock, evaluation "
        "design, competing risk and uncertainty.\n- Include contradictory "
        "and null findings; exclude unverifiable citations and papers with"
        " incompatible estimands from direct metric comparison.\n- "
        "Deduplicate working-paper/journal versions; preprint status is "
        "explicit.\n"
    )
    write_text("related_work_search_plan.md", literature)
    write_text(
        "arxiv_positioning.md",
        (
            "# Provisional arXiv positioning\n\n- Artifact target: research "
            "preprint; no peer-reviewed-publication claim.\n- Category "
            "candidates to investigate for fit after drafting: q-fin.RM (Risk "
            "Management), stat.AP (Applications), cs.LG (Machine Learning). "
            "Category names checked against the [official "
            "taxonomy](https://arxiv.org/category_taxonomy) on 7 October 2026;"
            " no final category selected.\n- Eligibility, endorsement, "
            "submission compliance and licensing: not established; verify "
            "separately before submission.\n- Scope: temporal probability "
            "validation in selected Freddie mortgage cohorts, with "
            "diagnostic/exploratory sections explicitly labeled.\n- No "
            "external-replication, causal-mechanism or regulatory-validation "
            "title claim.\n- Task 14 makes no manuscript submission and writes "
            "no abstract.\n"
        ),
    )
    # Audit coverage is structural: each required topic maps to exact copied values/design pointers.
    coverage = {
        "expanded_pd": [c["claim_id"] for c in claims if c["experiment_id"] == "B5"],
        "competing_risk": [c["claim_id"] for c in claims if c["experiment_id"] == "B6"],
        "pit_and_population": [c["claim_id"] for c in claims if c["experiment_id"] == "B7_9A"],
        "macro_primary_and_sensitivities": [
            c["claim_id"] for c in claims if c["experiment_id"] == "B10"
        ],
        "diagnostics_and_mechanisms": [
            c["claim_id"] for c in claims if c["experiment_id"] == "B11"
        ],
        "refinancing": [c["claim_id"] for c in claims if c["experiment_id"] == "B12"],
        "fannie": ["FANNIE_PROPOSED", "FANNIE_PROVENANCE"],
    }
    write(
        "task14_audit_coverage.json",
        {
            "coverage": coverage,
            "all_values_copied_not_recomputed": True,
            "missing_uncertainty": [
                "Task 5 calibration confidence intervals",
                "Task 11 diagnostic-refit coefficient confidence intervals",
                "Task 10/12 CIF confidence bands where not recorded",
            ],
            "headline_floating_precision": (
                "Exact JSON numbers; display rounding only, not new metrics"
            ),
        },
    )
    names = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .strip("\0")
        .split("\0")
    )
    prior = read("docs/track_b/fannie_source_preservation_manifest.json")
    write(
        "task14_preservation_manifest.json",
        {
            "base_commit": BASE,
            "public_lf_hashes": {n: sha(ROOT / n) for n in names},
            "private_byte_hashes": prior["private_byte_hashes"],
            "git_refs": prior["git_refs"],
            "external_archive_identity": read(
                "reports/track_b/fannie_release_archive_structure.json"
            ),
        },
    )
    write(
        "paper_evidence_manifest.json",
        {
            "version": "task14-v1",
            "base_commit": BASE,
            "freeze_meaning": (
                "Paper-facing evidence inventory only; NOT dataset "
                "pre-registration or external replication sealing"
            ),
            "source_artifact_hashes_lf": {p: sha(ROOT / p) for p in FILES.values()},
            "paper_artifact_hashes_lf": {
                p.relative_to(ROOT).as_posix(): sha(p) for p in PAPER.iterdir() if p.is_file()
            },
            "new_empirical_calculations": False,
        },
    )
    print(
        json.dumps(
            {
                "claims": len(claims),
                "figures": len(figures),
                "tables": len(tables),
                "paper_files": len(list(PAPER.iterdir())),
                "new_empirical_calculations": False,
            }
        )
    )


if __name__ == "__main__":
    build()
