"""Reproducible aggregate-only baseline report from the frozen private panel."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import pandas as pd

from credit_risk.track_b.data.schemas import digest, load_protocol
from credit_risk.track_b.pd.baseline import (
    DEV_END,
    EVAL_START,
    FEATURES,
    SALT,
    SEED,
    cohort,
    cohort_hash,
    counts,
    diagnostics,
    features,
    fit_baselines,
    gate,
    split,
)

SOURCE = "a82bc0f1efffccfb494b2b33f61877428bb4a6443c1a73d44bc7ee24d77aa45d"
SAMPLE = "b40de7596b0d4fbb189c1ceec5176f4461f7d290117250c1a7ec15b605ce7b11"
PANEL = "26a3b6eae7055ed6329b583e6b1e629d754a899ded29640f226530d9d8994172"


def frozen_inputs(root, archive):
    folder = root / "data/track_b/manifests/annual_2010_v1"
    panel = root / "data/track_b/processed/annual_2010_v1/panel.csv"
    ids = (folder / "selected_ids.txt").read_text(encoding="utf-8").splitlines()
    if (
        len(ids) != 1000
        or len(set(ids)) != 1000
        or hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() != SAMPLE
    ):
        raise ValueError("Frozen sample mismatch")
    if digest(archive) != SOURCE or digest(panel) != PANEL:
        raise ValueError("Frozen source/panel mismatch")
    manifest = json.loads((folder / "panel_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["output_sha256"] != PANEL
        or manifest["source_sha256"] != SOURCE
        or manifest["sample_set_sha256"] != SAMPLE
    ):
        raise ValueError("Panel provenance mismatch")
    _, protocol = load_protocol(root)
    tracked = [panel, *folder.iterdir(), archive]
    before = {str(p): digest(p) for p in tracked if p.is_file()}
    return panel, set(ids), protocol, before


def effective_status(f):
    return {str(status): counts(g) for status, g in f.groupby("outcome_status", sort=True)}


def table(rows, columns):
    return (
        "| "
        + " | ".join(columns)
        + " |\n| "
        + " | ".join(["---"] * len(columns))
        + " |\n"
        + "\n".join(
            "| "
            + " | ".join(
                (format(row[c], ".4g") if isinstance(row.get(c), float) else str(row.get(c, "")))
                for c in columns
            )
            + " |"
            for row in rows
        )
        + "\n"
    )


def run(root, archive):
    root, archive = Path(root), Path(archive)
    panel_path, ids, protocol, before = frozen_inputs(root, archive)
    panel = pd.read_csv(
        panel_path,
        dtype={
            "loan_id": "string",
            "t0": "string",
            "delinquency_state": "string",
            "payment_deferral_flag": "string",
            "modification_flag": "string",
            "assistance_plan": "string",
        },
        low_memory=False,
    )
    if set(panel.loan_id) != ids or len(panel) != 72232:
        raise ValueError("Panel identity/row mismatch")
    registry_path = root / "docs/track_b/pd_feature_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    primary = cohort(panel)
    eligible = panel.loc[panel.eligible].copy()
    dev, ev = split(primary)
    annual = [
        {"year": str(year), **counts(g), "statuses": effective_status(g)}
        for year, g in eligible.groupby(eligible.t0.str[:4])
    ]
    decision = gate(dev, ev)
    evidence = dict(
        schema_version="1.0",
        design_version="track-b-pd-v1",
        decision=decision,
        source_archive_sha256=SOURCE,
        sample_set_sha256=SAMPLE,
        panel_sha256=PANEL,
        panel_version="annual_2010_v1/parser-1.0.0",
        protocol_sha256_lf=protocol,
        feature_registry_sha256_lf=hashlib.sha256(
            registry_path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        design_sha256_lf=hashlib.sha256(
            (root / "docs/track_b/PD_COHORT_DESIGN.md").read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        cohort_rules=dict(
            known_statuses=["positive_default", "negative_survived_horizon", "competing_payoff"],
            horizon_months=12,
            minimum_history_months=6,
            payoff=(
                "facility default-before-payoff cumulative incidence; zero after compet"
                "ing termination"
            ),
            unknown="exclude; never label zero",
            primary_frequency="monthly",
            weighting="equal landmark",
        ),
        cohort_sha256=cohort_hash(primary),
        annual_feasibility=annual,
        eligible=counts(eligible),
        primary=counts(primary),
        eligible_by_status=effective_status(eligible),
        temporal_split=dict(
            salt=SALT,
            algorithm="SHA256(salt+loan_id) integer modulo 10; 0..6 development",
            development_end=DEV_END,
            evaluation_start=EVAL_START,
            purged_year="2015",
            development=counts(dev),
            evaluation=counts(ev),
            development_cohort_sha256=cohort_hash(dev),
            evaluation_cohort_sha256=cohort_hash(ev),
            unused_primary_landmarks=len(primary) - len(dev) - len(ev),
        ),
        feature_registry=registry,
        model_configuration=dict(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            tol=1e-8,
            class_weight=None,
            random_state=SEED,
            penalty="L2",
            features=list(FEATURES),
        ),
        preprocessing=dict(
            imputation="development median",
            scaling="development StandardScaler",
            sampling="none",
            calibrator="none",
        ),
        package_versions={
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "credit-risk-lab"]
        },
        seeds=dict(model=SEED, cluster_bootstrap=SEED),
        missingness={},
        baselines={},
        sensitivities={},
        limitations=[
            (
                "Historical operational knowledge time UNVERIFIED: nominal-time retrosp"
                "ective research only"
            ),
            (
                "Only 13 development and 5 evaluation default loans; coefficients and m"
                "etrics exploratory"
            ),
            (
                "Repeated overlapping windows; loan clustering cannot account for unkno"
                "wn borrower links"
            ),
            (
                "Fixed-fit bootstrap omits development uncertainty; percentile intervals "
                "unstable with five event loans"
            ),
            (
                "2010 mortgage cohort ageing/survival selection confounded with calendar drift; "
                "no new-vintage validation"
            ),
            "No regulatory/IFRS9/production/profit/fairness/external-validity claim",
            "No proper competing-risk or censoring-adjusted estimator fitted",
        ],
        next_task=(
            "Prespecified identifier-only sample-expansion feasibility study under a "
            "protocol amendment; no expansion in Task 3"
        ),
    )
    for name in FEATURES:
        x = features(eligible, registry)[name]
        evidence["missingness"][name] = dict(
            effective_sample=counts(eligible),
            missing=int(x.isna().sum()),
            rate=float(x.isna().mean()),
            by_year={
                str(k): dict(
                    effective_sample=counts(g),
                    missing=int(g[name].isna().sum()),
                    rate=float(g[name].isna().mean()),
                )
                for k, g in eligible.groupby(eligible.t0.str[:4])
            },
            by_outcome={
                str(k): dict(
                    effective_sample=counts(g),
                    missing=int(g[name].isna().sum()),
                    rate=float(g[name].isna().mean()),
                )
                for k, g in eligible.groupby("outcome_status")
            },
        )
    # Persist pre-fit gate separately, privately, before any fit.
    private = root / "data/track_b/processed/pd_baseline_v1"
    private.mkdir(exist_ok=True)
    (private / "prefit_gate.json").write_text(
        json.dumps(
            {
                k: evidence[k]
                for k in [
                    "decision",
                    "annual_feasibility",
                    "temporal_split",
                    "model_configuration",
                    "design_sha256_lf",
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if decision.startswith("BASELINE ESTABLISHED"):
        model, predictions = fit_baselines(dev, ev, registry)
        evidence["fitted_development"] = dict(
            effective_sample=counts(dev),
            medians=model[0].statistics_.tolist(),
            scaling_means=model[1].mean_.tolist(),
            scaling_scales=model[1].scale_.tolist(),
            standardized_coefficients=dict(zip(FEATURES, model[-1].coef_[0].tolist(), strict=True)),
            intercept=float(model[-1].intercept_[0]),
            iterations=int(model[-1].n_iter_[0]),
        )
        evidence["baselines"] = {name: diagnostics(ev, p) for name, p in predictions.items()}
        for name, kwargs in [
            ("S1_payoff_exclusion", dict(payoff=False)),
            ("S2_quarterly", dict(quarterly=True)),
            ("S3_physical_future_presence_known_labels", dict(physical=True)),
        ]:
            f = cohort(panel, **kwargs)
            d, e = split(f)
            result = dict(
                cohort=counts(f),
                cohort_sha256=cohort_hash(f),
                development=counts(d),
                evaluation=counts(e),
                decision=gate(d, e),
                removed_known_landmarks=len(primary) - len(f),
            )
            if result["decision"].startswith("BASELINE ESTABLISHED"):
                _, ps = fit_baselines(d, e, registry)
                result["baselines"] = {m: diagnostics(e, p) for m, p in ps.items()}
            evidence["sensitivities"][name] = result
    evidence["sensitivities"]["S3_unknown_exclusion"] = dict(
        loans_with_no_remaining_known_landmark=len(set(eligible.loan_id) - set(primary.loan_id)),
        excluded=counts(
            eligible.loc[
                ~eligible.outcome_status.isin(
                    ["positive_default", "negative_survived_horizon", "competing_payoff"]
                )
            ]
        ),
        by_status={
            k: v
            for k, v in effective_status(eligible).items()
            if k not in ["positive_default", "negative_survived_horizon", "competing_payoff"]
        },
        explanation=(
            "Loan removal counts overlap, are not additive; remaining labels never imputed"
        ),
    )
    if any(digest(Path(path)) != value for path, value in before.items()):
        raise ValueError("Frozen Task 2 inputs changed during run")
    evidence["preservation"] = dict(
        source_unchanged=True,
        sample_unchanged=True,
        panel_unchanged=True,
        task2_manifests_unchanged=True,
    )
    evidence["implementation_sha256_lf"] = {
        p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for p in (root / "src/credit_risk/track_b/pd").glob("*.py")
    }
    output = root / "reports/track_b/pd_baseline_validation.json"
    output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    render(root, evidence)
    return evidence


def render(root, e):
    parts = [
        "# Track B twelve-month PD baseline validation",
        "## Executive Summary",
        e["decision"],
        (
            "Historical operational knowledge time is **UNVERIFIED**. This is retrospective "
            "nominal-time mortgage research. Only five evaluation default loans support the "
            "results; point estimates are exploratory, not validation of lending performance."
        ),
        "## Cohort Definition",
        (
            "See docs/track_b/PD_COHORT_DESIGN.md for the fixed design. Six known clean "
            "months, no prevalent/prior event or reentry. Horizon months 1-12. Known "
            "facility outcomes only; no twelve-physical-row requirement."
        ),
        table(
            [dict(partition=k, **e[k]) for k in ["eligible", "primary"]],
            ["partition", "landmarks", "loans", "positive_landmarks", "default_loans"],
        ),
        table(
            [dict(status=k, **v) for k, v in e["eligible_by_status"].items()],
            ["status", "landmarks", "loans", "positive_landmarks", "default_loans"],
        ),
        "## Temporal Structure",
        (
            "Fixed identifier hash groups (70/30 expected, no outcome-based redraw). "
            "Development ends December 2014; evaluation starts January 2016; 2015 purges "
            "all overlapping development outcome windows. No shared loans. Earlier "
            "evaluation-group and later development-group observations are unused. "
            "Calendar, ageing and group-composition effects remain confounded."
        ),
        table(
            [
                dict(
                    year=r["year"],
                    **{k: v for k, v in r.items() if k not in ["year", "statuses"]},
                    payoff=r["statuses"].get("competing_payoff", {}).get("landmarks", 0),
                    censored=r["statuses"].get("right_censored", {}).get("landmarks", 0),
                    ambiguous=r["statuses"].get("ambiguous_event_order", {}).get("landmarks", 0),
                )
                for r in e["annual_feasibility"]
            ],
            [
                "year",
                "landmarks",
                "loans",
                "positive_landmarks",
                "default_loans",
                "payoff",
                "censored",
                "ambiguous",
            ],
        ),
        "## Competing Events",
        (
            "Primary zeros after documented payoff refer to default before payoff on this "
            "facility, not ordinary ongoing event-free exposure. S1 excludes payoff and "
            "changes selection/estimand; it is not a competing-risk estimator."
        ),
        "## Feature-Time Audit",
        (
            "Four fixed numerical features: released origination FICO/LTV; reporting-month "
            "loan age/delinquency 0-2. Exact historical availability/version is unverified. "
            "Every panel field is classified in the machine registry; "
            "unselected/unknown/future/outcome fields fail the predictor firewall."
        ),
        "## Missingness",
        (
            "Median imputation is prespecified and fitted only on development. Outcome "
            "stratification is descriptive only; no imputation choices use outcomes. "
            "Annual/outcome breakdowns are in JSON."
        ),
        table(
            [
                dict(feature=k, **v["effective_sample"], missing=v["missing"], rate=v["rate"])
                for k, v in e["missingness"].items()
            ],
            [
                "feature",
                "landmarks",
                "loans",
                "positive_landmarks",
                "default_loans",
                "missing",
                "rate",
            ],
        ),
        "## Baseline Models",
        (
            "Null: development landmark prevalence. Logistic: four standardized predictors, "
            "development medians/scales, L2 C=1, lbfgs, 2000 iterations, tolerance 1e-8, "
            "seed 31003. Natural prevalence; no class "
            "weighting/resampling/search/calibrator/threshold. Coefficients and "
            "preprocessing parameters are recorded in JSON."
        ),
        "## Temporal Validation",
        table(
            [dict(partition=k, **e["temporal_split"][k]) for k in ["development", "evaluation"]],
            ["partition", "landmarks", "loans", "positive_landmarks", "default_loans"],
        ),
    ]
    for title, keys in [
        ("Discrimination", ["roc_auc", "gini", "pr_auc_trapezoid", "average_precision"]),
        ("Probability Quality", ["brier", "log_loss", "observed_rate", "mean_probability"]),
    ]:
        parts.append("## " + title)
        rows = []
        for name, b in e["baselines"].items():
            for key in keys:
                ci = b["uncertainty"]["intervals"][key]
                rows.append(
                    dict(
                        model=name,
                        metric=key,
                        **b["effective_sample"],
                        estimate=b["metrics"][key],
                        lower=ci["lower"],
                        upper=ci["upper"],
                    )
                )
        parts.append(
            table(
                rows,
                [
                    "model",
                    "metric",
                    "landmarks",
                    "loans",
                    "positive_landmarks",
                    "default_loans",
                    "estimate",
                    "lower",
                    "upper",
                ],
            )
        )
    parts.extend(
        [
            (
                "PR-AUC uses trapezoidal interpolation; average precision is the step-weighted "
                "measure. A constant score's trapezoidal PR area can be misleading; do not "
                "interpret it as strong discrimination."
            ),
            "## Calibration",
            (
                "UNSTABLE / INSUFFICIENT EVENT SUPPORT. Numeric diagnostic fits below are "
                "exploratory and do not recalibrate predictions. Null slope is unidentifiable."
            ),
        ]
    )
    for name, b in e["baselines"].items():
        parts.append(
            name
            + ": "
            + json.dumps(
                {
                    k: b["calibration"][k]
                    for k in [
                        "calibration_intercept",
                        "joint_intercept",
                        "calibration_slope",
                        "slope_status",
                        "support_status",
                    ]
                }
            )
        )
        parts.append(
            table(
                b["reliability"],
                [
                    "bin",
                    "landmarks",
                    "loans",
                    "positive_landmarks",
                    "default_loans",
                    "observed_rate",
                    "mean_probability",
                    "sparse",
                ],
            )
        )
    parts.extend(
        [
            "## Clustered Uncertainty",
            (
                "500 loan-cluster draws with all landmarks retained. 95% percentile intervals "
                "condition on the fixed fitted model; no training uncertainty or borrower "
                "clustering. Single-class discrimination draws are omitted; probability-quality "
                "draws are retained. Sparse-event intervals are not confirmatory."
            ),
            "Bootstrap diagnostics: "
            + json.dumps(
                {
                    k: {x: v["uncertainty"][x] for x in ["draws", "seed", "single_class_draws"]}
                    for k, v in e["baselines"].items()
                }
            ),
            "## Sensitivity Analyses",
            (
                "Monthly is primary for monthly refresh, not chosen by AUC. Quarterly reduces "
                "repeated observations. S3 unknown statuses are excluded without labeling; the "
                "physical-presence sensitivity shows additional selection among already-known "
                "labels."
            ),
        ]
    )
    for name, s in e["sensitivities"].items():
        parts.append("### " + name)
        if "development" in s:
            parts.append(s["decision"])
            parts.append(
                table(
                    [dict(partition=k, **s[k]) for k in ["cohort", "development", "evaluation"]],
                    ["partition", "landmarks", "loans", "positive_landmarks", "default_loans"],
                )
            )
            if "baselines" in s:
                parts.append(
                    table(
                        [
                            dict(model=k, **b["effective_sample"], **b["metrics"])
                            for k, b in s["baselines"].items()
                        ],
                        [
                            "model",
                            "landmarks",
                            "loans",
                            "positive_landmarks",
                            "default_loans",
                            "roc_auc",
                            "average_precision",
                            "brier",
                            "log_loss",
                        ],
                    )
                )
        else:
            parts.append(
                table(
                    [dict(status=k, **v) for k, v in s["by_status"].items()],
                    ["status", "landmarks", "loans", "positive_landmarks", "default_loans"],
                )
            )
    parts.extend(
        [
            "## Limitations",
            "\n".join("- " + s for s in e["limitations"]),
            "## Decision",
            e["decision"],
            "Next task: " + e["next_task"],
            "## Reproducibility and preservation",
            (
                "Source/sample/panel/protocol/design/registry/cohort/code hashes, package "
                "versions, configuration and all sensitivity uncertainty details are recorded "
                "in pd_baseline_validation.json. Frozen Task 2 inputs were verified "
                "before/after fitting. Track A is untouched and its locked holdout is not "
                "evaluated. No models or raw licensed records are published."
            ),
        ]
    )
    (root / "reports/track_b/PD_BASELINE_VALIDATION.md").write_text(
        "\n\n".join(parts) + "\n", encoding="utf-8"
    )
