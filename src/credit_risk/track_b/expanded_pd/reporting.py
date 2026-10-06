"""Aggregate-only scientific report and exportable diagnostic figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def table(rows, keys):
    text = "| " + " | ".join(keys) + " |\n| " + " | ".join(["---"] * len(keys)) + " |\n"
    for row in rows:
        cells = [
            format(row[k], ".5g") if isinstance(row.get(k), float) else str(row.get(k, ""))
            for k in keys
        ]
        text += "| " + " | ".join(cells) + " |\n"
    return text


def render(root, r):
    root = Path(root)
    output = root / "reports/track_b"
    a = r["cohort_reproduction"]
    models = r["temporal_results"]
    parts = [
        "# Expanded-cohort PD temporal validation",
        "## Executive Summary",
        r["decision"] + "; research champion: " + r["champion"],
        "## Research Question",
        (
            "Twelve-month facility default before payoff in the frozen 2010 Freddie "
            "cohort. Historical operational knowledge time UNVERIFIED. No "
            "regulatory/production/IFRS9/fairness claim."
        ),
        "## Cohort",
        table(
            [dict(partition=k, **v) for k, v in a["cohorts"].items()],
            ["partition", "landmarks", "loans", "positive_landmarks", "default_loans"],
        ),
        "## Temporal Design",
        (
            "Original loan hash groups; development <=2014-12; all 2015 landmarks "
            "purged; evaluation >=2016-01. No overlapping loans. Dates and outcomes "
            "unchanged."
        ),
        "## Feature-Time Governance",
        (
            "Registry: expanded_pd_feature_registry.json. Eight numeric predictors: "
            "origination credit score/LTV/DTI/rate/term/borrowers, plus t0 loan "
            "age/delinquency. Two categorical predictors: purpose/occupancy. Exact "
            "provider historical vintage unverified. No raw-source joins, macro or "
            "post-t0 fields."
        ),
        "## Models",
        (
            "Null development prevalence; fixed C=1 logistic; one monthly logistic "
            "hazard; two shallow XGB configurations, with only one selected "
            "challenger. Static-only sensitivities are prespecified within these "
            "families. Median/missing-indicator, scaling, modal categorical imputation "
            "and reference OHE fit on fit groups only."
        ),
        "## Development Selection",
        table(
            [dict(partition=k, **v) for k, v in a["internal_partitions"].items()],
            ["partition", "landmarks", "loans", "positive_landmarks", "default_loans"],
        ),
        (
            "Identifier hash assigns disjoint 60% fit / 20% calibration / 20% "
            "selection groups. Internal groups share development calendar coverage; "
            "this is not internal temporal CV. No final refit. Sigmoid only retained "
            "when independent selection improves both Brier and log loss by >=1%, "
            "positive slope, and no material AUC loss."
        ),
        "XGBoost trials:",
        table(
            [
                dict(
                    depth=t["configuration"]["max_depth"],
                    trees=t["configuration"]["n_estimators"],
                    **t["effective_sample"],
                    **t["metrics"],
                )
                for t in r["xgboost_trials"]
            ],
            [
                "depth",
                "trees",
                "landmarks",
                "loans",
                "positive_landmarks",
                "default_loans",
                "roc_auc",
                "average_precision",
                "brier",
                "log_loss",
            ],
        ),
        "Development selection/raw versus sigmoid:",
        json.dumps(r["development_selection"], indent=2),
        "## Locked Temporal Evaluation",
        (
            "Task5 metrics opened once after model/artifact/calibration specifications "
            "froze. Tasks3/4 aggregate labels and nested Task3 predictions were "
            "previously inspected; this is NOT never-seen data. No post-evaluation "
            "model changes. Ledger and artifact/code hashes are in JSON. Prediction "
            "arrays and models remain private under Git-ignore."
        ),
    ]
    for title, keys in [
        ("Discrimination", ["roc_auc", "gini", "average_precision", "pr_auc_trapezoid"]),
        ("Probability Quality", ["brier", "log_loss", "observed_rate", "mean_probability"]),
    ]:
        rows = []
        for name, result in models.items():
            for key in keys:
                ci = r["clustered_uncertainty"]["intervals"][name][key]
                rows.append(
                    dict(
                        model=name,
                        metric=key,
                        **result["effective_sample"],
                        estimate=result["metrics"][key],
                        lower=ci["lower"],
                        upper=ci["upper"],
                    )
                )
        parts.extend(
            [
                "## " + title,
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
                ),
            ]
        )
    parts.extend(
        [
            (
                "Average Precision is primary PR summary. Trapezoidal PR area is "
                "secondary; constant-null area can be misleading. All intervals are 95% "
                "fixed-fit loan-cluster percentiles."
            ),
            "## Calibration",
            (
                "Raw probabilities retained alongside the development-selected variant. "
                "Calibration numbers are diagnostic point estimates; no iid Wald "
                "intervals. Reliability bins with fewer than 20 event loans are sparse."
            ),
            table(
                [
                    dict(
                        model=k,
                        **v["effective_sample"],
                        CITL=v["calibration"]["calibration_intercept"],
                        slope=v["calibration"]["calibration_slope"],
                        support=v["calibration"]["support_status"],
                    )
                    for k, v in models.items()
                ],
                [
                    "model",
                    "landmarks",
                    "loans",
                    "positive_landmarks",
                    "default_loans",
                    "CITL",
                    "slope",
                    "support",
                ],
            ),
        ]
    )
    for name in ["logistic", "xgboost"]:
        parts.extend(
            [
                name + ": " + r["development_selection"][name]["decision"],
                table(
                    models[name]["reliability"],
                    [
                        "bin",
                        "landmarks",
                        "loans",
                        "positive_landmarks",
                        "default_loans",
                        "observed_rate",
                        "mean_probability",
                        "support",
                    ],
                ),
            ]
        )
    parts.extend(
        [
            "## Hazard Model",
            r["hazard"]["estimand"],
            (
                "Each eligible t0 predicts the next known month. Default event appears "
                "once per facility. Payoff exits the risk set; first-month ambiguous/admin "
                "follow-up is excluded. Development t0<=2014-12 means monthly event labels "
                "end 2015-01, whereas twelve-month landmark labels can extend "
                "through2015-12. No purged t0 is reused. Future projection freezes t0 "
                "predictors except deterministic ageing. PD=1-product(1-h), not sum(h). "
                "Net-risk projection is NOT the primary default-before-payoff CIF and is "
                "excluded from champion selection."
            ),
            json.dumps(r["hazard"], indent=2),
            "## Nonlinear Challenger",
            (
                "Bounded depth2/120 and depth3/180 tree search; development selection "
                "only. All configurations recorded. Champion promotion requires paired "
                "probability-quality improvements, discrimination retention and "
                "supported-calendar stability under the prespecified rule."
            ),
            "## Paired Uncertainty",
            json.dumps(
                {k: v for k, v in r["clustered_uncertainty"].items() if k != "intervals"}, indent=2
            ),
            "## Interpretability",
            (
                "Logistic effects are associations, not causal. Numeric odds ratios use "
                "development-standardized units; categorical effects use stored "
                "references. TreeSHAP is raw-model log-odds contribution; any retained "
                "sigmoid adds another mapping. Deterministic bounded explanation sample, "
                "one earliest evaluation landmark per facility. Direction bins and rank "
                "stability are descriptive."
            ),
            table(
                r["logistic_coefficients"],
                ["feature", "coefficient", "odds_ratio", "direction", "scale"],
            ),
            "Categorical references: " + json.dumps(r["categorical_references"]),
            json.dumps(r["xgboost_explanations"], indent=2),
            "Model agreement: " + json.dumps(r["model_agreement"], indent=2),
            "## Delinquency Sensitivity",
            (
                "Origination-only sensitivity removes both age and current delinquency. "
                "Raw-only fixed pipelines; no champion selection from this sensitivity. "
                "Results appear in the core metric tables."
            ),
            "## Segment Diagnostics",
            (
                "Segments with fewer than 20 event loans suppress "
                "discrimination/calibration; descriptive exposure/event-rate counts "
                "remain. Loan counts can overlap across bins/blocks. No segment-specific "
                "optimized models."
            ),
            json.dumps(r["segments"], indent=2),
            "## Temporal Stability",
            (
                "Calendar blocks are fixed, not selected by performance. Sparse blocks are "
                "suppressed. PSI uses development-derived bins and has no universal "
                "pass/fail cutoff."
            ),
            json.dumps(r["calendar"], indent=2),
            json.dumps(r["stability"], indent=2),
            "## Limitations",
            "\n".join("- " + s for s in r["limitations"]),
            "## Decision",
            r["decision"] + "; champion=" + r["champion"],
            "Next task: " + r["next_task"] + ". Not implemented.",
            "![Aggregate diagnostic figure](expanded_pd_diagnostics.png)",
            (
                "References: [Competing-risk "
                "estimands](https://pubmed.ncbi.nlm.nih.gov/22253319/), [XGBoost "
                "contribution "
                "predictions](https://xgboost.readthedocs.io/en/stable/prediction.html)."
            ),
        ]
    )
    (output / "EXPANDED_PD_VALIDATION.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    coeff = sorted(r["logistic_coefficients"], key=lambda v: abs(v["coefficient"]))[-10:]
    axes[0, 0].barh([v["feature"] for v in coeff], [v["coefficient"] for v in coeff])
    axes[0, 0].set_title("Logistic standardized/reference coefficients")
    shap = sorted(r["xgboost_explanations"]["global_mean_abs"].items(), key=lambda v: v[1])[-10:]
    axes[0, 1].barh([k for k, v in shap], [v for k, v in shap])
    axes[0, 1].set_title("Mean absolute TreeSHAP (raw log odds)")
    for name in ["logistic", "xgboost"]:
        bins = models[name]["reliability"]
        axes[1, 0].plot(
            [b["mean_probability"] for b in bins],
            [b["observed_rate"] for b in bins],
            "o-",
            label=name,
        )
    axes[1, 0].plot([0, 0.2], [0, 0.2], "k--", alpha=0.4)
    axes[1, 0].set(
        xlabel="Mean predicted PD",
        ylabel="Observed landmark rate",
        title="Descriptive reliability; sparse bins marked in report",
    )
    axes[1, 0].legend()
    for name in ["logistic", "xgboost"]:
        rows = [
            (k, v["metrics"]["observed_rate"], v["metrics"]["mean_probability"])
            for k, v in r["calendar"][name].items()
            if "metrics" in v
        ]
        axes[1, 1].plot(
            [k for k, a, b in rows], [b for k, a, b in rows], "o-", label=name + " predicted"
        )
    rows = [
        (k, v["metrics"]["observed_rate"])
        for k, v in r["calendar"]["logistic"].items()
        if "metrics" in v
    ]
    axes[1, 1].plot([k for k, a in rows], [a for k, a in rows], "k--o", label="observed")
    axes[1, 1].legend()
    axes[1, 1].set_title("Supported temporal blocks")
    fig.suptitle("Frozen 2010 cohort: research diagnostics, historical availability UNVERIFIED")
    fig.tight_layout()
    fig.savefig(output / "expanded_pd_diagnostics.png", dpi=150)
    plt.close(fig)
