"""Aggregate competing-risk report and exportable diagnostic figures."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def table(rows, keys):
    text = "| " + " | ".join(keys) + " |\n| " + " | ".join(["---"] * len(keys)) + " |\n"
    for row in rows:
        text += (
            "| "
            + " | ".join(
                format(row[k], ".5g") if isinstance(row.get(k), float) else str(row.get(k, ""))
                for k in keys
            )
            + " |\n"
        )
    return text


def render(root, r):
    root = Path(root)
    output = root / "reports/track_b"
    audit = r["survival_audit"]
    parts = [
        "# Survival and competing-risk validation",
        "## Executive Summary",
        r["decision"],
        (
            "This establishes monthly first-event risk accounting, coherent "
            "competing-risk probabilities and censor-adjusted research evaluation. It "
            "does not establish calibrated lifetime PD or production validity. Dynamic "
            "risk updates are next-month forecasts; entry-time multi-month dynamic CIF "
            "is not identified without future-state assumptions."
        ),
        "## Research Question",
        (
            "Conditional on eligible entry into the assigned calendar window, estimate "
            "default before payoff/maturity, payoff before default and remaining "
            "event-free probability. Historical operational knowledge time UNVERIFIED."
        ),
        "## Event Definitions",
        r["protocol"]["events"],
        "Raw first endpoint audit: " + json.dumps(audit["raw_first_endpoints"]),
        "## Time Origin and Delayed Entry",
        r["protocol"]["origin"],
        r["protocol"]["origin_A_assessment"],
        json.dumps(audit["delayed_entry_age"], indent=2),
        "## Risk-Set Construction",
        r["protocol"]["risk_set"],
        table(
            [dict(partition=k, **v) for k, v in audit["partitions"].items()],
            [
                "partition",
                "facilities",
                "risk_intervals",
                "default_facilities",
                "payoff_facilities",
                "censored_facilities",
            ],
        ),
        "Global eligible-entry cohort: "
        + json.dumps(audit["global_cohort"])
        + "; risk intervals="
        + str(audit["global_risk_intervals"]),
        "Censor reasons: " + json.dumps(audit["censor_reasons"], indent=2),
        "Entry/zero-exposure flow: " + json.dumps(audit["flow"], indent=2),
        "## Transition Diagnostics",
        (
            "Transitions use next observed states as descriptive targets only; no "
            "future-state predictor or Markov forecasting model. Terminal causes shown "
            "separately."
        ),
        table(
            r["transition_diagnostics"]["evaluation"], ["current", "next", "count", "probability"]
        ),
        "## Nonparametric Survival",
        (
            "AJ event-free survival equals KM with default and payoff as exits. "
            "Default-only KM censors payoff and estimates net risk, not default CIF. "
            "All event/censor ties use the documented monthly convention."
        ),
        "## Default CIF",
        (
            "AJ reference and structural forecasts share conditional window-entry "
            "origin. Sparse default counts limit horizon discrimination/calibration."
        ),
        "## Payoff CIF",
        (
            "Payoff/maturity is a competing endpoint, not necessarily voluntary "
            "prepayment. Probability conservation verified: S+FD+FP=1."
        ),
        "## Cause-Specific Models",
        (
            "Structural and dynamic multinomial logistic models jointly fit two cause "
            "logits against no-event reference. Independent binary logits were avoided "
            "because their probabilities need not sum to <=1. All preprocessing from "
            "development; fixed C=1 and duration bands; grouped development diagnostic "
            "followed by pre-evaluation development refit."
        ),
        json.dumps(r["development_validation"], indent=2),
        (
            "Conditional cause/no-event odds ratios are not proportional hazard "
            "ratios, causal effects or direct CIF changes."
        ),
        table(r["cause_logits"]["structural"], ["cause", "feature", "log_odds", "odds_ratio"]),
        "## Dynamic State Model",
        (
            "Updated current-state predictions condition on observed t0 state, "
            "balance/rate/age/remaining term. They are rolling one-month updates. No "
            "observed future path or frozen-current-delinquency annual projection is "
            "scored as a prospective CIF."
        ),
        json.dumps(r["monthly_current_state"], indent=2),
        json.dumps(r["monthly_uncertainty"], indent=2),
        "## Temporal Validation",
        (
            "Development interval endpoints <=2014-12; no 2015 intervals; evaluation "
            "t0>=2016-01 with separate facilities. Task6 ledger frozen before "
            "construction and model specifications before temporal metrics. Prior "
            "Task4/5 outcomes already inspected; do not claim virgin data."
        ),
        "## Horizon-Specific Discrimination",
        r["protocol"]["metrics"],
        "## Probability Quality",
        (
            "IPCW Brier divides by total cohort size. Censoring KM re-estimated within "
            "each facility bootstrap. Competing payoff remains known non-default at "
            "later horizons. Integrated score over months1..36: "
        )
        + str(r["integrated_brier_1_36"]),
        "## CIF Calibration",
        (
            "Mean predicted CIF compared with observed AJ incidence. No binary "
            "calibration intercept/slope or post-evaluation recalibration."
        ),
    ]
    rows = []
    for h, v in r["horizon_results"].items():
        row = dict(
            horizon=h,
            status=v["status"],
            at_risk=v["observed"]["at_risk"],
            observed_default=v["observed"]["default_cif"],
            observed_payoff=v["observed"]["payoff_cif"],
            observed_survival=v["observed"]["survival"],
        )
        if "metrics" in v:
            row.update(
                predicted_default=v["metrics"]["mean_predicted_cif"],
                predicted_payoff=v["mean_payoff_cif"],
                AUC=v["metrics"]["cumulative_dynamic_auc"],
                Brier=v["metrics"]["ipcw_brier"],
                default_facilities=v["metrics"]["default_facilities_by_horizon"],
            )
        rows.append(row)
    parts.append(
        table(
            rows,
            [
                "horizon",
                "status",
                "at_risk",
                "default_facilities",
                "observed_default",
                "predicted_default",
                "observed_payoff",
                "predicted_payoff",
                "observed_survival",
                "AUC",
                "Brier",
            ],
        )
    )
    parts.extend(
        [
            "Facility-resampled 95% intervals:",
            json.dumps(r["uncertainty"], indent=2),
            (
                "Reliability groups use development-CIF quantiles; sparse groups are not "
                "reliable calibration evidence."
            ),
            json.dumps(r["reliability"], indent=2),
            "Ambiguity sensitivity: " + json.dumps(r["ambiguity_sensitivity"], indent=2),
            "## Task 5 Bridge",
            json.dumps(r["task5_bridge"], indent=2),
            (
                "Cached frozen scores only; Task5 ledger/evidence/models unchanged. "
                "Different conditioning, competing payoff and risk-set selection need not "
                "produce identical probabilities."
            ),
            "## Limitations",
            "\n".join("- " + s for s in r["limitations"]),
            "## Decision",
            r["decision"],
            "Next task: " + r["next_task"] + ". Not implemented.",
            "![Competing-risk diagnostics](survival_competing_risk_diagnostics.png)",
            (
                "Methods: [Discrete competing-risk "
                "validation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7217187/), "
                "[Time-dependent competing-risk "
                "discrimination](https://pmc.ncbi.nlm.nih.gov/articles/PMC4512205/), "
                "[Censor-adjusted "
                "validation](https://www.bmj.com/content/377/bmj-2021-069249)."
            ),
        ]
    )
    (output / "SURVIVAL_COMPETING_RISK_VALIDATION.md").write_text(
        "\n\n".join(parts) + "\n", encoding="utf-8"
    )
    # Figures contain aggregate curves/transition probabilities, never licensed identifiers.
    obs = r["nonparametric_evaluation"]
    t = [v["month"] for v in obs]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, key, label in [
        (axes[0, 0], "survival", "Event-free survival"),
        (axes[0, 1], "default_cif", "Default CIF"),
        (axes[0, 2], "payoff_cif", "Payoff/maturity CIF"),
    ]:
        ax.step(t, [v[key] for v in obs], where="post", label="Observed AJ")
        ax.set(xlabel="Months since conditional entry", title=label)
        ax.legend()
    axes[0, 1].plot(t, [v["naive_net_default"] for v in obs], ":", label="Naive net risk (NOT CIF)")
    axes[0, 1].legend()
    axes[1, 0].step(t, [v["at_risk"] for v in obs], where="post")
    axes[1, 0].axhline(200, color="red", linestyle=":")
    axes[1, 0].set(title="Facilities at risk", xlabel="Months")
    supported = [v for v in r["horizon_results"].values() if "metrics" in v]
    axes[1, 1].plot(
        [v["month"] for v in supported],
        [v["metrics"]["mean_predicted_cif"] for v in supported],
        "o-",
        label="Structural predicted",
    )
    axes[1, 1].plot(
        [v["month"] for v in supported],
        [v["observed"]["default_cif"] for v in supported],
        "o-",
        label="Observed AJ",
    )
    axes[1, 1].set(title="Supported default-CIF calibration", xlabel="Months")
    axes[1, 1].legend()
    states = ["00", "01", "02"]
    targets = ["00", "01", "02", "default", "payoff/maturity"]
    matrix = np.zeros((3, 5))
    for row in r["transition_diagnostics"]["evaluation"]:
        if row["current"] in states and row["next"] in targets:
            matrix[states.index(row["current"]), targets.index(row["next"])] = row["probability"]
    image = axes[1, 2].imshow(matrix, aspect="auto", vmin=0, vmax=1)
    axes[1, 2].set_xticks(range(5), targets, rotation=25)
    axes[1, 2].set_yticks(range(3), states)
    axes[1, 2].set_title("Observed transitions (descriptive)")
    fig.colorbar(image, ax=axes[1, 2])
    fig.suptitle("2010 vintage: conditional entry, no regulatory lifetime-PD claim")
    fig.tight_layout()
    fig.savefig(output / "survival_competing_risk_diagnostics.png", dpi=150)
    plt.close(fig)
