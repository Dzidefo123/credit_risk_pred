"""Generate Task 4 research diagnostics from the anchored original training partition."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from credit_risk.validation.pd_diagnostics import run_training_diagnostics


def table(headers, rows):
    return (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    )


def render_report(result):
    candidates = result["candidates"]
    comparisons, folds, bins, thresholds = [], [], [], []
    for name, candidate in candidates.items():
        m, cal = candidate["oof"]["metrics"], candidate["oof"]["calibration"]
        comparisons.append(
            [
                name,
                *[f"{m[k]:.6f}" for k in ("roc_auc", "gini", "pr_auc", "brier", "log_loss")],
                f"{cal['calibration_intercept']:.6f}",
                f"{cal['calibration_slope']:.6f}",
            ]
        )
        for metric, values in candidate["cv_summary"].items():
            folds.append(
                [
                    name,
                    metric,
                    f"{values['mean']:.6f}",
                    f"{values['std']:.6f}",
                    ", ".join(f"{v:.6f}" for v in values["fold_values"]),
                ]
            )
        for b in candidate["oof"]["reliability"]["bins"]:
            bins.append(
                [
                    name,
                    b["rows"],
                    f"{b['mean_probability']:.5f}" if b["rows"] else "undefined",
                    f"{b['observed_bad_rate']:.5f}" if b["rows"] else "undefined",
                    b["sparse"],
                ]
            )
        for t in candidate["threshold_analysis"]:
            thresholds.append(
                [
                    name,
                    t["threshold"],
                    t["confusion_matrix"],
                    *[
                        f"{t[k]:.4f}"
                        for k in ("precision", "recall", "specificity", "f1", "balanced_accuracy")
                    ],
                ]
            )
    uncertainty = result["uncertainty"]
    intervals = []
    for name, metrics in uncertainty["intervals"].items():
        for metric, bounds in metrics.items():
            intervals.append([name, metric, f"{bounds['lower']:.6f}", f"{bounds['upper']:.6f}"])
    paired = [
        [metric, f"{bounds['lower']:.6f}", f"{bounds['upper']:.6f}"]
        for metric, bounds in uncertainty["paired_differences"]["intervals"].items()
    ]
    outcome_rates = table(
        ["Model", "Observed event rate", "Mean predicted probability"],
        [
            [
                name,
                f"{candidate['oof']['metrics']['observed_bad_rate']:.6f}",
                f"{candidate['oof']['metrics']['mean_probability']:.6f}",
            ]
            for name, candidate in candidates.items()
        ],
    )
    logistic = candidates["logistic_regression"]["oof"]["metrics"]
    xgboost = candidates["xgboost"]["oof"]["metrics"]
    findings = (
        f"Pooled OOF AUC: logistic {logistic['roc_auc']:.6f}, "
        f"XGBoost {xgboost['roc_auc']:.6f}. Brier: {logistic['brier']:.6f} versus "
        f"{xgboost['brier']:.6f}; log loss: {logistic['log_loss']:.6f} versus "
        f"{xgboost['log_loss']:.6f}. The paired intervals below support the observed "
        "difference conditional on these predictions, without establishing future performance."
    )
    return (
        "\n\n".join(
            [
                "# PD validation diagnostics",
                "## Executive Summary",
                f"New evidence uses only {result['rows']:,} saved training rows "
                f"({result['events']:,} events), "
                "with five stratified group folds and fresh raw logistic/XGBoost candidates. "
                "No frozen model was loaded, no original holdout prediction accessed, and "
                "no registry entry changed. "
                "Compare ranking, probability quality and stability together; this research "
                "does not promote a new champion.",
                findings,
                "## Dataset / Target Reminder",
                "The target is the **two-year serious-delinquency outcome**, inherited from "
                "SeriousDlqin2yrs. "
                "It is not 12-month regulatory PD. Source attribution and timing "
                "limitations remain in "
                "[target definition](../../docs/TARGET_DEFINITION.md) and "
                "[leakage review](../data/LEAKAGE_REVIEW.md). Original development, "
                "calibration and consumed test rows are excluded.",
                "## Discrimination",
                "ROC-AUC measures ranking; Gini = 2 AUC - 1. PR-AUC uses trapezoidal integration, "
                "whereas average precision is a separate step-weighted measure. Both "
                "conventions are recorded in JSON. "
                "Ranking alone says nothing about probability accuracy or lending costs.",
                "## Calibration",
                "Calibration-in-the-large estimates intercept a with slope fixed at 1: "
                "logit(E[y]) = a + logit(p). "
                "The separate joint fit estimates a and b in logit(E[y]) = a + b logit(p); "
                "its intercept is stored separately. "
                "Ideal CITL intercept is 0 and slope is 1. Positive CITL means "
                "underprediction; negative means overprediction. "
                "Slope below 1 suggests overly extreme probabilities; above 1 suggests "
                "insufficient dispersion. "
                "Fits use unpenalized logistic maximum likelihood, rejecting "
                "separated/ill-conditioned joint fits. "
                "Only diagnostic logits clip p to [1e-12, 1-1e-12] to avoid infinity; "
                "scoring uses original probabilities. "
                "Coefficient CIs are omitted because shared training fits and unknown "
                "borrowers invalidate a simple iid claim. "
                "Bin Wilson intervals in JSON are approximate row-binomial intervals, not "
                "borrower-cluster robust. "
                "Sparse bins (<100 rows) are flagged; bin counts are always shown.",
                outcome_rates,
                table(["Model", "Bin rows", "Mean p", "Observed rate", "Sparse"], bins),
                "![Training OOF reliability and threshold curves](pd_diagnostics.png)",
                "## Threshold Diagnostics",
                "Predeclared research thresholds are 0.03, 0.05, 0.10, 0.20 and 0.50; "
                "predicted positive means p >= threshold. "
                "These are event classifications, not an approval policy. Confusion matrix "
                "is [[TN, FP], [FN, TP]]. "
                "Each fold also evaluates a threshold equal to its training prevalence (in "
                "JSON), without using evaluation labels "
                "to select it. Generic selection tooling supports explicitly labelled "
                "training/development max-F1 or Youden J "
                "and chooses the higher threshold on ties; no threshold was optimized on "
                "pooled OOF outcomes here. "
                "Precision/F1 are zero when no positives are predicted; "
                "recall/specificity/balanced accuracy are undefined "
                "when their actual class is absent. Lending thresholds require costs, risk "
                "appetite and policy constraints.",
                table(
                    [
                        "Model",
                        "Threshold",
                        "Confusion matrix",
                        "Precision",
                        "Recall",
                        "Specificity",
                        "F1",
                        "Balanced accuracy",
                    ],
                    thresholds,
                ),
                "## Cross-Validation",
                "StratifiedGroupKFold, five folds, shuffled seed 42, groups = "
                "saved-compatible exact raw predictor profiles. "
                "Duplicate profiles stay together; labels are excluded from group "
                "identities. Both candidates use identical folds "
                "and fixed existing configurations. Every fold refits capping, imputation "
                "and applicable scaling/log transforms "
                "using its training rows only. Fold standard deviation uses ddof=1 and is "
                "descriptive, not a standard error. "
                "**This is not out-of-time validation.** No usable temporal information exists. "
                "[Implementation "
                "reference](https://scikit-learn.org/stable/modules/cross_validation.html).",
                table(["Model", "Metric", "Mean", "SD", "Fold values"], folds),
                "## Statistical Uncertainty",
                f"Paired exact-predictor-group bootstrap: {uncertainty['requested_samples']} "
                f"replicates, seed "
                f"{uncertainty['seed']}, 95% percentile intervals; "
                f"{uncertainty['valid_samples']} valid, "
                f"{uncertainty['skipped_single_class']} single-class replicates skipped. "
                f"Both models receive identical group weights. "
                "Resampling assumes independence between observed profile groups and cannot "
                "account for unidentified repeated "
                "borrowers across profiles. A row bootstrap would likewise assume "
                "independent rows and miss those borrowers. "
                "These intervals are conditional on fixed OOF predictions, not refitting or "
                "future temporal uncertainty; "
                "overlapping fold training data limits inferential claims. No p-values, "
                "independent-fold t-test or general "
                "deployment significance claim is made.",
                table(["Model", "Metric", "Lower", "Upper"], intervals),
                "Paired differences: **XGBoost minus logistic**. Positive AUC/Gini/AP "
                "favors XGBoost; negative Brier/log loss favors it.",
                table(["Metric", "Difference lower", "Difference upper"], paired),
                "## Champion/Challenger Comparison",
                "The table contains pooled **new training-only OOF** results, not "
                "historical final-test scores. "
                "Logistic remains the interpretable benchmark; XGBoost offers nonlinear "
                "complexity. "
                "A champion decision must weigh discrimination, calibration, stability, "
                "interpretability, reproducibility "
                "and operational requirements. Neither maximum AUC nor a bootstrap interval "
                "alone authorizes promotion.",
                table(
                    [
                        "Model",
                        "AUC",
                        "Gini",
                        "PR-AUC",
                        "Brier",
                        "Log loss",
                        "CITL intercept",
                        "Joint slope",
                    ],
                    comparisons,
                ),
                "**Historical locked-holdout result, retained and not reevaluated:** "
                "XGBoost AUC 0.868152, "
                "Brier 0.048545, log loss 0.176030. Historical sigmoid calibration slightly "
                "improved log loss while "
                "slightly worsening Brier; that is not an improvement on every measure. "
                "Historical calibrated model and "
                "new raw CV models have different evaluation designs and cannot be treated "
                "as directly comparable.",
                "## Limitations",
                "No usable temporal information; no true out-of-time validation; borrower "
                "identity unknown; "
                "target not 12-month regulatory PD; holdout already consumed historically. "
                "Public source provenance and "
                "feature availability at an actual lending decision remain imperfect. OOF "
                "results reuse a previously explored "
                "training population and fixed candidate choices, so this is additional "
                "research evidence, not a new untouched test. "
                "No nested tuning or new calibration selection occurred. Probability fit "
                "coefficients are diagnostics, not deployed "
                "recalibration. No evidence of lender profitability, fairness, rejection "
                "performance or real-world rollout is claimed.",
                "## Reproduction and provenance",
                "Run `.venv/Scripts/python.exe scripts/pd_diagnostics.py --source "
                "cs-training.csv` from the repository root "
                "(substitute the verified local source path if different). It hashes source "
                "bytes and saved split metadata, "
                "then parses only anchored training positions. [Machine-readable "
                "results](pd_diagnostics.json) include "
                "source/split/training-position/registry/code hashes, dependency versions, "
                "configurations, seeds and fold identities. "
                "No row-level predictions or applicant records are committed. Results "
                "require the ignored original source "
                "and split artifacts, and the matching historical registry bridge version.",
            ]
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    result = run_training_diagnostics(args.root, args.source)
    output = args.root / "reports/model_validation"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, candidate in result["candidates"].items():
        bins = [b for b in candidate["oof"]["reliability"]["bins"] if b["rows"]]
        axes[0].plot(
            [b["mean_probability"] for b in bins],
            [b["observed_bad_rate"] for b in bins],
            "o-",
            label=name,
        )
        rows = candidate["threshold_analysis"]
        for metric, style in (("precision", "-"), ("recall", "--"), ("f1", ":")):
            axes[1].plot(
                [t["threshold"] for t in rows],
                [t[metric] for t in rows],
                style,
                label=f"{name} {metric}",
            )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set(
        xlabel="Mean predicted probability",
        ylabel="Observed event rate",
        title="Training-only OOF reliability",
    )
    axes[1].set(
        xlabel="Research threshold",
        ylabel="Metric",
        title="Training-only OOF threshold diagnostics",
    )
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "pd_diagnostics.png", dpi=150)
    plt.close(fig)
    (output / "pd_diagnostics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "PD_DIAGNOSTICS.md").write_text(render_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "models": {n: c["oof"]["metrics"] for n, c in result["candidates"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
