"""Generate a nested training-only calibration study, without promoting a model."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from credit_risk.validation.calibration_study import METHODS, MODELS, run_calibration_study


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
        ]
    )


def number(value):
    return "undefined" if value is None else f"{value:.6f}"


def comparison_rows(result):
    rows = []
    for name in MODELS:
        for method in METHODS:
            diag = result["models"][name][method]["pooled_oof"]
            metrics, calibration = diag["metrics"], diag["calibration"]
            rows.append(
                [
                    name,
                    method.upper(),
                    *[
                        number(metrics[k])
                        for k in ("roc_auc", "gini", "pr_auc", "brier", "log_loss")
                    ],
                    number(calibration["calibration_intercept"]),
                    number(calibration["calibration_slope"]),
                ]
            )
    return rows


def model_conclusion(result, name):
    assessment = result["recommendations"][name]
    raw = result["models"][name]["raw"]["pooled_oof"]["metrics"]
    gains = []
    for method in METHODS[1:]:
        metrics = result["models"][name][method]["pooled_oof"]["metrics"]
        gains.append(
            f"{method}: Brier delta {metrics['brier'] - raw['brier']:+.6f}, "
            f"log-loss delta {metrics['log_loss'] - raw['log_loss']:+.6f}"
        )
    extra = ""
    if assessment["recommendation"] == "ISOTONIC":
        isotonic = result["models"][name]["isotonic"]["pooled_oof"]["metrics"]
        extra = (
            f" The Brier reduction is {(raw['brier'] - isotonic['brier']) / raw['brier']:.2%} "
            f"relative, with AUC delta {isotonic['roc_auc'] - raw['roc_auc']:+.6f}. "
            "This improves probability quality while sacrificing some ranking through ties. "
            "Near-ideal global CITL/slope alone do not exclude nonlinear local miscalibration; "
            "inspect the reliability curve and sparse tails before interpreting the gain."
        )
    return (
        "; ".join(gains)
        + ". Lowest point-estimate Brier: "
        + assessment["lowest_point_brier_method"].upper()
        + ". "
        + assessment["reason"]
        + ". "
        + "Research recommendation: **"
        + assessment["recommendation"]
        + "**."
        + extra
    )


def render_report(result):
    settings = result["study_config"]
    uncertainty_rows, bin_rows, rates, fold_rows, ranking_rows, selected_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for name in MODELS:
        for method in METHODS[1:]:
            comparison = result["paired_comparisons"][name][method]
            for metric in ("brier", "log_loss", "roc_auc"):
                bounds = comparison["paired_differences"]["intervals"][metric]
                uncertainty_rows.append(
                    [
                        name,
                        method,
                        metric,
                        number(comparison["point_differences"][metric]),
                        number(bounds["lower"]),
                        number(bounds["upper"]),
                    ]
                )
        for method in METHODS:
            value = result["models"][name][method]
            diag = value["pooled_oof"]
            rates.append(
                [
                    name,
                    method,
                    number(diag["metrics"]["observed_bad_rate"]),
                    number(diag["metrics"]["mean_probability"]),
                    number(diag["reliability"]["ece"]),
                ]
            )
            fold_rows.append(
                [
                    name,
                    method,
                    *[
                        f"{value['cv_summary'][metric]['mean']:.6f} +/- "
                        f"{value['cv_summary'][metric]['std']:.6f}"
                        for metric in ("roc_auc", "brier", "log_loss")
                    ],
                ]
            )
            for b in diag["reliability"]["bins"]:
                bin_rows.append(
                    [
                        name,
                        method,
                        f"{b['lower']:.1f}-{b['upper']:.1f}",
                        b["rows"],
                        number(b["mean_probability"]),
                        number(b["observed_bad_rate"]),
                        b["sparse"],
                    ]
                )
            if method != "raw":
                changes = [record["ranking"]["auc_delta"] for record in value["folds"]]
                unique = [
                    record["ranking"]["calibrated_unique_probabilities"]
                    for record in value["folds"]
                ]
                pooled_delta = (
                    diag["metrics"]["roc_auc"]
                    - result["models"][name]["raw"]["pooled_oof"]["metrics"]["roc_auc"]
                )
                ranking_rows.append(
                    [
                        name,
                        method,
                        number(min(changes)),
                        number(max(changes)),
                        number(pooled_delta),
                        ", ".join(map(str, unique)),
                    ]
                )
        choices = Counter(
            fold["inner"]["models"][name]["selected_method"] for fold in result["fold_design"]
        )
        selected = result["models"][name]["inner_selected"]["pooled_oof"]["metrics"]
        selected_rows.append(
            [
                name,
                dict(sorted(choices.items())),
                *[number(selected[k]) for k in ("roc_auc", "brier", "log_loss")],
            ]
        )
    chunks = [
        "# Nested training-only probability calibration study",
        "## Executive decision",
        "This experiment compares six fixed model/method combinations on untouched outer folds, "
        "and separately evaluates a method-selection procedure whose decisions are "
        "made on inner data. "
        "Recommendations are research assessments; no model artifact or lending policy is changed.",
        "## Dataset and nesting",
        f"Only {result['rows']:,} original training rows ({result['events']:,} "
        f"two-year serious-delinquency outcomes) "
        "are parsed by the anchored Task 4 loader. Original development, "
        "calibration and consumed test partitions "
        "are excluded. Five outer StratifiedGroupKFold folds use seed 42 and exact "
        "raw predictor groups. "
        "Within each outer-training partition, another five grouped folds use seed "
        "42 + outer fold number: "
        "inner role 0 fits calibrators, role 1 selects methods, and roles 2-4 fit "
        "the base pipeline. "
        "This is roughly 60%/20%/20% of outer training, subject to group stratification. "
        "All preprocessing fits only on the base-fit rows. No base refit occurs after calibration. "
        "Both base candidates use fixed existing configurations and each mapping "
        "shares the same base fit. "
        "The fitting function receives only outer-training data; the prediction "
        "function receives predictors only. "
        "Outer labels are used for stratified fold allocation and scoring, never "
        "fitting or selection. "
        "Fold identities, role sizes/event counts, calibrator parameters and seeds "
        "are recorded in JSON.",
        "## Prespecified selection and practical importance",
        "Primary objective: minimum inner-selection Brier score. A calibrated "
        "method must improve absolute Brier "
        f"by at least {settings['minimum_brier_gain']:.4f} and not worsen "
        f"inner-selection log loss. Otherwise choose RAW. "
        "Ties among eligible calibrators favor sigmoid's lower complexity. This "
        "numerical margin was fixed before "
        "running the study; it is a research tolerance, not a lender-derived "
        "economic or regulatory threshold. "
        "No hyperparameters or classification thresholds are selected. Each "
        "selected mapping is frozen before "
        "outer prediction. The post-study research assessment additionally requires "
        "the conditional paired Brier "
        "interval to lie wholly beyond the minimum gain and the log-loss interval "
        "to show no deterioration. "
        "That assessment does not alter any outer prediction or turn the selected "
        "method's score into a fresh test.",
        "## Full model x calibration comparison",
        "Pooled Task 5 outer OOF predictions; raw probabilities bypass the existing "
        "calibrator's clipping, "
        "so they remain exactly as the base model emits them. Sigmoid fits a "
        "positive slope and intercept on "
        "logit probabilities (Platt-style log-odds scaling); isotonic fits a "
        "flexible nondecreasing mapping. "
        "The reused calibrator clips fit inputs and transformed outputs to [1e-6, 1-1e-6]. "
        "Calibration diagnostic logits separately use epsilon 1e-12. CITL fixes "
        "slope at 1; the reported slope "
        "comes from a joint intercept/slope fit. Joint intercepts and undefined fit "
        "reasons are stored in JSON.",
        table(
            ["Model", "Method", "AUC", "Gini", "PR-AUC", "Brier", "Log loss", "CITL", "Slope"],
            comparison_rows(result),
        ),
        "PR-AUC uses trapezoidal integration; average precision is stored separately. "
        "Raw Task 5 models use smaller base-fit sets than Task 4: their metrics are "
        "not a repeated Task 4 evaluation.",
        "## Does Logistic Regression benefit from post-hoc calibration?",
        model_conclusion(result, "logistic_regression"),
        "## Does XGBoost benefit from post-hoc calibration?",
        model_conclusion(result, "xgboost"),
        "## Which method gives the best probability quality?",
        "The lowest point-estimate Brier method is identified above for each model. "
        "Brier and log loss need not agree: report both and preserve the "
        "prespecified primary objective. "
        "Small numerical improvements alone do not justify additional complexity. "
        "Ranking quality (AUC/Gini) "
        "and probability quality (Brier/log loss/reliability) answer different questions.",
        "## Are improvements practically meaningful, and is complexity justified?",
        "The prespecified margin and paired intervals determine the research "
        "recommendations, rather than "
        "assuming a calibrated method must win. Even a statistical difference here "
        "does not establish "
        "economic value without a lender's exposure, loss severity, costs and decision policy. "
        "A raw model already calibrated in this population can be harmed by "
        "unnecessary recalibration. "
        "A logistic link does not guarantee calibration; tree boosting does not "
        "automatically require recalibration.",
        "## Honest evaluation of inner method selection",
        "Methods listed here were chosen on inner-selection data before outer evaluation. "
        "These scores evaluate the entire prespecified selection procedure, rather "
        "than hindsight selection "
        "of the best outer-fold method. Fixed-method comparisons above are reported separately.",
        table(
            ["Model", "Inner choices across folds", "OOF AUC", "OOF Brier", "OOF log loss"],
            selected_rows,
        ),
        "## Fold stability",
        "Five fold means +/- sample SD (ddof=1). Complete fold values for "
        "AUC/Gini/PR-AUC/AP/Brier/log loss "
        "and fold ranking checks are in JSON. SD is descriptive, not an "
        "independent-fold standard error.",
        table(["Model", "Method", "AUC", "Brier", "Log loss"], fold_rows),
        "## Paired uncertainty",
        f"{settings['bootstrap_samples']} paired exact-predictor-group bootstrap replicates, "
        f"seed {settings['seed'] + 50}, {settings['confidence_level']:.0%} percentile intervals. "
        "Each method uses the same sampled group weights as raw; all four "
        "comparisons use the same seed. "
        "Differences are calibrated minus raw: negative Brier/log loss favors calibration. "
        "These are marginal, conditional intervals on fixed OOF predictions, not "
        "simultaneous familywise "
        "intervals or proof of temporal generalization. Shared training folds and "
        "unidentified borrowers limit "
        "inference. No refitting uncertainty, multiple-testing-adjusted "
        "significance or regulatory claim is made. "
        "Valid/skipped replicate counts are recorded separately for every comparison in JSON.",
        table(
            ["Model", "Calibrator", "Metric", "Point delta", "95% lower", "95% upper"],
            uncertainty_rows,
        ),
        "## Ranking investigation",
        "A single monotone mapping preserves order; isotonic flats and output "
        "clipping introduce ties. "
        "The implementation checks for rank inversions and records AUC changes in "
        "every outer fold. "
        "Fold-specific monotone maps may reorder observations across folds, so "
        "pooled AUC can change even "
        "when every within-fold sigmoid AUC is identical. Material-change flags use "
        "|delta AUC| > 0.001. "
        "Isotonic changes are interpreted with its reduced unique-score counts, not "
        "automatically as better ranking. "
        "[Method reference](https://scikit-learn.org/stable/modules/calibration.html).",
        table(
            [
                "Model",
                "Calibrator",
                "Min fold AUC delta",
                "Max fold AUC delta",
                "Pooled AUC delta",
                "Unique scores per fold",
            ],
            ranking_rows,
        ),
        "## Reliability and bin support",
        "![Calibration comparison with bin support](calibration_comparison.png)",
        "Common fixed-width bins: [0, 0.1, ..., 1]; p=1 is in the last bin. "
        "ECE = sum of bin count / total count x absolute difference between bin "
        "observed rate and mean p. "
        "ECE depends on binning and is not a regulatory metric. Empty bins remain explicit. "
        "Solid markers/lines indicate bins with at least 100 rows; hollow markers "
        "indicate sparse bins "
        "and are not connected. Marker area reflects support; lower panels give log-scale counts. "
        "The sparse flag alone does not guarantee enough events. Wilson bounds in "
        "JSON are approximate "
        "row-binomial intervals, not borrower-cluster robust; no visual error bars "
        "imply otherwise.",
        table(["Model", "Method", "Observed event rate", "Mean predicted p", "ECE"], rates),
        table(
            ["Model", "Method", "Bin", "Rows", "Mean p", "Observed rate", "Sparse <100"], bin_rows
        ),
        "## Preserved evidence and scientific limitations",
        "Task 4 raw OOF evidence remains unchanged in [PD diagnostics](PD_DIAGNOSTICS.md). "
        "Historical locked-holdout XGBoost results remain separately retained: AUC "
        "0.868152, Brier 0.048545, "
        "log loss 0.176030. No historical holdout predictions or frozen models were "
        "loaded; registry history "
        "is unchanged. Public data have no usable temporal information or reliable "
        "borrower identities. "
        "**This is not out-of-time validation.** The target is not twelve-month regulatory PD. "
        "Exact profiles proxy grouping, not true borrowers. A single deterministic "
        "split design and smaller "
        "inner base-fit samples limit generalization. Previously explored training "
        "data are not a fresh holdout. "
        "Flexible isotonic fitting can overfit, especially at sparse tails; "
        "clipping avoids infinite log loss "
        "but does not fix poor tail estimates. No deployment readiness or "
        "regulatory calibration is claimed.",
        "## Reproduction",
        "Run `.venv/Scripts/python.exe scripts/calibration_study.py --source "
        "cs-training.csv` from the root. "
        "[Versioned results](calibration_study.json) include "
        "source/code/split/registry hashes, seeds, configurations, "
        "fold metrics, intervals and recommendations. No applicant rows or OOF "
        "prediction vectors are committed.",
        "## Final recommendations",
        "Logistic Regression: **"
        + result["recommendations"]["logistic_regression"]["recommendation"]
        + "**. "
        "XGBoost: **" + result["recommendations"]["xgboost"]["recommendation"] + "**. "
        "These retain or recommend research probability outputs only; frozen "
        "historical artifacts are untouched.",
    ]
    return "\n\n".join(chunks) + "\n"


def plot_reliability(result, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), height_ratios=[3, 1])
    colors = {"raw": "tab:blue", "sigmoid": "tab:orange", "isotonic": "tab:green"}
    for column, name in enumerate(MODELS):
        top, bottom = axes[:, column]
        top.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        for method in METHODS:
            bins = result["models"][name][method]["pooled_oof"]["reliability"]["bins"]
            supported = [b for b in bins if b["rows"] >= 100]
            sparse = [b for b in bins if 0 < b["rows"] < 100]
            top.plot(
                [b["mean_probability"] if b["rows"] >= 100 else np.nan for b in bins],
                [b["observed_bad_rate"] if b["rows"] >= 100 else np.nan for b in bins],
                color=colors[method],
                linewidth=1,
                alpha=0.65,
            )
            for values, hollow in ((supported, False), (sparse, True)):
                top.scatter(
                    [b["mean_probability"] for b in values],
                    [b["observed_bad_rate"] for b in values],
                    s=[15 + 2 * np.sqrt(b["rows"]) for b in values],
                    edgecolors=colors[method],
                    facecolors="none" if hollow else colors[method],
                    alpha=0.7,
                    label=method.upper() if not hollow else None,
                )
            bottom.step(
                [(b["lower"] + b["upper"]) / 2 for b in bins],
                [b["rows"] if b["rows"] else np.nan for b in bins],
                where="mid",
                color=colors[method],
                label=method,
            )
        top.set(
            title=name.replace("_", " ").title() + " - nested OOF",
            xlabel="Mean predicted probability",
            ylabel="Observed two-year event rate",
            xlim=(-0.02, 1.02),
            ylim=(-0.02, 1.02),
        )
        top.legend(fontsize=8)
        bottom.set(yscale="log", xlabel="Fixed probability-bin midpoint", ylabel="Bin rows (log)")
        for axis in (top, bottom):
            axis.grid(alpha=0.2)
    fig.suptitle(
        "Filled bins: >=100 rows; hollow bins: <100 rows. Marker area reflects support.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    result = run_calibration_study(
        args.root, args.source, progress=lambda message: print(message, flush=True)
    )
    output = args.root / "reports/model_validation"
    output.mkdir(parents=True, exist_ok=True)
    plot_reliability(result, output / "calibration_comparison.png")
    (output / "calibration_study.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "CALIBRATION_STUDY.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result["recommendations"], indent=2))


if __name__ == "__main__":
    main()
