"""Aggregate-only explanation reporting; no borrower data or models are saved."""

import numpy as np

LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Unsecured utilization",
    "age": "Age",
    "NumberOfTime30_59DaysPastDueNotWorse": "30-59 day past-due count",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines/loans",
    "NumberOfTimes90DaysLate": "90+ day past-due count",
    "NumberRealEstateLoansOrLines": "Real estate loans/lines",
    "NumberOfTime60_89DaysPastDueNotWorse": "60-89 day past-due count",
    "NumberOfDependents": "Dependents",
}


def label(name):
    return LABELS.get(name, LABELS.get(name.removeprefix("missingindicator_"), name) + " missing")


def number(value):
    return "undefined" if value is None else f"{value:.4f}"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
        ]
    )


def render_report(result):
    logistic = sorted(
        result["logistic_stability"],
        key=lambda row: (-abs(row["coefficient"]["mean"]), row["feature"]),
    )
    tree = sorted(result["xgboost_stability"], key=lambda row: (row["global_rank"], row["feature"]))
    flipped = [label(row["feature"]) for row in logistic if row["sign_flip"]]
    agreement = result["xgboost_rank_agreement"]["mean_rho"]
    tops_lr = ", ".join(label(r["feature"]) for r in logistic[:5])
    tops_tree = ", ".join(label(r["feature"]) for r in tree[:5])
    coefficient_rows = [
        [
            label(r["feature"]),
            number(r["coefficient"]["mean"]),
            number(r["coefficient"]["std"]),
            number(r["coefficient"]["min"]),
            number(r["coefficient"]["max"]),
            number(np.exp(r["coefficient"]["mean"])),
            r["dominant_sign"],
            number(r["sign_consistency"]),
            r["sign_flip"],
            number(r["rank"]["mean"]),
        ]
        for r in logistic
    ]
    shap_rows = [
        [
            label(r["feature"]),
            number(r["pooled_mean_abs_shap"]),
            number(r["fold_importance"]["std"]),
            number(r["global_rank"]),
            ", ".join(number(v) for v in r["fold_ranks"]),
            number(r["rank"]["mean"]),
            number(r["rank"]["std"]),
        ]
        for r in tree
    ]
    comparison = [
        [
            label(r["feature"]),
            number(r["logistic_standardized_rank"]),
            r["logistic_direction"],
            r["logistic_sign_flip"],
            number(r["xgboost_shap_rank"]),
        ]
        for r in result["cross_model_comparison"]
    ]
    dependence_rows = []
    for name, diagnostics in result["dependence"].items():
        for b in diagnostics["bins"]:
            dependence_rows.append(
                [
                    label(name),
                    number(b["median_input"]),
                    b["rows"],
                    number(b["mean_shap"]),
                    b["sparse"],
                ]
            )
    local_rows = []
    for example in result["local_examples"]:
        for name, model in example["models"].items():
            ordered = sorted(
                model["contributions_log_odds"].items(), key=lambda pair: (-pair[1], pair[0])
            )
            positive = [f"{label(k)} {v:+.3f}" for k, v in ordered if v > 0][:3]
            negative = [f"{label(k)} {v:+.3f}" for k, v in reversed(ordered) if v < 0][:3]
            local_rows.append(
                [
                    example["example"],
                    name,
                    number(model["baseline_log_odds"]),
                    number(model["raw_probability"]),
                    "; ".join(positive),
                    "; ".join(negative),
                ]
            )
    interaction_rows = [
        [label(r["feature_a"]), label(r["feature_b"]), number(r["mean_abs_pair_contribution"])]
        for r in result["leading_interactions"]
    ]
    chunks = [
        "# Training-only model explainability and explanation stability",
        "## Executive Summary",
        f"Fresh fixed-specification models explain {result['rows']:,} saved training "
        f"rows across five grouped evaluation folds. "
        f"Leading logistic associations: {tops_lr}. Leading XGBoost contributions: {tops_tree}. "
        f"Mean pairwise fold Spearman agreement for the ten original XGB predictors "
        f"is {number(agreement)}. "
        "Explanations describe the fitted models, not causal drivers or lending decisions. "
        "No frozen model or consumed holdout prediction was loaded; Tasks 4/5 and "
        "registry history are unchanged.",
        "## Methodology",
        "The anchored Task 4 training-only loader checks source/split/sample "
        "identities and excludes consumed rows "
        "before parsing. StratifiedGroupKFold, five folds, shuffle seed 42 and "
        "exact raw predictor groups match Task 4. "
        "Both candidates use the existing fixed model configuration and model seed "
        "42 in every fold. "
        "Every pipeline fits only on that fold's training rows; explanation "
        "functions accept evaluation predictors only. "
        "All evaluation rows receive exact native TreeSHAP. No new tuning, "
        "calibration or predictive model is introduced. "
        "The source target is inherited two-year serious delinquency, not "
        "twelve-month regulatory PD.",
        "## Logistic Regression Interpretation",
        "The pipeline caps inputs at training quantiles 0.001/0.999, applies log1p "
        "to all original fields except age, "
        "median-imputes missing values, adds missing indicators, and standardizes "
        "all resulting columns using training "
        "means and population SDs. A stored coefficient beta therefore changes raw "
        "log-odds per one training SD "
        "of that imputed/transformed column, conditional on the other columns. "
        "exp(beta) is the corresponding modeled "
        "odds ratio. beta / training_scale is the coefficient per transformed unit; "
        "exp(beta / scale) is its odds ratio. "
        "These are not raw currency-unit effects for log-transformed income or "
        "literal count effects after capping. "
        "For age, a transformed unit is a year only within the fitted caps. "
        "Indicator 0-to-1 changes use its "
        "transformed-unit coefficient. A constant feature has no observed SD "
        "contrast and is flagged. "
        "Per-fold coefficients, scales and odds ratios are in JSON. exp(mean beta) "
        "below summarizes fold associations; "
        "it is neither a pooled-model estimate nor mean fold odds ratio. All "
        "associations are noncausal. "
        "Task 5's isotonic recommendation concerns logistic probability quality; "
        "coefficients/local logit sums here "
        "explain the raw base logistic model, not odds ratios of "
        "isotonic-calibrated probabilities.",
        "![Standardized logistic coefficients](logistic_coefficients.png)",
        "## Logistic Coefficient Stability",
        "Sample SD uses ddof=1; no independent-fold confidence interval is claimed. "
        "Sign consistency is the fraction "
        "in the modal positive/negative/near-zero category (tolerance 1e-8). A sign "
        "flip requires both positive and "
        "negative fitted coefficients. Absent indicator columns are not estimated, "
        "rather than imputed as zero coefficients. "
        "Rank ties use average ranks; presentation ties are ordered by feature name.",
        "Direction-flip flags: " + (", ".join(flipped) if flipped else "none") + ". "
        "A flipped association should not be presented as having a robust direction.",
        table(
            [
                "Feature",
                "Mean beta",
                "SD",
                "Min",
                "Max",
                "exp(mean beta)",
                "Direction",
                "Sign consistency",
                "Sign flip",
                "Mean rank",
            ],
            coefficient_rows,
        ),
        "![Fold coefficient stability](logistic_coefficient_stability.png)",
        "## XGBoost Global SHAP",
        "Native XGBoost pred_contribs=True, approx_contribs=False computes exact "
        "tree-path-dependent TreeSHAP. "
        "Training tree cover provides the reference weighting; no external or "
        "evaluation background is fitted. "
        "The scale is raw binary margin/log-odds: baseline + sum(feature SHAP) "
        "approximately equals the margin, "
        "and sigmoid(margin) matches pipeline probability. Additivity and feature "
        "alignment are checked on all rows. "
        "Mean absolute SHAP ranks contribution magnitude to model output. Signed "
        "distribution summaries do not "
        "establish a universal feature direction or causal importance. The native "
        "XGBoost version is recorded; "
        "no separate SHAP package is required. Missing indicators retain their own columns.",
        "![Global SHAP magnitude and signed distribution](xgboost_shap_summary.png)",
        "Whiskers are empirical 5th/95th percentiles, boxes 25th/75th and centers "
        "medians of full evaluation-row "
        "contributions. They are distribution summaries, not uncertainty intervals. "
        "[XGBoost prediction reference](https://xgboost.readthedocs.io/en/stable/prediction.html); "
        "[TreeSHAP research](https://www.nature.com/articles/s42256-019-0138-9).",
        "## XGBoost SHAP Stability",
        "Global magnitude is row-weighted pooled mean absolute SHAP. SD describes "
        "unweighted fold-level means. "
        "Absent tree columns contribute zero to the model, with presence counts "
        "recorded. Average rank and rank SD "
        "summarize folds; original-feature Spearman comparisons use all ten "
        "predictors. Fold fits overlap in training "
        "rows, so agreement is descriptive and does not establish future temporal stability.",
        table(
            [
                "Feature",
                "Pooled mean abs SHAP",
                "Fold SD",
                "Global rank",
                "Fold ranks",
                "Mean rank",
                "Rank SD",
            ],
            shap_rows,
        ),
        "![Fold SHAP ranks](xgboost_shap_stability.png)",
        "## Cross-Model Comparison",
        "Ranks below compare the ten original features only. Logistic ranks use "
        "mean absolute standardized "
        "coefficients; XGBoost ranks use pooled mean absolute SHAP. Indicator "
        "explanations remain visible separately "
        "above. Ranking units differ; their magnitudes are not directly comparable. "
        "Disagreement does not make "
        "either model wrong: scaling, correlations, thresholds and interactions can "
        "redistribute attribution.",
        table(
            ["Feature", "Logistic rank", "Logistic direction", "Sign flip", "XGB SHAP rank"],
            comparison,
        ),
        "![Cross-model original-feature ranks](cross_model_ranks.png)",
        "## Nonlinear Model Behavior",
        "Only the top five original XGB predictors are investigated. Exact input "
        "values are used when at most 24 "
        "values occur; otherwise 12 unique-edge quantile bins of nonmissing raw "
        "inputs are used. The displayed x "
        "coordinate is the bin median input; y is mean SHAP, with contribution "
        "quartiles for supported bins. "
        "These are associations across observed contexts, not partial dependence or "
        "counterfactual intervention. "
        "Hollow markers indicate fewer than 100 rows and are not connected. "
        "Quartiles are not confidence intervals. "
        "Missing inputs are excluded from dependence bins and counted in JSON, "
        "while global SHAP retains them. "
        "The pipeline caps/imputes inputs; explanations are therefore about the "
        "fitted capped representation. "
        "Past-due counts >=90 are source anomalies/sentinels and should not be read "
        "as literal delinquency counts.",
        "![Limited SHAP dependence diagnostics](shap_dependence.png)",
        table(["Feature", "Bin median input", "Rows", "Mean SHAP", "Sparse <100"], dependence_rows),
        "### Limited interaction analysis",
        "Native pred_interactions values are computed on 200 evaluation rows per "
        "fold, sampled proportionally "
        "from predicted-score deciles (seed 142 in each fold). No labels or "
        "interesting explanations select samples. "
        "Global SHAP uses all rows; this subset is only for exploratory "
        "interactions. Pair magnitude below is "
        "2 x mean absolute off-diagonal allocation, accounting for both symmetric "
        "halves. Symmetry and row-sum "
        "additivity to main SHAP contributions are verified. Only the top three "
        "pairs are discussed; these do not "
        "prove a causal interaction or explain a measured share of the performance advantage.",
        table(["Feature A", "Feature B", "Mean abs two-sided interaction"], interaction_rows),
        "## Illustrative Local Explanations",
        "The same three evaluation examples from fold 1 are chosen at XGB score "
        "percentiles 10, 50 and 90, "
        "with ties resolved by evaluation order. Raw values, source-row IDs and "
        "outcomes are not committed. "
        "Contributions raise or lower each raw model's log-odds relative to its own "
        "baseline. Logistic's baseline "
        "is its intercept at centered transformed inputs; XGB's is the tree "
        "reference expectation. They are different "
        "reference points, not identical borrower archetypes. These are model "
        "explanations, not approval/decline "
        "decisions or borrower quality labels. Complete contribution sums and "
        "residuals are in JSON.",
        table(
            [
                "Example",
                "Model",
                "Baseline log-odds",
                "Raw p",
                "Leading upward contributions",
                "Leading downward contributions",
            ],
            local_rows,
        ),
        "## Scientific questions",
        "RQ1: 30-59 and 90+ day past-due counts, age and utilization have "
        "consistent coefficient directions across all five folds. The first three "
        "also retain ranks 1-3. Income missingness remains negative but its "
        "magnitude varies substantially; it is a source/model association, not a "
        "reason to prefer incomplete applications.",
        "RQ2: utilization, 30-59 day past-due count, 90+ day past-due count and "
        "age hold XGB ranks 1-4 in every fold. 60-89 day past-due count and open "
        "credit lines swap ranks 5/6 in the first fold. The other original "
        "predictor ranks are unchanged.",
        "RQ3: both families share the same top five original predictors. "
        "Utilization ranks fourth in logistic versus first in XGB; open credit "
        "lines rank tenth versus sixth. Logistic places more standardized "
        "magnitude on debt ratio and dependents. Different attribution units "
        "and correlated inputs prevent direct magnitude comparisons.",
        "RQ4: monthly income, open credit lines, real estate loans and the "
        "dependents-missing indicator change logistic sign. Their weak mean "
        "associations should not be assigned a robust direction. Debt ratio "
        "and income missingness retain signs but show appreciable magnitude "
        "variation. XGB rank agreement is high, without proving unchanged "
        "attribution sizes or future stability.",
        "RQ5: utilization contributions rise from negative at low nonzero "
        "values toward positive at high values, with an exception in the "
        "zero-input bin. Past-due contributions jump between zero and one "
        "and then flatten; the 90+ count has further steps through roughly "
        "four. Age contributions fall most visibly across middle-to-older "
        "age bins and flatten later. These observed-context shapes and the "
        "leading age/utilization-by-delinquency interactions are richer than "
        "an additive capped/log-transformed logistic specification. They "
        "do not establish lending cutoffs or causal effects. Sparse tails "
        "and counts 96/98 cannot support literal count extrapolation.",
        "RQ6: the shared predictor families, nonlinear shapes and limited "
        "interactions are consistent with nonlinear modeling benefit, but "
        "cannot prove absence of unidentified leakage. No feature-removal "
        "ablation or interaction-removal experiment establishes the source "
        "of a performance advantage. Unknown decision-time availability, "
        "borrower identities and dates remain unresolved; neither these "
        "explanations nor Tasks 4/5 random-fold performance resolve them.",
        "## Limitations",
        "Association is not causation. SHAP explains the fitted model, not the "
        "real-world data-generating process. "
        "Correlated predictors can share or shift attribution. Borrower identity is "
        "unknown, no usable temporal "
        "structure exists, and the inherited target is two-year serious "
        "delinquency. Fold stability does not prove "
        "future stability. Feature explainability is not a fairness assessment; "
        "absent protected characteristics "
        "do not establish absence of discrimination or bias. Sparse bins, source "
        "anomalies and unknown decision-time "
        "availability constrain interpretation. No causal, regulatory-calibration "
        "or deployment-readiness claim is made.",
        "## Governance Interpretation",
        "Tasks 4/5 remain separate research evidence: [PD diagnostics](PD_DIAGNOSTICS.md) and "
        "[calibration study](CALIBRATION_STUDY.md). Raw XGBoost is the development "
        "recommendation; logistic isotonic "
        "calibration is not refitted here. Historical locked-holdout XGBoost "
        "metrics remain separately retained: "
        "AUC 0.868152, Brier 0.048545, log loss 0.176030. Frozen artifacts and "
        "registry history are unchanged. "
        "Explanations do not become operational reason codes or automatically change policy.",
        "## Reproduction",
        "Run `.venv/Scripts/python.exe scripts/explainability_study.py --source "
        "cs-training.csv` from the repository "
        "root. [Versioned aggregate results](explainability_stability.json) record "
        "source/split/code/registry hashes, "
        "model configuration, versions, fold definitions, output scale and "
        "interaction sampling. No trained model "
        "or full row-level attribution/prediction matrix is written.",
    ]
    return "\n\n".join(chunks) + "\n"


def generate_figures(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logistic = sorted(
        result["logistic_stability"], key=lambda r: (-abs(r["coefficient"]["mean"]), r["feature"])
    )
    tree = sorted(result["xgboost_stability"], key=lambda r: (r["global_rank"], r["feature"]))

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=150)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    means = [r["coefficient"]["mean"] for r in logistic]
    axis.barh(
        [label(r["feature"]) for r in logistic],
        means,
        color=["tab:blue" if v >= 0 else "tab:orange" for v in means],
    )
    axis.invert_yaxis()
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Mean standardized coefficient across folds (raw log-odds)",
        title="Logistic conditional associations per training SD",
    )
    save(fig, "logistic_coefficients.png")

    fig, axis = plt.subplots(figsize=(10, 6))
    coefficients = np.array([r["fold_coefficients"] for r in logistic], dtype=float)
    extent = float(np.nanmax(np.abs(coefficients))) or 1
    image = axis.imshow(coefficients, aspect="auto", cmap="RdBu_r", vmin=-extent, vmax=extent)
    axis.set_yticks(
        range(len(logistic)),
        [label(r["feature"]) + (" [sign flip]" if r["sign_flip"] else "") for r in logistic],
    )
    axis.set_xticks(
        range(coefficients.shape[1]), [f"Fold {i + 1}" for i in range(coefficients.shape[1])]
    )
    for i in range(len(logistic)):
        for j in range(coefficients.shape[1]):
            axis.text(
                j,
                i,
                f"{coefficients[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(coefficients[i, j]) > extent * 0.55 else "black",
            )
    axis.set_title("Logistic coefficient direction and magnitude across folds")
    fig.colorbar(image, ax=axis, label="Standardized coefficient")
    save(fig, "logistic_coefficient_stability.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    y = np.arange(len(tree))
    axes[0].barh(y, [r["pooled_mean_abs_shap"] for r in tree], color="tab:blue")
    axes[0].set_yticks(y, [label(r["feature"]) for r in tree])
    axes[0].invert_yaxis()
    axes[0].set(
        xlabel="Pooled mean absolute SHAP (log-odds)", title="Global model contribution magnitude"
    )
    stats = [
        {
            "med": r["distribution"]["median"],
            "q1": r["distribution"]["p25"],
            "q3": r["distribution"]["p75"],
            "whislo": r["distribution"]["p05"],
            "whishi": r["distribution"]["p95"],
            "fliers": [],
        }
        for r in tree
    ]
    axes[1].bxp(stats, positions=y, orientation="horizontal", showfliers=False, manage_ticks=False)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="Signed SHAP contribution (log-odds)",
        title="OOF distribution: whiskers 5th/95th percentile",
    )
    save(fig, "xgboost_shap_summary.png")

    fig, axis = plt.subplots(figsize=(10, 6))
    ranks = np.array([r["fold_ranks"] for r in tree])
    image = axis.imshow(ranks, aspect="auto", cmap="viridis_r", vmin=1, vmax=len(tree))
    axis.set_yticks(range(len(tree)), [label(r["feature"]) for r in tree])
    axis.set_xticks(range(ranks.shape[1]), [f"Fold {i + 1}" for i in range(ranks.shape[1])])
    for i in range(len(tree)):
        for j in range(ranks.shape[1]):
            axis.text(
                j,
                i,
                f"{ranks[i, j]:g}",
                ha="center",
                va="center",
                color="black" if ranks[i, j] < len(tree) / 2 else "white",
            )
    axis.set_title("XGBoost feature-rank stability on unseen evaluation folds")
    fig.colorbar(image, ax=axis, label="Mean absolute SHAP rank (1 = highest)")
    save(fig, "xgboost_shap_stability.png")

    rows = sorted(
        result["cross_model_comparison"], key=lambda r: (r["xgboost_shap_rank"], r["feature"])
    )
    fig, axis = plt.subplots(figsize=(10, 6))
    for index, row in enumerate(rows):
        axis.plot(
            [row["logistic_standardized_rank"], row["xgboost_shap_rank"]],
            [index, index],
            color="gray",
            alpha=0.6,
        )
    axis.scatter(
        [r["logistic_standardized_rank"] for r in rows],
        range(len(rows)),
        label="Logistic standardized magnitude",
        color="tab:blue",
    )
    axis.scatter(
        [r["xgboost_shap_rank"] for r in rows],
        range(len(rows)),
        label="XGB mean absolute SHAP",
        color="tab:orange",
    )
    axis.set_yticks(range(len(rows)), [label(r["feature"]) for r in rows])
    axis.invert_yaxis()
    axis.set(
        xticks=range(1, len(rows) + 1),
        xlabel="Rank among the ten original predictors (1 = highest)",
        title="Cross-model agreement: compare ranks, not magnitude units",
    )
    axis.legend(fontsize=8)
    save(fig, "cross_model_ranks.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (name, summary) in zip(axes.flat, result["dependence"].items(), strict=False):
        bins = summary["bins"]
        x = [b["median_input"] for b in bins]
        axis.plot(
            x,
            [b["mean_shap"] if not b["sparse"] else np.nan for b in bins],
            color="tab:blue",
            linewidth=1,
        )
        for b in bins:
            if not b["sparse"]:
                axis.vlines(
                    b["median_input"], b["shap_q25"], b["shap_q75"], color="tab:blue", alpha=0.3
                )
            axis.scatter(
                b["median_input"],
                b["mean_shap"],
                s=20 + min(80, np.sqrt(b["rows"])),
                facecolors="none" if b["sparse"] else "tab:blue",
                edgecolors="tab:blue",
                alpha=0.7,
            )
        if "PastDue" in name or name == "NumberOfTimes90DaysLate":
            axis.set_xscale("symlog", linthresh=1)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(
            title=label(name),
            xlabel="Observed raw input (bin median)",
            ylabel="SHAP (raw log-odds)",
        )
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0,
        0.85,
        "Top five original features only.\n\nFilled: >=100 rows; hollow: "
        "sparse.\nVertical bars: contribution quartiles.\nNot confidence intervals "
        "or causal effects.\n\nPast-due x-axis: symlog; >=90 values\nare source "
        "anomalies/sentinels.",
        va="top",
        fontsize=10,
    )
    save(fig, "shap_dependence.png")
