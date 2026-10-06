"""Canonical MDVR and traceability manifest assembled from validated aggregates."""

import hashlib
import json
import math
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from credit_risk.reporting.evidence import Evidence, require

REPORT = "reports/model_validation/MODEL_DEVELOPMENT_VALIDATION_REPORT.md"
MANIFEST = "reports/model_validation/model_validation_manifest.json"
LINKS = {
    "audit": "../data/DATA_AUDIT.md",
    "training_audit": "../data/TRAINING_DATA_AUDIT.json",
    "dictionary": "../../docs/DATA_DICTIONARY.md",
    "target": "../../docs/TARGET_DEFINITION.md",
    "suitability": "../data/DATASET_SUITABILITY.md",
    "leakage": "../data/LEAKAGE_REVIEW.md",
    "registry": "../../docs/HOLDOUT_REGISTRY.md",
    "ledger": "../holdout_registry.json",
    "task4": "pd_diagnostics.json",
    "task5": "calibration_study.json",
    "task6": "explainability_stability.json",
    "historical": "../phase5_validation_summary.json",
    "experiment": "../phase4_experiment.json",
}
NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "Unsecured utilization",
    "age": "Age",
    "NumberOfTime30_59DaysPastDueNotWorse": "30-59-day past-due count",
    "NumberOfTimes90DaysLate": "90+-day past-due count",
    "NumberOfTime60_89DaysPastDueNotWorse": "60-89-day past-due count",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines/loans",
    "NumberRealEstateLoansOrLines": "Real estate loans/lines",
    "NumberOfDependents": "Dependents",
}
LIMITATIONS = [
    "No usable dates: true out-of-time validation and temporal calibration stability unavailable.",
    "Unknown borrower identity: exact-profile groups do not establish borrower independence.",
    "Two-year serious delinquency is not regulatory default, 12-month PD or lifetime PD.",
    "Pristine-download equivalence, source selection and measurement timing remain unverified.",
    "Training records were previously explored; repeated research is not independent validation.",
    "Sparse/anomalous tails and unknown missingness mechanisms constrain interpretation.",
    "Fairness, lending profitability, external transportability and deployment readiness unproven.",
]


def label(name):
    if name.startswith("missingindicator_"):
        return NAMES.get(name.removeprefix("missingindicator_"), name) + " missing"
    return NAMES.get(name, name)


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def num(value):
    return "undefined" if value is None else f"{value:.6f}"


def interval(value):
    return f"[{num(value['lower'])}, {num(value['upper'])}]"


def cite(*keys):
    return "Evidence: " + ", ".join(f"[{key}]({LINKS[key]})" for key in keys) + "."


def render(e, metadata):
    t4, t5, t6 = e.task4, e.task5, e.task6
    lr = t4["candidates"]["logistic_regression"]["oof"]
    xgb = t4["candidates"]["xgboost"]["oof"]
    features = e.experiment["feature_names"]
    coefficients = {r["feature"]: r for r in t6["logistic_stability"]}
    tree = sorted(t6["xgboost_stability"], key=lambda r: (r["global_rank"], r["feature"]))
    comparison = sorted(t6["cross_model_comparison"], key=lambda r: r["xgboost_shap_rank"])
    parts = [
        "# Model Development & Validation Report",
        "## 1. Executive Summary",
        "**Objective:** estimate the probability of a two-year serious-delinquency "
        "outcome using the repository-attributed Give Me Some Credit benchmark. "
        "This is a credit-risk ML research benchmark, not a lender deployment or "
        "regulatory PD validation. " + cite("target", "suitability"),
        "**Development conclusion:** raw XGBoost is the development champion and "
        "Logistic Regression is the interpretable challenger. Task 4 supports "
        "stronger XGBoost discrimination and probability quality; Task 5 retains "
        "RAW for XGBoost and recommends ISOTONIC for Logistic Regression as research. "
        "Task 6 finds broadly shared predictor families and consistent tree "
        "attribution rankings. These conclusions do not replace frozen historical "
        "models, calibrators or policy. " + cite("task4", "task5", "task6"),
        table(
            ["Evidence category", "Meaning"],
            [
                [
                    "TRAINING_ONLY",
                    "Tasks 4-6 results from saved training records; no consumed holdout access",
                ],
                [
                    "HISTORICAL_LOCKED_HOLDOUT",
                    "Retained results from the already-consumed final evaluation",
                ],
                [
                    "DESCRIPTIVE",
                    "Retained provenance/data-quality evidence; not predictive validation",
                ],
                [
                    "GOVERNANCE",
                    "Registry, hashes, code, experimental controls and verification evidence",
                ],
            ],
        ),
        "All result tables carry a category. Sections containing recommendations "
        "are proposed practice, not measured production evidence. Category labels "
        "apply to the claim, not to every incidental field in a mixed historical artifact.",
        "## 2. Model Purpose and Scope",
        "The decision supported is a development choice between fixed candidate "
        "families for this benchmark. The repository has not established a "
        "real lending approval, risk-based price, capital requirement or "
        "provisioning decision from this evidence. A source record is treated "
        "as a financial snapshot before a prospective outcome; verified "
        "origination time, population eligibility and economic decision costs "
        "are absent. " + cite("target", "suitability"),
        "### Visible IFRS 9 boundary",
        "**Track A does not implement or empirically validate 12-month PD, "
        "lifetime PD, LGD, EAD, SICR, Stage 1/2/3 or ECL.** Give Me Some Credit "
        "does not supply the dated event, facility, exposure, cash-flow and "
        "recovery records required to support those components. Separate "
        "synthetic portfolio/loss demonstrations in the repository are "
        "educational and do not fill these empirical gaps. These components "
        "belong to a future longitudinal/facility-level research track. "
        + cite("suitability", "target"),
        "## 3. Dataset and Provenance",
        "[DESCRIPTIVE] Repository history attributes the data to Kaggle Give "
        "Me Some Credit. The intended competition observation unit is a "
        "borrower/person profile; the local unit that can actually be verified "
        "is a source record. The row index does not establish borrower identity. "
        "Pristine-download equivalence remains unverified. Source dates, "
        "supplier sampling, acceptance/rejection lineage and currency are "
        "unverified. No download or new source inspection was performed to "
        "generate this report. " + cite("audit", "dictionary", "target"),
        "[GOVERNANCE] Recorded source SHA-256: `" + e.experiment["source_sha256"] + "`.",
        "## 4. Target Definition",
        "> The target represents a two-year serious-delinquency outcome and "
        "should not be interpreted as a regulatory 12-month or lifetime "
        "Probability of Default measure.",
        "[DESCRIPTIVE] SeriousDlqin2yrs is the inherited binary label for "
        "serious delinquency at the 90-day threshold or a worse credit outcome "
        "within the source two-year task. The event was not reconstructed "
        "from dated histories. A binary prediction ranks/scores the label; "
        "a calibrated event probability measures agreement with its observed "
        "frequency in a specified population. Neither establishes regulatory "
        "default adjudication, horizon equivalence or compliant PD estimation. "
        "Negative-label maturity, censoring and reporting lags cannot be "
        "verified at record level. " + cite("target"),
        "## 5. Data Quality Assessment",
        "[DESCRIPTIVE] The table below reuses the saved training-only audit. "
        "Older whole-source descriptions remain descriptive historical audit "
        "evidence and were not recomputed. Flags request investigation rather "
        "than automatic deletion or recoding. " + cite("training_audit", "audit", "dictionary"),
        table(
            ["Predictor", "Missing rows", "Missing fraction", "Finite minimum", "Finite maximum"],
            [
                [
                    label(n),
                    e.audit["columns"][n]["missing_count"],
                    num(e.audit["columns"][n]["missing_fraction"]),
                    num(e.audit["columns"][n]["numeric"]["minimum"]),
                    num(e.audit["columns"][n]["numeric"]["maximum"]),
                ]
                for n in features
            ],
        ),
        "Past-due counts 96/98 may be anomalous or sentinel-like; their meaning "
        "is not established. Age zero is implausible for an adult lending "
        "applicant. Utilization above one can represent over-limit balances, "
        "while extreme utilization/debt/income tails need supplier definitions. "
        "Observed missingness does not identify its mechanism. " + cite("audit", "dictionary"),
        "## 6. Leakage Assessment",
        "Specific legacy defects included learned imputation before splitting, "
        "outcome-dependent exclusions and in-sample evaluation. The current "
        "pipeline fits transformations inside training roles and excludes "
        "the target and row ID from predictors. Exact-profile grouping "
        "controls observed duplicate-profile overlap. It does not establish "
        "borrower independence or authenticate feature measurement dates. "
        "Historical delinquency counts can be legitimate predictors, but "
        "unknown lookback boundaries leave upstream temporal leakage unresolved. "
        + cite("leakage", "task4", "target"),
        "The Task 2 leakage document describes the then-missing cross-run "
        "registry. That specific engineering gap was subsequently addressed "
        "by Task 3; the older document is retained rather than rewritten. "
        "No explanation result proves absence of unidentified leakage. "
        + cite("registry", "task6"),
        "## 7. Development Population",
        table(
            ["Category", "Population", "Rows", "Positive outcomes", "Event fraction"],
            [
                [
                    "TRAINING_ONLY",
                    "Saved training records",
                    t4["rows"],
                    t4["events"],
                    num(t4["observed_event_rate"]),
                ]
            ],
        ),
        "The existing training population has previously been explored. "
        "Cross-validation within it is research/development evidence, not "
        "a newly independent final test or evidence of expansion to thin-file "
        "applicants. Population selection and rejected-applicant performance "
        "are unknown. " + cite("training_audit", "task4", "target"),
        "## 8. Data Splitting and Experimental Governance",
        "[GOVERNANCE] Dataset -> source fingerprint -> candidate split -> "
        "sample fingerprint -> holdout reservation -> training/development "
        "-> calibration -> freeze selection -> consumed status before first "
        "final prediction -> final evaluation -> frozen artifacts. Failed "
        "access remains consumed conservatively. " + cite("registry"),
        table(
            ["Category", "Historical partition", "Rows", "Events"],
            [
                ["GOVERNANCE", name, p["rows"], p["bad_count"]]
                for name, p in e.historical["partitions"].items()
            ],
        ),
        "Source fingerprints protect byte identity; raw predictor-profile "
        "fingerprints protect profile reuse across source versions without "
        "including outcomes or row IDs. The historical bridge uses saved "
        "pandas profile hashes, not cryptographic borrower identities. "
        "Missing/malformed or altered anchored ledgers fail closed. Atomic "
        "replacement under a bounded exclusive lock prevents partial updates. "
        "Hash collisions, unidentified changed borrower snapshots and deliberate "
        "joint alteration of code/evidence remain limitations. The ledger "
        "tracks final-holdout access, not every historical research inspection. "
        + cite("registry", "ledger"),
        "## 9. Candidate Models",
        "Logistic Regression supplies an additive, interpretable benchmark "
        "on transformed inputs. XGBoost tests flexible nonlinear structure "
        "and interactions using a fixed boosted-tree specification. Only "
        "these two families enter this development decision. Legacy models "
        "are repository history, not additional validated candidates. "
        + cite("task4", "experiment"),
        "## 10. Preprocessing and Feature Treatment",
        f"[GOVERNANCE] Training-fitted caps use quantiles "
        f"{t4['config']['clip_lower_quantile']} and {t4['config']['clip_upper_quantile']}. "
        "Inputs except age receive log1p; training medians impute missing "
        "values and missing indicators are retained. Logistic standardization "
        "uses training means/population SDs. These transforms are modeling "
        "choices, not verified corrections of source truth. All learned "
        "preprocessing remains confined to fitting roles. " + cite("task4", "task6"),
        "## 11. Model Development Methodology",
        f"[GOVERNANCE] Task 4 uses {t4['fold_count']} StratifiedGroupKFold "
        f"folds with shuffle seed {t4['seed']}, grouping exact raw predictor "
        "profiles. Fresh fixed-specification candidates fit fold training "
        "rows and predict held-out training-population folds. Tasks 5/6 "
        "reuse the recorded outer evaluation positions. No hyperparameter "
        "optimization, new model specification or historical-champion refit "
        "was performed for this report. " + cite("task4", "task5", "task6"),
        "## 12. Discrimination Performance",
        "[TRAINING_ONLY] Task 4 pooled out-of-fold results. Higher AUC, Gini "
        "and PR measures indicate stronger ranking; lower Brier and log loss "
        "indicate better probability quality. Average precision and trapezoidal "
        "PR-AUC use different definitions and are reported separately. " + cite("task4"),
        table(
            ["Model", "AUC", "Gini", "Average precision", "PR-AUC", "KS", "Brier", "Log loss"],
            [
                [
                    m,
                    *[
                        num(c["oof"]["metrics"][k])
                        for k in (
                            "roc_auc",
                            "gini",
                            "average_precision",
                            "pr_auc",
                            "ks",
                            "brier",
                            "log_loss",
                        )
                    ],
                ]
                for m, c in t4["candidates"].items()
            ],
        ),
        "## 13. Probability Calibration",
        "[TRAINING_ONLY] Task 4 calibration-in-the-large (CITL) estimates an "
        "intercept with logit predictions as an offset; slope comes from a "
        "joint intercept/slope fit. Values near zero/one respectively are "
        "desirable, but do not establish segment, tail or future calibration. "
        "Reliability tables retain their bin-level qualifications. " + cite("task4"),
        table(
            ["Model", "Mean prediction", "Observed event fraction", "CITL", "Slope"],
            [
                [
                    m,
                    num(c["oof"]["metrics"]["mean_probability"]),
                    num(c["oof"]["metrics"]["observed_bad_rate"]),
                    num(c["oof"]["calibration"]["calibration_intercept"]),
                    num(c["oof"]["calibration"]["calibration_slope"]),
                ]
                for m, c in t4["candidates"].items()
            ],
        ),
    ]

    parts += [
        "## 14. Nested Calibration Study",
        "[TRAINING_ONLY] Task 5 separates outer evaluation from inner "
        "base fitting, calibration and method selection. The inner-selected "
        "row evaluates that selection procedure; fixed-method rows compare "
        "the same nested base fits. Reduced base-fit populations mean these "
        "values should not be treated as a direct repeat of Task 4. " + cite("task5"),
        table(
            ["Model", "Method", "AUC", "Brier", "Log loss", "CITL", "Slope"],
            [
                [
                    m,
                    method,
                    *[num(v["pooled_oof"]["metrics"][k]) for k in ("roc_auc", "brier", "log_loss")],
                    num(v["pooled_oof"]["calibration"]["calibration_intercept"]),
                    num(v["pooled_oof"]["calibration"]["calibration_slope"]),
                ]
                for m, methods in t5["models"].items()
                for method, v in methods.items()
            ],
        ),
        "[TRAINING_ONLY] Calibrated minus raw paired differences and conditional "
        "percentile intervals. Negative Brier/log-loss differences favor "
        "calibration. AUC differences show any ranking cost. " + cite("task5"),
        table(
            ["Model", "Method", "Metric", "Point difference", "Interval"],
            [
                [
                    m,
                    method,
                    k,
                    num(v["point_differences"][k]),
                    interval(v["paired_differences"]["intervals"][k]),
                ]
                for m, methods in t5["paired_comparisons"].items()
                for method, v in methods.items()
                for k in ("roc_auc", "brier", "log_loss")
            ],
        ),
        "[GOVERNANCE] The prespecified Brier gain margin was "
        + num(t5["study_config"]["minimum_brier_gain"])
        + "; the paired evidence must also pass the log-loss guard. "
        "Logistic ISOTONIC improves probability quality with a modest ranking "
        "cost. Logistic SIGMOID slightly helps log loss while worsening Brier, "
        "so it is not an improvement on every measure. XGBoost remains RAW "
        "because calibrated methods do not meet both requirements. These "
        "are post-study research recommendations, without artifact promotion. " + cite("task5"),
        "## 15. Threshold Diagnostics",
        "[TRAINING_ONLY] The predeclared Task 4 diagnostic grid illustrates "
        "precision/recall trade-offs. Predicted-positive fraction is a "
        "classification quantity, not a decline rate. Fold-training "
        "prevalence thresholds are separately retained in Task 4 evidence. "
        "No optimized pooled-OOF cutoff, lending profitability or operational "
        "approval band is established. " + cite("task4"),
        table(
            [
                "Model",
                "Threshold",
                "Precision",
                "Recall",
                "Specificity",
                "Predicted-positive fraction",
                "Confusion matrix TN/FP/FN/TP",
            ],
            [
                [
                    m,
                    *[
                        num(row[k])
                        for k in (
                            "threshold",
                            "precision",
                            "recall",
                            "specificity",
                            "predicted_positive_rate",
                        )
                    ],
                    "/".join(str(v) for pair in row["confusion_matrix"] for v in pair),
                ]
                for m, c in t4["candidates"].items()
                for row in c["threshold_analysis"]
            ],
        ),
        "## 16. Statistical Uncertainty",
        f"[TRAINING_ONLY] Task 4 intervals use confidence level "
        f"{num(t4['uncertainty']['confidence_level'])} and "
        f"{t4['uncertainty']['valid_samples']} valid resamples of "
        f"{t4['uncertainty']['resampling_unit']}. They are "
        + t4["uncertainty"]["interval_type"]
        + ". "
        + t4["uncertainty"]["limitations"]
        + ". "
        + cite("task4"),
        table(
            ["Model", "Metric", "Conditional interval"],
            [
                [m, k, interval(v)]
                for m, metrics in t4["uncertainty"]["intervals"].items()
                for k, v in metrics.items()
            ],
        ),
        "[TRAINING_ONLY] Paired Task 4 differences: "
        + t4["uncertainty"]["paired_differences"]["direction"]
        + ".",
        table(
            ["Metric", "Paired conditional interval"],
            [
                [k, interval(v)]
                for k, v in t4["uncertainty"]["paired_differences"]["intervals"].items()
            ],
        ),
        "Task 5 intervals are likewise conditional on fixed nested OOF "
        "predictions. Neither captures full refitting/selection uncertainty, "
        "unidentified borrower dependence, economic regime changes or external "
        "transportability. Reliability-bin Wilson intervals are approximate "
        "row-binomial summaries, not borrower-cluster-robust intervals. "
        "No new uncertainty calculation was performed. " + cite("task4", "task5"),
        "## 17. Explainability",
        "[TRAINING_ONLY] Leading original logistic predictors and their "
        "associations. beta is a standardized coefficient per one training "
        "SD of a capped/imputed/transformed input, conditional on other "
        "columns. exp(mean beta) summarizes fold coefficients; it is neither "
        "a pooled fitted odds ratio nor mean fold odds ratio. Log1p inputs "
        "do not imply raw currency/count-unit effects. Indicator 0-to-1 "
        "contrasts require the transformed-unit coefficient. These raw-model "
        "coefficients do not explain isotonic-calibrated odds. " + cite("task6"),
        table(
            ["Predictor", "Mean beta", "Direction", "exp(mean beta)", "Sign flip"],
            [
                [
                    label(r["feature"]),
                    num(coefficients[r["feature"]]["coefficient"]["mean"]),
                    coefficients[r["feature"]]["dominant_sign"],
                    num(math.exp(coefficients[r["feature"]]["coefficient"]["mean"])),
                    coefficients[r["feature"]]["sign_flip"],
                ]
                for r in sorted(comparison, key=lambda r: r["logistic_standardized_rank"])[:5]
            ],
        ),
        "[TRAINING_ONLY] Exact native TreeSHAP contributions are in raw "
        "margin/log-odds, using tree-path-dependent training cover. All "
        "evaluation rows receive aligned feature attribution and additivity "
        "checks. Mean absolute SHAP describes model contribution magnitude, "
        "not causation or a universal signed effect. " + cite("task6"),
        table(
            ["Predictor", "Pooled mean absolute SHAP", "Rank"],
            [
                [label(r["feature"]), num(r["pooled_mean_abs_shap"]), num(r["global_rank"])]
                for r in tree[:5]
            ],
        ),
        "Both families share the same leading original predictor families. "
        "Utilization and open credit lines receive different ranks; coefficient "
        "and SHAP units are not directly comparable. Correlated inputs can "
        "redistribute attribution. " + cite("task6"),
        "Observed-context dependence diagnostics show increasing utilization "
        "contributions with a zero-input exception, delinquency-count jumps "
        "and flattening, and a curved age relationship. Sparse tails and "
        "anomalous 96/98 counts limit interpretation. Limited interactions "
        "are exploratory; no ablation establishes the causal source of a "
        "performance advantage. These findings are consistent with nonlinear "
        "benefit but cannot rule out unidentified leakage. " + cite("task6"),
        "## 18. Explanation Stability",
        "[TRAINING_ONLY] Mean pairwise fold Spearman agreement for original "
        "XGB features is "
        + num(t6["xgboost_rank_agreement"]["mean_rho"])
        + ". Overlapping fold fits make this descriptive rank agreement, "
        "not proof of temporal stability, stable attribution sizes or fairness. " + cite("task6"),
        table(
            ["XGB predictor", "Ranks by fold"],
            [[label(r["feature"]), ", ".join(f"{v:g}" for v in r["fold_ranks"])] for r in tree],
        ),
        "[TRAINING_ONLY] Logistic direction changes occurred in: "
        + ", ".join(label(r["feature"]) for r in t6["logistic_stability"] if r["sign_flip"])
        + ". Their weak mean associations should not be assigned a robust "
        "direction. Debt ratio and income missingness retain signs while "
        "showing magnitude variation. " + cite("task6"),
        "## 19. Champion/Challenger Assessment",
        "This qualitative decision matrix avoids arbitrary composite scoring. "
        "Performance entries are TRAINING_ONLY; interpretability, complexity "
        "and governance burden are reasoned assessments of the documented "
        "specifications. " + cite("task4", "task5", "task6"),
        table(
            ["Dimension", "Logistic Regression", "Raw XGBoost"],
            [
                [
                    "Discrimination",
                    "AUC " + num(lr["metrics"]["roc_auc"]),
                    "AUC " + num(xgb["metrics"]["roc_auc"]),
                ],
                [
                    "Probability quality",
                    "Brier " + num(lr["metrics"]["brier"]),
                    "Brier " + num(xgb["metrics"]["brier"]),
                ],
                ["Calibration", "ISOTONIC research recommendation", "RAW retained in nested study"],
                [
                    "Interpretability",
                    "Conditional transformed-input coefficients",
                    "TreeSHAP plus dependence diagnostics",
                ],
                [
                    "Explanation stability",
                    "Strong leading signs; weaker sign flips",
                    "High descriptive fold-rank agreement",
                ],
                [
                    "Nonlinear modeling",
                    "Additive on specified transformed inputs",
                    "Thresholds and interactions",
                ],
                ["Complexity", "Simpler functional form", "More complex tree ensemble"],
                [
                    "Governance burden",
                    "Transforms/calibration require review",
                    "Transforms, attribution and nonlinear behavior require review",
                ],
            ],
        ),
        "**Raw XGBoost is the development champion. Logistic Regression "
        "remains the interpretable challenger.** This decision is confined "
        "to available development evidence; it does not replace the frozen "
        "historical selected model or establish deployment approval.",
        "---",
        "## 20. Historical Locked-Holdout Evidence",
        "> These metrics are historical retained evidence from an "
        "already-consumed holdout. They were not regenerated during the "
        "current validation program.",
        table(
            ["Category", "Historical model/method", "AUC", "Brier", "Log loss"],
            [
                [
                    "HISTORICAL_LOCKED_HOLDOUT",
                    "XGBoost / historically selected sigmoid",
                    *[
                        num(e.historical["final_metrics"]["xgboost"]["sigmoid"][k])
                        for k in ("roc_auc", "brier", "log_loss")
                    ],
                ]
            ],
        ),
        "These are frozen historical results, not the raw-XGBoost development "
        "recommendation evaluated on new data. No prediction file, raw "
        "holdout record or model bundle was opened for this report. No new "
        "holdout metric or confidence interval was calculated. " + cite("historical", "ledger"),
        "---",
        "## 21. Model Limitations",
        "\n".join("- " + value for value in LIMITATIONS),
        cite("target", "suitability", "leakage", "task4", "task5", "task6"),
        "## 22. Model Risk Assessment",
        "The following HIGH/MEDIUM/LOW/INFORMATIONAL labels are a local "
        "qualitative research-review framework, not a bank or regulator's "
        "classification. HIGH means unresolved evidence prevents a stated "
        "use claim; MEDIUM means interpretation or robustness needs further "
        "review; LOW means a specific demonstrated control limits a risk; "
        "INFORMATIONAL describes a documented boundary, without certifying "
        "residual safety.",
        table(
            ["Concern", "Level", "Evidence/reason", "Required resolution"],
            [
                [
                    "Temporal generalization",
                    "HIGH",
                    "No usable observation/event dates",
                    "Dated independent out-of-time evaluation",
                ],
                [
                    "Borrower independence",
                    "HIGH",
                    "Row/profile keys are not borrower IDs",
                    "Stable borrower/facility lineage",
                ],
                [
                    "Regulatory target mismatch",
                    "HIGH",
                    "Two-year serious delinquency only",
                    "Separate dated target/default contract before regulatory use",
                ],
                [
                    "Provenance and selection",
                    "HIGH",
                    "Pristine file and sampling unverified",
                    "Authenticated source and acceptance/measurement lineage",
                ],
                [
                    "Historical exploration",
                    "MEDIUM",
                    "Training population previously explored",
                    "Independent future/external validation",
                ],
                [
                    "Sparse/anomalous tails",
                    "MEDIUM",
                    "Unknown coded values and small support",
                    "Supplier definitions and justified robustness assessment",
                ],
                [
                    "Calibration transportability",
                    "HIGH",
                    "No temporal calibration evidence",
                    "Mature dated outcome cohorts",
                ],
                [
                    "Fairness",
                    "HIGH",
                    "Not assessed; explanations are insufficient",
                    "Appropriate population/group data and fairness review",
                ],
                [
                    "Consumed-data reuse mechanism",
                    "LOW",
                    "Anchored fail-closed registry",
                    "Maintain ledger and independent evidence review",
                ],
                [
                    "Research-only scope",
                    "INFORMATIONAL",
                    "No operational approval claimed",
                    "Independent governance before a deployment decision",
                ],
            ],
        ),
        cite("target", "suitability", "leakage", "registry", "task4", "task5", "task6"),
        "## 23. Monitoring Recommendations",
        "Proposed practice for a future real deployment; no infrastructure "
        "or alert threshold is implemented or endorsed by this report. "
        "A risk owner must approve populations, baselines, maturity rules, "
        "review cadence and actions before deployment. " + cite("suitability", "target"),
        table(
            ["Monitoring layer", "Measures", "Interpretation/action"],
            [
                [
                    "Data",
                    "Schema, missingness, ranges, category/value validity",
                    "Investigate source/pipeline changes and affected cohorts",
                ],
                [
                    "Population",
                    "Feature and score distributions; PSI or appropriate drift measures",
                    "Investigate sustained/material change with sample-size context",
                ],
                [
                    "Mature performance",
                    "AUC/Gini, PR-AUC/average precision, Brier, log loss",
                    "Use completed target windows; do not relabel immature outcomes",
                ],
                [
                    "Calibration",
                    "Observed vs predicted rates, CITL, slope, reliability",
                    "Investigate stable ranking with systematic probability error",
                ],
                [
                    "Explanation",
                    "Feature-importance drift and rank stability",
                    "Review shifts with feature correlation and cohort context",
                ],
                [
                    "Governance",
                    "Lineage breaks, sustained deterioration, new population or target",
                    "Investigate; consider approved recalibration or redevelopment",
                ],
            ],
        ),
        "No universal PSI or regulatory trigger is invented. Drift alone "
        "does not prove performance failure. Mature outcomes are required "
        "for performance/calibration claims, and observed accepted-loan "
        "performance cannot identify rejected-applicant outcomes without "
        "additional evidence. Recalibration or redevelopment requires "
        "documented review and fresh independent validation.",
        "## 24. Governance and Reproducibility",
        "[GOVERNANCE] The generator reads an explicit allowlist of committed "
        "aggregates, documentation, configuration and source bytes. It "
        "checks cross-artifact targets, provenance, historical metrics, "
        "fold identities, calibration comparisons, feature rankings, code "
        "hashes and the consumed registry anchor. Missing, malformed or "
        "conflicting evidence causes failure; it is not silently reconciled.",
        "[GOVERNANCE] Repository package version: `"
        + metadata["repository_version"]
        + "`; generation-time Git baseline: `"
        + str(metadata["git_commit"])
        + "`. "
        "This commit identifies the checkout used to assemble evidence, not "
        "an independent model validation sign-off. The manifest records "
        "generation time, source identities, generator hashes, commands "
        "and per-claim categories. " + cite("registry", "experiment", "historical"),
        "[GOVERNANCE] Task 1 bounded container-startup retry is documented "
        "by [smoke-test code](../../scripts/smoke_container.py) and "
        "[regression tests](../../tests/test_container_smoke.py). CI "
        "configuration demonstrates intended checks, not a newly successful "
        "hosted run. This generator does not execute pytest, CI or Docker; "
        "their current results must be reported separately by the verification run.",
        "[GOVERNANCE] Retained experiment environments (not regenerated):",
        table(
            ["Package", "Task 4", "Task 5", "Task 6"],
            [
                [
                    name,
                    t4["versions"].get(name, "not recorded"),
                    t5["versions"].get(name, "not recorded"),
                    t6["versions"].get(name, "not recorded"),
                ]
                for name in sorted(set(t4["versions"]) | set(t5["versions"]) | set(t6["versions"]))
            ],
        ),
        "[GOVERNANCE] Tasks 4/6 model/fold seed: "
        + str(t4["seed"])
        + "; Task 4 bootstrap seed: "
        + str(t4["uncertainty"]["seed"])
        + "; Task 5 study seed: "
        + str(t5["study_config"]["seed"])
        + ". Inner and resampling seeds are retained in the source JSON. "
        "The lockfile records reproducible dependency resolution.",
    ]

    parts += [
        "## Appendix A - Metrics",
        table(
            ["Metric", "Meaning and boundary"],
            [
                [
                    "AUC / Gini",
                    "Ranking discrimination; Gini = 2*AUC - 1; not probability calibration",
                ],
                [
                    "Average precision / PR-AUC",
                    "Non-interpolated summary vs trapezoidal curve area; not interchangeable",
                ],
                ["KS", "Maximum absolute TPR-FPR; directionless discrimination diagnostic"],
                ["Brier / log loss", "Probability errors; lower is better; population-dependent"],
                [
                    "CITL / slope",
                    "Offset intercept near zero; joint-fit slope near one are desirable",
                ],
                [
                    "Precision / recall",
                    "Threshold-dependent classification behavior, not lending profit",
                ],
                [
                    "SHAP magnitude / rank agreement",
                    "Model contribution and descriptive fold agreement, not causality",
                ],
            ],
        ),
        cite("task4", "task5", "task6"),
        "## Appendix B - Artifact Manifest",
        "[Machine-readable manifest](model_validation_manifest.json) records "
        "source evidence categories, file SHA-256 and LF-normalized "
        "SHA-256, generator hashes, frozen artifact digests copied from "
        "committed historical manifests, quantitative results and "
        "limitations. Non-frozen text hash checks allow only Git LF/CRLF "
        "conversion; frozen model source remains byte-exact. File hashes "
        "establish consistency with recorded evidence, not independent "
        "authentication or pristine-source equivalence. Local model bundles "
        "and historical raw data/prediction files are outside the "
        "generator's read allowlist.",
        "## Appendix C - Reproduction Commands",
        "Generate the report from committed evidence without the original "
        "dataset or ignored model artifacts:",
        "```console\nuv sync --locked\n"
        "uv run --no-sync python scripts/generate_model_validation_report.py\n"
        "uv run --no-sync pytest tests/test_model_validation_report.py -q\n"
        "uv run --no-sync pytest -q\n"
        "uv run --no-sync ruff check src tests scripts api\n"
        "uv run --no-sync ruff format --check src tests scripts api\n"
        "uv run --no-sync python scripts/check_governance.py\n"
        "uv run --no-sync python scripts/check_repository.py\n```",
        "Run the generator twice on the same checkout to compare Markdown "
        "bytes and manifest content excluding generated_at. For fully "
        "identical metadata, supply the same timezone-aware ISO timestamp "
        "via --generated-at. A changed Git baseline or evidence/code "
        "identity appropriately changes metadata. Historic source "
        "artifacts need not be present, and no training command is needed.",
        "## 25. Final Development Conclusion",
        "Track A establishes a reproducible research framework for developing "
        "and validating models of two-year serious delinquency. Within "
        "available TRAINING_ONLY evidence, raw XGBoost provides stronger "
        "discrimination and probability quality than the Logistic Regression "
        "benchmark while exhibiting consistent feature-attribution rankings "
        "across grouped folds. Logistic Regression remains the interpretable "
        "challenger, with an ISOTONIC research recommendation from nested "
        "calibration. Historical artifacts remain frozen. The project "
        "does not establish regulatory PD validity, temporal generalization, "
        "lending profitability or deployment readiness. "
        + cite("task4", "task5", "task6", "suitability"),
    ]
    return "\n\n".join(parts) + "\n"


def git_commit(root):
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(top).resolve() != root:
            return None
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(root, *, generated_at=None, commit=None):
    e = Evidence(root)
    timestamp = datetime.now(UTC) if generated_at is None else datetime.fromisoformat(generated_at)
    require(timestamp.tzinfo is not None, "Generation timestamp must be timezone-aware")
    metadata = {
        "schema_version": 1,
        "generated_at": timestamp.astimezone(UTC).isoformat(),
        "git_commit": git_commit(e.root) if commit is None else commit,
        "repository_version": tomllib.loads(e.payloads["pyproject.toml"].decode("utf-8"))[
            "project"
        ]["version"],
    }
    claims = {
        "development_performance": {
            "evidence_category": "TRAINING_ONLY",
            "path": "reports/model_validation/pd_diagnostics.json",
            "pointer": "/candidates",
        },
        "nested_calibration": {
            "evidence_category": "TRAINING_ONLY",
            "path": "reports/model_validation/calibration_study.json",
            "pointer": "/models",
        },
        "calibration_decision": {
            "evidence_category": "TRAINING_ONLY",
            "path": "reports/model_validation/calibration_study.json",
            "pointer": "/recommendations",
        },
        "explanation_ranks": {
            "evidence_category": "TRAINING_ONLY",
            "path": "reports/model_validation/explainability_stability.json",
            "pointer": "/cross_model_comparison",
        },
        "rank_agreement": {
            "evidence_category": "TRAINING_ONLY",
            "path": "reports/model_validation/explainability_stability.json",
            "pointer": "/xgboost_rank_agreement",
        },
        "training_data_quality": {
            "evidence_category": "DESCRIPTIVE",
            "path": "reports/data/TRAINING_DATA_AUDIT.json",
            "pointer": "/columns",
        },
        "target_scope": {"evidence_category": "DESCRIPTIVE", "path": "docs/TARGET_DEFINITION.md"},
        "regulatory_boundary": {
            "evidence_category": "DESCRIPTIVE",
            "path": "reports/data/DATASET_SUITABILITY.md",
        },
        "consumed_status": {
            "evidence_category": "GOVERNANCE",
            "path": "reports/holdout_registry.json",
            "pointer": "/entries",
        },
        "historical_xgboost": {
            "evidence_category": "HISTORICAL_LOCKED_HOLDOUT",
            "path": "reports/phase5_validation_summary.json",
            "pointer": "/final_metrics/xgboost/sigmoid",
        },
    }
    try:
        report = render(e, metadata)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError(f"Malformed report evidence: {exc}") from exc
    # Hash actual generator bytes; never insert machine-specific paths.
    generator_names = [
        "scripts/generate_model_validation_report.py",
        "src/credit_risk/reporting/__init__.py",
        "src/credit_risk/reporting/evidence.py",
        "src/credit_risk/reporting/mdvr.py",
    ]
    generator_hashes = {name: hashlib.sha256(e.read(name)).hexdigest() for name in generator_names}
    for name in generator_names:
        e.categories[name] = "GOVERNANCE"
    manifest = {
        **metadata,
        "report_path": REPORT,
        "report_sha256": hashlib.sha256(report.encode("utf-8")).hexdigest(),
        "source_evidence": e.manifest_sources(),
        "generator_sha256": generator_hashes,
        "claims": claims,
        "quantitative_evidence": {
            "task4": {
                "evidence_category": "TRAINING_ONLY",
                "results": {m: v["oof"] for m, v in e.task4["candidates"].items()},
                "uncertainty": e.task4["uncertainty"],
            },
            "task5": {
                "evidence_category": "TRAINING_ONLY",
                "results": {
                    m: {method: v["pooled_oof"] for method, v in methods.items()}
                    for m, methods in e.task5["models"].items()
                },
                "paired_comparisons": e.task5["paired_comparisons"],
                "recommendations": e.task5["recommendations"],
            },
            "task6": {
                "evidence_category": "TRAINING_ONLY",
                "logistic_stability": e.task6["logistic_stability"],
                "xgboost_stability": e.task6["xgboost_stability"],
                "rank_agreement": e.task6["xgboost_rank_agreement"],
            },
            "historical": {
                "evidence_category": "HISTORICAL_LOCKED_HOLDOUT",
                "model": "xgboost",
                "method": "sigmoid",
                "metrics": {
                    k: e.historical["final_metrics"]["xgboost"]["sigmoid"][k]
                    for k in ("roc_auc", "brier", "log_loss")
                },
                "regenerated": False,
            },
        },
        "frozen_artifact_digests": {
            "evidence_category": "GOVERNANCE",
            "verification": "recorded manifest values only; local payloads not opened",
            "phase4": e.experiment["artifacts_sha256"],
            "phase5": e.historical["artifacts_sha256"],
        },
        "registry": {
            "evidence_category": "GOVERNANCE",
            "historical_anchor_verified": True,
            "entries": [
                {"run": v.run, "status": v.status, "sample_count": len(v.samples)}
                for v in e.ledger.entries
            ],
        },
        "limitations": LIMITATIONS,
        "verification": {
            "evidence_consistency": "passed",
            "frozen_source_hashes": "passed",
            "holdout_access": False,
            "model_loading": False,
            "training": False,
            "pytest_or_hosted_ci_run_by_generator": False,
        },
        "reproduction_command": (
            "uv run --no-sync python scripts/generate_model_validation_report.py"
        ),
    }
    e.assert_unchanged()
    return report, manifest


def generate(root, **kwargs):
    root = Path(root).resolve()
    report, manifest = build(root, **kwargs)
    for name in (REPORT, MANIFEST):
        require((root / name).resolve().is_relative_to(root), "Report output escapes repository")
    (root / REPORT).parent.mkdir(parents=True, exist_ok=True)
    # Explicit LF output makes report_sha256 independent of the host newline convention.
    (root / REPORT).write_text(report, encoding="utf-8", newline="\n")
    (root / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report, manifest
