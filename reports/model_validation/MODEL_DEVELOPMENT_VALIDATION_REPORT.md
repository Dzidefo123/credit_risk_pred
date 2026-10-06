# Model Development & Validation Report

## 1. Executive Summary

**Objective:** estimate the probability of a two-year serious-delinquency outcome using the repository-attributed Give Me Some Credit benchmark. This is a credit-risk ML research benchmark, not a lender deployment or regulatory PD validation. Evidence: [target](../../docs/TARGET_DEFINITION.md), [suitability](../data/DATASET_SUITABILITY.md).

**Development conclusion:** raw XGBoost is the development champion and Logistic Regression is the interpretable challenger. Task 4 supports stronger XGBoost discrimination and probability quality; Task 5 retains RAW for XGBoost and recommends ISOTONIC for Logistic Regression as research. Task 6 finds broadly shared predictor families and consistent tree attribution rankings. These conclusions do not replace frozen historical models, calibrators or policy. Evidence: [task4](pd_diagnostics.json), [task5](calibration_study.json), [task6](explainability_stability.json).

| Evidence category | Meaning |
| --- | --- |
| TRAINING_ONLY | Tasks 4-6 results from saved training records; no consumed holdout access |
| HISTORICAL_LOCKED_HOLDOUT | Retained results from the already-consumed final evaluation |
| DESCRIPTIVE | Retained provenance/data-quality evidence; not predictive validation |
| GOVERNANCE | Registry, hashes, code, experimental controls and verification evidence |

All result tables carry a category. Sections containing recommendations are proposed practice, not measured production evidence. Category labels apply to the claim, not to every incidental field in a mixed historical artifact.

## 2. Model Purpose and Scope

The decision supported is a development choice between fixed candidate families for this benchmark. The repository has not established a real lending approval, risk-based price, capital requirement or provisioning decision from this evidence. A source record is treated as a financial snapshot before a prospective outcome; verified origination time, population eligibility and economic decision costs are absent. Evidence: [target](../../docs/TARGET_DEFINITION.md), [suitability](../data/DATASET_SUITABILITY.md).

### Visible IFRS 9 boundary

**Track A does not implement or empirically validate 12-month PD, lifetime PD, LGD, EAD, SICR, Stage 1/2/3 or ECL.** Give Me Some Credit does not supply the dated event, facility, exposure, cash-flow and recovery records required to support those components. Separate synthetic portfolio/loss demonstrations in the repository are educational and do not fill these empirical gaps. These components belong to a future longitudinal/facility-level research track. Evidence: [suitability](../data/DATASET_SUITABILITY.md), [target](../../docs/TARGET_DEFINITION.md).

## 3. Dataset and Provenance

[DESCRIPTIVE] Repository history attributes the data to Kaggle Give Me Some Credit. The intended competition observation unit is a borrower/person profile; the local unit that can actually be verified is a source record. The row index does not establish borrower identity. Pristine-download equivalence remains unverified. Source dates, supplier sampling, acceptance/rejection lineage and currency are unverified. No download or new source inspection was performed to generate this report. Evidence: [audit](../data/DATA_AUDIT.md), [dictionary](../../docs/DATA_DICTIONARY.md), [target](../../docs/TARGET_DEFINITION.md).

[GOVERNANCE] Recorded source SHA-256: `9d863f88e2ef2190ed608c15c3dd362d1833d940e35f4f2bf22660e04b5f05cb`.

## 4. Target Definition

> The target represents a two-year serious-delinquency outcome and should not be interpreted as a regulatory 12-month or lifetime Probability of Default measure.

[DESCRIPTIVE] SeriousDlqin2yrs is the inherited binary label for serious delinquency at the 90-day threshold or a worse credit outcome within the source two-year task. The event was not reconstructed from dated histories. A binary prediction ranks/scores the label; a calibrated event probability measures agreement with its observed frequency in a specified population. Neither establishes regulatory default adjudication, horizon equivalence or compliant PD estimation. Negative-label maturity, censoring and reporting lags cannot be verified at record level. Evidence: [target](../../docs/TARGET_DEFINITION.md).

## 5. Data Quality Assessment

[DESCRIPTIVE] The table below reuses the saved training-only audit. Older whole-source descriptions remain descriptive historical audit evidence and were not recomputed. Flags request investigation rather than automatic deletion or recoding. Evidence: [training_audit](../data/TRAINING_DATA_AUDIT.json), [audit](../data/DATA_AUDIT.md), [dictionary](../../docs/DATA_DICTIONARY.md).

| Predictor | Missing rows | Missing fraction | Finite minimum | Finite maximum |
| --- | --- | --- | --- | --- |
| Unsecured utilization | 0 | 0.000000 | 0.000000 | 50708.000000 |
| Age | 0 | 0.000000 | 0.000000 | 107.000000 |
| 30-59-day past-due count | 0 | 0.000000 | 0.000000 | 98.000000 |
| Debt ratio | 0 | 0.000000 | 0.000000 | 329664.000000 |
| Monthly income | 13430 | 0.198780 | 0.000000 | 1560100.000000 |
| Open credit lines/loans | 0 | 0.000000 | 0.000000 | 58.000000 |
| 90+-day past-due count | 0 | 0.000000 | 0.000000 | 98.000000 |
| Real estate loans/lines | 0 | 0.000000 | 0.000000 | 54.000000 |
| 60-89-day past-due count | 0 | 0.000000 | 0.000000 | 98.000000 |
| Dependents | 1754 | 0.025961 | 0.000000 | 13.000000 |

Past-due counts 96/98 may be anomalous or sentinel-like; their meaning is not established. Age zero is implausible for an adult lending applicant. Utilization above one can represent over-limit balances, while extreme utilization/debt/income tails need supplier definitions. Observed missingness does not identify its mechanism. Evidence: [audit](../data/DATA_AUDIT.md), [dictionary](../../docs/DATA_DICTIONARY.md).

## 6. Leakage Assessment

Specific legacy defects included learned imputation before splitting, outcome-dependent exclusions and in-sample evaluation. The current pipeline fits transformations inside training roles and excludes the target and row ID from predictors. Exact-profile grouping controls observed duplicate-profile overlap. It does not establish borrower independence or authenticate feature measurement dates. Historical delinquency counts can be legitimate predictors, but unknown lookback boundaries leave upstream temporal leakage unresolved. Evidence: [leakage](../data/LEAKAGE_REVIEW.md), [task4](pd_diagnostics.json), [target](../../docs/TARGET_DEFINITION.md).

The Task 2 leakage document describes the then-missing cross-run registry. That specific engineering gap was subsequently addressed by Task 3; the older document is retained rather than rewritten. No explanation result proves absence of unidentified leakage. Evidence: [registry](../../docs/HOLDOUT_REGISTRY.md), [task6](explainability_stability.json).

## 7. Development Population

| Category | Population | Rows | Positive outcomes | Event fraction |
| --- | --- | --- | --- | --- |
| TRAINING_ONLY | Saved training records | 67562 | 4514 | 0.066813 |

The existing training population has previously been explored. Cross-validation within it is research/development evidence, not a newly independent final test or evidence of expansion to thin-file applicants. Population selection and rejected-applicant performance are unknown. Evidence: [training_audit](../data/TRAINING_DATA_AUDIT.json), [task4](pd_diagnostics.json), [target](../../docs/TARGET_DEFINITION.md).

## 8. Data Splitting and Experimental Governance

[GOVERNANCE] Dataset -> source fingerprint -> candidate split -> sample fingerprint -> holdout reservation -> training/development -> calibration -> freeze selection -> consumed status before first final prediction -> final evaluation -> frozen artifacts. Failed access remains consumed conservatively. Evidence: [registry](../../docs/HOLDOUT_REGISTRY.md).

| Category | Historical partition | Rows | Events |
| --- | --- | --- | --- |
| GOVERNANCE | train | 67562 | 4514 |
| GOVERNANCE | development | 22483 | 1504 |
| GOVERNANCE | calibration | 29964 | 2005 |
| GOVERNANCE | test | 29991 | 2003 |

Source fingerprints protect byte identity; raw predictor-profile fingerprints protect profile reuse across source versions without including outcomes or row IDs. The historical bridge uses saved pandas profile hashes, not cryptographic borrower identities. Missing/malformed or altered anchored ledgers fail closed. Atomic replacement under a bounded exclusive lock prevents partial updates. Hash collisions, unidentified changed borrower snapshots and deliberate joint alteration of code/evidence remain limitations. The ledger tracks final-holdout access, not every historical research inspection. Evidence: [registry](../../docs/HOLDOUT_REGISTRY.md), [ledger](../holdout_registry.json).

## 9. Candidate Models

Logistic Regression supplies an additive, interpretable benchmark on transformed inputs. XGBoost tests flexible nonlinear structure and interactions using a fixed boosted-tree specification. Only these two families enter this development decision. Legacy models are repository history, not additional validated candidates. Evidence: [task4](pd_diagnostics.json), [experiment](../phase4_experiment.json).

## 10. Preprocessing and Feature Treatment

[GOVERNANCE] Training-fitted caps use quantiles 0.001 and 0.999. Inputs except age receive log1p; training medians impute missing values and missing indicators are retained. Logistic standardization uses training means/population SDs. These transforms are modeling choices, not verified corrections of source truth. All learned preprocessing remains confined to fitting roles. Evidence: [task4](pd_diagnostics.json), [task6](explainability_stability.json).

## 11. Model Development Methodology

[GOVERNANCE] Task 4 uses 5 StratifiedGroupKFold folds with shuffle seed 42, grouping exact raw predictor profiles. Fresh fixed-specification candidates fit fold training rows and predict held-out training-population folds. Tasks 5/6 reuse the recorded outer evaluation positions. No hyperparameter optimization, new model specification or historical-champion refit was performed for this report. Evidence: [task4](pd_diagnostics.json), [task5](calibration_study.json), [task6](explainability_stability.json).

## 12. Discrimination Performance

[TRAINING_ONLY] Task 4 pooled out-of-fold results. Higher AUC, Gini and PR measures indicate stronger ranking; lower Brier and log loss indicate better probability quality. Average precision and trapezoidal PR-AUC use different definitions and are reported separately. Evidence: [task4](pd_diagnostics.json).

| Model | AUC | Gini | Average precision | PR-AUC | KS | Brier | Log loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.829387 | 0.658774 | 0.329161 | 0.328910 | 0.513891 | 0.053323 | 0.206154 |
| xgboost | 0.861182 | 0.722365 | 0.387057 | 0.386904 | 0.561994 | 0.049605 | 0.179870 |

## 13. Probability Calibration

[TRAINING_ONLY] Task 4 calibration-in-the-large (CITL) estimates an intercept with logit predictions as an offset; slope comes from a joint intercept/slope fit. Values near zero/one respectively are desirable, but do not establish segment, tail or future calibration. Reliability tables retain their bin-level qualifications. Evidence: [task4](pd_diagnostics.json).

| Model | Mean prediction | Observed event fraction | CITL | Slope |
| --- | --- | --- | --- | --- |
| logistic_regression | 0.066826 | 0.066813 | -0.000259 | 0.988270 |
| xgboost | 0.066791 | 0.066813 | 0.000433 | 1.001734 |

## 14. Nested Calibration Study

[TRAINING_ONLY] Task 5 separates outer evaluation from inner base fitting, calibration and method selection. The inner-selected row evaluates that selection procedure; fixed-method rows compare the same nested base fits. Reduced base-fit populations mean these values should not be treated as a direct repeat of Task 4. Evidence: [task5](calibration_study.json).

| Model | Method | AUC | Brier | Log loss | CITL | Slope |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | inner_selected | 0.831077 | 0.051587 | 0.191449 | 0.011591 | 0.938122 |
| logistic_regression | isotonic | 0.831077 | 0.051587 | 0.191449 | 0.011591 | 0.938122 |
| logistic_regression | raw | 0.832935 | 0.053077 | 0.206306 | -0.003473 | 0.947478 |
| logistic_regression | sigmoid | 0.833021 | 0.053321 | 0.205907 | 0.011426 | 1.040826 |
| xgboost | inner_selected | 0.860465 | 0.049778 | 0.180376 | -0.000628 | 0.992087 |
| xgboost | isotonic | 0.859319 | 0.049868 | 0.182308 | 0.002169 | 0.927989 |
| xgboost | raw | 0.860465 | 0.049778 | 0.180376 | -0.000628 | 0.992087 |
| xgboost | sigmoid | 0.859982 | 0.049797 | 0.180492 | 0.003263 | 0.975086 |

[TRAINING_ONLY] Calibrated minus raw paired differences and conditional percentile intervals. Negative Brier/log-loss differences favor calibration. AUC differences show any ranking cost. Evidence: [task5](calibration_study.json).

| Model | Method | Metric | Point difference | Interval |
| --- | --- | --- | --- | --- |
| logistic_regression | isotonic | roc_auc | -0.001858 | [-0.002910, -0.000978] |
| logistic_regression | isotonic | brier | -0.001490 | [-0.001837, -0.001180] |
| logistic_regression | isotonic | log_loss | -0.014857 | [-0.019835, -0.010702] |
| logistic_regression | sigmoid | roc_auc | 0.000086 | [-0.000270, 0.000485] |
| logistic_regression | sigmoid | brier | 0.000244 | [0.000156, 0.000349] |
| logistic_regression | sigmoid | log_loss | -0.000399 | [-0.001164, 0.000206] |
| xgboost | isotonic | roc_auc | -0.001147 | [-0.001955, -0.000372] |
| xgboost | isotonic | brier | 0.000090 | [-0.000022, 0.000202] |
| xgboost | isotonic | log_loss | 0.001931 | [0.000967, 0.003083] |
| xgboost | sigmoid | roc_auc | -0.000484 | [-0.000777, -0.000221] |
| xgboost | sigmoid | brier | 0.000019 | [-0.000013, 0.000052] |
| xgboost | sigmoid | log_loss | 0.000116 | [0.000031, 0.000223] |

[GOVERNANCE] The prespecified Brier gain margin was 0.000100; the paired evidence must also pass the log-loss guard. Logistic ISOTONIC improves probability quality with a modest ranking cost. Logistic SIGMOID slightly helps log loss while worsening Brier, so it is not an improvement on every measure. XGBoost remains RAW because calibrated methods do not meet both requirements. These are post-study research recommendations, without artifact promotion. Evidence: [task5](calibration_study.json).

## 15. Threshold Diagnostics

[TRAINING_ONLY] The predeclared Task 4 diagnostic grid illustrates precision/recall trade-offs. Predicted-positive fraction is a classification quantity, not a decline rate. Fold-training prevalence thresholds are separately retained in Task 4 evidence. No optimized pooled-OOF cutoff, lending profitability or operational approval band is established. Evidence: [task4](pd_diagnostics.json).

| Model | Threshold | Precision | Recall | Specificity | Predicted-positive fraction | Confusion matrix TN/FP/FN/TP |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.030000 | 0.089474 | 0.955472 | 0.303848 | 0.713478 | 19157/43891/201/4313 |
| logistic_regression | 0.050000 | 0.147379 | 0.817900 | 0.661226 | 0.370785 | 41689/21359/822/3692 |
| logistic_regression | 0.100000 | 0.296986 | 0.547851 | 0.907150 | 0.123250 | 57194/5854/2041/2473 |
| logistic_regression | 0.200000 | 0.454301 | 0.336952 | 0.971022 | 0.049554 | 61221/1827/2993/1521 |
| logistic_regression | 0.500000 | 0.532721 | 0.102791 | 0.993545 | 0.012892 | 62641/407/4050/464 |
| xgboost | 0.030000 | 0.139279 | 0.899202 | 0.602144 | 0.431352 | 37964/25084/455/4059 |
| xgboost | 0.050000 | 0.177324 | 0.828755 | 0.724718 | 0.312261 | 45692/17356/773/3741 |
| xgboost | 0.100000 | 0.272139 | 0.661719 | 0.873287 | 0.162458 | 55059/7989/1527/2987 |
| xgboost | 0.200000 | 0.393568 | 0.490696 | 0.945867 | 0.083301 | 59635/3413/2299/2215 |
| xgboost | 0.500000 | 0.582709 | 0.189632 | 0.990277 | 0.021743 | 62435/613/3658/856 |

## 16. Statistical Uncertainty

[TRAINING_ONLY] Task 4 intervals use confidence level 0.950000 and 200 valid resamples of exact-predictor group. They are percentile, conditional on fixed cross-fitted predictions. Shared training folds; no refitting uncertainty or unidentified borrower dependence. Evidence: [task4](pd_diagnostics.json).

| Model | Metric | Conditional interval |
| --- | --- | --- |
| logistic_regression | average_precision | [0.315526, 0.344829] |
| logistic_regression | brier | [0.052073, 0.054720] |
| logistic_regression | gini | [0.645546, 0.673275] |
| logistic_regression | log_loss | [0.200019, 0.212353] |
| logistic_regression | roc_auc | [0.822773, 0.836637] |
| xgboost | average_precision | [0.371769, 0.403661] |
| xgboost | brier | [0.048385, 0.050752] |
| xgboost | gini | [0.711991, 0.734971] |
| xgboost | log_loss | [0.175922, 0.183630] |
| xgboost | roc_auc | [0.855996, 0.867486] |

[TRAINING_ONLY] Paired Task 4 differences: xgboost minus logistic_regression.

| Metric | Paired conditional interval |
| --- | --- |
| average_precision | [0.049048, 0.067599] |
| brier | [-0.004259, -0.003264] |
| gini | [0.056796, 0.071280] |
| log_loss | [-0.031713, -0.021896] |
| roc_auc | [0.028398, 0.035640] |

Task 5 intervals are likewise conditional on fixed nested OOF predictions. Neither captures full refitting/selection uncertainty, unidentified borrower dependence, economic regime changes or external transportability. Reliability-bin Wilson intervals are approximate row-binomial summaries, not borrower-cluster-robust intervals. No new uncertainty calculation was performed. Evidence: [task4](pd_diagnostics.json), [task5](calibration_study.json).

## 17. Explainability

[TRAINING_ONLY] Leading original logistic predictors and their associations. beta is a standardized coefficient per one training SD of a capped/imputed/transformed input, conditional on other columns. exp(mean beta) summarizes fold coefficients; it is neither a pooled fitted odds ratio nor mean fold odds ratio. Log1p inputs do not imply raw currency/count-unit effects. Indicator 0-to-1 contrasts require the transformed-unit coefficient. These raw-model coefficients do not explain isotonic-calibrated odds. Evidence: [task6](explainability_stability.json).

| Predictor | Mean beta | Direction | exp(mean beta) | Sign flip |
| --- | --- | --- | --- | --- |
| 30-59-day past-due count | 0.438442 | positive | 1.550289 | False |
| 90+-day past-due count | 0.399121 | positive | 1.490514 | False |
| Age | -0.333461 | negative | 0.716440 | False |
| Unsecured utilization | 0.229823 | positive | 1.258377 | False |
| 60-89-day past-due count | 0.135629 | positive | 1.145257 | False |

[TRAINING_ONLY] Exact native TreeSHAP contributions are in raw margin/log-odds, using tree-path-dependent training cover. All evaluation rows receive aligned feature attribution and additivity checks. Mean absolute SHAP describes model contribution magnitude, not causation or a universal signed effect. Evidence: [task6](explainability_stability.json).

| Predictor | Pooled mean absolute SHAP | Rank |
| --- | --- | --- |
| Unsecured utilization | 0.753197 | 1.000000 |
| 30-59-day past-due count | 0.355633 | 2.000000 |
| 90+-day past-due count | 0.312064 | 3.000000 |
| Age | 0.219265 | 4.000000 |
| 60-89-day past-due count | 0.160930 | 5.000000 |

Both families share the same leading original predictor families. Utilization and open credit lines receive different ranks; coefficient and SHAP units are not directly comparable. Correlated inputs can redistribute attribution. Evidence: [task6](explainability_stability.json).

Observed-context dependence diagnostics show increasing utilization contributions with a zero-input exception, delinquency-count jumps and flattening, and a curved age relationship. Sparse tails and anomalous 96/98 counts limit interpretation. Limited interactions are exploratory; no ablation establishes the causal source of a performance advantage. These findings are consistent with nonlinear benefit but cannot rule out unidentified leakage. Evidence: [task6](explainability_stability.json).

## 18. Explanation Stability

[TRAINING_ONLY] Mean pairwise fold Spearman agreement for original XGB features is 0.995152. Overlapping fold fits make this descriptive rank agreement, not proof of temporal stability, stable attribution sizes or fairness. Evidence: [task6](explainability_stability.json).

| XGB predictor | Ranks by fold |
| --- | --- |
| Unsecured utilization | 1, 1, 1, 1, 1 |
| 30-59-day past-due count | 2, 2, 2, 2, 2 |
| 90+-day past-due count | 3, 3, 3, 3, 3 |
| Age | 4, 4, 4, 4, 4 |
| 60-89-day past-due count | 6, 5, 5, 5, 5 |
| Open credit lines/loans | 5, 6, 6, 6, 6 |
| Monthly income | 7, 7, 7, 7, 7 |
| Debt ratio | 8, 8, 8, 8, 8 |
| Real estate loans/lines | 9, 9, 9, 9, 9 |
| Dependents | 10, 10, 10, 10, 10 |
| Monthly income missing | 11, 11, 11, 11, 11 |
| Dependents missing | 12, 12, 12, 12, 12 |

[TRAINING_ONLY] Logistic direction changes occurred in: Monthly income, Open credit lines/loans, Real estate loans/lines, Dependents missing. Their weak mean associations should not be assigned a robust direction. Debt ratio and income missingness retain signs while showing magnitude variation. Evidence: [task6](explainability_stability.json).

## 19. Champion/Challenger Assessment

This qualitative decision matrix avoids arbitrary composite scoring. Performance entries are TRAINING_ONLY; interpretability, complexity and governance burden are reasoned assessments of the documented specifications. Evidence: [task4](pd_diagnostics.json), [task5](calibration_study.json), [task6](explainability_stability.json).

| Dimension | Logistic Regression | Raw XGBoost |
| --- | --- | --- |
| Discrimination | AUC 0.829387 | AUC 0.861182 |
| Probability quality | Brier 0.053323 | Brier 0.049605 |
| Calibration | ISOTONIC research recommendation | RAW retained in nested study |
| Interpretability | Conditional transformed-input coefficients | TreeSHAP plus dependence diagnostics |
| Explanation stability | Strong leading signs; weaker sign flips | High descriptive fold-rank agreement |
| Nonlinear modeling | Additive on specified transformed inputs | Thresholds and interactions |
| Complexity | Simpler functional form | More complex tree ensemble |
| Governance burden | Transforms/calibration require review | Transforms, attribution and nonlinear behavior require review |

**Raw XGBoost is the development champion. Logistic Regression remains the interpretable challenger.** This decision is confined to available development evidence; it does not replace the frozen historical selected model or establish deployment approval.

---

## 20. Historical Locked-Holdout Evidence

> These metrics are historical retained evidence from an already-consumed holdout. They were not regenerated during the current validation program.

| Category | Historical model/method | AUC | Brier | Log loss |
| --- | --- | --- | --- | --- |
| HISTORICAL_LOCKED_HOLDOUT | XGBoost / historically selected sigmoid | 0.868152 | 0.048545 | 0.176030 |

These are frozen historical results, not the raw-XGBoost development recommendation evaluated on new data. No prediction file, raw holdout record or model bundle was opened for this report. No new holdout metric or confidence interval was calculated. Evidence: [historical](../phase5_validation_summary.json), [ledger](../holdout_registry.json).

---

## 21. Model Limitations

- No usable dates: true out-of-time validation and temporal calibration stability unavailable.
- Unknown borrower identity: exact-profile groups do not establish borrower independence.
- Two-year serious delinquency is not regulatory default, 12-month PD or lifetime PD.
- Pristine-download equivalence, source selection and measurement timing remain unverified.
- Training records were previously explored; repeated research is not independent validation.
- Sparse/anomalous tails and unknown missingness mechanisms constrain interpretation.
- Fairness, lending profitability, external transportability and deployment readiness unproven.

Evidence: [target](../../docs/TARGET_DEFINITION.md), [suitability](../data/DATASET_SUITABILITY.md), [leakage](../data/LEAKAGE_REVIEW.md), [task4](pd_diagnostics.json), [task5](calibration_study.json), [task6](explainability_stability.json).

## 22. Model Risk Assessment

The following HIGH/MEDIUM/LOW/INFORMATIONAL labels are a local qualitative research-review framework, not a bank or regulator's classification. HIGH means unresolved evidence prevents a stated use claim; MEDIUM means interpretation or robustness needs further review; LOW means a specific demonstrated control limits a risk; INFORMATIONAL describes a documented boundary, without certifying residual safety.

| Concern | Level | Evidence/reason | Required resolution |
| --- | --- | --- | --- |
| Temporal generalization | HIGH | No usable observation/event dates | Dated independent out-of-time evaluation |
| Borrower independence | HIGH | Row/profile keys are not borrower IDs | Stable borrower/facility lineage |
| Regulatory target mismatch | HIGH | Two-year serious delinquency only | Separate dated target/default contract before regulatory use |
| Provenance and selection | HIGH | Pristine file and sampling unverified | Authenticated source and acceptance/measurement lineage |
| Historical exploration | MEDIUM | Training population previously explored | Independent future/external validation |
| Sparse/anomalous tails | MEDIUM | Unknown coded values and small support | Supplier definitions and justified robustness assessment |
| Calibration transportability | HIGH | No temporal calibration evidence | Mature dated outcome cohorts |
| Fairness | HIGH | Not assessed; explanations are insufficient | Appropriate population/group data and fairness review |
| Consumed-data reuse mechanism | LOW | Anchored fail-closed registry | Maintain ledger and independent evidence review |
| Research-only scope | INFORMATIONAL | No operational approval claimed | Independent governance before a deployment decision |

Evidence: [target](../../docs/TARGET_DEFINITION.md), [suitability](../data/DATASET_SUITABILITY.md), [leakage](../data/LEAKAGE_REVIEW.md), [registry](../../docs/HOLDOUT_REGISTRY.md), [task4](pd_diagnostics.json), [task5](calibration_study.json), [task6](explainability_stability.json).

## 23. Monitoring Recommendations

Proposed practice for a future real deployment; no infrastructure or alert threshold is implemented or endorsed by this report. A risk owner must approve populations, baselines, maturity rules, review cadence and actions before deployment. Evidence: [suitability](../data/DATASET_SUITABILITY.md), [target](../../docs/TARGET_DEFINITION.md).

| Monitoring layer | Measures | Interpretation/action |
| --- | --- | --- |
| Data | Schema, missingness, ranges, category/value validity | Investigate source/pipeline changes and affected cohorts |
| Population | Feature and score distributions; PSI or appropriate drift measures | Investigate sustained/material change with sample-size context |
| Mature performance | AUC/Gini, PR-AUC/average precision, Brier, log loss | Use completed target windows; do not relabel immature outcomes |
| Calibration | Observed vs predicted rates, CITL, slope, reliability | Investigate stable ranking with systematic probability error |
| Explanation | Feature-importance drift and rank stability | Review shifts with feature correlation and cohort context |
| Governance | Lineage breaks, sustained deterioration, new population or target | Investigate; consider approved recalibration or redevelopment |

No universal PSI or regulatory trigger is invented. Drift alone does not prove performance failure. Mature outcomes are required for performance/calibration claims, and observed accepted-loan performance cannot identify rejected-applicant outcomes without additional evidence. Recalibration or redevelopment requires documented review and fresh independent validation.

## 24. Governance and Reproducibility

[GOVERNANCE] The generator reads an explicit allowlist of committed aggregates, documentation, configuration and source bytes. It checks cross-artifact targets, provenance, historical metrics, fold identities, calibration comparisons, feature rankings, code hashes and the consumed registry anchor. Missing, malformed or conflicting evidence causes failure; it is not silently reconciled.

[GOVERNANCE] Repository package version: `0.12.0`; generation-time Git baseline: `b70b08ff45cec260d279a51146ece9b7202f0287`. This commit identifies the checkout used to assemble evidence, not an independent model validation sign-off. The manifest records generation time, source identities, generator hashes, commands and per-claim categories. Evidence: [registry](../../docs/HOLDOUT_REGISTRY.md), [experiment](../phase4_experiment.json), [historical](../phase5_validation_summary.json).

[GOVERNANCE] Task 1 bounded container-startup retry is documented by [smoke-test code](../../scripts/smoke_container.py) and [regression tests](../../tests/test_container_smoke.py). CI configuration demonstrates intended checks, not a newly successful hosted run. This generator does not execute pytest, CI or Docker; their current results must be reported separately by the verification run.

[GOVERNANCE] Retained experiment environments (not regenerated):

| Package | Task 4 | Task 5 | Task 6 |
| --- | --- | --- | --- |
| numpy | 2.4.6 | 2.4.6 | 2.4.6 |
| pandas | 3.0.6 | 3.0.6 | 3.0.6 |
| scikit-learn | 1.9.1 | 1.9.1 | 1.9.1 |
| scipy | 1.17.1 | 1.17.1 | 1.17.1 |
| xgboost | 3.2.0 | 3.2.0 | 3.2.0 |

[GOVERNANCE] Tasks 4/6 model/fold seed: 42; Task 4 bootstrap seed: 43; Task 5 study seed: 42. Inner and resampling seeds are retained in the source JSON. The lockfile records reproducible dependency resolution.

## Appendix A - Metrics

| Metric | Meaning and boundary |
| --- | --- |
| AUC / Gini | Ranking discrimination; Gini = 2*AUC - 1; not probability calibration |
| Average precision / PR-AUC | Non-interpolated summary vs trapezoidal curve area; not interchangeable |
| KS | Maximum absolute TPR-FPR; directionless discrimination diagnostic |
| Brier / log loss | Probability errors; lower is better; population-dependent |
| CITL / slope | Offset intercept near zero; joint-fit slope near one are desirable |
| Precision / recall | Threshold-dependent classification behavior, not lending profit |
| SHAP magnitude / rank agreement | Model contribution and descriptive fold agreement, not causality |

Evidence: [task4](pd_diagnostics.json), [task5](calibration_study.json), [task6](explainability_stability.json).

## Appendix B - Artifact Manifest

[Machine-readable manifest](model_validation_manifest.json) records source evidence categories, file SHA-256 and LF-normalized SHA-256, generator hashes, frozen artifact digests copied from committed historical manifests, quantitative results and limitations. Non-frozen text hash checks allow only Git LF/CRLF conversion; frozen model source remains byte-exact. File hashes establish consistency with recorded evidence, not independent authentication or pristine-source equivalence. Local model bundles and historical raw data/prediction files are outside the generator's read allowlist.

## Appendix C - Reproduction Commands

Generate the report from committed evidence without the original dataset or ignored model artifacts:

```console
uv sync --locked
uv run --no-sync python scripts/generate_model_validation_report.py
uv run --no-sync pytest tests/test_model_validation_report.py -q
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts api
uv run --no-sync ruff format --check src tests scripts api
uv run --no-sync python scripts/check_governance.py
uv run --no-sync python scripts/check_repository.py
```

Run the generator twice on the same checkout to compare Markdown bytes and manifest content excluding generated_at. For fully identical metadata, supply the same timezone-aware ISO timestamp via --generated-at. A changed Git baseline or evidence/code identity appropriately changes metadata. Historic source artifacts need not be present, and no training command is needed.

## 25. Final Development Conclusion

Track A establishes a reproducible research framework for developing and validating models of two-year serious delinquency. Within available TRAINING_ONLY evidence, raw XGBoost provides stronger discrimination and probability quality than the Logistic Regression benchmark while exhibiting consistent feature-attribution rankings across grouped folds. Logistic Regression remains the interpretable challenger, with an ISOTONIC research recommendation from nested calibration. Historical artifacts remain frozen. The project does not establish regulatory PD validity, temporal generalization, lending profitability or deployment readiness. Evidence: [task4](pd_diagnostics.json), [task5](calibration_study.json), [task6](explainability_stability.json), [suitability](../data/DATASET_SUITABILITY.md).
