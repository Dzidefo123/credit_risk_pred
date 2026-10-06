# Baseline model report

Audit date: 2026-10-05; current inspected commit `a099ed1e5ea9451e3cf6d2972bbd3ced89b34704`. This report separates the preserved V1 historical baseline from the retained modern PD experiment. It reports stored model results and newly verified integrity checks, not a fresh training run or new final-test evaluation.

## V1 historical baseline: results unverified

The original implementation is preserved in `credit_risk(1).ipynb`, `app.py`, `main.py`, templates and Git history at `811a2fef379975f9e67a8d61199259c2a11cb5d3`. The original README is archived in `docs/history/README_v1.md`.

| Item | Traceable historical evidence |
| --- | --- |
| Dataset | Notebook initial stored shape 125,113 rows and 11 columns after index removal; retained CSV instead has 150,000 rows and 12 columns |
| Outcome prevalence | Exact notebook training-population prevalence not reliably reconstructed; retained source has 6.684% label-1 rate |
| Features | Ten financial/demographic predictors, with target and index removed |
| Preprocessing | Deduplication, income/dependent imputation and row filtering before the train/test split; later standard scaling for the NN |
| Split | Random 80/20, random_state=42; no stratify argument, borrower grouping or temporal validation in cell 57 |
| Exact final train/test counts | Not certified: cleaned population and execution state differ from retained source; do not substitute 120,000/30,000 |
| XGBoost | `XGBClassifier(tree_method='exact')`; other parameters left to unpinned library defaults; initial fit/evaluation on same data, then refit on training split |
| Neural network | Dense 128/64 ReLU units, sigmoid output, Adam, binary cross-entropy, ten epochs, batch_size=128, validation_split=0.2; no explicit TensorFlow seed |
| Ensemble | Arithmetic mean of NN and XGBoost test probabilities, binary threshold 0.5 |
| Saved artifact | `joblib.dump(model, combined_model.pkl)` saves the XGBoost variable, not the NN/ensemble/scaler; static artifact inspection corroborates its class |
| Initial stored XGBoost results | Accuracy 0.9481, positive-class recall 0.29; in-sample, not independent validation |
| Later stored test results | NN and averaged ensemble each show accuracy 0.9350; these outputs were not reproduced or endorsed |
| Calibration/ranking evidence | No verified historical ROC-AUC, KS, PR-AUC, Brier, log loss or calibration study; do not fabricate them |

Reproduction is **not established**. The source population mismatch, pre-split imputation, outcome-dependent exclusions, mutable notebook state, Colab paths, missing NN seed and incompatible/incomplete old environment prevent a fair controlled replay. The bundled environment targets Python 3.8.2 and contains pandas 2.0.3 even though the notebook calls removed DataFrame.append; TensorFlow, seaborn and matplotlib metadata were absent. No original pickle was deserialized and no historical model was retrained for this audit. Historical serving bypasses the notebook transforms and describes an ensemble that was not persisted.

Preservation is intentional, not acceptance of the original methodology. These historical accuracy values must not be compared directly with modern holdout metrics: population, evaluation design, preprocessing and saved object differ.

## Modern frozen PD baseline/challenger

Retained source SHA-256: `9d863f88e2ef2190ed608c15c3dd362d1833d940e35f4f2bf22660e04b5f05cb`. Source rows 150,000; ten predictors; 10,026 inherited positive outcomes (6.684%). Target: `SeriousDlqin2yrs`, source two-year delinquency label, not a newly defined twelve-month or regulatory default target.

| Partition | Rows | Positive outcomes | Positive rate |
| --- | --- | --- | --- |
| train | 67,562 | 4,514 | 6.6813% |
| development | 22,483 | 1,504 | 6.6895% |
| calibration | 29,964 | 2,005 | 6.6914% |
| test | 29,991 | 2,003 | 6.6787% |

Seed 42; exact-predictor grouping and group-level stratification, with no group crossing partitions. Source row indices are excluded from predictors and do not identify borrowers. There are no dates for true out-of-time testing.

Preprocessing is inside fitted sklearn pipelines: training quantile caps 0.001/0.999; median imputation with missing indicators and retained empty features. Logistic also uses log1p except for age and StandardScaler. XGBoost does not apply that logistic log transform. No preprocessing is learned from the development, calibration or test partitions.

| Candidate | Declared parameters |
| --- | --- |
| Constant benchmark | Training prevalence 0.06681270536692223; evaluated on development only |
| Logistic | L2, C=1, solver=lbfgs, max_iter=3000, tol=1e-5, controlled seed |
| XGBoost | n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, n_jobs=2, tree_method=hist, objective=binary:logistic, eval_metric=logloss, scale_pos_weight=1 |
| Calibration | raw; sigmoid on clipped logits with positive slope (minimum 1e-8), L-BFGS-B; isotonic with out-of-bounds clipping. Fit on calibration, choose on development log loss/Brier tie-breaker |

There is no current NN ensemble, LightGBM run, model hyperparameter search or CV experiment. Legacy NN remains preserved. Champion/candidate and calibration methods were selected on development, not final-test AUC. Selected logistic uses isotonic; selected XGBoost uses sigmoid.

## Recorded comparison

The constant benchmark below is a **development** reference, not a final-test result. AP and trapezoidal PR-AUC are separate quantities.

| Development model | ROC-AUC | Gini | KS | PR-AUC (trapezoid) | AP | Brier | Log loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constant_train_prevalence | 0.500000 | 0.000000 | 0.000000 | 0.533447 | 0.066895 | 0.062420 | 0.245532 |
| logistic_regression | 0.842176 | 0.684353 | 0.551760 | 0.348922 | 0.349383 | 0.052661 | 0.199094 |
| xgboost | 0.868726 | 0.737453 | 0.587303 | 0.403949 | 0.404617 | 0.048595 | 0.176048 |

The tied constant-score PR curve has trapezoidal area 0.533447 due to endpoint interpolation; AP=0.066895 is the more useful no-ranking reference. Do not present its trapezoidal area as superior discrimination.

| Selected variant: recorded final holdout | ROC-AUC | Gini | KS | PR-AUC (trapezoid) | AP | Brier | Log loss | Precision @0.10 | Recall @0.10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression / isotonic | 0.834776 | 0.669551 | 0.512370 | 0.354090 | 0.345232 | 0.050967 | 0.188287 | 0.288547 | 0.581128 |
| xgboost / sigmoid | 0.868152 | 0.736304 | 0.577380 | 0.408330 | 0.408630 | 0.048545 | 0.176030 | 0.287788 | 0.672991 |

Threshold 0.10 is diagnostic classification only, not an underwriting rule. F1 and explicit confusion matrices are not currently recorded. The retained final test has already been consumed. The validation JSON field `fresh_test_access=true` describes the historical first evaluation, not permission for another independent evaluation today.

Raw XGBoost Brier=0.048537874864 and log loss=0.176073441532; selected sigmoid Brier=0.048544557231 and log loss=0.176029924503. Sigmoid slightly improves log loss while slightly worsening Brier. Selection preceded test access; no claim of universal or statistically significant calibration improvement follows from this tiny change.

The recorded 200 exact-predictor-group bootstrap replicates give selected XGBoost conditional 95% percentile intervals: AUC [0.859273, 0.875408], Brier [0.046743, 0.050395], log loss [0.170154, 0.182341]. These are conditional on frozen fitted models and exclude retraining uncertainty, temporal shifts and unidentified borrower dependence. Reliability/segment diagnostics and plots exist in the validation evidence; they are not fairness certification or lender approval.

## What this audit legitimately reproduced

- Source row/column counts, target prevalence, distributions, duplicates, missingness and exact-profile partition overlap were recomputed from the retained data/assignments.
- `verify_experiment` passed its source, artifact, code-byte, dependency-version and split-assignment checks without model deserialization or prediction.
- Every artifact listed in the Phase 5 validation manifest matched its retained SHA-256, including selected bundles, plots, selection and saved predictions.
- Existing tests passed: 261, with two upstream deprecation warnings; Ruff and governance/repository/preservation guards passed.
- An isolated artifact-free API returned 503 for health, score and decision. Hosted seven Python lanes passed; container smoke failed on an uncaught startup connection reset. This is not a completed container verification.

The final model numbers above were **read from hash-verified retained evidence**, not newly recomputed from original holdout predictions. No training, recalibration, original holdout scoring or champion reselection occurred. Fixture tests train small separate synthetic/test fixtures; they do not independently validate the original lending-like population.

## Lineage and limitations

Selected XGBoost bundle SHA-256: `17030fc4d0a5cf0e905f11933e97542b0afcccba789fdfcc481c0bc9347d7d75`.
Selected logistic bundle SHA-256: `4687a557b52a0242bc38bfb17c14afff0fefbf0e6b2bbba7a6038fa56ed135b3`.
Selection file SHA-256: `7bd1c00846f90aab0452a3ed5e128cba950ece2ad7bad0f8e9ee9c34cf237c12`.
Whole consumed-test marker SHA-256: `07379a470cf63ba5edda21a9d321639935033d460e6cc89282561b2d72fee0ee`. The marker/validation embedded selection digest is a different object and must not be confused with this whole-file digest.

Frozen training environment: Python 3.11.4, NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, XGBoost 3.2.0 and joblib 1.6.0; validation also records SciPy 1.17.1 and matplotlib 3.11.2. New supported environment lanes establish package tests, not frozen-model serialization portability. Original training Git identity was not recorded; the audit commit is not retroactive training provenance.

Remain research-only. Predictor timestamps, source sampling, borrower identity, outcome maturity/contractual definition and income units are unverified. Random grouped validation cannot establish future-vintage performance. No live lending decisions or regulatory approval are claimed. No fitted LGD/EAD, IFRS 9 compliance, policy effectiveness, NN champion or production monitoring is inferred from these results.

See the [repository audit](../../docs/REPOSITORY_AUDIT.md) and [data audit](../data/DATA_AUDIT.md) for ranked gaps and the approval-gated migration plan.
