# Phase 4: logistic baseline and XGBoost challenger

Date: 2026-10-04 (America/Sao_Paulo). Branch: feature/risk-modeling-lab.
Package 0.4.0. Local experiment: artifacts/phase4-origination-001.

## Implementation and measured scope

Implemented train-fitted origination transforms, complete logistic/XGBoost
pipelines, grouped stratified partitions, an experiment runner, persisted
preprocessing/model artifacts, a training CLI and initial ranking/probability
metrics. Configured candidates were trained once with fixed parameters on the
inherited cross-sectional source. No final-test or calibration predictions were
made, no champion was promoted, and no calibration claim is made.

The source SHA-256 is unchanged from Phase 1. All 150,000 rows are retained.
Partition counts: train 67,562 (4,514 bad), development 22,483 (1,504 bad),
calibration 29,964 (2,005 bad), final test 29,991 (2,003 bad). Predictor-duplicate
groups cannot cross partitions. Actual fractions differ slightly from requested
group fractions. This is not out-of-time validation or proof of borrower-level
independence. The original data has no reliable dates/customer identifiers.

## Development-only results (raw probabilities)

| Candidate | ROC-AUC | Gini | Two-sided KS | Average precision | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant train prevalence | 0.5000 | 0.0000 | 0.0000 | 0.0669 | 0.06242 | 0.24553 |
| Logistic regression | 0.8422 | 0.6844 | 0.5518 | 0.3494 | 0.05266 | 0.19909 |
| XGBoost | 0.8687 | 0.7375 | 0.5873 | 0.4046 | 0.04859 | 0.17605 |

At the preset diagnostic threshold 0.10, logistic precision/recall are
0.3074/0.5691; XGBoost precision/recall are 0.2827/0.6855. The challenger ranks
better and has lower raw probability loss here, but higher recall comes with
lower precision. Do not infer a universal superiority or a business approval
threshold. Observed development event rate is 0.06689; mean probabilities are
0.06759 logistic and 0.06803 XGBoost. Aggregate agreement does not prove
calibration at individual risk levels or within segments.

The full measured experiment metadata and exact metric definitions are committed
in phase4_experiment.json. `pr_auc` is trapezoidal PR area; primary comparison
uses average precision because constant scores expose interpolation artifacts.
The target remains SeriousDlqin2yrs, not newly constructed 12-month default.
These results are not portfolio performance, production PDs or regulatory evidence.

## Evidence and limitations

The phase's test suite verifies train-only preprocessing, exact-predictor grouping
including opposite-label duplicates, deterministic splits, changed seeds,
missing/all-missing feature handling, column ordering, scoring-field allowlists,
full pipeline serialization, reproducible candidate metrics and holdout access.
Hand-computed tests validate Gini, KS, Brier, ranking ties/reversal, single-class
undefined metrics and invalid inputs. Strong ranking versus probability loss is
explicitly demonstrated. Calibration remains a separate implementation phase.

Data selection/provenance, unresolved sentinel values, simple clipping assumptions,
unknown borrower independence, unknown temporal stability, sample selection and
fairness remain limitations. Development is used for candidate comparison; only
an untouched final test can support subsequent independent comparison. Logistic
coefficients and tree importances do not establish causation or fairness.

See docs/pd_modeling.md for architecture, assumptions and reproduction commands.
Complete test/lint/preservation, locked install, packaging and isolated wheel
checks before commit. Large CSVs, serialized pipelines and per-row prediction
artifacts remain ignored. No remote push or V1 source change is part of this phase.

## Completed verification

All 74 tests passed on Python 3.11.4, with scikit-learn 1.9.1 and XGBoost 3.2.0.
Ruff lint/format, Phase 1 preservation and locked sync passed. Saved real-source
pipelines reproduced development predictions; artifact and code hashes matched
the recorded run. No real calibration or final-test observations were scored.
The wheel and source distribution built and exclude local data, per-row exports,
models and environments. An isolated wheel import checked all 18 submodules
and fitted/scored both candidates on a tiny synthetic fixture; external config
validation succeeded. Generated artifact ignore rules and the phase diff were
reviewed before commit. Python 3.12-3.14 remain declared but unverified.
