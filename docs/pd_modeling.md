# Phase 4: origination risk candidates

## Business interpretation and scope

This phase implements an interpretable logistic benchmark and an XGBoost
challenger for the inherited SeriousDlqin2yrs label. That label is a source
serious-delinquency outcome whose name describes a two-year horizon. It is not
our synthetic next-12-month target, and its equivalence to a contractual default
or an applicant-level production PD has not been established. Source-population
selection and the dataset/notebook provenance mismatch remain documented risks.
No longitudinal synthetic observations are used to train these candidates.

Both outputs are raw, uncalibrated probabilities of the inherited label. No
underwriting recommendation, limit, ECL or regulatory compliance is implied.
Calibration, independent final-test validation and governance follow later.

## Grouped split and holdout discipline

The pipeline retains all 150,000 valid source rows. Exact predictor duplicates
are assigned a shared 64-bit feature hash using pandas.hash_pandas_object with
index=False. Target labels and source indices are never inputs to this key.
Mixed-outcome duplicates stay together. Source_row_id is retained for lineage,
not used as a predictor or assumed to be a borrower/customer identifier.

Groups are stratified by whether they contain any positive label, then allocated
by seed-42 sequential train_test_split calls: final-test groups first (20%),
calibration groups next (20% of the original groups), development groups next
(15% of the original groups), and training groups last (45%). Fractions apply to
groups, not rows; actual row counts and event rates are recorded. The split
requires both classes in all partitions and fails on insufficient group coverage.
No rows or groups are discarded based on outcomes.

This prevents exact-duplicate leakage but cannot establish real borrower
independence or provide out-of-time validation without reliable IDs and dates.
Hash collisions would conservatively keep unrelated records together; locked
pandas versions and recorded assignments support reproducibility.

Training is the only partition used for capping, imputation, scaling and model
fit. Development is the only scored partition in Phase 4. Calibration and final
test are reserved and exported as split assignments without predictions. Their
labels are used for stratification and count diagnostics, not model fitting or
candidate selection. The development set is reusable for development, so its
performance is preliminary; it cannot become the final independent evaluation.

## Preprocessing and candidates

OriginationFeatures requires exactly the ten declared predictors, orders them
consistently and rejects target, identifier or future-outcome fields at scoring.
It validates the input contract and learns lower/upper quantiles on training only
(default 0.001/0.999). It caps rather than drops records. Missing values remain
missing through this step. A feature entirely missing in training has zero bounds
and cannot contribute a newly observed value at scoring without retraining.

The logistic pipeline applies log1p to non-age numeric predictors after capping;
age remains linear. Training-fitted median imputation adds missing indicators,
retains entirely missing features, and is followed by StandardScaler and L2
logistic regression (C=1, lbfgs, max_iter=3000). Convergence warnings fail the run
rather than silently accepting an unconverged candidate.

The XGBoost pipeline uses the same training-fitted caps and median/missingness
preprocessing without logs or scaling. Initial fixed settings are hist trees,
250 estimators, depth 3, learning rate 0.05, row/column subsampling 0.8 and L2
regularization 5, with two CPU threads and a fixed seed. No hyperparameter search
or early stopping uses the reserved calibration or test sets.

Neither model uses resampling, class weights or scale_pos_weight above 1. This
keeps the observed training prevalence rather than deliberately changing it for
classification objectives. It does not guarantee calibration; both still need
independent probability calibration and validation. Learned caps, transformations
and candidate settings are experimental assumptions, not approved credit policy.

Logistic coefficients are exported per standardized transformed unit and missing
indicator, not raw currency units. The intercept is present in the persisted
pipeline. XGBoost gain importance is a training-derived heuristic. Neither is
causal evidence, a regulatory scorecard or a fairness assessment; WOE/IV and
further interpretability remain later work.

## Metric definitions

ROC-AUC, Gini=2*AUC-1, and two-sided KS measure ranking/separation. KS is maximum
absolute TPR-FPR and remains high for reversed ranking, so consider AUC/Gini
alongside it. Brier and binary log loss assess probability error; average predicted
risk versus observed event rate is only an aggregate diagnostic, not a complete
calibration assessment. Reliability curves and subgroup validation follow in
Phase 5. Undefined one-class ranking metrics return null, not invented numbers.

`pr_auc` is trapezoidal area under the precision-recall curve. `average_precision`
is the non-interpolated PR summary used as the primary PR comparison in this
experiment. Trapezoidal interpolation is especially misleading for a constant
score: the constant benchmark has PR area about 0.5334 while its average precision
is only the 0.0669 event rate. Reporting both definitions avoids confusing that
interpolation artifact with useful ranking.

Precision/recall use a preset classification threshold of 0.10, inclusive for
positive classification. This is diagnostic only; it is not an optimized or
validated underwriting threshold. Changing thresholds trades precision/recall.
A constant training-prevalence benchmark on development provides a probability
loss comparison without estimating its probability from development outcomes.

## Reproduce the experiment

From the checkout root:

```console
uv sync --locked
uv run --locked credit-risk-lab train --csv cs-training.csv --config configs/model.yaml --output-dir artifacts/phase4-origination-001
uv run --locked python scripts/train.py --csv cs-training.csv --output-dir artifacts/another-run
uv run --locked pytest
uv run --locked ruff check src tests
```

A local authorized CSV is required for this real-source experiment. Tests use
clearly synthetic applicant fixtures and do not require Kaggle access. Training
refuses a nonempty output directory; choose a new run path to reproduce. A run
persists full pipelines, split assignments, development probabilities, coefficient/
importance CSVs and experiment.json under the ignored artifacts directory.
The experiment metadata records input hash, code fingerprints, configuration,
seed, Python/dependency versions, partitions, development metrics and artifact
hashes. Own newly produced joblib artifacts are verified in tests; the inherited
pickle is never loaded. Serialized models require trusted inputs and compatible
package versions; this phase does not claim cross-version artifact portability.

OneDrive left stale empty dist-info directories during package upgrades. The
local new .venv metadata was repaired with scoped removal and locked reinstall;
V1 venv was untouched. Import now fails if installed version metadata is missing,
and CLI tests check its presence. If OneDrive repeatedly interferes, prefer an
unsynced uv environment via UV_PROJECT_ENVIRONMENT while keeping source here.

Read reports/phase4_modeling.md for measured results and limitations. Final-test
metrics, calibration comparisons, bootstrap uncertainty, segments and model
promotion are intentionally deferred to Phase 5.
