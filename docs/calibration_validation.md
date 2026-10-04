# Calibration and validation

Phase 5 evaluates the frozen Phase 4 origination candidates. Run from the repository root:

```console
uv sync --locked
uv run --locked python scripts/validate.py --csv cs-training.csv --run-dir artifacts/phase4-origination-001 --config configs/validation.yaml --output-dir artifacts/phase5-validation-001
```

Use a new output directory for each reproduction. The command loads joblib files;
use only trusted experiments produced locally by this lab. Manifest checksums
identify accidental changes, not malicious pickle content.

## Partition discipline

Base models and preprocessing fit only the training partition in Phase 4.
Before loading them, validation checks source and artifact hashes, base-model
source hashes, dependency versions, predictor columns and recomputed grouped
partition assignments. Neither base estimator nor its preprocessing is refitted.

Each probability calibrator fits only the calibration partition. Development
log loss selects one method per model; Brier score breaks ties, then the declared
method order. The preferred candidate uses development performance too.
`selection.json` and the selected model wrappers are saved before final-test
predictions. Reliability-bin edges are quantiles of calibration predictions,
not quantiles learned from the final test.

A persistent `test_consumption.json` in the training run marks test access
before scoring. Reproduction with the same source, models, methods, numerical
clipping, selected choices and calibration code is permitted and marked as
repeat access. Different choices are rejected; subsequent selection needs a
new independent holdout. This is a local workflow guard, not a security boundary.
Do not remove the marker or tune models using the reported test results.

## Calibration methods

Raw probabilities are the identity with the same numerical clipping used for
all methods (epsilon 1e-6). Sigmoid calibration fits
`sigmoid(a * logit(raw_probability) + b)` by unweighted binary log loss,
with positive slope to preserve ranking apart from clipping ties. It is a
Platt-style log-odds mapping; it does not refit the base classifier.
Isotonic regression learns a nondecreasing piecewise interpolation, clips outside
the fitted domain, and may introduce ties that change AUC and average precision.
The original inherited two-year delinquency label remains the target for all
comparisons. It is not relabeled as the synthetic portfolio's 12-month default.

Strong ranking does not imply accurate probabilities: a monotone transformation
can leave AUC unchanged while altering Brier score, log loss and expected events.
The independent synthetic miscalibration test verifies this distinction; the
measured report shows whether calibration helps on the actual source.

## Evidence and interpretation

The output contains aggregate `validation.json`, the pre-test `selection.json`,
selected full-pipeline joblib wrappers, ignored row-level test predictions and
reliability/ROC/precision-recall figures. `artifacts_sha256` covers generated
artifacts; the validation JSON does not include its own hash.

Metrics include ROC AUC, Gini, two-sided KS, trapezoidal PR AUC, average precision,
Brier, log loss and precision/recall at the Phase 4 diagnostic threshold (0.1).
Average precision is the primary PR summary; trapezoidal area can be misleading
for coarse or constant scores. Threshold 0.1 is not a validated lending policy.

Reliability diagnostics include calibration-bin event rates, approximate Wilson
intervals, ECE and observed/expected events. ECE depends on the bins and is not a
universal model acceptance criterion. Wilson bars assume independent rows and
are approximate, not borrower-cluster robust. Empty bins remain explicit.

Paired percentile bootstrap intervals resample exact-predictor groups, retaining
all rows in each selected group and using the same draws for both selected
models. This matches the available split unit. Source indices are not borrower
IDs; residual dependence may remain. Intervals are conditional on frozen fitted
models and exclude training and calibration-estimation uncertainty. Replicates
without both outcomes are counted and skipped. The default is 200 draws, seed 43,
95% confidence; more draws improve Monte Carlo precision without changing choices.

Age, income missingness and utilization segments use prespecified boundaries.
Low support flags mean fewer than 100 rows or fewer than 20 outcomes in either
class. One-class ranking metrics are unavailable, not zero. These checks do not
establish fairness or legal eligibility. Missingness rates are reported for every
partition; drift/PSI monitoring belongs to Phase 10.

## Dated backtesting and limits

`backtest_predictions(frame, as_of)` requires genuine observation dates,
performance-window ends, probabilities and binary labels. It reports metrics by
observation month only where outcomes are known and the full performance window
has matured by the cutoff, including the exact end-date boundary. Missing and
immature outcomes are excluded. It does not train models or establish that scores
were generated prospectively; callers must provide that provenance.

The inherited origination CSV has no observation dates, so this experiment
provides grouped cross-sectional development/final-test comparisons. It cannot
establish out-of-time performance, temporal stability, contractual-default PD
accuracy, or behavior on declined applicants. The dated helper is tested with
explicitly synthetic fixtures; no dates are fabricated for the source data.

Measured evidence is in [the validation report](../reports/validation_report.md).
