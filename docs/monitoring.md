# Population monitoring

Freeze a reference once from the original development partition. Then supply a
comparison origination CSV with the same ten raw feature columns and optional
source-row provenance ID. Targets are not required or used. Extra fields are
rejected by the origination contract. The comparison command verifies the frozen
source/model artifacts but only scores comparison rows, without fitting.

```powershell
.venv\Scripts\python.exe -m credit_risk.cli freeze-monitor-reference --csv cs-training.csv --run-dir artifacts/phase4-origination-001 --validation-dir artifacts/phase5-validation-001 --config configs/monitoring.yaml --output-dir artifacts/phase10-reference-001
.venv\Scripts\python.exe scripts/make_monitoring_demo.py --csv cs-training.csv --reference-dir artifacts/phase10-reference-001 --output-dir data/raw/phase10-monitoring-demo
.venv\Scripts\python.exe -m credit_risk.cli monitor --source-csv cs-training.csv --run-dir artifacts/phase4-origination-001 --validation-dir artifacts/phase5-validation-001 --reference-dir artifacts/phase10-reference-001 --current-csv data/raw/phase10-monitoring-demo/unchanged.csv --current-manifest data/raw/phase10-monitoring-demo/demo_manifest.json --output-dir artifacts/phase10-control-002
.venv\Scripts\python.exe -m credit_risk.cli monitor --source-csv cs-training.csv --run-dir artifacts/phase4-origination-001 --validation-dir artifacts/phase5-validation-001 --reference-dir artifacts/phase10-reference-001 --current-csv data/raw/phase10-monitoring-demo/perturbed.csv --current-manifest data/raw/phase10-monitoring-demo/demo_manifest.json --output-dir artifacts/phase10-drift-002
```

Use fresh output directories. For an external comparison, omit the demo manifest
or supply its checksum/provenance manifest. Without metadata, the population is
marked unverified external comparison; observation dates remain unavailable.
The source has no dated cohorts, so neither the default reference nor the
controlled demo supports an out-of-time drift claim. Comparison row identifiers
matching reserved final-test provenance IDs are forbidden, even if their features
were altered. Predictor-group matches are also forbidden without IDs. A new
external source must use unambiguous provenance IDs or omit that optional field;
these IDs never identify borrowers. This conservative guard may refuse genuinely
new applicants whose predictors exactly match reserved historical profiles.

`fit_reference(features, pd, config)` and `compare_reference(...)` are pure
measurement APIs; the CLI adds origination contracts, model integrity and holdout
protection. Reference quantiles use finite raw values only. Internal cut points
are unique; bins are (-infinity, cut1], (cut1, cut2], ..., (last, infinity).
Constant references instead have below/equal/above categories. An all-missing
reference reserves a nonmissing category, so newly populated values cannot vanish
from the calculation. Missing values always occupy the last bucket. Infinite
values and empty populations are rejected. No feature imputation precedes drift.

For reference counts r and comparison counts c in K common bins:

```
p_k = (r_k / sum(r) + epsilon) / (1 + K * epsilon)
q_k = (c_k / sum(c) + epsilon) / (1 + K * epsilon)
PSI = sum((q_k - p_k) * log(q_k / p_k))
```

Proportion smoothing (default epsilon=1e-6) makes empty-bin contributions finite
and preserves zero drift when proportions are identical but population sizes
differ. Values depend on binning/smoothing; per-bin counts, smoothed shares and
contributions are emitted for audit. Feature PSI is a CSI equivalent, not a
WOE-weighted characteristic measure. Numeric KS and Wasserstein comparisons use
nonmissing observations separately. Wasserstein normalization uses reference IQR,
falling back to reference standard deviation; a zero scale gives null normalized
distance. Reference-range excursions are the fraction of finite comparison values
outside the reference min/max; no reference support yields unavailable.

Score is illustrative, not a fitted scorecard:

```
factor = points_to_double_odds / log(2)
score = base + factor * (log((1-PD)/PD) - log(base_good_bad_odds))
```

Defaults are base 600, good:bad odds 20, and 20 points for doubling those odds.
PD endpoints are clipped to [1e-6,1-1e-6] for this transform only. PD drift uses
unmodified probabilities. High score means lower risk. Score and PD convey the
same ranking information and their alerts must not be treated as independent.

Thresholds are configurable with inclusive boundaries: >= critical is CRITICAL,
else >= warning is WARNING. PSI, absolute missingness change, KS distance,
out-of-support share and mean PD/score changes have separate thresholds.
Minimum population size defaults to 500; fewer rows in either population return
INSUFFICIENT_DATA. Numeric diagnostics require at least 50 nonmissing rows in each
population, otherwise null/insufficient status. PSI and missingness still describe
all-row distributions; any low-sample alerts are exploratory. An OK status only
means monitored available measurements are below the configured limits.

Bins, smoothing and score mapping form a frozen measurement contract. Changing
these settings requires an explicit new reference; alert thresholds may be
changed without recomputing the reference. JSON checksums protect reference data,
reference row positions and selected model identity. Changes to the measurement
code also require an explicit new reference. Trusted-local checksum checks detect
alteration, not authenticity of arbitrary pickles. No original final-test scoring
or recalibration occurs; no inherited V1 model is deserialized.

Outputs include monitoring.json, label-free row PD/score outputs and a figure.
The summary records quality checks, source/model/reference/code hashes, metric
statuses, alerts, provenance and unavailable outcome checks. It never interprets
distribution movement as deteriorated model calibration or observed defaults.
With real histories, first establish comparable dated cohorts and mature target
windows, then link the Phase 5 discrimination/calibration diagnostics before a
governed recalibration or retraining decision. PSI is not a universal hypothesis
test: the [population-stability review](https://arxiv.org/abs/2303.01227) discusses
limitations of common measures and large-sample goodness-of-fit tests. The
[SciPy KS documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
assumes independent continuous samples; ties and the paired demo make its p-values
unsuitable as formal significance evidence here. Alerts use distances instead.
[Wasserstein distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)
supplements binned PSI with a comparison of the numeric empirical distributions.
