# Phase 13: model governance documentation

Documentation revision: 2026-10-04. Software package remains 0.12.0 because this
phase does not change model, policy, monitoring or serving behavior.

Added the [model card](model_card.md), [machine-readable governance register](governance_register.json),
[open risk register](model_risk_register.md) and [governance process](../docs/model_governance.md).
The existing [validation](validation_report.md), [monitoring](monitoring_report.md)
and [credit policy](credit_policy.md) reports retain their historical measurements
and now explain current status, proposed review/incident actions and required
approval evidence. Nine unresolved findings remain open; owners/reviewers are
unassigned and production use is not approved. No independent sign-off, executed
operational workflow, regulatory compliance or future performance is fabricated.

The model card separates the selected XGBoost/sigmoid origination model and
logistic/isotonic benchmark from the synthetic, uncalibrated portfolio Markov
model and selection-bias experiments. It documents target/population uncertainty,
preprocessing/split discipline, measured calibration and uncertainty, limitations,
excluded uses, artifact identities and monitoring/retraining conditions. The
policy and monitoring sections distinguish executable checks from proposed
operating procedures and human decisions. Drift does not automatically imply
performance failure or authorize retraining/policy changes.

The governance checker reads seven committed aggregate evidence files and typed
policy/monitoring configuration, compares the register and key metric/partition/
threshold tables, and verifies local document links. Canonical JSON hashes support
cross-platform document checks; original model/data artifact hashes retain their
raw-byte meaning. CI now includes this check. No historical artifact or original
applicant CSV is loaded or scored by the checker.

## Completed verification

- Governance identities, selected metric/partition/threshold tables and local
  document links passed against the committed aggregate evidence and configs.
- An artifact-free documentation copy passed even after JSON whitespace and
  newline normalization, without original data/models or deserialization.
- Five deliberate inconsistencies were rejected: altered metric, model hash,
  production approval, model-card metric table and a broken evidence link.
- Full existing suite: 261 passed, with the same two upstream deprecation warnings
  documented in Phase 11. Tests only fitted/scored their own synthetic fixtures.
- Ruff lint/format, configuration, V1 preservation, uv lock consistency, workflow
  actionlint and release archive build/content checks passed. Hosted CI has not run.
- Historical report result sections were retained; frozen model source, dependency
  lock, policies, reference config and original experiment/test-consumption files
  were not changed. No behavioral package version bump was made.

The original final test remains consumed; this phase performs no retraining,
recalibration, final-test scoring, promotion, deployment or external publication.
Phase 12's Docker engine and hosted CI execution limitations remain unresolved.
Phase 14 is the README rewrite and final repository audit.
