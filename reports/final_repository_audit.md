# Final repository audit — Phase 14

Audit date: 2026-10-04. Repository: credit_risk_pred; working branch:
`feature/risk-modeling-lab`. Software version remains 0.12.0; Phase 14 changes
presentation, packaging metadata and audit guards, not model behavior.

The fourteen requested phases now have implementation or documentation evidence.
This is a research lab with explicitly bounded demonstrations, not a production
credit system. Some aspirational mission capabilities remain partial or deferred.
Nine findings are open, owners are unassigned and production approval is absent.
See the [risk register](model_risk_register.md) and
[governance register](governance_register.json).

## Mission traceability

| Mission area | Delivered evidence | Status and boundary |
| --- | --- | --- |
| 1: audit and cleanup | [Phase 1 audit](phase1_audit.md), inventory, preservation checks | Current index excludes inherited environments, caches, raw data and pickle; earlier history retains them |
| 2–3: business architecture and engineering layout | [README](../README.md), [architecture](../docs/architecture.md), src package, typed configs | Delivered with separate origination and synthetic portfolio tracks; CLI/reports replace proposed ten new notebooks |
| 4–5: data and targets | [Data contracts](../docs/data_contracts.md), loaders, generator, censored targets | Inherited two-year label kept; synthetic forward default targets separate; no invented borrower/date lineage |
| 6: PD modeling | [Phase 4 evidence](phase4_experiment.json), logistic and XGBoost pipelines | Delivered; preprocessing fitted on train; predictor duplicates grouped; original final test excluded from selection |
| 7: scorecard/interpretability | Coefficient/gain exports, grades, monitoring odds-to-score scaling | Partial: no fitted WOE/IV or monotonic binning library, per-feature points scorecard, SHAP or validated adverse-action explanations; GOV-004/009 |
| 8–9: calibration and validation | [Validation report](validation_report.md), [model card](model_card.md), [Phase 5 evidence](phase5_validation_summary.json) | Delivered cross-sectional metrics, segments, reliability and conditional group bootstrap; no out-of-time or external backtest |
| 10: drift | [Monitoring report](monitoring_report.md), [Phase 10 evidence](phase10_monitoring_summary.json) | Delivered frozen-reference feature/PD/score PSI, KS and missingness; controlled perturbation is not observed production drift |
| 11–12: vintage and roll rates | [Portfolio methods](../docs/portfolio_analytics.md), Python analytics and SQL | Delivered synthetic coverage-aware cohorts and consecutive-month account/balance transitions; no real cohort performance |
| 13: expected loss | [Loss methods](../docs/expected_loss.md), [Phase 7 evidence](phase7_expected_loss_summary.json) | Delivered Markov PD and scenario PD x LGD x EAD; LGD/CCF assumptions, not fitted recoveries or accounting ECL certification |
| 14–15: decisions and limits | [Credit policy](credit_policy.md), [Phase 8 evidence](phase8_policy_summary.json), YAML appetite strategies | Delivered grades, review guards, hypothetical caps and policy comparisons; no funded outcomes or optimized economic frontier |
| 16: reject inference | [Reject methods](../docs/reject_inference.md), [Phase 9 evidence](phase9_reject_summary.json) | Delivered masked-outcome simulation and propensity weighting; no identification of actual rejected applicants' outcomes |
| 17: behavioral features | Synthetic histories include state, utilization, DPD and months on book; transition PD uses current state | Partial: no reusable lagged 3/6/12-month payment/utilization feature transformer or fitted behavioral ML challenger; GOV-009 |
| 18: SQL | Four scripts under sql; DuckDB comparisons in portfolio tests | Delivered portfolio counterparts, not a production database deployment |
| 19–20: MLflow and API | [Integration methods](../docs/api_tracking.md), [Phase 11 evidence](phase11_integration_summary.json) | Delivered optional local evidence tracking and FastAPI score/decision; bundle checks before deserialize, no network tracking dependency or production auth/audit trail |
| 21–22: tests and engineering | [Test/CI/container methods](../docs/testing_ci_docker.md), fixture tests, locked releases and audit guards | Local checks executed; hosted matrix and Docker runtime pending execution |
| 23: governance | [Governance process](../docs/model_governance.md), model card, machine-readable register, consistency guard | Delivered evidence and proposed controls; independent review/operating approvals not executed |
| 24–25: README and interview narrative | [Business walkthrough](../README.md), linked methodology and result reports | Delivered business-first narrative, separate-track Mermaid, install/train/validate/test/API commands and explicit limitations |
| 26–27: principles and phased work | Phase reports and commit history | Small modules, measured evidence, ignored outputs and meaningful phase commits; no fabricated compliance or portfolio results |

The original architecture's notebook names were a proposed organization, not
new scientific results. Executable modules, CLI commands, tests and reports are
the maintained interfaces. The original 73-cell notebook remains V1 evidence.
No empty notebook suite was created to imply an implemented behavioral model.

## Historical evidence and model integrity

The root README was rewritten after its original raw bytes were archived at
[README_v1.md](../docs/history/README_v1.md). SHA-256 is
`de5efdfaac9233d2847623b07b9d0b952ea8f4881ab0190bb424f51750b4062f`,
matching the immutable [Phase 1 inventory](phase1_inventory.json).
The archive is marked binary in Git to preserve that hash across platforms.
The local Phase 1 checker now follows this explicit archive mapping. Original
app.py, main.py, notebook and templates remain unchanged historical evidence;
release and container runtimes exclude them. Their old claims are not endorsed
by the rebuilt README. Earlier Git commits still contain generated files and
raw artifacts: index cleanup is not a history rewrite or secret-scanning claim.

The four Phase 4 training-source byte hashes are checked by the portable final
audit. Five frozen model/calibration sources retain their existing CRLF policy.
Phase 14 does not edit any of them, the dependency lock, policy thresholds,
monitoring reference config or recorded experiment results. The selected model,
selection artifact and selection lock have distinct identities, documented in
the model card. The README metric table is checked against committed aggregate
evidence by the governance checker.

No original applicant model was retrained or recalibrated in this phase. No
original final-test predictions were made; the existing test-consumption marker
was retained. New computations use the README's synthetic portfolio only.

## Final verification

The following local checks passed. They do not certify the unexecuted hosted
CI matrix or container.

<!-- verification:phase14 -->
- Full suite: **261 passed**, with two existing upstream deprecation warnings
  from Starlette/HTTPX and MLflow/SQLAlchemy; no test failures.
- Ruff lint and formatting passed for 87 Python files; typed configuration,
  uv lock consistency and actionlint workflow validation passed.
- Governance checks passed for seven aggregate evidence files, eight linked
  governance documents and the new README metric table. The portable final
  audit passed for 171 tracked files, six preserved V1 files, four frozen
  training-source hashes and 51 current documentation links. The separate
  local Phase 1 check verified retained data/pickle/environment/cache evidence.
- An artifact-free staged checkout with LF defaults passed both audit guards;
  it contained no original CSV or fitted model. Five deliberate errors were
  rejected: changed archive bytes, changed frozen source bytes, a broken README
  link, a tracked CSV and a stale README AUC value.
- All four README synthetic portfolio commands completed: 500 accounts,
  15,142 snapshots/targets and 14,642 consecutive transition pairs; all 500
  accounts were covered in the expected-loss snapshot. Outputs are ignored
  local files. These are synthetic demonstrations, not real portfolio results.
- Wheel/source builds and release content checks passed. The wheel includes
  the revised README metadata; the source archive includes README.md and
  data/README.md. The rebuilt wheel passed the isolated core-only smoke check
  outside the checkout with Python isolation and no MLflow installed.
- Frozen calibration source bytes matched the original test-consumption
  record. No frozen model source, policy/reference configuration, original
  experiment output, dependency lock or consumed final-test marker was edited.
- Reviewed the staged diff and passed its whitespace check. The V1 README uses
  binary diff treatment to preserve its original CRLF/trailing-space bytes.
- Hosted CI has not run. Docker build/runtime remain unverified because the
  local engine is unavailable. No push, deployment or image publication occurred.
<!-- /verification:phase14 -->

The root README is now wheel metadata and is included in the source archive,
with data/README.md included explicitly rather than the entire data directory.
Docker's builder copies the new README and its context allowlist admits it.
The release guard rejects private data, model blobs, environments and V1 runtime
content. The portable repository guard checks current tracked-file hygiene,
archive preservation, frozen training-source bytes and current local links;
CI runs this guard and the evidence-backed README metric check. This is not an
exhaustive security or licensing audit.

## Remaining work before operational use

1. Resolve source provenance, target/default definitions, units, dated borrower
   lineage and independent temporal/external validation (GOV-001/002).
2. Assign owners and independent reviewers; complete approvals and assess
   fairness, explanations, affordability and funded economics (GOV-003/004/005).
3. Establish actual rejected-population evidence, mature production outcomes,
   controlled alert handling and recalibration/retraining gates (GOV-006/007).
4. Execute hosted CI and Docker checks; establish access, audit, artifact trust,
   deployment and rollback controls (GOV-008). Docker CLI is installed, but its
   local engine is unavailable; no image deployment or publication occurred.
5. Implement and validate rolling behavioral features and any WOE/IV/monotonic
   scorecard layer on suitable data before claiming those capabilities (GOV-009).

## Interview walkthrough

Start with the portfolio/business objective and distinguish the two populations.
Explain target horizons and censoring, then train-only features, the logistic
benchmark and XGBoost challenger. Discuss calibration separately from ranking,
use the recorded final holdout and its uncertainty, and identify why temporal
validation is still missing. Show how grades, review guards and limit assumptions
change decisions; distinguish hypothetical origination loss proxies from the
portfolio transition model. Demonstrate vintages, migration, scenario loss and
drift controls, then explain the evidence required for independent approval,
recalibration, retraining and rollback. The remaining gaps are part of the risk
assessment, not capabilities to claim as finished.
