# Repository audit: PD modeling and validation milestone

Audit date: 2026-10-05 (America/Sao_Paulo). Inspected commit: `a099ed1e5ea9451e3cf6d2972bbd3ced89b34704`, branch `feature/risk-modeling-lab`; package version `0.12.0`. Repository: https://github.com/Dzidefo123/credit_risk_pred.

The supplied prompt calls this milestone v0.2. That is its roadmap label, not an instruction to downgrade the existing package or repeat completed phases. This is an audit-only change. No modeling implementation, configuration, dependency, artifact, holdout lock, or historical implementation was changed. Reusable data-audit code is proposed for the approved implementation phase, because the first-execution instruction explicitly says to stop before coding.

## Evidence and verification

Reviewed the tracked inventory (171 files before these reports), authored modules, tests, CLI scripts, configurations, notebook source and stored outputs, original API/templates, packaging/lock, Docker/CI, governance evidence, training and validation manifests, and useful history back to `811a2fe`. Historical third-party environment contents were inventoried, not executed. The original serialized model was not deserialized.

Local results on this audit date: **261 pytest tests passed in 53.43 seconds**, with two upstream deprecation warnings (Starlette/httpx and MLflow/SQLAlchemy). Ruff, `check_governance.py`, `check_repository.py` and `check_phase1.py` passed. The frozen experiment verifier confirmed source checksum, artifact hashes, frozen source bytes, exact dependency versions and split assignments. All Phase 5 output hashes matched the saved validation manifest. This is verification of retained evidence, not a fresh independent model validation: no original model fitting or new holdout predictions occurred.

[Hosted main CI run 37213595474](https://github.com/Dzidefo123/credit_risk_pred/actions/runs/37213595474) at the inspected commit completed with **seven successful Python test lanes and one failed container lane**. [Container job logs](https://github.com/Dzidefo123/credit_risk_pred/actions/runs/37213595474/job/111469523963) show a successful image build followed by `ConnectionResetError: [Errno 104] Connection reset by peer` on the first `/health` request, at `scripts/smoke_container.py:56`. Its retry handler catches URLError and TimeoutError, but not that exception. Container checks did not complete; this is not evidence that the image cannot start, nor that the container API passed. An isolated local TestClient with unavailable artifacts returned 503 for health, score and decision using the smoke payload. Empty features are valid under the nullable schema, so a 422 payload error is not the observed cause.

## 1. Current architecture

The active system is a src-layout package, not the original notebook ensemble. `src/credit_risk/data/` owns source adapters, validation, target contracts and separately labeled synthetic histories; `features/` owns training-fitted transforms; `models/` owns PD candidates, calibrators and synthetic transition PD; `validation/` owns metrics, diagnostics, plots, segmentation and the evidence/holdout runner. `decisioning/`, `portfolio/` and `monitoring/` contain the completed research policy, reject-inference, vintage, roll-rate, assumed-loss and drift modules. `api/` owns the current FastAPI factory, schemas and verified service; `tracking/` exports optional local MLflow evidence. `utils/` supplies configuration and logging.

`configs/`, `scripts/`, `tests/`, `sql/`, `docs/` and `reports/` support the package. `pyproject.toml`, `uv.lock`, Dockerfile and GitHub Actions provide build/install/check paths. Local datasets and fitted artifacts are ignored. The wheel includes the current package only. Legacy `app.py`, `main.py`, `credit_risk(1).ipynb` and templates remain inspectable historical files; the original README is preserved in `docs/history/README_v1.md`. Git history was not rewritten.

## 2. Current modeling pipeline

CSV ingestion normalizes declared aliases and validates ten canonical predictors plus `SeriousDlqin2yrs`. The source row index is lineage only, never a predictor. Quality anomalies are recorded; valid duplicate rows are retained. Exact-predictor hashes form indivisible groups, stratified using the maximum label within each group, with seed 42. Group fractions are 45% training, 15% development, 20% calibration and 20% final test; row fractions differ slightly.

The training-fitted feature transformer caps at the 0.001/0.999 quantiles. Logistic uses log1p for all predictors except age, median imputation with missing indicators, scaling and regularized logistic regression. XGBoost uses capping, median imputation with indicators and the declared fixed tree parameters. Neither learns preprocessing from development, calibration or test. A constant training-prevalence benchmark is evaluated on development.

Base candidates fit training only. Raw, positive-slope sigmoid and isotonic variants are fitted on calibration where required, then selected on development log loss with Brier tie-breaking. The selected candidate and calibrators are locked before the one recorded final-test evaluation. Final-test alternatives are descriptive evidence and cannot reverse development selection. Bundles preserve fitted preprocessing and calibration for scoring.

## 3. Dataset

The retained CSV has 150,000 rows, 12 columns including a source index, ten predictors and one binary outcome. Positive count is 10,026 (6.684%). See the [formal data audit](../reports/data/DATA_AUDIT.md) for distributions, missingness, anomalies, conflicting duplicate labels, correlations and partition overlap. There are no reliable observation dates or borrower identifiers. Income units and source population provenance are unverified. The old notebook's initial stored shape is 125,113 rows, which does not match this CSV.

## 4. Target definition

The inherited label is `SeriousDlqin2yrs`, documented as a source two-year delinquency outcome. It has not been reconstructed from dated performance records. A row represents a source record; borrower/account uniqueness cannot be established. Prediction at an origination-like snapshot is a research assumption, not a verified observation point. Predictor timestamps, contractual default criteria, cures, censoring and sampling/acceptance mechanism are unknown. The output is a model-estimated probability of the inherited outcome, not a regulatory PD or a twelve-month default probability.

## 5. Features

Ten predictors cover utilization, age, 30-59/60-89/90-day delinquency counts, debt ratio, monthly income, open credit lines, real-estate lines and dependents. Exact canonical names, dtypes and descriptive statistics are in the data report. No feature includes the target or source index. Absence of direct label input does not prove prediction-time availability; dated lineage would be required for that.

## 6. Existing models

The active candidates are the constant benchmark, L2 logistic regression and XGBoost. Logistic C=1, max_iter=3000, tol=1e-5; XGBoost has 250 depth-3 trees, learning_rate=0.05, subsample=colsample_bytree=0.8, reg_lambda=5, n_jobs=2 and histogram tree construction. There is no tuning search or cross-validation experiment. LightGBM and TensorFlow are not current dependencies; no dependency was added during the audit.

The historical notebook contains a 128/64-unit ReLU neural network, sigmoid output, Adam/binary cross-entropy, ten epochs, batch size 128, and an arithmetic average of NN/XGBoost probabilities. Its saved `combined_model.pkl` is the XGBoost variable alone, not the ensemble or scaler. Preserve the notebook as historical evidence; do not market the current champion as a neural ensemble.

## 7. Evaluation methodology

Current outputs include ROC-AUC, Gini, KS, trapezoidal PR-AUC, average precision, precision/recall at explicit diagnostic threshold 0.10, Brier, log loss, reliability/ECE, observed-versus-predicted rates, segment diagnostics and 200 exact-predictor-group bootstrap replicates. Bootstrap intervals are conditional on already fitted models and omit training uncertainty. The benchmark's trapezoidal PR-AUC is 0.533447 despite average precision of 0.066895: the tied-score PR endpoint interpolation is misleading if presented as ranking strength. Keep AP and trapezoidal area distinct and disclose this convention.

F1 and explicit confusion matrices are absent. Calibration-in-the-large intercept and validation calibration slope are absent; the sigmoid fitting coefficient is not that diagnostic. There is no true out-of-time validation and no group-aware development-only CV study. Recorded modern and historical results are separated in the [baseline report](../reports/baseline/Baseline_Model_Report.md).

## 8. Deployment/API architecture

The current factory has no import-time model/data I/O. Lifespan loads verified trusted-local evidence once; absent or inconsistent artifacts leave health/scoring unavailable with 503. `/score` returns probability, grade, artifact hash, model/package version, calibration and target semantics with `research_only=true`. `/decision` uses the existing research policy and explicitly assumed EAD/loss proxy; these are not fitted LGD/EAD models or IFRS 9 ECL. Default verified startup also needs the original CSV and base/validation manifests to verify lineage and block held-out predictor profiles.

The image is configured for UID 10001; the smoke harness requests read-only operation, dropped capabilities and no-new-privileges. Hosted completion is currently blocked by the smoke startup retry defect. Production authentication, lending audit trails, promotion approval and live population/outcome monitoring are not established. Legacy `/predict/` is not the packaged API.

## 9. Reproducibility issues

Modern manifests pin source/artifact/code hashes and record seeds, parameters, feature names and dependency versions. Exact retained evidence passes verification. Original training Git commit is not captured in training manifests/MLflow tags; today's audited commit must not be backfilled as the historical training identity. Generic supported Python/dependency ranges do not establish compatibility with frozen serialized bundles.

V1 has mutable notebook execution, Colab paths, no reliable lock or NN seed, a population mismatch, pandas append incompatibility and missing environment metadata for imports. Its stored scores cannot be certified or fairly compared with the modern pipeline. Historical model reproduction remains unverified.

## 10. Leakage risks

Confirmed historical failures include in-sample evaluation, preprocessing before the split, outcome-dependent row exclusion and serving without the fitted notebook transformations. These are not findings against the current train-only pipeline. Current exact-predictor grouping prevents observed exact duplicate overlap, but cannot prove borrower separation or exclude post-observation information without identifiers/dates. The final-test consumption lock is scoped to a run directory: a new run directory can bypass a repository-wide consumed-sample rule. Never create a supposedly fresh final holdout from records already viewed.

## 11. Modeling weaknesses

Unverified timing and outcome semantics limit PD interpretation. Random group holdouts do not establish performance through time or on accepted/rejected populations. No group-aware CV, calibration slope/intercept, complete threshold diagnostics or borrower explanation framework is implemented. Coefficients and tree gain importance exist, but transformed/standardized coefficients are not raw-unit odds ratios, and gain is not SHAP. The tiny sigmoid log-loss change is not uniform calibration improvement: Brier worsens slightly. No regulatory, fairness or production approval follows from these metrics.

## 12. Engineering weaknesses

Highest immediate engineering issue is the failing container smoke startup check. Strict byte/version coupling is useful evidence protection but makes edits to frozen model/feature/config/metric/calibration files incompatible with existing bundles. Prefer additive modules and versioned new experiments. Current MLflow is optional evidence export, not champion promotion. The automatic validation outputs are JSON/plots; a reusable complete Markdown report renderer is missing. No distributed platform is needed for this milestone.

## 13. Testing weaknesses

The 261 tests verify substantial mathematical, contract, isolation and integration behavior; successful local execution did not catch the real HTTP startup reset. Add a meaningful bounded-retry regression test and retain a real container integration gate. Synthetic test behavior and dependency matrix checks cannot certify actual lending outcomes, artifact portability, fairness or dated validation.

## 14. Documentation weaknesses

README's statement that no hosted run occurred for unpushed commits is now stale. The new evidence is seven passed Python lanes and a failed container lane; older phase reports remain dated historical evidence. Explain clearly that phases already built beyond PD use separately synthetic histories and assumed loss/policy experiments. The milestone label must not obscure package version or imply another wholesale migration. Automated report lineage and missing metrics need explicit status.

## 15. Preserve

Preserve V1 paths/archive/history; canonical contracts; train-only fitted pipelines; exact-predictor grouping; benchmark/challenger design; independent calibration partition; selection lock and consumed test marker; immutable artifact manifests; research-only API semantics; optional MLflow; PSI bins; existing tests and governance evidence. Preserve all frozen sources and artifacts during this audit.

## 16. Ranked problems and refactoring candidates

| Rank | Finding | Required response |
| --- | --- | --- |
| Critical for lending use | Target timing, borrower identity, source sampling and contractual default provenance unverified | Remain research-only; obtain lender contracts/new dated data before lending validation |
| Critical historical evidence | V1 leakage/selection and saved-model/ensemble mismatch | Preserve and label historical metrics unverified; never endorse them as holdout performance |
| High | Container job fails on uncaught initial connection reset | Fix bounded startup retries with regression tests; rerun real hosted smoke |
| High | Consumed holdout protection is per-run, not global | Add a source/sample ledger; never reuse viewed records as independent validation |
| High | Missing calibration intercept/slope, F1/confusion and development-only CV | Add validated diagnostics and a separately approved training-only CV design |
| Medium | No reusable full Markdown validation renderer or SHAP explanations | Add report generation and optional explanation adapters with lineage and limitations |
| Medium | Git training identity absent; README CI status stale; PR-area convention easy to misread | Record provenance for new runs, update status and distinguish AP/PR-area |
| Low | Upstream deprecation warnings and residual legacy paths | Track dependency maintenance; keep historical paths intentionally |

## 17. Exact proposed architecture and migration

Retain `models/pd.py`; moving it to a new `pd/` directory supplies no functionality and risks artifact compatibility. Create no empty directories. Proposed changes are incremental, subject to approval:

| Sequence | Files to add/modify | Acceptance gate |
| --- | --- | --- |
| 1 | Modify `scripts/smoke_container.py`; add `tests/test_container_smoke.py` | Bounded retry covers startup reset/disconnect/timeout, permanent failures still fail; existing suite and actual hosted container smoke pass |
| 2 | Add `src/credit_risk/data/audit.py`, `tests/test_data_audit.py` | Read-only aggregates distinguish facts/assumptions, duplicates and units; repeatable report input without silent cleaning |
| 3 | Add `src/credit_risk/validation/holdout_registry.py`, `tests/test_holdout_registry.py` | New run directories cannot represent previously consumed source samples as fresh; trusted existing evidence remains readable |
| 4 | Add `src/credit_risk/validation/discrimination.py`, `src/credit_risk/validation/calibration.py`, `tests/test_pd_diagnostics.py` | Explicit thresholds, F1/confusion, AP/PR convention and correctly specified intercept/slope diagnostics; degenerate data and fit failures handled |
| 5 | Add `src/credit_risk/validation/cross_validation.py`, `configs/pd_cv.yaml`, `tests/test_pd_cross_validation.py` | Group-aware folds only within original training records; all preprocessing refit per fold, no original final-test access or champion reselection |
| 6 | Add `src/credit_risk/explainability/logistic.py`, `src/credit_risk/explainability/tree.py`, functional `__init__.py`, `tests/test_explainability.py`; modify `pyproject.toml`/`uv.lock` only if an optional SHAP extra is approved | Transformed-feature units clear, local/global outputs reconcile with declared model scale, SHAP labeled noncausal, no dependency hacks |
| 7 | Modify `src/credit_risk/tracking/mlflow.py`; add `src/credit_risk/tracking/provenance.py`; extend `tests/test_tracking.py` | New experiments record actual Git identity/dirty status; missing old provenance is labeled unknown, never invented |
| 8 | Add `src/credit_risk/reporting/pd_validation.py`, functional `__init__.py`, `scripts/report_pd.py`, `tests/test_pd_reporting.py`; generate `reports/model_validation/PD_MODEL_VALIDATION_REPORT.md`; modify `README.md` | Report values derived from verified evidence, absent candidates marked unavailable, conventions/limitations and lineage included; documentation checks pass |

Each row is a small implementation task/commit with an inspect-change-test-report gate. New fits must use approved training-only experiments or genuinely new data; the consumed final holdout stays closed. The reporting step must accept historical evidence without implying a fresh validation. Existing loss, portfolio and reject-inference demonstrations stay explicitly separate; do not add LGD/EAD estimation, IFRS 9 stages/SICR, macro scenarios, PySpark or Databricks in this milestone.

**Exactly one first coding task:** repair the container smoke test's bounded startup retry behavior and add a regression test, then demonstrate successful real hosted container smoke. Do not hide an assertion failure or retry indefinitely. The current log proves an uncaught transport reset; it does not yet prove the full container check will pass after the fix.

Implementation is paused at the supplied prompt's first-execution gate: “STOP THERE. Wait for my approval before beginning major refactoring.”
