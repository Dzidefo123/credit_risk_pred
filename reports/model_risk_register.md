# Model risk register

As of 2026-10-04 (Phase 13). Scope: origination candidate, synthetic portfolio
benchmark, policy, monitoring and serving. All findings below are **open**.
Severity is a lab judgment of intended-use risk, not an institution's rating.
Suggested accountable roles have no named assignees. Closure requires a linked
remediation artifact, review decision and named reviewer; this document supplies
no such sign-off. No deadline or owner is invented.

| ID / severity | Finding and consequence | Existing control / evidence | Required closure evidence | Suggested role / gate |
| --- | --- | --- | --- | --- |
| GOV-001 / High | Source acquisition, eligible population, label/default semantics and feature units are unverified | Exact source hash, ten-feature contract, [audit](phase1_audit.md) | Authorized data lineage, verified target and feature dictionary, censoring/window rules | Data steward + model owner; before external application |
| GOV-002 / High | No dated external/temporal validation or true borrower grouping; future PD validity unknown | Grouped splits and conditional [validation](validation_report.md) | New borrower-keyed dated cohorts, mature outcomes, frozen independent evaluation and uncertainty | Independent validator; before production consideration |
| GOV-003 / High | No named accountable owner, independent validation opinion or deployment/policy approvals | [Register](governance_register.json) records null owners and not_approved | Named accountable roles, effective challenge, risk acceptance and signed stage decisions | Risk sponsor; before production consideration |
| GOV-004 / High | Segment differences, age use and incomplete individual explanations; fairness not established | Segment metrics, input allowlist, illustrative reason codes | Intended-use legal/domain review, suitable protected-group data, fairness/error analysis, explanation validation | Model owner + qualified legal/domain reviewer; before real decisions |
| GOV-005 / High | Income/DebtRatio units and affordability are unverified; LGD/CCF/drawdown are assumptions | Raw-input guards and [policy](credit_policy.md) / [loss](phase7_expected_loss_report.md) separation | Verified cash-flow/exposure/recovery data, economic objectives, independent limit/policy/loss validation | Credit policy owner; before funded offers or loss reporting |
| GOV-006 / High | Unknown acceptance bias and rejected outcomes; IPW identification cannot be assumed | Separate synthetic [reject-inference evidence](reject_inference_report.md); no invented observed labels | Financing indicators/overlap diagnostics, defensible identification and sensitivity review, independent evaluation | Model owner + validator; before population expansion |
| GOV-007 / High | Historical drift demonstration lacks mature dated performance, live operation and evidence-based trigger calibration | Frozen bins, insufficient-data status and [monitoring report](monitoring_report.md) | Comparable live cohorts, maturity/coverage checks, calibrated review thresholds, operating ownership and incident records | Monitoring owner; before operational reliance |
| GOV-008 / High | No production access control/audit workflow, proven deployment/rollback or executed hosted CI/container checks | Integrity guards, [Phase 12 local tests](phase12_engineering_report.md), configured CI/Docker | Executed Linux/container/CI results, deployment-specific authentication and audit controls, verified incident/rollback exercise | Engineering owner; before exposure beyond controlled research |
| GOV-009 / Medium | State-only uncalibrated portfolio model and incomplete planned interpretability/behavioral layers | Explicit synthetic labels and documented framework boundaries | Validated behavioral features/model and explanations if required by the intended use; restrict scope otherwise | Model owner; before claiming these capabilities |

Research restriction is the current response to unresolved findings; it is not
formal acceptance of risk for real lending. Each future closure must specify
model/policy/reference hashes, population, scope, remaining risk and approver.
Changing a finding to closed without its evidence is not sufficient.
