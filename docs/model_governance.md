# Model governance and change control

Revision: Phase 13, 2026-10-04. The lab is research only. These are documented
procedures and future approval requirements, not implemented institutional
workflows, regulatory compliance or evidence of completed independent review.
No model owner, independent validator, credit approver or deployment approver
has been assigned. The [register](../reports/governance_register.json) explicitly
records that production approval is absent.

## Inventory, evidence and responsibility

The [model card](../reports/model_card.md) defines the selected origination
XGBoost/sigmoid candidate, logistic/isotonic benchmark, source/target, performance,
calibration, artifact identities and excluded uses. The synthetic portfolio
Markov benchmark, illustrative policies and reject-inference experiments are
separate research components. The inherited V1 pickle is preserved but excluded
from new serving and validation. Model, policy and monitoring reference identities
are distinct; software releases do not retroactively version historical runs.

| Proposed accountable role | Responsibility | Current assignment |
| --- | --- | --- |
| Model owner | Intended use, feature/target contract, change request, limitations | Unassigned |
| Data steward | Authorized provenance, keys/dates/units, outcome maturity and access | Unassigned |
| Independent validator | Effective challenge, independent acceptance criteria and evaluation | Unassigned |
| Credit policy approver | Risk appetite, limits, economics, fairness/domain review and exceptions | Unassigned |
| Engineering/deployment owner | Access/audit controls, artifact release, reproducibility, rollback | Unassigned |
| Monitoring owner | Comparable cohorts, alert triage, outcome review and incident records | Unassigned |
| Risk sponsor | Assign roles and approve scope/remaining risk before operational use | Unassigned |

Current evidence was produced and self-checked within this lab. Successful unit
or integration tests are engineering evidence, not independent validator sign-off.
The [risk register](../reports/model_risk_register.md) identifies unresolved
findings and required closure evidence. Assigning a role or adding a signature
cannot replace missing empirical validation.

## Lifecycle gates

1. **Research**: fixed target/population, data contracts, source/model lineage,
   development/calibration separation and documented limitations. This is the
   current state. Research API decisions are hypothetical.
2. **Validation-ready candidate**: verified observation/target semantics, authorized
   representative data, dated borrower-keyed splits, missingness/selection review,
   prespecified performance/calibration/fairness/economic criteria and fresh
   independent holdout. Required evidence is not currently complete.
3. **Independent review**: a named validator challenges design, data, uncertainty,
   segments, calibration, robustness, intended use and open findings. Approval
   criteria must be set before independent evaluation, not selected from the
   already-consumed Phase 5 test results.
4. **Policy/deployment review**: separately approve credit thresholds, affordability,
   limits and assumed/estimated loss components; verify operational controls,
   hosted CI/container results, audit records and rollback exercise. No real
   offers, exposure changes or financial reporting are authorized by model selection.
5. **Controlled operational use**: only after named approvals specify model/policy/
   reference hashes, population, access, effective date, monitoring and restrictions.
   This state has not been reached or implemented in the lab.
6. **Review/retirement**: investigate incidents and material changes, preserve
   evidence and retire/replace only through a recorded review decision.

These gates are procedural. The API does not enforce organizational approval,
collect sign-offs or implement a lending workflow. MLflow export flags do not
constitute promotion controls. research_only=true is a disclosure, not an ACL.

## Change classification and holdout protection

| Change | Minimum review/evidence before broader use |
| --- | --- |
| Documentation correction | Trace changed claim to source; run governance/link checks; retain historical evidence |
| Threshold, grade or limit assumption | Version policy separately; analyze development/new eligible data, exposure/mix/loss sensitivity and safeguards; obtain policy review |
| Monitoring thresholds | Record reason and old/new configuration; keep measurement/reference definitions unchanged; review historical/new comparable cohorts |
| Monitoring bins, reference, score transform or model identity | New reference version and compatible measurement identity; retain old reference; no silent rebinning to hide alerts |
| Recalibration | Fresh representative mature calibration data, frozen candidate-selection rules and new independent evaluation; validation and policy impact review |
| Retraining, features, target, population or dependency/source change | New experiment/lineage, preprocessing fit separation, fresh calibration and independent evaluation, full validation/serving checks |

Original final-test access was consumed in Phase 5. Do not delete its persistent
marker, tune from published final metrics, or describe reproduction as a new
independent test. The code permits unchanged recorded reproductions; Phase 13
reads prior aggregate JSON only and makes no such scoring call. A future model-selection or calibration change
needs a genuinely new independent holdout, preferably dated and borrower-keyed.
Research development/policy/monitoring populations may overlap; that does not
turn them into independent policy validation or temporal evidence.

## Monitoring, incidents and model review

The [monitoring report](../reports/monitoring_report.md) specifies current metrics,
units, thresholds and proposed response. Its development reference is historical,
not temporal. Distribution changes alone cannot establish calibration decay,
concept drift or increasing realized defaults. Thresholds are configurable lab
choices, not universal risk limits.

Proposed operating cadence after a valid future deployment: data/schema/lineage
checks per batch; monthly comparable population/segment drift review; monthly
outcome review only for mature cohorts with sufficient support; and a scheduled
comprehensive review at least annually or sooner on material changes. This cadence
is a proposal: no scheduler, notification service or review meeting exists here.
Any sponsor must set and approve a context-specific cadence and support limits.

Investigate data failures before considering model change. Integrity failures
already prevent ready scoring; monitoring insufficient-data results require more
coverage rather than a green status. Material persistent population shifts,
verified mature calibration deterioration, new segment failures, target/product
changes or unusable data trigger review. They do not automatically trigger fit,
recalibration, threshold change or promotion. Drift thresholds can prioritize
triage; outcome and intended-use evidence determine remediation.

A future incident record should capture detection time and batch/cohort IDs,
model/policy/reference hashes, affected inputs/decisions, completeness/maturity,
metrics, root cause, containment, accountable role, review decision and follow-up.
Protect applicant data and keep personally identifying row data out of Git and
aggregate reports. No retention schedule, personal-data handling authorization
or immutable decision-log service is established by this repository.

## Release, containment and rollback

Serving verifies the trusted bundle's source, models, calibration/code identities
and frozen dependency versions before deserialization. Missing/invalid artifacts
return unavailable rather than a legacy fallback. Joblib hashes detect changed
bytes; they do not authenticate untrusted files. Package/core-wheel tests,
read-only mounts and non-root Docker design support reproducibility, but local
Docker and hosted CI execution remain pending in the recorded Phase 12 evidence.

For a future approved release, preserve the entire model/policy/reference bundle,
source manifests, environment/lock, release commit, approvals and previous valid
bundle. Exercise rollback before operational reliance. If no approved compatible
bundle exists, stop issuing automatic decisions and route to the agreed manual
process; never fall back to the inherited V1 pickle. The current lab has no
approved prior production bundle, traffic router, automatic rollback or incident
notification implementation. A proposed manual fallback is not a live process.

## Documentation checks and approval record

```console
uv run --no-sync python scripts/check_governance.py
```

The checker compares the machine register with committed aggregate evidence and
effective typed policy/monitoring configuration; it also verifies selected metric,
partition and threshold tables plus local evidence links. Canonical JSON evidence
hashes ignore whitespace and platform line endings; model/source/artifact hashes
remain the original raw-byte identities. The check does not load models, read
original applicant records, score any holdout, establish independent validation or
approve use. CI includes it so changed evidence/configurations require an explicit
reviewed documentation update. It cannot establish the truth of an edited source
manifest or detect every misleading prose claim.

A future approval record must contain a named reviewer/role, decision and date,
allowed population/use, model/policy/reference hashes, independent evidence,
findings/conditions, effective version and next review trigger. The current
register records not_approved and null reviewers. No approval is fabricated.

## Cross-experiment final-holdout integrity (Task 3)

The [repository-level holdout registry](HOLDOUT_REGISTRY.md) now checks source
and raw-profile identities across run directories. Its tracked
[ledger](../reports/holdout_registry.json) retains the migrated historical consumed
holdout; the existing governance checker validates that baseline. New final
access is a one-way consumption transition, including interrupted scoring.
The per-run consumption marker remains intact. Retrospective review uses stored
evidence rather than rescoring via the validation runner.

This addresses the earlier per-directory enforcement gap within the supported
local workflow. It does not establish borrower identity, temporal independence,
regulatory approval or transformed-data equivalence. See the registry document
for the weaker historical 64-bit bridge, concurrency/Git limitations and lifecycle.
Historical governance snapshots and model metrics remain unchanged.
