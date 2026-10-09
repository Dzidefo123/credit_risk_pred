# Task 19 — Manuscript v0.3 major revision and evidence reconciliation

## A. Repository state

Started from clean `main` at `3098da7bf17dffb7f0d0f5b2a0732b6311402864`, equal to fetched `origin/main`. The Task 17, Task 18 and correction commits `b64c20f`, `fde602d` and `7ea0bb4` are ancestors. Work uses the local branch `codex/task19-manuscript-v03`. No history repair or push is performed.

The existing hosted run `37864681856` completed with a successful container job and failures in all seven test-matrix jobs at the private-data-free test stage. The API identifies the failed stage, not its exact assertion; this task does not claim to have independently retrieved the job logs. Infrastructure is outside this writing task. Hosted CI remains an explicit reproducibility issue, distinct from the local checks below.

## B. Task 17 erratum

`TASK17_ERRATUM.md` was created before the revised manuscript. It withdraws the incorrect structural claim that month-constant macro terms cannot change facility ranking in a multinomial model. Different competing-cause denominators and jointly refitted mortgage coefficients permit ranking changes. The empirical decomposition does not depend on that argument. The historical review and registers remain unchanged.

## C. Manuscript sections rewritten

The abstract, introduction, related-work positioning, methods, results, discussion, limitations, conclusion and reproducibility statement are rebuilt. Results now follow development fit, temporal scores, calendar contribution, stratified discrimination, unseen transport, all-horizon CIF and statistical-unit sensitivity. Earlier broad benchmark narration and incomplete appendix headings are omitted. The draft is additive at `paper/main_v0.3.md`; v0.1 and v0.2 are preserved.

## D. Claims strengthened

Traceability and disclosure are strengthened. The weighted pair decomposition, distinct ex-period denominators, countervailing unseen-vintage scores and admitted supported-month/entry summaries now appear centrally. This does not strengthen a causal mechanism claim. All empirical values are retained observations or already executed post-hoc summaries, not new measurements in Task 19.

## E. Claims narrowed

The study-level description is `MIXED_TEMPORAL_TRANSPORT_ACROSS_POPULATIONS`, while the original frozen primary decision remains unchanged. The negative primary result is scoped to a seasoned seen-vintage population and its observed calendar. Within-month AUC is support-qualified. Macro information is vintage/revision-aware, not provider-release-certified. CIF paths are retrospective, concentrated and overlapping. Facility uncertainty is conditional on the realized calendar.

## F. Claims removed

Removed from v0.3: structural rank impossibility, an unweighted gain ratio presented as contribution share, superiority of M2 at the shortest payoff-CIF horizon, independent CIF corroboration, universal macro failure and an identified 2020 cause of the whole pooled AUC gain. Mathematical propagation and diagnostic hypotheses remain distinguished from causal identification.

## G. New limitations added

Explicit limitations cover supported-month exclusions, different population weighting/encoding, origination-information predictors, current-state omission, survivor/burnout hypotheses, payoff/maturity ambiguity, distinct months versus independent observations, concentrated entry, overlapping realized macro information and the retained aggregate bootstrap's limited support. Unseen annual decomposition and peer-review strengthening are not invented.

## H. Abstract changes

The abstract is completely rewritten. It reports the development/temporal contrast, calendar concentration, weighted ranking decomposition, support-qualified month summaries and slight unseen joint-loss reversal. It ends with uncertainty, exposure and causal limitations. Current word count: 223 under the documented whitespace convention, excluding traceability comments.

## I. Title decision

Primary title: **Population- and Regime-Dependent Transport of Vintage-Aware Macroeconomic Features in Mortgage Competing-Risk Models**.

Three alternatives are recorded in the revision register:

- When Pooled Mortgage Discrimination Improves but Probability Quality Does Not: An Audited Temporal Evaluation
- Calendar Separation and Population Transport in Vintage-Aware Mortgage Competing-Risk Models
- Pooled Ranking, Within-Period Discrimination and Probability Transport in Longitudinal Mortgage Risk

## J. Seen/unseen transport treatment

Both populations and their complete retained score vectors appear in the central results. The unseen joint-loss reversal is not demoted to a footnote; remaining unfavorable Brier/default-AUC comparisons are shown beside it. Population, seasoning, calendar and encoding differences are hypotheses/limitations, not demonstrated causes. No unseen uncertainty interval or per-year result is fabricated.

## K. Pooled/within-period AUC treatment

The year decomposition uses actual pair weights and corrected contribution shares. The local month analysis is admitted with its original registration, script hash, input/output checks and aggregate-only output. It was registered and executed before Task 19 under separate user authorization; this revision does not run it again. Sparse months and the eligible-population decomposition are clearly separated from the full-primary pooled statistic. The earlier original Task 18 registration and the later local registration are distinct.

`reports/paper/task19_evidence/admission.json` pins the copied evidence bytes. A narrow additive `.gitattributes` preserves the hash chain across LF/CRLF checkouts; no prior attributes or preservation rules are changed. No loan identifiers or individual records are admitted.

## L. Calendar concentration treatment

The full yearly contribution table shows interval weights and both directions. The renormalized ex-period mean is distinct from the full-denominator residual. The retained aggregate year-block interval is reported as sensitivity with its limited-block and method qualification; it is not regenerated or substituted for the original interval-array bootstrap. Calendar concentration is not a COVID causal effect.

## M. CIF reframing

All frozen horizons and both causes are shown. M1 is correctly described as closer than M2 at the shortest payoff horizon. The admitted entry histogram and modal endpoint define the dominant historical window. Projections can extend past realized facility follow-up. The CIF and calendar findings share substantially the same population and period and are not independent validation. Conditional-entry/censoring implementation review remains open.

## N. Uncertainty/statistical-unit treatment

Both joint-loss and payoff-Brier intervals are reported side by side under facility and calendar resampling. `ROBUST_FACILITY_ONLY` is retained as a scoped descriptive classification; no general robustness or broad macro sampling uncertainty is asserted. Equal interval weighting is distinguished from an equal-facility estimand. The development-month count is not described as a count of independent economic observations.

## O. PIT terminology changes

Unqualified PIT terminology is replaced with vintage-aware and revision-aware language. The macro information contract uses ALFRED vintage dates and values; exact provider first-release timestamps and lags were not certified. Retrospective mortgage knowledge time is separately acknowledged. CG06 remains partially resolved.

## P. Governance/reproducibility treatment

The manuscript describes deterministic facility/cohort rules, frozen models/predictions, artifact lineage, evidence reuse, registered post-hoc summaries, correction ledgers and hostile review. These are audit controls, not regulatory certification or proof of scientific validity. Every strong central interpretation remains bounded. No model fit, prediction regeneration, calibration, source acquisition, outcome-array scoring pass or holdout-ledger API is invoked by Task 19.

## Q. Literature/novelty changes

Existing verified Task 16 references support the methodological context. No new bibliography entry is added and `references.bib` is unchanged. The manuscript cites the relevant subset of the verified registry. The closest-paper full-text gap remains concise and explicit. `DISTINCTIVE COMBINATION PLAUSIBLE WITH MATERIAL LIMITATIONS` is unchanged; tests do not upgrade novelty or establish priority.

## R. Evidence traceability status

Every displayed quantitative statement is tagged and maps to an exact public source field, display format, source hash, retained/admitted status, historical review concern, Task 18 status and correction scope. Original Task 14 claim IDs and Task 17 survival verdicts are included when an exact point-estimate source matches. Newly admitted post-hoc fields are not misrepresented as original primary claims. Mathematical definitions, model identifiers, headings and reference metadata are explicitly distinguished from empirical measurements.

The deterministic renderer and checker report 257 quantitative occurrences, all mapped, with zero orphan numeric claims. Body word count excluding reference list and traceability comments is 5,995. The original 123 v0.1/v0.2 numeric bindings remain unchanged. Six proposed figure specifications and seven bound tables are recorded separately; no figures or predictions are generated.

## S. Test results

Task 19 targeted regressions initially passed 27 tests. The first full run found two legacy assertions that globally prohibited v0.3: 1,397 passes and two failures. They now inspect the original Task 17/18 commit trees, permitting later writing tasks while retaining the historical no-manuscript guarantee. The exact two test-code amendments are pinned separately in the additive Task 19 preservation manifest; original science and historical manifests are unchanged. A regression rejects subsequent unapproved edits to these amended tests. Final targeted result: 182 passed; full suite: 1,401 passed with four known warnings. Lint, format, configuration, governance, hygiene and preservation passed. Exact timings and staged-diff outcomes are recorded in `task19_verification.json` after the final run. Tests mutate display values, source fields, hashes, markers, unsupported language and required qualifications, and check rejection of changed frozen evidence without modifying real artifacts. Automated checks establish traceability/consistency, not scientific acceptance.

## T. Remaining arXiv blockers

Author identities, affiliations and acknowledgments remain `[AUTHOR REVIEW REQUIRED]`. `[PUBLICATION TERMS REVIEW REQUIRED]` and `[ETHICS AND AUTHOR REVIEW REQUIRED]` remain. A further hostile manuscript review and author review of the conditional-entry/censoring implementation are required before treating this as a submission-ready paper. The incomplete closest-paper comparison, support qualifications and hosted CI issue remain visible. This task does not submit to arXiv, certify publication rights or declare all scientific questions closed.

## U. Remaining peer-review strengthening analyses

Unexecuted: complexity adjustment, facility-weighted scoring, current-state conditioning, support-restricted evaluation, a flexible challenger, maturity separation, the unseen population's per-year decomposition and authorized Fannie replication. CG03 implementation review and CG06/closest-paper literature review remain. These are not silently performed. Fannie remains `DRAFT_NOT_YET_AUTHORIZED`; no Fannie outcome is used.

## V. Recommendation

**READY_FOR_HOSTILE_MANUSCRIPT_REVIEW**.

This is a writing-stage recommendation, not arXiv readiness, model approval or an upgrade to the literature decision. No new empirical task begins. Commit only after local validation passes, with `paper: revise manuscript after sensitivity closure`. **Do not push.**
