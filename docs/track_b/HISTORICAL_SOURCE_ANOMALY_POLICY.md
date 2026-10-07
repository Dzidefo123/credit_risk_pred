# Track B Task 8B — Historical Source Anomaly Robustness Protocol

**ROBUSTNESS POLICY READY WITH MATERIAL LIMITATIONS.** Version 1.0.0, 7 October 2026. This is a governance and counterfactual sensitivity design. No policy is authorized or applied; Task 8 remains stopped. No parser change, new frozen sample, performance access for the stopped vintages, model run, or macro acquisition.

## Source ambiguity and scientific usability

Task 8A inspected 12,169,807 origination records and identified four unresolved records. Their prevalence is **0.0000328682%**, or **0.328682 per million**. Low frequency does not establish immateriality. Source semantics can remain unresolved while a dataset has a defensible restricted use under a reviewed, prespecified policy that avoids inventing values. A dataset is scientifically unusable for a particular claim when identity, event observability, cohort assignment or required features cannot be established under such restrictions. Usability is claim-specific, not a blanket assertion about every field or vintage.

The frozen register is A1: 2006Q3 line 107274, original rate `.`; A2: 2008Q4 line 206261, embedded 2009Q1; A3: 2020Q4 line 17103, embedded 2020Q3; A4: 2022Q4 line 98552, embedded 2022Q3. Raw source archives and identifiers remain immutable and private. A1 stays MALFORMED_ISOLATED_VALUE; the intended value/token semantics and three quarter placements remain unresolved. No public licensed identifier appears in these artifacts.

## Candidate policies — design only

| Policy | Rule | Conditions and limitations |
|---|---|---|
| A — strict stop | One unresolved source violation blocks its vintage | No sample exists, so a selected-ID difference is undefined, not zero |
| B — record quarantine | Exclude the frozen anomalous record from eligibility before ranking | Preserve raw record/register/count; apply source/schema criteria uniformly; no outcomes or post-freeze substitution |
| C — field quarantine | Retain facility; make malformed non-identity predictor unusable | Only A1 in this register; preserve raw `.`; hypothetical null reason `UNRESOLVED_SOURCE_ANOMALY:ORIGINAL_RATE_DOT`; never claim a documented missing enumeration or invent a rate |
| D — cohort-attribute quarantine | Retain ID and both metadata attributes; prohibit disputed-quarter use | Another authoritative cohort basis must be justified. Annual agreement makes A3/A4 candidates for restricted annual analysis; A2 crosses the annual boundary and is blocked without clarification |

**Recommended fallback for later approval: B.** It makes no claim about the unknown value or true quarter. This recommendation does not authorize exclusion, a parser amendment, resumption, sample freeze or performance access. Existing policy A continues to govern execution.

## Eligibility before ranking: mathematical distinction

For a frozen anomaly predicate a(x) that depends only on origination/source-schema evidence, define E_B={i in E_ID: a(x_i)=false}. Rank E_B by (SHA256(`track-b-multivintage-v1:{archive_vintage}:{immutable_loan_id}` encoded as UTF-8), loan_id), then take the first 20,000. This is sampling from a prespecified eligible population, not outcome-adaptive replacement: neither E_B nor ranking uses outcomes. It changes the target eligibility population and can still introduce source-quality selection bias. Hash sampling does not cure that bias.

If an anomaly lies inside the unrestricted first 20,000, pre-ranking exclusion produces one entrant and one departure (symmetric difference 2) relative to unrestricted inclusion. If it lies outside, sample composition is unchanged. After a sample has been frozen, do not substitute a new facility; a separately approved versioned sampling redesign would be needed. No future favorable performance may determine the anomaly criterion or policy.

## Counterfactual identifier-only impact

These are diagnostics, not newly created samples. Only read-only Task 8A identifier census databases were ranked; the A1 key was obtained from its origination member/line. Temporary top-rank sets were held in memory, never saved as selected-ID files or registered as samples. No performance member was opened.

“Strict-valid-only reference” here means the surveyed identifier universe minus its frozen anomaly. **It is conditional on other records passing the later full origination gate.** Task 8A was a convention census, not full schema eligibility certification. Task 8B does not rerun or resume that gate. Additional exclusions could change ranks; a later authorized gate must recertify them. New anomalies trigger a new versioned review before any performance access, never silent expansion of this four-record policy.

| Anomaly | Archive | Audited ID universe | Reference universe (B) | Rank if retained | Within 20,000 | Inclusion vs B: entrants / departures / symmetric difference |
|---|---:|---:|---:|---:|---|---|
| A1 | 2006 | 1,193,544 | 1,193,543 | 474,260 | No | 0 / 0 / 0 |
| A2 | 2008 | 1,233,146 | 1,233,145 | 378,990 | No | 0 / 0 / 0 |
| A3 | 2020 | 3,913,741 | 3,913,740 | 1,413,681 | No | 0 / 0 / 0 |
| A4 | 2022 | 1,580,753 | 1,580,752 | 646,011 | No | 0 / 0 / 0 |

The JSON supplies the full four-policy × four-vintage table. A blocks every vintage and has no selected set. B has the reference universe, anomaly excluded, difference 0 by definition. C: 2006 has 1,193,544 candidate IDs, A1 eligible but outside top 20,000, difference 0; C is inapplicable to identity/cohort anomalies. D: conditional annual-only scenarios have 3,913,741 IDs in 2020 and1,580,753 in 2022, their anomalies eligible but outside top 20,000, difference 0. D has no admissible 2008 universe without another authoritative annual assignment. Even a diagnostic unrestricted inclusion of A2 gives rank 378,990 and difference 0; that does not legitimize annual reassignment. Inapplicable policies report null, not fabricated eligibility counts. Rank if retained is hypothetical for B/A, not eligibility under those policies.

## Rate-record impact and minimum schema

A1 ranks 474,260 and is not selected under either exclusion or hypothetical field retention. Original rate is a FEATURE in the canonical schema, not an identity key or required non-null incident-PD/event eligibility field. Existing parser accepts a blank as null but rejects dot. Task 8B does not change that behavior. A missing column and a nullable field are different: keep the canonical column and raw lineage even if a later approved field-quarantine design disables its value.

A future PD/hazard model could use a reduced predictor set or explicit missing category/indicator with training-only preprocessing and validation; this is scientific feasibility, not evidence of validated predictions. No rate is imputed now. Unknown rate cannot support rate-dependent amortization, discounting or ECL calculations; an original coupon is not automatically accounting EIR. Missingness can be informative and requires sensitivity assessment.

## Prefix, uniqueness and annual/quarter distinction

All three identifiers are syntactically well formed and unique in their annual census; every audited annual universe had zero duplicates. Distinct embedded years rule out cross-vintage duplication for this four-record set: A2's embedded 2009 is not another included archive year. This is facility-key evidence, not borrower identity.

A3/A4 agree on annual 2020/2022 but disagree on quarterly membership. A2 disputes both 2008 archive vintage and embedded 2009 year, requiring stricter treatment. Source-member quarter and identifier quarter remain separate metadata; neither is rewritten. First-payment month is not exact origination date.

Current Task 8 sampling/reporting groups are annual archive vintages. Calendar support comes from reporting months and age from the provider clock, not an exact origination-quarter fixed effect. Thus quarter-specific classification is not fundamental to those descriptive summaries if an approved annual definition can be justified; it remains essential for any future quarter-level cohort claim. Annual grouping does not fix unrestricted age–period–cohort identification. A2 cannot enter a true annual origination cohort under D until its annual assignment is authoritatively resolved.

## Materiality and future robustness

Review frequency/concentration, identity/linkage, selected-ID composition, eligibility-population change, predictor usability, event definitions/coverage, calendar/cohort classification, raw provenance and downstream model sensitivity separately. No arbitrary regulatory threshold is created. Zero direct sample-content difference is demonstrated only for the conditional identifier universe; it says nothing about event or prediction effects, which were not observed.

Future primary B versus separately approved C/D or later provider-supported compatibility should preserve sources, ranking namespace, event definitions, temporal partitions and evaluation discipline. Separate common-ID prediction/feature effects from population-composition effects. Prespecify counts/missingness, hazard/CIF/calibration and temporal-transport comparisons; disclose changed populations. Never choose the policy giving the best outcomes or silently reuse a consumed evaluation as fresh validation. No model was run now.

## External clarification and versioning

Two clarification records exist in the JSON: original-rate dot and the three quarter placements. Provider Freddie Mac; submission/response dates and text/hash are null; status NO_SUBMISSION_RECORDED / NO_RESPONSE_RECORDED. This records absence of evidence in this workspace, not a fabricated claim that a support request was made. No external message was sent.

A later response requires verified provider authority, immutable response content/hash, date, affected convention, old/new rules, protocol impact and a new amendment version. If parser behavior changes, increment its version. Preserve Task 8 STOP → Task 8A UNRESOLVED → Task 8B design → later clarification/amendment. A response never overwrites Task 8A or automatically authorizes execution. Unverified answers, missing hashes or a reused amendment version must fail the governance gate.

## Preservation, publication boundary and verification

Task 8 and 8A files remain byte-identical. Their local uncommitted artifacts are deliberately excluded from this focused policy commit; the JSON pins lineage paths and Task 8A SHA-256. No claim is made that this commit publishes or completes Task 8. Public reproducibility is limited to aggregate diagnostics and standalone synthetic policy contracts until the eventual cohort/audit publication. Private census IDs and raw mortgage data remain excluded.

Verification: 17 targeted tests passed; full local suite 822 passed with four existing warnings (67.73 seconds); lint/format, governance and repository checks passed. All 349 pre-existing public files, protected private artifacts/panels/ledgers, seven archive hashes and the Track A tag were verified unchanged. Full results are recorded in the JSON. Tests exercise hypothetical policies only on synthetic records, without wiring them into any production parser or pipeline. Preservation uses opaque byte hashes, not outcome evaluation. Frozen samples, panels, source archives, Track A tag/artifacts, Tasks 2–7 and consumed ledgers remain protected. Retained XGBoost AUC 0.868152, Brier 0.048545, log loss0.176030 remain historical results, not newly evaluated metrics.

**Decision: ROBUSTNESS POLICY READY WITH MATERIAL LIMITATIONS.** Governance design is valid; source semantics and full eligibility remain unresolved. Task 8 resumption is not authorized.

**Exactly one next action:** review and explicitly authorize or reject candidate B as the future primary handling policy. No processing follows that recommendation in Task 8B.
