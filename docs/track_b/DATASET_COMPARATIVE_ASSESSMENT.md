# Track B mortgage candidates: comparative assessment

Review date: 2026-10-06. Official documentation review only. No loan/sample records downloaded or inspected, accounts registered, terms accepted, models fitted or development source formally selected. [Complete field crosswalk](FIELD_DICTIONARY_COMPARISON.md), [machine-readable evidence](field_dictionary_comparison.json), [unchanged contract](LONGITUDINAL_DATA_CONTRACT.md).

## Decision

Freddie Mac Standard remains a provisional leading candidate for mortgage PD/term-structure and simplified disposition-loss research; Fannie Mae Primary remains a plausible external replication candidate. This is a shortlist direction, not an approved dataset acquisition or a finding of full 77-field support. Neither source supplies the whole accounting-oriented contract. Current evidence does not justify calling either IFRS 9 ready.

The strongest difference from the static Track A benchmark is documented loan-month history. That enables research designs to be considered; it does not establish target validity, complete follow-up, real-time information sets or actual model performance. Track A stays frozen and its outcomes/probabilities are not reused as mortgage/default estimates.

## Current documentation and scope

The [Freddie landing page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset) reports coverage through March 2026 and approximately 56 million loans across its collections. Those are provider descriptions, not locally verified row counts. This review uses Standard Release 47. The [July guide](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf) says Non-Standard remains at Release 30; the headline range must not imply all subdatasets have current histories.

The user's January Freddie guide is historical context. The [Release 47 change notice](https://www.freddiemac.com/fmac-resources/research/pdf/disclosure-changes-summary.pdf) changes names/positions, adds fields and changes gain/loss signs. Future parsers must pin a release-specific layout; loss signs cannot be copied from an older reader. A documentation version is not an as-of record vintage.

Fannie documents a [July 2026 update with Q1 2026 information](https://capitalmarkets.fanniemae.com/credit-risk-transfer/fannie-mae-single-family-loan-performance-data). Its [shared glossary, dated September 10, 2026](https://capitalmarkets.fanniemae.com/resources/file/credit-risk/pdf/crt-file-layout-and-glossary.pdf), distinguishes SF Loan Performance from CAS/CIRT. This assessment uses only applicable SF fields. Existing dictionaries do not prove that a future acquired file matches that release/schema.

## Material findings against the contract

1. **Loan identity is not borrower identity.** Both support a disclosed mortgage key. No stable cross-loan obligor key was identified. Borrower counts, masked geography and refinance mappings cannot establish independence across all facilities or providers. No identity reconstruction is proposed.
2. **Dates differ.** Freddie's first-payment month and encoded origination quarter are not a verified note date. Fannie documents a month-level origination field. Monthly delinquency supports bands/intervals, not exact days. Termination month is not the first qualifying default month.
3. **Monthly records are not automatically point-in-time.** Release lags and retrospective corrections leave row/attribute first-availability unresolved. A later study needs archived releases or a clearly restricted retrospective information-set assumption. Acquisition entry creates delayed-entry/early-history questions; missing records cannot become healthy months.
4. **Exit and default semantics need harmonization.** Define a research event from monthly states separately from payoff, defects, sales and foreclosure-related exits. Same numeric termination codes can have different provider meanings. A newly declared event proxy must not be presented as regulatory default.
5. **Loss components are not a full cash-flow ledger.** The crosswalk finds aggregate proceeds/cost information, not individual transaction/reversal identities and timing. Disposition/reporting dates cannot silently become recovery dates. Trailing revisions make ultimate workout completion a separate evidence gate. Observed accounting/downturn LGD remains unsupported; a narrower research loss measure may be feasible later.
6. **Mortgage exposure is not revolving CCF.** Loan balances/removal amounts are useful candidates, but first-default exposure needs its own event/balance/accrual/masking reconciliation. Neither supplies revolving commitments and draw behavior. Original balance is not an undrawn credit limit.
7. **Public Fannie applicability matters.** Current credit scores and several payment/modification/interest metrics in the shared glossary are CAS/CIRT-only. Specifically, positions 48/50, 71/113, 75/76, 77/78 and 85 are not evidence of those fields in SF Loan Performance. Do not infer a current/original PD comparison or a public net-loss series from their names.
8. **Accounting and macro gaps remain.** Note rates are not accounting EIR; origination scores are not recognition-time probability term structures. Actual SICR policy, stages, impairment adjudication, scenario paths/weights and macro vintages need other evidence. Delinquency/assistance data can support only explicitly labeled later heuristics.

These are contract-based research judgments, with field-level evidence pointers in the crosswalk. SUPPORTED field presence is not a successful quality/coverage check. DERIVABLE describes future design possibilities; no labels or values were derived now.

## Component conclusions

The categories below apply to the named scope, not to an overall dataset score. PARTIALLY SUPPORTED means a narrower possible research design; missing core gates still block full contract acceptance and any current modeling authorization.

| Component | Freddie Standard | Fannie Primary | Scope restriction |
| --- | --- | --- | --- |
| 12-month PD | PARTIALLY SUPPORTED | PARTIALLY SUPPORTED | Loan-month proxy research; event/coverage/PIT gates unresolved |
| Lifetime PD | PARTIALLY SUPPORTED | PARTIALLY SUPPORTED | Mortgage hazards/competing exits; no automatic maturity or borrower-independence claim |
| LGD | PARTIALLY SUPPORTED | PARTIALLY SUPPORTED | Simplified aggregate disposition-loss research only; no full timed workout/downturn LGD |
| EAD | PARTIALLY SUPPORTED | PARTIALLY SUPPORTED | Mortgage principal-at-declared-event study only; revolving CCF unsupported |
| SICR | PARTIALLY SUPPORTED | PARTIALLY SUPPORTED | Future declared heuristic only; actual recognition/current-risk/policy evidence missing |
| IFRS 9 staging | UNSUPPORTED | UNSUPPORTED | Accounting scope, impairment, recognition and stage decisions missing |
| ECL | UNSUPPORTED | UNSUPPORTED | Full portfolio/accounting loss contract not met; later simplified integration would be assumption-based |
| Stress testing | PARTIALLY SUPPORTED | PARTIALLY SUPPORTED | External macro-vintage/scenario/regime evidence required |

## Access and reproducibility gates

[Freddie's public page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset) requires registration and applicable terms; commercial redistribution has separate licensing. [Fannie access information](https://capitalmarkets.fanniemae.com/credit-risk-transfer/fannie-mae-single-family-loan-performance-data) also requires registration/terms. Its [FAQ](https://capitalmarkets.fanniemae.com/resources/file/credit-risk/pdf/sf-loan-performance-dataset-faqs.pdf) describes internal-use, redistribution/derived-product and re-identification restrictions. Public repository rights must be resolved separately before acquisition or any distribution. No terms were accepted in this review; do not publish raw data, sample records or vendor documents by default.

For later authorized inspection, pin provider/product/release, dictionary, source byte hashes, acquisition timestamp and license decision. Preserve revisions and declared availability assumptions. Check missing/special values, masking, duplicate loan-months, gaps, exposure definitions and completed/censored events/workouts. No empirical quality claim follows from reading a dictionary.

## External replication architecture

A possible later design is Freddie Standard development, Fannie Primary external replication, and separate public macro vintages. Lock Freddie estimands, features, labels, evaluation windows and comparison protocol before looking at replication outcomes. Harmonize dates, masks, events, modifications, entry/exit and population exclusions. Do not tune to Fannie and then describe it as untouched external validation.

Replication can test a prespecified methodology or compatible frozen model, depending on the declared protocol; these are different claims. Different portfolios can share economic regimes and unknown borrowers/properties, so cross-provider replication is not proof of independence or of transportability to consumer/revolving portfolios. Coarse geography/time may support macro linkage, but never borrower re-identification. A separate source would still be needed for revolving exposure research.

## Baseline publication and CI gate

The approved baseline through `398ed7f` was fast-forwarded to both main and the feature branch. [Main CI](https://github.com/Dzidefo123/credit_risk_pred/actions/runs/37484321465) passed Windows and container jobs but failed Linux at one non-frozen CLI CRLF/LF preservation assertion. This initial failed run is not a green baseline. No completion tag is justified until the approved baseline or an explicitly authorized compatibility successor passes all hosted jobs. The proposed fix preserves exact frozen-source checks and does not change recorded model evidence.

## Next decision

Resolve the baseline CI gate and formally approve a **mortgage-only dataset selection and acquisition plan**, including release/version, lawful repository sharing, target/PIT limits and external replication protocol. Freddie remains preferred provisionally; this document does not select/download a source or authorize model implementation.
