# Track B Task 1: mortgage research design

Protocol: freddie_mortgage_research_v1. Prespecified before loan records, 2026-10-06. [Selection decision](DATASET_SELECTION_DECISION.md), [machine-readable protocol](mortgage_research_protocol.json), [77-field evidence](FIELD_DICTIONARY_COMPARISON.md). Documentation only: no ingestion, target construction or model fitting.

## 1. Unit, population and information set

The analysis unit is a disclosed mortgage loan, not an identified borrower. Initial scope is Freddie Standard, using the pinned Release 47 layout family and a separately authorized official sample. Account for acquisition entry, modifications, early masking and provider selection; these records are not all applications, all borrowers or all mortgages. First-payment month is never renamed origination_date. Calendar granularity remains monthly.

Let t0 be the endpoint of a nominal reporting month. X_t0 contains only permitted origination/static information under a declared availability assumption and monthly information from at/before t0. The retained release is a retrospective revised view: actual available_at remains UNKNOWN unless authentic historical release evidence exists. Do not fabricate it by copying observation_date. Preserve release-as-of/acquisition provenance separately. Passing temporal feature-window tests establishes no-future-row use within this representation, not historical real-time knowledge.

Require at least six consecutive known monthly observations ending at t0, continuous ascertainment from first observed entry to t0, and no prior observed qualifying event. Already-defaulted t0 observations are outside incident risk. This establishes first observed event in the disclosed process, not never-defaulted-since-origination. No primary re-entry after unknown/gap intervals; cure/re-default needs a separate future protocol. Eligibility depends on pre-t0 history, never on future maturity or event presence.

The six-month lookback is a prespecified engineering/research choice to exercise behavioral windows and coverage, not a regulatory requirement or a threshold selected for model performance. The 2010 pilot offers a mature vintage to inspect long histories; it is not assumed representative or guaranteed to have sufficient events for modeling.

## 2. Research default and termination semantics

The target is **first observed composite adverse mortgage event**. This name must accompany PD results. It combines a monthly severe-delinquency/REO observation with specified credit-related terminal events, and therefore must not be called solely a 90-day delinquency target or a verified regulatory/accounting default.

| Evidence in a future month | Primary interpretation |
| --- | --- |
| Numeric delinquency code 03 through 99 | Qualifying adverse-state event; use documented monthly bands, never months times 30 as exact DPD |
| RA state | Qualifying REO-state event |
| Terminal code 02, 03 or 09 | Qualifying credit-exit proxy, even if earlier severe-state evidence is absent; event month is first observed qualifying month, not backdated economic default |
| Terminal code 01 without competing adverse-state evidence | Combined payoff/maturity competing event; not pure voluntary prepayment |
| Terminal code 15, 16 or 96 without qualifying adverse-state evidence | Administrative/sale/defect exit; censor at last confirmed risk interval, not default merely because a loss amount exists |
| XX, missing state, missing expected month | Unknown ascertainment; no conversion to current/non-default |
| Unrecognized state/exit code or unparseable date | Fail closed; investigate source/layout and revise protocol explicitly |

These source-code interpretations are release-specific; see the [July guide](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf) and [Release 47 changes](https://www.freddiemac.com/fmac-resources/research/pdf/disclosure-changes-summary.pdf). Research event choice and censoring treatment are our declared assumptions, not provider approval.

Use the first qualifying month across observed continuous history. A later cure does not erase an incident event. Later disposition, recovery or loss does not move the event backward. Original/current score missing codes must not be confused with delinquency codes. A zero balance alone is not a payoff reason.

If a qualifying default state and code 01 occur in the same month, within-month ordering is unknown: classify ambiguous_event_order, retain both observations and quarantine from the primary labeled sample. Do not silently give one cause precedence. Terminal-date/report-month contradictions and evidence of a gap before first event also require quarantine/review. A direct qualifying credit-exit code can establish an event despite an unknown state in the same record, but cannot resolve an earlier unobserved interval. A qualifying state with an administrative sale code remains an observed research event; the administrative code alone is not its cause.

## 3. Twelve-month labels and censoring

For a landmark t0, evaluate monthly offsets k=1,...,12. The interval is (t0, t0+12 calendar months]; t0 events are prevalent, not new positives. A first event at k=12 is in-window; k=13 is outside it. Do not substitute 365 days or infer precise dates from month codes.

| Outcome status | Binary default-by-12 indicator | Evidence requirement |
| --- | --- | --- |
| positive_default | 1 | Valid first qualifying event in months 1..12 before payoff and any earlier ascertainment failure |
| negative_survived_horizon | 0 | Twelve consecutive ascertainable non-event months with no unresolved exit |
| competing_payoff | 0 for the specified default-before-payoff estimand | Verified payoff/maturity before default; preserve cause/time; this is not proof of twelve months of observed survival |
| right_censored | Unknown | Administrative cutoff/exit or first follow-up gap/unknown before horizon; retain last confirmed risk interval |
| insufficient_followup | Unknown | Valid coverage/risk interval cannot be established |
| ambiguous_event_order | Unknown | Same-month incompatible causes or unresolved dates; never resolved from later loss values |
| not_incident_risk_eligible | Not applicable | Prevalent/prior event, failed lookback or unresolved pre-t0 history |

A positive observed before twelve months can be known before full horizon maturity. Payoff extinguishes this facility's future exposure, so the binary competing-payoff value is a structural non-default for the declared facility estimand, not imputed missing performance. It would not be a justified negative for a borrower-default-regardless-of-facility-closure target. Censored/unknown outcomes are not training zeros.

Right censoring is distinct from the observed payoff competing event. Stop primary risk-time accumulation at first ascertainment failure; resumed later rows do not silently bridge it. Administrative censoring can be informative, and a hazard estimator does not cure that assumption. Include exit/censor reasons, unknown counts and sensitivity plans in future reporting. Delayed entry is measured from first observable risk period; exclude claims about unobserved pre-acquisition survival.

## 4. Hazard and competing-risk estimands

For this facility process the first objective is P(T_D<=12, T_D<T_P given X_t0), the default cumulative incidence before competing payoff. In a setting without a competing exit it reduces to P(T<=12 given X_t0). Net default risk under hypothetical elimination of payoff is a different, generally unidentified target and is not the primary output.

At future month k, define h_D(k) and h_P(k) as default and payoff probabilities conditional on the loan being event-free and observed at the interval start. They must be nonnegative and jointly sum to at most one. With the declared causes: S_k = S_(k-1)*(1-h_D(k)-h_P(k)); m_D(k)=S_(k-1)*h_D(k); F_D(K)=sum_(k<=K) m_D(k). Do not use conditional hazard or cumulative probability as marginal default mass in a later ECL sum. Ignoring payoff and treating it as an ordinary noninformative censor estimates a different object.

Two separately declared future designs are possible: fixed landmark covariates X_t0 for horizon forecasts, and dynamic hazards with covariates measured before each interval. Never put information from the interval whose outcome is predicted into its predictors. No hazard classifier, survival model, boosting model or performance comparison is implemented now. Lifetime curves beyond observed support/remaining mortgage life require explicit extrapolation assumptions; contractual maturity is not guaranteed economic exposure life.

## 5. Exposure and loss objects permitted by the fields

Mortgage EAD research initially means a **monthly principal-exposure proxy at the observed research event**. Report observation month, measurement basis, masking/deferral treatment and missingness. Current balance at a removal month may be zero; do not report that automatically as zero exposure at default. Removal principal can support a separately named removal-exposure measure but must not silently replace first-event principal. If event-time balance cannot be reconciled, keep the value unknown; do not fill it with a future/latest/nonzero balance. Fees, accrued interest, contractual commitments and precise event-day balances are not supplied by that proxy. No revolving CCF or undrawn-limit model.

The initial loss object is an **aggregate disposition-loss severity proxy**, not timed workout LGD. If later authorized, use an explicitly defined provider loss numerator and a positive matching removal-principal denominator within the same disposition/release cohort. That denominator is not automatically EAD at the primary event. Separate terminations, defects, unreported components and trailing revisions; no recovery-completion inference from disposition date. No hidden discount assumption, accounting EIR from coupon, or clipping of gains/severity above one. Do not add an already aggregated loss to its component losses or count net selling costs twice.

The reviewed guide notes release-specific signs, incomplete recent loss disclosures and later revisions. Acquisition must preserve those masks and source definitions. Selection into completed/disclosed dispositions is not representative of all incident defaults. Report coverage by loss-eligibility/censoring/exit status, and label any later simplified severity calculation as such. Full timed workout, downturn and IFRS LGD remain prohibited; aggregate proceeds cannot be converted into dated transaction recoveries.

SICR may later compare explicitly reconstructed research risk under declared policy heuristics, with horizon/model-version comparability. Origination FICO alone is not initial lifetime PD; current state is not a matched current probability. Actual bank SICR/stage decisions, accounting recognition/credit impairment, POCI and accounting ECL remain outside current evidence. ECL waits for compatible risk/exposure/loss definitions, scenario/discount choices and dependence/uncertainty review.

## 6. Task 2 pilot and engineering acceptance

Prespecify the official 2010 Standard sample as one pilot vintage. Retain at most 1,000 loan IDs using ascending SHA256(protocol_id + colon + loan_identifier), a UTF-8 deterministic key independent of outcomes, scores, balances and maturity. Break a hash tie by provider ID. Stream and retain all available histories for these IDs. No outcome-enriched sampling, quiet additional vintages or synthetic augmentation. This engineering pilot is not a selected population for validated model-performance claims.

Before any acquisition, confirm access/terms, permitted derived outputs, exact archive/schema/release and resource budget. Preserve hashes and documentation provenance; do not publish raw records or credentials. A future sample-layout mismatch is a stop condition, not permission to guess columns. The entire historical collection is not authorized. The original official sample may contain more records than retained: agree transfer/storage limits separately, then filter without losing selected-loan history.

Task 2 must demonstrate the following with synthetic fixtures and then authorized-record checks, not by fitting a model:

- Unique source/loan/month keys; distinguish exact duplicates from conflicting revisions; no last-row-wins reconciliation.
- Calendar/month parsing, chronology and source-specific reporting-cycle metadata; no fabricated origination/default days.
- Special/masked values retained as missing/flagged, not arithmetic codes; unknown states and gaps never healthy months.
- Feature lineage and t0 cutoffs; future payments, terminal reasons, future peak delinquency, loss totals and future macro releases blocked as predictors.
- Label boundary cases at t0, month 12/13, payoff, gap, administrative exit, prevalent event and ambiguous same-month causes.
- No predictor changes when future rows/losses are changed; future outcomes may change labels only. Preserve the distinction between nominal no-future-row tests and unverified real-time availability.
- Coverage/censoring and event/exit counts with explicit denominators; no label-zero imputation for censored records.
- Reproducible ID selection and schema/source hashes, row accounting and separation of predictor versus outcome/metadata columns.
- Exact Track A tag, frozen model/evidence hashes, historical metrics and consumed-registry preservation; no Track A fits or scoring.

## 7. Future modeling/validation gate

Before fitting, freeze modeling cohorts, temporal splits, feature allowlists and censoring/competing-risk estimands. Repeated landmarks and their overlapping twelve-month outcomes are dependent: purge overlapping outcome windows across chronological boundaries and apply a declared loan-level separation policy for new-loan tests. A dynamic known-loan forecasting test is a different estimand and must be labeled separately. Borrower independence across loans cannot be established.

Chronological modeling comparisons must evaluate discrimination, probability quality and horizon calibration with methods appropriate to censoring and competing risks, plus conditional uncertainty and observed population support. Ordinary AUC on arbitrarily completed survivors cannot silently stand in for that evaluation. Fannie replication follows a locked protocol after Freddie development; no replication-outcome tuning portrayed as external validation. No observed accuracy, profitability, causal macro effect or production readiness claim is made in Task 1.

## 8. Gate conclusion

The selection permits a narrower, explicitly retrospective mortgage research design to proceed to separately authorized engineering. It does not certify all field or modeling assumptions. Protocol revisions require reasons and disclosure of what was already inspected. The next task is small-sample panel engineering, not XGBoost training or IFRS 9 implementation.
