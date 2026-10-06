# Track B dataset suitability framework

Status: Task 0 assessment design. No candidate selected or evaluated. Apply the [longitudinal contract](LONGITUDINAL_DATA_CONTRACT.md) before any empirical modeling. The [search strategy](DATASET_SEARCH_STRATEGY.md) defines the next task.

Track A remains COMPLETE and unchanged. Its two-year benchmark cannot satisfy missing longitudinal gates or provide a new Track B holdout.

## Separate component decisions, no overall score

Use SUPPORTED, PARTIALLY SUPPORTED or UNSUPPORTED **for a named component and stated estimand/population**. Never average components, award arbitrary numeric points or call a dataset IFRS 9 ready from a combined score.

- SUPPORTED: inspectable documented evidence meets every core gate for the stated research component and use scope. This means data feasibility, not model quality, deployment approval, regulatory suitability or unbiased identification.
- PARTIALLY SUPPORTED: a narrower clearly stated estimand/design meets its own core gates but full requested scope does not; limitations and excluded claims are explicit. Example: contract-life default modeling supported but behavioral-life extrapolation unsupported. A missing core gate cannot be excused by a partial label.
- UNSUPPORTED: a required identity, timing, event, ascertainment or component-specific measurement gate fails or cannot be evidenced. Other components remain independently assessable.
- UNASSESSED is only a workflow state before evidence review, not an acceptance category. Unknown core evidence stays pending/unsupported for modeling; popularity, row count and an attractive AUC cannot substitute for it.

Acceptance requires named source/version, cited dictionary/schema/coverage evidence, legal/reproducibility review, decision owner, unresolved gaps and restricted scope. Task 0 contains no actual candidate ratings.

## Global gates and hard rejection

| Gate | Necessary evidence | Failure consequence |
| --- | --- | --- |
| Lawful accessible use | License/terms, research/redistribution limits, secure access and allowed transformations | No acquisition/use beyond rights; do not bypass registration or restricted access |
| Usable dates | Observation/event time and meaningful calendar/frequency | Reject all proposed longitudinal empirical analyses without genuine dates |
| Stable analysis-unit identity | Facility/borrower keys and documented changes/linkage | Reject individual longitudinal components if no stable unit; profile hashes/row order do not qualify |
| Outcome semantics | Explicit default event/date/definition or derivable dated history | Reject PD/default-conditioned LGD/EAD if qualifying events cannot be identified |
| Point-in-time reconstruction | Event/availability/revision lineage, or justified conservative availability bounds | Reject predictive real-time claims if reconstruction impossible; only a separately labeled retrospective study may remain |
| Ascertainment and exits | Coverage, follow-up cutoff, gaps, closure/transfer/prepayment definitions | Reject binary negative-label claims without completion; reject hazard work if risk intervals cannot be recovered |
| Population and sample lineage | Entry/eligibility, represented products, vintages, reporting and selection mechanism | Restrict estimates to observed scope; unresolved selection prevents transportability claims |
| Integrity/reproducibility | Versioned schema, source fingerprints, units, documented corrections and reproducible transformations | Resolve material inconsistencies before acceptance |

A missing recovery history rejects LGD, not necessarily PD. Missing exposures rejects EAD and monetary ECL, not necessarily event-only PD. Stable facility identity without borrower linkage can support a specifically facility-level limited study, but cannot claim borrower independence or full borrower-linked acceptance. An aggregate time series cannot replace loan-level panels; it may support contextual macro research only.

If access terms cannot be confirmed, do not download. If a data gap is potentially remediable, record what evidence is needed, not a speculative supported rating. No automatic positive decision from dataset scale, dates in filenames or vendor marketing.

## Component acceptance matrix

| Component | Mandatory concepts for full stated research scope | Conditional extensions / quality questions | Reject or restrict when |
| --- | --- | --- | --- |
| 12-month PD | Stable risk unit, t0/PIT predictors, qualifying future event, event scope, reliable follow-up/coverage/exits and eligibility | Interval-censored dates, administrative versus competing exits, censoring dependence, maturity and representative events | No future window or unknown outcome timing; incomplete records never silently negative |
| Lifetime PD | PD gates plus multi-period risk history, entry/exit/censoring, usable remaining-life basis and temporal depth | Delayed entry, recurring defaults/cures, behavioral life, regime coverage and extrapolation | Only one aggregate horizon label; no risk duration; lifetime beyond observed support must be explicitly assumed |
| LGD | Dated default episodes, defined nonzero EAD, allocated timed recoveries/costs, currency, completion/cutoff and loss/discount convention | Collateral priority, cure/debt sales, open-workout censoring, expenses and downturn support | No recoveries or denominator; missing costs/timing may support only declared simplified loss, never full workout/downturn LGD |
| EAD | Product family, t0 drawn balance, default timing/matched event exposure and exposure basis | Term schedules/payments/accrual; revolving undrawn limits/draws/limit changes, cancellations and CCF denominators | No actual exposure at default; event indicator/utilization alone is not EAD; revolving study without limits/draw behavior unsupported |
| SICR | Initial-recognition date/risk evidence, current comparable remaining-life risk, temporal instrument linkage and policy inputs | Rating changes, qualitative/forbearance evidence, model-version and horizon bridges | Only current score or delinquency; a disclosed current-state heuristic is not full SICR evidence |
| IFRS 9 staging | Instrument scope, recognition lineage, defensible SICR evidence, current credit-impaired evidence, dated policy and reasons | POCI/simplified approaches, modifications, cures and overrides | Default flag alone cannot identify all stages; performing alone does not distinguish Stage 1/2 |
| ECL | Compatible scoped PD term structures/marginal mass, default-conditioned LGD/EAD or cash shortfalls, timing/currency/discounting, stage/horizon and scenarios/weights | Dependence, revolving life, prepayments, collateral, scenario calibration, uncertainty and reconciliation to observed loss | Any core monetary/horizon component missing; components from unrelated datasets do not establish one observed portfolio's ECL |
| Stress testing | Dated credit outcomes/components, relevant historical macro/exposure linkage, defined stress target/horizon and justified scenario paths | Macro vintages, regime coverage, tail identification, structural assumptions and distinction from weighted ECL | No macro/credit timing link or insufficient regime support; assumed shocks can be sensitivity demonstrations only |

Temporal depth is assessed relative to the claim: observable 12-month follow-up is not lifetime coverage, a long calendar without defaults is not event support, and a short completed workout history is not ultimate recovery coverage. Record eligible entities, observed/censored durations, events by vintage/product, missingness, gaps and economic coverage if later authorized inspection permits. Do not invent universal minimum sample counts or maturity cutoffs in Task 0.

## Evidence checklist for a later candidate dossier

| Evidence area | What the assessor must capture |
| --- | --- |
| Provenance/access | Publisher/underlying provider, exact version, access route, terms, permissible sharing and authenticated dictionary |
| Unit/linkage | Entity keys, relation cardinalities, transfers, multiple facilities/episodes and mapping completeness |
| Time | Calendar, observation/event/availability dates, frequency, publication lags, revisions and left truncation |
| Outcome | Default definition/scope, derivation, eligibility, cure and repeated-event treatment |
| Follow-up | Ascertainment endpoint, missing reporting intervals, censoring/competing exits and denominator definitions |
| Exposures | Drawn/undrawn/event balances, limits, schedule, currency, accrued amounts and reconciliations |
| Workouts | Recovery/cost/proceeds dates, allocations, completeness, write-off/debt-sale policy and discount basis |
| Macroeconomic linkage | Geography/frequency, real-time vintages, forecast-as-of, paths, horizon, weights and unit transformations |
| Selection | Acceptance/rejection, reporting coverage, exclusions, originations versus survivors and censoring mechanisms |
| Reproducibility | Source fingerprints/version, restricted/public replication plan, derivations and eventual time/entity validation design |

Each conclusion must name the supporting page/schema/table, distinguish verified fact from supplier assertion, and list unresolved questions. Metadata review is not inspection of actual record quality. No source has been empirically accepted in Task 0.

## Assessment record template

Use one row per candidate version and component; do not aggregate statuses.

| Candidate/version | Component + estimand | Status | Evidence references | Passed/failed gates | Scope restrictions | Missing evidence/action | Reviewer/date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Not yet researched | Component to be assessed | UNASSESSED | None yet | Not checked | No modeling authorized | Read-only discovery/dictionary review | To be assigned |

Supplement with license/access decision, source-unit/geography, target/coverage definitions, source compatibility and planned independent validation. A dataset can receive mixed component decisions. Acceptance does not authorize acquisition or model implementation when either is outside the current task.

## Multiple datasets and non-equivalence

Separate populations may support PD/lifetime, workouts/LGD and revolving EAD, while public macro data provides context. Full portfolio ECL requires common event/product/currency/horizon definitions and demonstrated transportability/dependence assumptions. Without linked facility-level evidence, label assembly as a cross-dataset research integration or sensitivity exercise, never an empirical ECL estimate for one real portfolio. Unrelated IDs, similar feature profiles and common month labels do not establish borrower joins.

Synthetic data is acceptable for schema/PIT/censoring/scenario-interface tests only. Synthetic data must NEVER be used to claim empirical model performance. It cannot satisfy missing real recovery, exposure or economic-regime evidence.

## Decision and governance boundary

Hold a recorded review before any next-stage modeling: accept a named component/scope, partially accept a narrower estimand, or reject it with reasons. Document developer response and unresolved gaps. Later approval exercises are simulations, not bank approvals. Accounting boundaries follow the [contract's sourced discussion](LONGITUDINAL_DATA_CONTRACT.md); suitability is a research judgment, not a compliance opinion.

Exactly one next task: **Candidate Dataset Discovery and Comparative Suitability Assessment**, under the [search strategy](DATASET_SEARCH_STRATEGY.md). No downloads or models now.
