# Track B Task 7 — Macro Identification Design

## Decision and estimand

**MACRO DESIGN READY WITH MATERIAL IDENTIFICATION LIMITATIONS**. Preferred strategy is **A: multiple origination vintages**, with national point-in-time macro data and constrained duration/cohort representation in a new predictive joint competing-risk study. No causal effect, calibrated stress response, lifetime regulatory PD or ECL is established. The [protocol](macro_research_protocol.json) and [provenance report](MACRO_DATA_PROVENANCE.md) govern future authorization gates.

Future estimands distinguish rolling next-month default/payoff hazards given observed current state and known macro information from prospective default/payoff cumulative incidence conditional on scenario and future-state assumptions. They are not interchangeable. Task 6 remains a frozen historical study, including its 54-month training-duration limit and material payoff calibration error.

## Single-vintage confounding and age–period–cohort structure

In an exact age/cohort clock, calendar period = origination cohort + age. With essentially the 2010 annual vintage, older mortgages almost necessarily appear in later economic environments. A coefficient on unemployment alongside age cannot be interpreted as a separately identified stress effect merely because the optimizer returns numbers. Within-year origination differences supply limited overlap, not independent regimes. Freddie first-payment month is not a proven exact origination date; provider loan_age is an observed clock, and study-entry duration is another clock. Do not conflate these.

Even with multiple cohorts the algebraic identity creates the classical APC rank problem for unrestricted age, period and cohort terms. More cohorts improve support but do not eliminate this identity. Saturated calendar fixed effects also absorb national macro series. The preferred future predictive design must constrain duration representation, limit cohort effects and avoid arbitrary unrestricted period effects. Such constraints are scientific assumptions, not a discovered solution or causal instrument. Precise degrees of freedom, penalty choices and interactions require prespecification in the later modeling protocol; no parameters are selected now.

## Survivor selection and underwriting cohort

The 2010 mortgages observed in 2018 are survivors of previous default, payoff and censoring. Entry conditional on survival changes credit quality and refinancing incentives. Task 6 reset its evaluation clock at conditional entry and recorded much older entry ages than development. This does not represent an unconditional origination-life cohort.

Different vintages also differ in underwriting, acquisition policy, origination rates, leverage, credit score distribution, servicing and disclosure histories. Multi-vintage pooling must audit these differences, use only common documented field definitions and maintain source-specific missingness. No borrower independence is proven by a facility identifier. Unknown pre-first-observation defaults, cures and gaps remain research limitations. Harmonization must not retroactively rewrite the consumed 2010 outcome protocol or ledgers.

## Payoff competition

Macro conditions affect both cause-specific hazards and who remains at risk. Lower market rates can induce refinancing/payoff and change the remaining default population. Task 6's aggregate 60-month default incidence was roughly3.10% using AJ versus5.21% naive default-only KM; these are retained results, not newly calculated here. Treating payoff as ordinary censoring targets a different net-risk quantity. Termination01 remains the inherited payoff/maturity proxy; it does not identify pure voluntary refinance.

## Dynamic states and forecasting

For the next observed interval, proposed joint hazards are hD(t | X(t),M(t)) and hP(t | X(t),M(t)); X is current eligible mortgage state and M is information available at t. Macro scenarios alone do not determine future delinquency, balances, modifications or payoff composition.

For prospective month j, a later authorized model would obtain the scenario macro state, generate/assume loan state conditional on survival, obtain jointly valid default/payoff hazards, and update S_j=S_(j-1)*(1-hD_j-hP_j), FD_j=FD_(j-1)+S_(j-1)*hD_j and FP_j analogously. Probability conservation requires hD+hP<=1. These equations are documented, not executed. With stochastic loan-state paths, averaging conditional trajectories requires the joint state/path distribution; plugging mean covariates or averaging hazards and then multiplying survival is generally not equivalent. Interactions and nonlinearities matter.

No observed future states may be fed into a prospective forecast. Freezing current delinquency over an annual path also needs an explicitly labeled assumption and sensitivity study. Future transition modeling, censoring assumptions and state support must be separately reviewed. Existing Task 6 dynamic scores remain rolling next-month diagnostics only.

## Alternatives assessed

| Strategy | Strength | Material weakness | Task 7 choice |
| --- | --- | --- | --- |
| A: multiple origination vintages | Same ages in different periods and different ages in the same period; common Freddie event framework; stronger calendar/seasoning support | APC constraints still necessary; cohort underwriting differences; more data/provenance and regime harmonization | **Preferred**, with national macro and disclosed restrictions |
| B: geography variation | Within-calendar unemployment/HPI variation can help avoid purely national time aliasing | Sparse events; correlated local conditions; historic LAUS/HPI vintages and MSA mapping not certified; borrower geographic selection | Not primary; no state/MSA join now |
| C: constrained structure in current vintage | Small computational footprint and compatibility with existing hazards | Identification comes mainly from imposed age/macro restrictions; weak transport support; constraints cannot be validated across cohorts | Insufficient as the sole preferred design |
| D: historical replay | Transparent descriptive path and conditional prediction interpretation | Observed economic path not an ex-ante forecast; future loan states/censoring and selection unresolved; extrapolation does not identify causality | Possible later sensitivity only |

## Prespecified multi-vintage expansion

Conceptual cohorts: **2006, 2008, 2010, 2014, 2018, 2020, 2022**. The rationale is early housing contraction, crisis underwriting, existing reference cohort, recovery, later rate environment, pandemic originations and transition out of very low rates. Selection is economic/time-based before looking at new outcomes or model performance. The existing 2010 sample remains exactly unchanged.

Freddie's official [dataset page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset) documents historical Standard origination years and monthly performance availability. That establishes conceptual availability, not successful acquisition or sufficient common support. Later cohorts have shorter follow-up; 2022 cannot support the same long horizons as2006. Do not compare maturity-dependent CIFs as if follow-up were identical. A future data task must tabulate age/calendar/event support and censoring, prior to model fitting.

Acquisition should begin with a deliberately small deterministic pilot per additional cohort under a separately approved resource/sample budget, then a harmonization/support decision. No archives downloaded in Task7. The design does not mandate a large full-history extraction or sampling expansion based on favorable outcomes. Estimate archive sizes before approval and stream selected records as in earlier tasks. Freeze sample seeds/ranking and source release/layout; publish aggregate coverage only.

Pre-2010 mortgage cohorts are useful but the conditional primary HPI and mortgage-rate ALFRED metadata start in2010. Historic agency archives/value versions must resolve earlier information sets, or those features/periods must be expressly excluded. Entering older cohorts only at2010 produces survivor-selected groups and does not recover crisis-period evidence. A coverage gate may require narrowing the scientific claim; it must not silently fill the gaps with current revised history.

## Pandemic regime and measurement breaks

2020–2021 are not ordinary unemployment experiments. Payment assistance, forbearance, fiscal transfers, foreclosure/collection interventions and rates can change links between delinquency and loss/default proxies. A later protocol must harmonize disclosed assistance/deferral fields, distinguish missingness from no assistance, prespecify regime indicators or separate regime analyses, and assess transport. A macro coefficient trained on recovery cannot be assumed valid for pandemic conditions. PMMS changed methodology in November2022, adding another measurement boundary.

## Validation and computational feasibility

National macro values have effective replication by calendar date/regime, not by loan-month row. Future uncertainty needs calendar-block methods and regime-aware temporal evaluation alongside facility separation; facility-only bootstrap understates common-macro uncertainty. Shared macro shocks across facilities violate a naive independent-row story. No confidence intervals are generated in Task7.

A new study should freeze calendar windows, purge logic, facility assignment, eligible states and evaluation consumption before fitting. Neither Task5 nor Task6 evaluations become fresh holdouts by adding features. New cohort acquisition is not permission to optimize against already consumed outcomes. Development-only preprocessing and contemporaneous macro vintage selection are mandatory. Future-period and held-cohort evaluation answer different transport questions and should be named separately.

The Task6 joint multinomial competing-hazard representation is a feasible low-dimensional starting architecture. Additional cohorts multiply rows, so aggregate macro versions should be resolved once per assessment date and then joined to risk intervals only after equivalence tests, with controlled memory and no identifier-bearing public output. An explicit age/calendar overlap matrix, cohort feature/missingness diagnostics, actual macro version coverage and event counts must decide feasible horizons. No model family is selected based on new results here.

## Claims and governance gates

Permitted claim now: a tested point-in-time information architecture and a prespecified preferred expansion design exist. Not permitted: a macro stress model has been validated; unemployment coefficients are causal; macro histories are complete; mortgage operational knowledge times are verified; dynamic multi-year CIFs are available; regulatory staging/EAD/LGD or ECL is ready.

Independent future review must approve APC restrictions, macro coverage exclusions, economic consistency of scenarios, cohort/regime harmonization, resource limits and a new validation protocol. [Acquisition/modeling gates](macro_research_protocol.json) are explicit. Exactly one next task: **Track B Task 8 — Multi-Vintage Mortgage Cohort Acquisition and Harmonization**. Task7 stops here without implementing it.
