# Task 16 — literature grounding and novelty audit

## Executive Summary

Scientific positioning decision: **DISTINCTIVE COMBINATION PLAUSIBLE WITH MATERIAL LIMITATIONS**.

Integration status: **VERIFIED READY FOR COMMIT — Task 16V engineering compatibility amendment applied**.

The evidence supports an empirical validation paper rather than a new mortgage-risk method. Joint mortgage termination, macro conditioning, temporal validation, probability calibration, and refinancing incentives all have substantial precedents. The possible contribution is the documented combination of revision-aware macro information, natural-calendar transport evaluation, joint default/payoff probability assessment, cumulative-incidence consequences, and support diagnostics. Priority is not established. In particular, unavailable details of Bu et al. (2026) cannot be treated as absent methods.

This task did not rerun experiments, inspect new loan outcomes, fit models, or change results. The additive draft retains all 123 numeric bindings and the original abstract. Fannie remains a proposed external replication: `DRAFT_NOT_YET_AUTHORIZED`; Task 13A remains unauthorized. Publication terms review remains required.

## Search Strategy

The structured search covers mortgage termination, survival and competing risks, macro conditioning, information vintages, temporal transport, distribution shift, probability quality, refinancing and burnout. Seventy-one queries are recorded in `docs/paper/literature_search_log.json`, using web discovery followed by publisher, institutional, author-manuscript and official repository verification. This is not a systematic review, and no exhaustive database coverage or retrieval count is claimed.

The candidate log records 41 items: 31 retained references, six excluded/deferred candidates and four predecessor versions. These are candidate records, not 41 unique papers or every returned search hit. Thirty retained references have verified peer-reviewed publication status; OPSurv is explicitly a preprint. Publisher metadata identifies the final publication where an author version supplies inspected content. Online-first and issue dates are distinguished. Missing DOI or pagination is left missing rather than invented. Crossref content negotiation yielded 21 metadata records; four unsuccessful attempts are retained and independently supported by primary metadata. Six additional primary records complete the registry.

The registry, primary verification notes and closest-paper matrix state inspected scope. `UNKNOWN` means unverified, not absent. Tests establish consistency with these snapshots; they cannot independently establish the correctness of every scientific interpretation.

## Literature Buckets

| Bucket | Main anchors |
| --- | --- |
| L1 Classical termination | Deng1996, Deng2000, Schwartz1989, Stanton1995 |
| L2 Competing mortgage risks | Bhattacharya2019, Bu2026, OPSurv2024 |
| L3 Mortgage/credit machine learning | Sadhwani2021, Chen2021, Peng2026, Wang2024 |
| L4 Macro conditioning | Bellotti2009, Breeden2020, Breeden2022, Breeden2023 |
| L5 Temporal evaluation | Li2023, Chen2021, Peng2026, Breeden2023 |
| L6 Shift and transport | Gama2014, Ovadia2019, Roschewitz2025 |
| L7 Calibration and scoring | Brier1950, Gneiting2007, VanCalster2019 |
| L8 Survival and CIF | Aalen1978, Fine1999, Austin2016, Graf1999, Gerds2012, Blanche2013, Heyard2020 |
| L9 Refinancing | Schwartz1989, Stanton1995, Deng2000 |
| L10 Monitoring implications | Peng2026, Gama2014, VanCalster2019 |

Reference identifiers are bibliography keys. Every retained reference is classified CORE or SUPPORTING and has verification sources; no UNVERIFIED reference is admitted to v0.2.

## Classical Mortgage Competing Risks

[Deng, Quigley and Van Order (1996)](https://www.nber.org/papers/w5184) establish a competing-hazard mortgage setting with equity and unemployment. Their [2000 paper](https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00110) treats dependent mortgage termination options and heterogeneity. This defeats component claims about competing default/prepayment events, option-based economic motivation and macro-sensitive mortgage termination. It does not establish that our precise information-timing and evaluation contract was used.

## Modern Mortgage Competing Risks

[Bhattacharya, Wilson and Soyer (2019), author manuscript](https://arxiv.org/html/1706.07677) provides a Bayesian competing-risk approach on Freddie data with posterior model assessment. A comparable exact calendar holdout and revision-aware macro contract were not verified. [Bu, Wang and Yang (2026)](https://www.sciencedirect.com/science/article/abs/pii/S016766872600017X) is the highest-priority unresolved overlap: Freddie data, joint termination, subdistribution modeling, copula dependence and changing housing-price/interest-rate covariates. Neither method family should be described as our invention.

## Freddie-Based Studies

Sixteen closest papers are compared in `docs/paper/closest_paper_matrix.json`. Freddie and combined GSE studies include Bhattacharya, Bu, Breeden, Wang and Peng; OPSurv is a preprint benchmark contribution. Sadhwani et al. use CoreLogic and must not be presented as a Freddie study. Li et al. use LendingClub and provide adjacent lending evidence rather than a mortgage replication. Shared provider data alone does not establish shared event labels, populations or comparability of numerical results.

## Macroeconomic Mortgage Risk

Macro variables are well established in credit survival models. [Bellotti and Crook (2009)](https://www.pure.ed.ac.uk/ws/files/8628798/Credit_scoring_with_macroeconomic_variables_v10_4_1.pdf) supply adjacent credit-card predictive evidence. [Breeden and Vaskouski (2020)](https://www.risk.net/journal-of-credit-risk/7516241/current-expected-credit-loss-procyclicality-it-depends-on-the-model) supplies mortgage evidence using historical macro scenarios. These are predictive modeling references, distinct from merely establishing economic association. Macro conditioning is not novel; particular availability/revision rules require separate verification.

## PIT/Vintage-Aware Information

[Croushore and Stark (2001)](https://www.sciencedirect.com/science/article/pii/S0304407601000720) provides the real-time vintage foundation. Bianchi and Jiao's 2026 sovereign-credit work supplies a credit-wide real-time macro precedent; Breeden's historical mortgage forecast scenarios threaten any broad claim that mortgage research never uses as-of information. Forecast vintages and revised observed-series vintages are different contracts. The search has not settled whether a prior mortgage competing-risk study implements the same observed ALFRED release/revision reconstruction. This remains a material gap, not evidence of priority.

## Temporal Validation

[Peng and Lessmann, inspected author version](https://arxiv.org/html/2601.20533v2) combines Freddie survival prediction, calibration and simulated drift. It is a close threat to broad mortgage-transport claims. Its documented simulated drift design differs from our frozen natural-calendar evaluation; extending to macro variables and competing prepayment is discussed as future work in that version. Li, Chen and Breeden add credit or mortgage temporal assessment precedents. We did not establish a prior empirical result with the exact same development-improvement/later-score-deterioration pattern and contract, but temporal holdouts themselves are standard practice.

## Calibration and Proper Scoring

[Gneiting and Raftery (2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) grounds proper scoring; Brier supplies foundational attribution. Van Calster, Ovadia and Roschewitz establish calibration and shift concerns. Heyard supplies especially close discrete-time competing-risk assessment including discrimination, probability quality and calibration. Thus the scoring combination and discrimination/calibration distinction are established. General shift evidence supports the possibility of divergence; the search does not certify that every cited paper demonstrates our exact AUC-up/Brier-worse mortgage pattern.

The role of competing hazards in cumulative incidence is an established mathematical property. Austin, Fine–Gray and related references explain the estimands; our fitted multinomial hazard is not a Fine–Gray model. The possible distinctive contribution concerns the observed transport pattern and magnitude, not the CIF identity.

## Refinancing/Prepayment Literature

[Stanton (1995), author manuscript](https://faculty.haas.berkeley.edu/stanton/pdf/prepay.pdf) treats refinancing costs, heterogeneous exercise and selection/burnout. Schwartz–Torous and Deng provide further option-based antecedents. Contract-versus-market rate incentive is established. The Task 12 proxy is an exploratory test within a previously inspected population; it does not identify individual refinance motives or establish a new feature family. Omitted burnout is a material literature-informed limitation.

## Closest Papers

| Work | Verified overlap | Important unresolved difference |
| --- | --- | --- |
| Bu2026 | Freddie; default/prepayment; housing prices/rates; copula/subdistribution joint model | Full sample, event rules, PIT/revisions, temporal splitting, proper scores, calibration and transport diagnostics unknown |
| Bhattacharya2019 | Freddie Bayesian competing hazards | Exact information-time and temporal scoring contract unverified |
| Peng2026 | Freddie survival, drift and calibration | Simulated drift/default focus; documented extensions differ from our joint natural-calendar question |
| Heyard2020 | Discrete competing-risk assessment and calibration | Clinical setting; establishes methods rather than mortgage finding |
| Breeden2020/2022/2023 | GSE macro, discrete-time and temporal stabilization research | Exact observed-macro vintage and joint evaluation combination unverified |
| Bianchi2026 | Real-time credit macro information and later instability | Sovereign CDS, not mortgage competing events |

The JSON matrix contains all sixteen entries and source-specific scope. Bu's unavailable fields remain UNKNOWN. The comparison does not claim the broad macro model or our event definitions are absent from Bu's full text.

## Novelty Threats

The strongest skeptical objection is that the study assembles established components on a widely studied dataset, and reports a negative transport result in one research cohort. Recent work already joins mortgage survival, macro conditioning, drift and calibration. Outcome-exposed diagnostics and exploratory refinancing cannot repair the lack of an independent confirmation. Bu's unavailable full text may substantially overlap with the remaining combination.

This objection defeats methodological invention and priority claims. It does not erase the documented negative empirical result. The remaining case is a transparent evaluation of a specific joint-probability failure with appropriately limited inference. It must survive closer literature inspection and hostile scientific review.

## What Is Not Novel

N1 competing risks, N2 Freddie use, N3 macro predictors, N5 temporal evaluation, N6 proper scores, N7 calibration, N8 ranking/probability divergence, and N11 refinance-gap features are ESTABLISHED. N4 PIT construction, N10 support diagnostics and N12 governance are INCREMENTAL in this context. N9 empirical CIF propagation under temporal shift is POTENTIALLY_DISTINCTIVE; its mathematical basis is established. The combination is provisionally DISTINCTIVE_COMBINATION, with priority explicitly false.

## What May Be Distinctive

The integrated empirical chain may be distinctive: information available at assessment time; development versus forward probability quality; joint default/payoff scoring and calibration; competing-CIF consequences; and diagnostic support comparisons. The scoped search did not identify a fully verified exact match, but incomplete access is a reason to qualify this result. It is not a first-paper claim.

## Revised Contribution Positioning

1. A reproducible PIT-aware temporal evaluation of mortgage macro augmentation within a competing-risk research contract.
2. A documented distinction between payoff ranking and probability quality during transport.
3. Quantified competing-CIF consequences using the frozen results, without causal attribution to individual macro variables.
4. Post-hoc characterization of temporal support failure, with exploratory refinancing evidence explicitly separated from confirmation.

The paper should lead with an empirical model-risk question rather than an algorithm invention. No new numerical result is asserted by this literature audit.

## Unresolved Literature Questions

- Obtain Bu's full published text legally and complete sample/event/evaluation/PIT cells before asserting a difference.
- Determine whether mortgage observed-macro vintage reconstruction already appears in closely related studies; historical forecast scenarios alone do not settle this.
- Complete author review of conditional-entry and IPCW implementation compatibility. Foundational references are not implementation validation.
- Separate observed covariate support shift from established concept drift; current diagnostics do not identify a causal mechanism.
- Assess dynamic borrower information, local macro variation, burnout, true current coupon, unobserved refinance motive and dependence assumptions as limitations; do not add empirical features in this task.

CG01, CG02, CG04 and CG05 have verified contextual support. CG03 and CG06 remain PARTIALLY_RESOLVED. All six retain explicit residual limitations and citation keys in `citation_gap_resolution.json`.

## Recommendation

Prepare Task 17 as hostile scientific review, with emphasis on novelty overlap, conditional entry, censoring assumptions, temporal sample composition, exposure of diagnostics to outcomes, and publication/data terms. Do not begin Task 17 here.

## Task 16V engineering compatibility amendment

The original full-suite run recorded 1,253 passes, one failure and four warnings. Task 14's file-only guard rejected the required `docs/paper/literature/` directory as `Unreviewed paper file type`. The user subsequently authorized an engineering compatibility amendment while expressly retaining the scientific freeze.

`check_paper_evidence.py` now admits only seven named Task 16 JSON files. Unlisted files, nested directories and symlink additions fail. Every admitted file still undergoes the existing public-text scan. Frozen scientific artifact hashes are checked exactly as before. The sole code-hash exception within that checker is its own explicitly approved original/amended hash pair; the historical Task 14 manifest is unchanged.

The existing three-line `references.bib` allowance in `check_manuscript.py` is necessary because Task 16 requires a bibliography outside the earlier Markdown-only workflow. It affects only that filename; Task 16 separately requires the bibliography to equal the verified registry exactly. It does not permit altered empirical results or bindings.

Downstream Task 13T, session-amendment and Task 15 preservation checks also pinned the old checker bytes. Their engineering adapters now recognize exact approved code hashes while continuing to compare scientific evidence against original manifests. The approval record contains six existing-code hash pairs: `scripts/check_paper_evidence.py`, `scripts/check_manuscript.py`, `scripts/fannie_provider_provenance.py`, `tests/test_manuscript.py`, `tests/test_fannie_provider_provenance.py`, and `tests/test_fannie_session_amendment.py`. It is stored separately in `task16_compatibility_amendment.json`; no historical manifest or verification report is rewritten.

Eleven new regressions verify legitimate additions, rejection of mutations to the frozen claim/design/boundary/figure/table artifacts, rejection of unlisted files and directories, retained privacy scanning, and rejection of further checker edits. The expanded targeted run passed 138 tests. The intermediate full run found the two downstream stale-code checks; those have been adapted. The final full suite passed all 1,265 tests with the same four known non-failing warnings. Lint, formatting, configuration, governance, the 581-file staged hygiene audit, numeric binding validation and preservation checks passed. The original failed run and intermediate compatibility run remain recorded in `task16_verification.json`.

All 123 bindings, the original abstract, v0.1, Task 14 scientific evidence, empirical artifacts, archives and governance boundaries remain unchanged. No experiment was rerun, no model regenerated, and no holdout consumed. This compatibility work does not upgrade the scientific positioning decision or resolve the remaining literature questions. No push is authorized.
