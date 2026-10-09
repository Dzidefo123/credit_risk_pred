# Task 20 — Hostile Manuscript Review of v0.3

**Review target:** `paper/main_v0.3.md` at `1a42d505c6652fecf384d00b2d19d737c4b8524a`
**Manuscript sha256:** `dd7738c5726b27b703747b253f5e641bb9e3b32ba8507a18d2ba375ba765f17a` (verified)
**Repository state:** HEAD matches, working tree clean, all six required files present, all 21 imported Task 17/18 artifacts reachable.
**Scope:** review only. No manuscript or empirical artifact modified.

---

## 0. Independence declaration — read this first

**This reviewer is not independent, and the brief's most important design feature cannot be satisfied by me.**

I authored Tasks 17 and 18 in the same working session and carry their conclusions in context. The brief states "You did NOT write this manuscript" and "Do NOT begin from the conclusions of Tasks 17–19." I cannot comply with either. Section 2's provisional cold read — write a review from the manuscript alone before opening the prior artifacts — is **NOT_SATISFIABLE_BY_THIS_REVIEWER**, and its stated purpose ("whether the manuscript itself communicates the correct scientific scope without relying on repository history") is exactly what I am least able to judge, because I already know the history.

What remains valid: the technical audit. Prior knowledge of the evidence helps rather than hinders estimand analysis, independent re-derivation of the numbers, verification of the evidence map, and detection of findings the prior tasks missed. Several of the concerns below are new and none depends on my having read v0.2.

What is invalid: any claim from me that the paper does or does not communicate its scope to a fresh reader.

**Recommendation:** have §2 executed by a reviewer with no exposure to this project — a fresh session, a different model, or a human given only `main_v0.3.md` and the bibliography. Concerns **M5** and **M7** below are precisely the ones a cold reader would weigh differently, and I flag them without being able to settle them.

---

## 0a. Corrections after audit (2026-10-09)

A user-side audit of this review upheld its main disclosure concerns and found four of my interpretations wrong. All four were verified against the sources before amending; all four are accepted. Ledger: `task20_corrections.json`. The pre-correction review is preserved in history at `ee5049a`.

| ID | Original statement | Correction |
|---|---|---|
| T20-C01 | Minor 6: Table F compares different functionals "without a note" | **Withdrawn.** The Table F caption states that the observed reference is entry-conditioned competing incidence and the model columns are mean historical-path projections, and §3.6 names the Aalen–Johansen reference. |
| T20-C02 | Minor 5: §8 should disclose that the local closure cannot be re-executed | **Withdrawn.** §3.6 and §8 already disclose that private arrays are not distributed and that verifying public arithmetic is distinct from reproducing private-data processing. |
| T20-C03 | "Seven macro coefficients" / "seven parameters" | **Corrected.** Seven macro *predictors* in a three-class multinomial with a reference class carry two cause-specific coefficients each: at least 14 macro coefficients. The error originated in my Task 17 report and was carried here; v0.3 never made it. **Superseded by T20-C05.** |
| T20-C04 | M6: M1's unseen-vintage default calibration is "near-ideal" | **Corrected.** A mean and slope close to observed values (0.000858 vs 0.000901; slope 1.0091) are weak-calibration evidence in the Van Calster et al. hierarchy [VanCalster2019], not complete calibration. M6's substance — default calibration is never reported, and M2 under-predicts unseen-vintage default incidence 3.5× with slope 0.7639 — is unaffected. |

Net effect of round one: major concerns unchanged at 7; minor concerns 7 → 5; interpretation errors in the claim audit 2 → 1. No decision changes.

### Second round (2026-10-09, full audit package)

The packaged audit found further errors in the review, including in my round-one correction T20-C03. All factual points were verified against the repository before amending.

| ID | Original statement | Correction |
|---|---|---|
| T20-C05 | T20-C03: "at least 14" coefficients, possibly more via macro missing indicators "not verifiable from public artifacts"; "~88 serially-dependent observations" as effective support; "strengthens the concern" | **Corrected.** Public code settles it: missing indicators are built only for mortgage numerics, and non-finite macro values are rejected. The frozen M2 feature order (37 features) has no macro `:missing` columns. Exactly **14 cause-versus-no-event macro contrasts** and **21 stored class coefficients**. Penalized effective degrees of freedom are not estimated. 88 distinct development months is not an estimated effective sample size, and the corrected count does not by itself quantify how much stronger any overfitting argument is. |
| T20-C06 | M3: within-stratum gains are "resolution-free" | **Withdrawn.** +0.00636 within year and −0.00040 within supported months condition on different comparisons and populations; neither is invariant. |
| T20-C07 | M3 and abstract audit set month 1.979%/98.021% beside year 1.81%/98.19% as if the same quantity | **Corrected.** At month level 1.979%/98.021% are *pair weights*; the *gain contributions* are −0.01633% and 100.01633%. Weights and contributions are reported separately below. |
| T20-C08 | M2: missing calendar AUC intervals make the AUC claim "an internal inconsistency"; ARXIV_BLOCKER | **Downgraded to MODERATE.** The decomposition is an exact descriptive identity of a fixed dataset and needs no interval to be correct. Missing calendar uncertainty limits inferential and generalisation claims; disclosure is the required fix, a new interval is optional registered work. No arXiv blocker remains. |
| T20-C09 | M4: 2020 is "the only period in which national macro variables moved substantially"; M2 is "never materially better" | **Premises withdrawn.** Frozen Task 11 summaries show 2022 mortgage rates of 3.11–7.08 and Treasury rates of 1.55–4.02, with 2023 mortgage rates reaching 7.79. No materiality threshold is registered; 2021's −0.00608 is −4.5% of that year's M1 score. The recommendation to discuss both readings stands, without those premises. |
| T20-C10 | M6 presented default calibration as uniformly worse under M2 | **Corrected.** In seen vintages M2's absolute mean error improves (0.000475 → 0.000239) while its slope worsens (0.482 → 0.350). The full calibration vector should be reported, not a blanket claim either way. |
| T20-C11 | §10: period ranking is economically "negligible" and "not actionable because the calendar is observed" | **Withdrawn.** Economic usefulness was not evaluated; an observed calendar does not make its association with risk automatically non-actionable. |
| T20-C12 | §9, minor 1, §18: "'transport' is the wrong word", conflicts with §4.5 | **Softened.** Prediction-model validation literature uses transportability for performance assessment beyond the development setting; lack of causal attribution does not by itself rule the term out. It is a terminology preference, not a contradiction. Same-provider, non-virgin comparison must still not be called independent external validation. |
| T20-C13 | Abstract sentence 1 TOO_BROAD; central claim "C — OVERSTATED as written" | **Reclassified.** "Can distinguish" is a modal statement of possibility that the study does demonstrate, a conceptual motivation rather than a universal assertion. Tighter wording is editorial, not a scope error. |
| T20-C14 | Several statements of certainty | **Qualified.** "No target leakage" means none *identified*, not proof of an operationally available information set. The 143-statement pass is a reviewer attestation: its working file was not committed, so the sequence is not independently established. DISTINCTIVE_COMBINATION does not upgrade the frozen Task 16 conclusion. "Interval" for the 8-block scheme is a presentation choice, not a mathematical error. Decisions and blocker labels are this reviewer's judgments, not objective gates. |

Net effect of round two: concerns M2 and M3 downgraded to MODERATE (major 7 → 5); arXiv blockers 1 → 0; claim-audit interpretation errors 1 → 0 and scope errors 2 → 1. Decisions unchanged, and labelled as judgments.

---

## 1. Summary

The manuscript is a substantial improvement on what the evidence record implies it replaced, and its quantitative spine is the most rigorously verified I have examined. I attacked the numbers in two passes and they held completely: **143 of 143** headline statements re-derived from frozen artifacts (a reviewer attestation — the working file for this pass was not committed, T20-C14), and **257 of 257** evidence bindings resolved against their own named artifact and pointer with exact values matching to 1×10⁻¹². Zero orphans, zero value errors, zero source errors. Every binding carries a source hash and commit.

So this review cannot attack the arithmetic. It attacks emphasis, estimand discipline in three specific places, and one unexamined argument that I think is the paper's biggest missed opportunity.

**Five major and two moderate concerns** (seven raised as major; M2 and M3 downgraded after audit, T20-C07/C08). The most consequential: a robust 7-point fall in **default** discrimination is absent from the abstract and from the uncertainty table, while the payoff-AUC gain that the paper spends its longest results subsection dismantling is in both. For a credit-risk audience the loss-relevant event's discrimination is the headline, and it is buried.

**No fatal flaw.** I searched specifically and found none.

## 2. Overall recommendation

| | Decision |
|---|---|
| **Preprint** | **REQUIRES_MAJOR_REVISION** |
| **Peer review** | **BORDERLINE** |

These are not calibrated to each other. These are this reviewer's judgments, not objective gates (T20-C14). The preprint verdict is major-revision because the abstract omits a principal result and default calibration is unreported — but **every major concern is resolvable by manuscript text and by reporting frozen values that already exist.** No new computation is required for any of them. That is an unusually good position for a major-revision verdict.

The peer-review verdict is borderline rather than weak-reject because the contribution is real but narrow, the novelty is combinational, and two of the closest papers remain incompletely compared.

## 3. Central contribution

**Derived from the evidence, not the authors' wording:** in one frozen monthly competing-risk specification on one Freddie cohort, adding a national macro block raises a pooled payoff concordance statistic almost entirely through cross-period pairs while degrading probability quality, and the direction of that degradation does not hold across origination-vintage populations.

**Central claim classification: B — SUPPORTED_WITH_IMPORTANT_QUALIFICATION**, read at the scope §§3–4 establish. *(Corrected per T20-C13: an earlier version classed abstract sentence 1 as C — OVERSTATED. Its "can distinguish" is a modal statement of possibility that the study demonstrates, not a universal assertion; tighter wording is editorial.)*

The genuinely transferable contribution is methodological: pooled concordance can improve through period separation alone, so validation should report the stratified decomposition. That lesson generalises beyond mortgages.

## 4. Major strengths

1. **The evidence chain withstands independent verification.** 257/257 bindings resolve exactly. This is not a governance ornament; it let me attack interpretation without wondering whether the numbers drifted.
2. **Estimand discipline is mostly excellent.** The interval-versus-pair shift (§4.4), the eligible-month subpopulation (§4.4), the retrospective-not-prospective CIF framing (§3.6, §4.6) and the facility-versus-calendar uncertainty distinction (§4.7) are all stated explicitly. Most papers state none of these.
3. **The PIT retreat is exemplary.** "Vintage-aware and revision-aware… exact provider first-release timestamps were not certified" is precisely what the evidence supports, stated in the abstract rather than buried.
4. **Negative and countervailing results are disclosed**, including the 2021 sign reversal, the unseen-vintage reversal, and the ex-2020 interval containing zero.
5. **The erratum culture is real.** §3.3 and §5 record a withdrawn structural argument and two corrections rather than quietly dropping them.

## 5. Major concerns

### M1. The default-discrimination result is buried — MAJOR

Default AUC falls **0.693508 → 0.622831**, a change of **−0.070676** with facility interval **[−0.093906, −0.045251]**, which excludes zero. It persists on unseen vintages (0.777935 → 0.736496, −0.041439).

This is **larger in magnitude than the payoff-AUC gain the abstract does report**, it concerns the event that generates loss, and it is robust under the only resampling scheme applied to it. It appears once in Table B, once as a four-word clause in §4.5 ("as does default AUC"), and **nowhere in the abstract, §4.2, Table G, §5 or the conclusion**.

I do not allege cherry-picking: this result *supports* the paper's thesis, so omitting it is not self-serving. The problem is that the paper organises itself around a methodological puzzle (why did pooled payoff AUC rise?) and in doing so under-reports the finding a model-risk reader would act on first.

**Fix, manuscript-only:** add default AUC to the abstract, add its interval to Table G, and give it a paragraph in §4.2.

### M2. No calendar-unit uncertainty is reported for either AUC — MODERATE *(downgraded from MAJOR, T20-C08)*

§3.5 and §4.7 argue, correctly and repeatedly, that facility resampling "condition[s] on the realized shared calendar" and that calendar-dependent claims need calendar-block assessment. Table G applies both units to joint log loss and payoff Brier.

**Neither AUC has any calendar-block interval frozen.** `paired_calendar` contains only `joint_log_loss`, `default_brier`, `payoff_brier`. Facility AUC intervals exist but are also absent from Table G.

The decomposition in §4.4 is an exact descriptive identity of a fixed observed dataset; it does not need an interval to be correct, and v0.3 does not claim it estimates macroeconomic sampling uncertainty. *(An earlier version called this an internal inconsistency and an arXiv blocker; withdrawn per T20-C08.)* What the absence limits is inference: any statement that the AUC pattern would recur over other calendar paths is unsupported.

**Fix, manuscript-only:** state beside the AUC results and in Table G that calendar-block AUC uncertainty was not estimated, and report the existing facility AUC intervals labelled as conditional on the realized calendar. Computing calendar intervals is optional, separately registered work.

### M3. Decomposition shares depend on resolution and population, and should be presented that way — MODERATE *(downgraded, T20-C06/C07)*

The decomposition identity is exact and I verified it. v0.3 labels the year decomposition as year-stratified and defines the identity at a chosen resolution. The point is presentational: pair weights and gain contributions are different quantities, both depend on resolution, and the two decompositions are on different populations.

| Quantity | Year, full primary population | Month, eligible-month subset |
|---|---|---|
| Within-pair weight | 17.195% | 1.979% |
| Between-pair weight | 82.805% | 98.021% |
| Within gain contribution | 1.808% | −0.016% |
| Between gain contribution | 98.192% | 100.016% |
| Within-stratum gain | +0.00636 | −0.00040 |
| Pooled gain of that population | +0.06044 | +0.04831 |

The month between-contribution exceeds 100% because the within-month contribution is negative. *(Corrected per T20-C07: an earlier version set the month pair weights beside the year contribution shares as if they were the same quantity.)*

Finer strata leave fewer within-stratum pairs, so pair weights shift with resolution. **Neither the contribution shares nor the within-stratum gains are resolution-free.** *(An earlier version called the within-stratum gains resolution-free; withdrawn per T20-C06.)* Because the populations also differ, these comparisons do not isolate resolution alone.

**Fix, manuscript-only:** report pair weights and gain contributions separately; name the resolution and population with every share; describe gains as "within-period gain at the stated resolution and population."

### M4. The paper does not confront the strongest argument against it — MAJOR

§4.3 establishes that 2020 supplies 89.34% of the deterioration and that ex-2020 the difference is +0.00217 (2.8% relative, interval containing zero). §5 and §6 treat this as a **scope limitation** — "restrict generalization to other regimes."

A hostile referee can read it the other way. A model meant to respond to economic shocks failed badly in a shock period, and handling such periods is a stated purpose of macroeconomic conditioning. On that reading the concentration is not mitigating. Both readings are bounded model-risk interpretations; neither identifies a cause or a business consequence.

*(Corrected per T20-C09.)* An earlier version rested this on two unsupported premises. "Only 2020 saw substantial macro movement" is false: frozen Task 11 summaries show 2022 mortgage rates of 3.11–7.08 and Treasury rates of 1.55–4.02, with 2023 mortgage rates reaching 7.79. "M2 is never materially better" has no registered materiality threshold; 2021's −0.00608 is −4.5% of that year's M1 score, and the unseen population has a countervailing joint-loss result. Descriptively, the large 2020 deterioration did not recur in the 2022–23 rate rise (deltas +0.00581 and +0.00750) — an observation, not an explanation.

The manuscript never states this. Its careful hedging ("not disappearance outside the pandemic", "a dominant contribution in this observed calendar, not an estimate of a COVID effect") keeps it from overclaiming, but also keeps it from making the argument that would most strengthen it. The ex-2020 sensitivity is currently used in one direction only.

**Fix, manuscript-only:** add a paragraph to §5 presenting both readings and declining to adjudicate. This is the paper's largest available gain in significance.

### M5. Internal process vocabulary throughout the manuscript — MAJOR

The manuscript body refers to **Task 17** (§3.3, §5 twice), **Task 19** (§4.3), **Task 18** (§5), **Task 13A** (§6), **CG03** and **CG06** (§6), and uses the tokens `MIXED_TEMPORAL_TRANSPORT_ACROSS_POPULATIONS`, `ROBUST_FACILITY_ONLY`, `EXPLORATORY_ONLY`, `DRAFT_NOT_YET_AUTHORIZED` and `DISTINCTIVE COMBINATION PLAUSIBLE WITH MATERIAL LIMITATIONS` as if they were established terms.

"Task 19 does not regenerate it," inside a results section, is workflow narration, not science. A referee has no idea what Task 19 is. The **substance** belongs in the paper — an erratum exists, a correction was made, a classification was reached — but the identifiers do not.

This is also the concern my compromised independence most affects: these tokens are transparent to me and opaque to a cold reader. I flag it; a fresh reviewer should weight it.

### M6. Default calibration is never reported, and it undercuts the unseen-vintage reversal — MAJOR

§4.2 reports payoff calibration in detail (observed 1.536%, M1 1.138%, M2 2.830%, slopes 0.509 and 0.249). Default calibration appears nowhere, in either population. From the frozen artifacts:

| Population | Model | Observed | Mean predicted | Abs. mean error | Intercept | Slope |
|---|---|---|---|---|---|---|
| Seen | M1 | 0.001125 | 0.000649 | 0.000475 | −2.935 | 0.4821 |
| Seen | M2 | 0.001125 | 0.000886 | **0.000239** | −4.049 | 0.3501 |
| Unseen | M1 | 0.000901 | 0.000858 | 0.000043 | 0.109 | 1.0091 |
| Unseen | M2 | 0.000901 | **0.000258** | 0.000643 | −0.580 | 0.7639 |

On unseen vintages M1's mean prediction and slope are close to observed values — weak-calibration evidence only, not complete calibration (corrected per T20-C04) — while **M2 under-predicts default incidence by a factor of 3.5**. §4.5 says "Default and payoff Brier still favor M1, as does default AUC" — and omits the most damaging item. The slight unseen joint-loss advantage is therefore not a uniform default-risk improvement. Calibration is not uniformly worse under M2 either: in seen vintages its absolute mean error improves (0.000475 → 0.000239) while its slope worsens (0.482 → 0.350). *(Corrected per T20-C10.)*

**Fix, manuscript-only:** report the full default calibration vector — mean, mean error, intercept and slope — for both populations in §4.2 and §4.5, keeping improvements and deteriorations visible.

### M7. No figures — MAJOR for presentation

Seven tables, zero figures. The paper's central finding is a decomposition, and the per-year contribution plot is the single most informative display available from the existing frozen data. A reader must reconstruct the 2020 concentration mentally from Table C's eight rows.

Minimum set, all from existing values: (i) per-year contribution bars with interval weights; (ii) pooled / within / between AUC with the pair-weight annotation; (iii) the four-horizon CIF comparison for both causes. No new computation.

Again a cold-reader judgment I am poorly placed to make, but the absence is objective.

## 6. Minor concerns

1. **Title.** "Regime-Dependent" is not established — there is no regime model, change-point test or formal regime definition; the evidence is calendar-year concentration. "Transport" is a terminology preference rather than an error: prediction-model validation uses transportability for performance beyond the development setting (T20-C12). If kept, it must not suggest independent external validation. Possible alternative: *Calendar- and Population-Dependent Behaviour of Vintage-Aware Macroeconomic Features in Mortgage Competing-Risk Models*.
2. **VIFs omitted.** §4.3 cites condition number 69.1 but not the development VIFs — unemployment 10.21, Treasury 11.56, mortgage rate 11.57, HPI growth 11.22. The VIFs are more interpretable and strengthen the paper's own instability argument.
3. **Effective macro sample size never stated.** §3.4 says 88 distinct macro months and that rows do not create independent observations, but does not state the parameter count: seven macro predictors give **14 cause-versus-no-event contrasts (21 stored class coefficients)**; the frozen specification has no macro missing indicators. 88 distinct months is not an estimated effective sample size, and penalized effective degrees of freedom are not estimated. *(Corrected per T20-C03 and T20-C05; originally "seven coefficients", carried from my Task 17 report.)*
4. **"Interval" for the 8-block scheme.** A presentation choice, not a mathematical error (T20-C14); v0.3 already qualifies coverage and conditioning. With eight blocks, one partial, percentile coverage is poor, so consider reserving "interval" for the 5,619-cluster facility scheme and using **"calendar-block sensitivity range"** for the other. The paper already says "sensitivity" in prose; Table G's "lower/upper" headers undo it.
5. ~~**22 of 257 statements rest on an unreproducible run.**~~ **Withdrawn (T20-C02).** §3.6 and §8 already disclose the private-data reproduction limit.
6. ~~**Table F comparator mismatch.**~~ **Withdrawn (T20-C01).** The caption labels both quantities and §3.6 names the observed estimator.
7. **Bu2026 comparison still incomplete** (§2, §6). Survivable for arXiv, not for a journal where the referee may be an author.

## 7. Statistical design

Sound, with one unresolved tension the paper itself names. §3.1 declares facilities the loan-level statistical unit; §3.2 weights intervals equally. Because surviving longer is the complement of the modelled payoff event, the weighting is **outcome-correlated** — a facility contributing 86 intervals is by construction one that did not pay off. §3.2 states the issue and that a facility-weighted sensitivity is unexecuted. That disclosure is adequate for a preprint; a referee will want the sensitivity.

Pseudo-replication risk is **acknowledged and correctly handled in the text** (§3.4, §4.7) but only partially in the reporting — see M2. The paper never claims macroeconomic sampling uncertainty, which is right.

## 8. Temporal design

**I tried to break this and could not.** Development 2010-09–2017-12, purge 2018, evaluation 2019-01–2026-02, with facility roles deterministically disjoint — so the split is both calendar-separated *and* facility-disjoint, which is stronger than most temporal validation and stronger than the paper claims for itself. No facility history bridges the purge because no facility appears on both sides.

No target leakage identified — which is not proof of an operationally available historical information set; v0.3 keeps mortgage knowledge-time and macro-release limits (T20-C14). Macro inputs are vintage-restricted, loan covariates are origination-only, current state is excluded from predictors while remaining in ascertainment (§3.3 states this distinction precisely). CIF paths use realized future macro and the paper says so four times.

"Vintage-aware" is **not** assumed to mean point-in-time safe — §3.4 is explicit that release lags were not certified. Correct.

Residual: horizon truncation and censoring depend on an unverified conditional-independent-censoring assumption, and CG03 remains open.

## 9. Population and transport

Survivor conditioning is disclosed thoroughly (§3.1, §3.2, §5, §6). The seen cohort is 2006–2014 originations surviving unprepaid to 2019 — a burnout-selected population — and the paper says burnout is omitted.

The two populations differ simultaneously in origination vintage, seasoning at entry, survivor selection, calendar exposure (a 2022-vintage facility cannot contribute intervals in 2020), cohort encoding (frozen zero-reference fallback) and duration support, so no difference can be attributed to any one of them; §4.5 says so. Lack of causal attribution does not by itself rule out a scoped predictive transport comparison, so "transport" versus "population sensitivity" is a terminology choice. *(An earlier version called "transport" the wrong word and in conflict with §4.5; softened per T20-C12.)*

`MIXED_TEMPORAL_TRANSPORT_ACROSS_POPULATIONS` is justified as a *description* of divergent observed comparisons, and the paper is careful that it does not displace the original primary decision. But it inherits the "transport" problem and is an internal token (M5).

## 10. Pooled discrimination

The decomposition is exact, verified, and the most distinctive methodological element in the paper. Terminology (brief §8):

| Candidate | Assessment |
|---|---|
| "between-period discrimination" | Conflates two things: ranking periods by risk level, and the contribution of cross-period pairs to a pooled statistic |
| "calendar-regime separation" | Imports "regime", which is unestablished |
| "temporal ordering" | Suggests sequence rather than concordance |
| **"cross-period concordance"** | **Recommended** for the statistic — names the pair structure, implies no capability |
| **"calendar-level separation"** | **Recommended** for the model behaviour producing it |

Economic usefulness was not evaluated, and period-related risk information is not automatically non-actionable just because the calendar is observed. *(An earlier version called it negligible; withdrawn per T20-C11.)*

## 11. Probability quality

Metric hierarchy is the right question and the answer is mostly reassuring. Joint log loss was the **pre-specified** primary decision metric in the frozen protocol, not selected after seeing results — so its primacy is not post-hoc. The manuscript nonetheless declines to treat it as a complete summary (§3.5) and reports it beside cause-specific scores and calibration. Good.

§4.2's "No-event prevalence is not itself the fraction of total log loss attributable to no-event observations" is a correct and welcome refinement of a loose earlier claim.

Two gaps: default calibration entirely (M6), and no base-rate reference Brier, which would show how little room either model occupies on that scale.

## 12. CIF analysis

**Classification: B — SUPPORTING RESULT**, correctly placed in main text but not independent evidence.

The measured entry distribution settles it: **5,564 of 5,619 landmarks (99.02%) enter in 2019-01**, 5,615 in calendar 2019, latest entry 2020-01, nine distinct entry months. 24-month path endpoints run 2020-12 to 2021-12. So every 24-month window contains essentially all of calendar 2020, and the CIF comparison is the **same cohort over the same period** as §4.3. §4.6 states this plainly ("not independent corroboration of a separate transport experiment"), which is the right call and avoids the pseudo-replication trap.

It earns main-text placement because it conveys magnitude that monthly scores cannot: observed 0.3367 against M2's 0.7581 at 24 months. Keep it, framed as consequence — which is what the paper does.

The 12-month row shows **M1 closer than M2** (errors 0.00547 vs 0.01081); §4.6 states this correctly.

## 13. Uncertainty

The hierarchy is correctly articulated: facility resampling conditions on the realized calendar and composition; calendar blocks probe period sensitivity; neither gives macroeconomic sampling uncertainty; training and selection uncertainty is unestimated. Stated in §3.5, §4.7 and §6.

Three points: no calendar unit for either AUC, which limits inference (M2); "interval" for an 8-block percentile output, a presentation choice (minor 4); and no uncertainty at all attached to the unseen-vintage reversal, which the paper notes but which leaves its most interesting result as a bare point estimate.

## 14. Literature and novelty

**Classification: DISTINCTIVE_COMBINATION** on the brief's scale. This does not upgrade the frozen Task 16 conclusion, "distinctive combination plausible with material limitations", which stands given the closest-work gaps and this review's compromised independence (T20-C14).

Every component is established: competing mortgage terminations (Deng1996, Deng2000, Bhattacharya2019, Bu2026); macro conditioning in credit survival (Bellotti2009, Breeden2022, Breeden2023); real-time/revision-aware macro in credit (Croushore2001, Bianchi2026); competing-risk scoring and calibration (Heyard2020, Blanche2013, Gerds2012, VanCalster2019); Freddie survival under drift with calibration (Peng2026). Pair-decomposition of a concordance statistic is itself not new.

Closest five: **Peng2026** (nearest neighbour — Freddie, survival, calibration, drift, with macro conditioning and competing termination named as future work, which makes this paper a direct answer to a stated open question); **Bu2026** (Freddie, competing risks, macro covariates, CIF — decisive cells unverified); **Breeden2023** (Freddie, APC inputs, temporal stability); **Bianchi2026** (release-aligned revision-aware macro in credit, sovereign not mortgage); **Sadhwani2021** (multistate mortgage with economic predictors, not Freddie).

§2 is markedly better than the evidence record suggests its predecessor was — the search-completeness narration is much reduced, though traces remain (§2, §6). The honest "no priority or first-study claim is made" is the right posture.

## 15. Governance and reproducibility

**Genuinely scientifically useful, not merely elaborate engineering** — and I say that having just relied on it. The claim-evidence chain let me verify 257 statements in minutes rather than trusting a prose assertion. The frozen protocols, prediction hashes, consumption ledger, hostile review and correction ledgers materially reduce the risk that reported numbers drifted from the experiment.

Two limits the paper should keep stating. First, as the integration record for the imported work honestly notes, registration-before-metrics chronology **is not independently provable** from a commit — a hash proves content, not ordering. So the freeze is weaker than prospective preregistration, and §3.6's "frozen but non-virgin" is the right framing. Second, the freeze constrained model and feature search but did **not** prevent reporting selection, which is what M1 and M6 are.

§5's current treatment is about right in length. I would not expand it.

## 16. Regulatory framing

**Clean.** §5 mentions IFRS 9 and IRB only to disclaim: "implements neither a production ECL system nor regulatory PD/LGD/EAD validation and makes no compliance claim," plus explicit exclusion of decision thresholds, approval policy and loss economics. §6 and §8 repeat the boundary. §3.1 is careful that the default proxy is not a certified institution-specific definition. No implication of approval, production suitability or supervisory validation anywhere.

## 17. Abstract audit

Full sentence-level audit in `task20_abstract_audit.json`. Tally over 14 sentences: **GOOD 4, SUPPORTED 5, MISSING_QUALIFICATION 3, TOO_BROAD 1, AMBIGUOUS 1** (sentence 1 reclassified per T20-C13).

The abstract survives a reader who never reaches §6 on most counts. One material exception: the **absence of the default-discrimination result** (M1). Sentence 1 would benefit from design-specific wording, an editorial improvement.

## 18. Title audit

Covered in minor 1. "Vintage-aware" is accurate and well-defended. "Regime-dependent" is inferred, not established. "Transport" is a terminology choice (T20-C12). Population generality is adequately bounded by "Mortgage Competing-Risk Models" being specification-level rather than universal.

## 19. Fatal-flaw assessment

**NO FATAL FLAW.** Searched specifically for each:

| Candidate | Verdict |
|---|---|
| Target leakage | None identified. Macro vintage-restricted; covariates origination-only; current state excluded from predictors. Not proof of an operationally available information set (T20-C14) |
| Invalid temporal ordering | None. Calendar-separated *and* facility-disjoint |
| Incorrect competing-risk mathematics | None. Recursion standard; softmax guarantees coherence |
| Invalid probability construction | None. Coherent monthly cause probabilities, shared denominator |
| Wrong statistical unit | Tension exists and is disclosed; not fatal |
| Impossible estimand | None |
| Corrupted cohort | None. Counts reconcile across every artifact |
| Unsupported headline arithmetic | None. 143/143 and 257/257 verified |
| Severe selection bias | Survivor conditioning is real, disclosed, and the claim is scoped to it |

## 20. Required revisions

**All manuscript-only. None requires new computation.**

| # | Revision | Concern |
|---|---|---|
| R1 | Add default AUC (−0.070676, facility interval excludes zero) to the abstract, Table G and §4.2 | M1 |
| R2 | State in Table G and §4.4 that calendar-block AUC uncertainty was not estimated, and bound inferential language; report facility AUC intervals labelled as calendar-conditional | M2 |
| R3 | Report pair weights and gain contributions separately, naming resolution and population with each; describe gains as within-period at the stated resolution | M3 |
| R4 | Add a §5 paragraph presenting both bounded readings of the 2020 concentration, without sole-macro-movement or materiality premises | M4 |
| R5 | Remove all Task/CG identifiers and internal tokens from the manuscript; keep the substance | M5 |
| R6 | Add the full default calibration vector for both populations to §4.2 and §4.5, keeping improvements and deteriorations visible | M6 |
| R7 | Add three figures from existing frozen values | M7 |
| R8 | Retitle: replace "Regime-Dependent" and "Transport" | minor 1 |
| R9 | Add VIFs; state 14 cause-versus-no-event macro contrasts (21 stored coefficients) and 88 distinct months without presenting either as an effective sample size; optionally rename the 8-block output | minors 2–4 |

## 21. Recommended additional analyses

Full triage in `task20_analysis_triage.json`.

**ARXIV_BLOCKER (0).** *(Was 1; the calendar-AUC item is now disclosure via R2 plus optional analysis, T20-C08.)*
**HIGH_VALUE_BEFORE_SUBMISSION (4):** SA04 complexity adjustment; SA05 facility-weighted rescoring; CG03 implementation review; CG06 Bu2026 full text.
**PEER_REVIEW_RESPONSE (3):** calendar-block AUC intervals; SA08 support-restricted evaluation; maturity/refinancing separation.
**FUTURE_WORK (3):** nonlinear baseline; current-state conditioning; Fannie replication.

## 22. Preprint readiness

**REQUIRES_MAJOR_REVISION** — but the revision is writing and disclosure, not analysis. R1–R9 are achievable without touching the empirical layer. Once done, this would be a defensible preprint.

Non-scientific blockers remain: `[PUBLICATION TERMS REVIEW REQUIRED]`, `[ETHICS AND AUTHOR REVIEW REQUIRED]`, author identity and affiliations.

## 23. Peer-review readiness

**BORDERLINE.** The contribution is real, the verification is exceptional, and the methodological lesson travels. Against it: the specification is deliberately narrow (one linear multinomial, no interactions, no current state), the novelty is combinational, the closest paper is incompletely compared, and the central empirical result is dominated by a single calendar year the paper has not yet argued about properly.

Addressing M4 would move this most. A paper that confronts "is 2020 a failure case or the whole point?" is more interesting than one that reports a concentration and moves on.

---

### Decision summary

- **Preprint:** REQUIRES_MAJOR_REVISION
- **Peer review:** BORDERLINE
- **Fatal flaw:** NO
- **Major concerns:** 5; **moderate:** 2 (M2, M3, downgraded after audit)
- **Minor concerns:** 5 (7 raised; 2 withdrawn after audit)
- **Claim mismatches:** 0 value errors, 0 source errors, 0 interpretation errors, 1 scope error (2 interpretation errors and 1 scope error withdrawn after audit)
- **arXiv blockers:** 0 (was 1)
- **Manuscript editing alone can resolve the issues:** YES for all seven concerns
- **Decisions are reviewer judgments**, not objective gates
- **Independence:** COMPROMISED AND DISCLOSED; §2 cold read not satisfiable by this reviewer and should be repeated by one with no project exposure
