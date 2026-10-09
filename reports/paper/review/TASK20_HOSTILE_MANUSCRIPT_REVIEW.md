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
| T20-C03 | "Seven macro coefficients" / "seven parameters" | **Corrected.** Seven macro *predictors* in a three-class multinomial with a reference class carry two cause-specific coefficients each: at least 14 macro coefficients. The error originated in my Task 17 report and was carried here; v0.3 never made it. The correction strengthens the effective-sample-size concern. |
| T20-C04 | M6: M1's unseen-vintage default calibration is "near-ideal" | **Corrected.** A mean and slope close to observed values (0.000858 vs 0.000901; slope 1.0091) are weak-calibration evidence in the Van Calster et al. hierarchy [VanCalster2019], not complete calibration. M6's substance — default calibration is never reported, and M2 under-predicts unseen-vintage default incidence 3.5× with slope 0.7639 — is unaffected. |

Net effect: major concerns unchanged at 7; minor concerns 7 → 5; interpretation errors in the claim audit 2 → 1. No decision changes.

---

## 1. Summary

The manuscript is a substantial improvement on what the evidence record implies it replaced, and its quantitative spine is the most rigorously verified I have examined. I attacked the numbers in two independent passes and they held completely: **143 of 143** headline statements re-derived from frozen artifacts, and **257 of 257** evidence bindings resolved against their own named artifact and pointer with exact values matching to 1×10⁻¹². Zero orphans, zero value errors, zero source errors. Every binding carries a source hash and commit.

So this review cannot attack the arithmetic. It attacks emphasis, estimand discipline in three specific places, and one unexamined argument that I think is the paper's biggest missed opportunity.

**Seven major concerns.** The most consequential: a robust 7-point fall in **default** discrimination is absent from the abstract and from the uncertainty table, while the payoff-AUC gain that the paper spends its longest results subsection dismantling is in both. For a credit-risk audience the loss-relevant event's discrimination is the headline, and it is buried.

**No fatal flaw.** I searched specifically and found none.

## 2. Overall recommendation

| | Decision |
|---|---|
| **Preprint** | **REQUIRES_MAJOR_REVISION** |
| **Peer review** | **BORDERLINE** |

These are not calibrated to each other. The preprint verdict is major-revision because the abstract omits a principal result and one claim rests on an uncertainty unit the paper itself disowns — but **every major concern is resolvable by manuscript text and by reporting frozen values that already exist.** No new computation is required for any of them. That is an unusually good position for a major-revision verdict.

The peer-review verdict is borderline rather than weak-reject because the contribution is real but narrow, the novelty is combinational, and two of the closest papers remain incompletely compared.

## 3. Central contribution

**Derived from the evidence, not the authors' wording:** in one frozen monthly competing-risk specification on one Freddie cohort, adding a national macro block raises a pooled payoff concordance statistic almost entirely through cross-period pairs while degrading probability quality, and the direction of that degradation does not hold across origination-vintage populations.

**Central claim classification: B — SUPPORTED_WITH_IMPORTANT_QUALIFICATION**, provided the claim is read at the scope §§3–4 establish. As written in abstract sentence 1 it is **C — OVERSTATED**, because that sentence states a general capability with no population, model or period attached.

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

### M2. The headline discrimination claim has no calendar-unit uncertainty, which contradicts the paper's own argument — MAJOR

§3.5 and §4.7 argue, correctly and repeatedly, that facility resampling "condition[s] on the realized shared calendar" and that calendar-dependent claims need calendar-block assessment. Table G applies both units to joint log loss and payoff Brier.

**Neither AUC has any calendar-block interval frozen.** `paired_calendar` contains only `joint_log_loss`, `default_brier`, `payoff_brier`. So the pooled payoff-AUC gain — the subject of the abstract, §4.4 and §5's lead discussion — is supported *only* by the facility interval the paper itself says cannot speak to calendar questions. For a claim whose entire content is that the gain is **between-calendar-period**, that is an internal inconsistency.

I class this the one **ARXIV_BLOCKER**, and it is resolvable either way: compute the calendar-block AUC intervals, or state plainly in Table G and §4.4 that they do not exist and that the claim is correspondingly bounded. The second option is manuscript-only.

### M3. The 98.19% contribution share is resolution-dependent and is presented as a model property — MAJOR

The decomposition identity is exact and I verified it. But the within/between split depends on the stratum resolution chosen:

| Resolution | Within-pair weight | Between-pair weight |
|---|---|---|
| Calendar year | 17.195% | 82.805% |
| Calendar month | 1.979% | 98.021% |

As strata get finer the between share rises **mechanically**, because within-stratum pairs become scarcer. "98.19% of the gain is between-year" is therefore partly a statement about how coarse a year is relative to this panel, not solely about the models. The abstract leads with it.

The resolution-free, interpretable quantities are the **within-stratum gains**: +0.00636 at year level, −0.00040 at month level. Those carry the finding. Note also that the two decompositions are on different populations, with pooled gains of +0.06044 and +0.04831 — stated in §4.4, easily missed.

**Fix, manuscript-only:** lead with the within-stratum gains; present the contribution shares as resolution-conditional; state the resolution dependence in one sentence.

### M4. The paper does not confront the strongest argument against it — MAJOR

§4.3 establishes that 2020 supplies 89.34% of the deterioration and that ex-2020 the difference is +0.00217 (2.8% relative, interval containing zero). §5 and §6 treat this as a **scope limitation** — "restrict generalization to other regimes."

A hostile referee will invert it. The only period in which national macro variables moved substantially is the period in which the macro-augmented model failed catastrophically. Outside it, M2 is still *mildly worse* (positive in 6 of the 7 remaining years; better in 2021 alone, itself an unusual refinancing year). So across the evaluated calendar M2 is **never materially better and sometimes catastrophically worse**. On that reading the 2020 concentration is not mitigating — it is the indictment, because handling exactly such a period is the stated purpose of conditioning on macroeconomic information.

The manuscript never states this. Its careful hedging ("not disappearance outside the pandemic", "a dominant contribution in this observed calendar, not an estimate of a COVID effect") keeps it from overclaiming, but also keeps it from making the argument that would most strengthen it. The ex-2020 sensitivity is currently used in one direction only.

**Fix, manuscript-only:** add a paragraph to §5 presenting both readings and declining to adjudicate. This is the paper's largest available gain in significance.

### M5. Internal process vocabulary throughout the manuscript — MAJOR

The manuscript body refers to **Task 17** (§3.3, §5 twice), **Task 19** (§4.3), **Task 18** (§5), **Task 13A** (§6), **CG03** and **CG06** (§6), and uses the tokens `MIXED_TEMPORAL_TRANSPORT_ACROSS_POPULATIONS`, `ROBUST_FACILITY_ONLY`, `EXPLORATORY_ONLY`, `DRAFT_NOT_YET_AUTHORIZED` and `DISTINCTIVE COMBINATION PLAUSIBLE WITH MATERIAL LIMITATIONS` as if they were established terms.

"Task 19 does not regenerate it," inside a results section, is workflow narration, not science. A referee has no idea what Task 19 is. The **substance** belongs in the paper — an erratum exists, a correction was made, a classification was reached — but the identifiers do not.

This is also the concern my compromised independence most affects: these tokens are transparent to me and opaque to a cold reader. I flag it; a fresh reviewer should weight it.

### M6. Default calibration is never reported, and it undercuts the unseen-vintage reversal — MAJOR

§4.2 reports payoff calibration in detail (observed 1.536%, M1 1.138%, M2 2.830%, slopes 0.509 and 0.249). Default calibration appears nowhere, in either population. From the frozen artifacts:

| Population | Model | Observed | Mean predicted | Slope |
|---|---|---|---|---|
| Seen | M1 | 0.001125 | 0.000649 | 0.4821 |
| Seen | M2 | 0.001125 | 0.000886 | 0.3501 |
| Unseen | M1 | 0.000901 | 0.000858 | **1.0091** |
| Unseen | M2 | 0.000901 | **0.000258** | 0.7639 |

On unseen vintages M1's mean prediction and slope are close to observed values — weak-calibration evidence only, not complete calibration (corrected per T20-C04) — while **M2 under-predicts default incidence by a factor of 3.5**. §4.5 says "Default and payoff Brier still favor M1, as does default AUC" — and omits the most damaging item. The slight joint-loss reversal that motivates `MIXED_TEMPORAL_TRANSPORT` is accompanied by a default-calibration collapse the reader never sees.

**Fix, manuscript-only:** add a default-calibration row set to §4.2 and §4.5.

### M7. No figures — MAJOR for presentation

Seven tables, zero figures. The paper's central finding is a decomposition, and the per-year contribution plot is the single most informative display available from the existing frozen data. A reader must reconstruct the 2020 concentration mentally from Table C's eight rows.

Minimum set, all from existing values: (i) per-year contribution bars with interval weights; (ii) pooled / within / between AUC with the pair-weight annotation; (iii) the four-horizon CIF comparison for both causes. No new computation.

Again a cold-reader judgment I am poorly placed to make, but the absence is objective.

## 6. Minor concerns

1. **Title.** "Regime-Dependent" is not established — there is no regime model, change-point test or formal regime definition; the evidence is calendar-year concentration. "Transport" over-claims relative to the paper's own §4.5 concession that this "is not a controlled experiment that isolates a single population difference." Suggested: *Calendar- and Population-Dependent Behaviour of Vintage-Aware Macroeconomic Features in Mortgage Competing-Risk Models*. "Behaviour" or "sensitivity" is defensible where "transport" is not.
2. **VIFs omitted.** §4.3 cites condition number 69.1 but not the development VIFs — unemployment 10.21, Treasury 11.56, mortgage rate 11.57, HPI growth 11.22. The VIFs are more interpretable and strengthen the paper's own instability argument.
3. **Effective macro sample size never stated.** §3.4 says 88 distinct macro months and that rows do not create independent observations, but never draws the conclusion: at least 14 cause-specific macro coefficients — seven predictors, two cause logits each against the no-event reference — estimated from ~88 serially-dependent monthly observations. *(Corrected per T20-C03; originally stated as seven coefficients, an error carried from my Task 17 report.)*
4. **"Interval" for the 8-block scheme.** With eight blocks, one partial, percentile coverage is poor. Recommend reserving "interval" for the 5,619-cluster facility scheme and using **"calendar-block sensitivity range"** for the other. The paper already says "sensitivity" in prose; Table G's "lower/upper" headers undo it.
5. ~~**22 of 257 statements rest on an unreproducible run.**~~ **Withdrawn (T20-C02).** §3.6 and §8 already disclose the private-data reproduction limit.
6. ~~**Table F comparator mismatch.**~~ **Withdrawn (T20-C01).** The caption labels both quantities and §3.6 names the observed estimator.
7. **Bu2026 comparison still incomplete** (§2, §6). Survivable for arXiv, not for a journal where the referee may be an author.

## 7. Statistical design

Sound, with one unresolved tension the paper itself names. §3.1 declares facilities the loan-level statistical unit; §3.2 weights intervals equally. Because surviving longer is the complement of the modelled payoff event, the weighting is **outcome-correlated** — a facility contributing 86 intervals is by construction one that did not pay off. §3.2 states the issue and that a facility-weighted sensitivity is unexecuted. That disclosure is adequate for a preprint; a referee will want the sensitivity.

Pseudo-replication risk is **acknowledged and correctly handled in the text** (§3.4, §4.7) but only partially in the reporting — see M2. The paper never claims macroeconomic sampling uncertainty, which is right.

## 8. Temporal design

**I tried to break this and could not.** Development 2010-09–2017-12, purge 2018, evaluation 2019-01–2026-02, with facility roles deterministically disjoint — so the split is both calendar-separated *and* facility-disjoint, which is stronger than most temporal validation and stronger than the paper claims for itself. No facility history bridges the purge because no facility appears on both sides.

No target leakage: macro inputs are vintage-restricted, loan covariates are origination-only, current state is excluded from predictors while remaining in ascertainment (§3.3 states this distinction precisely). CIF paths use realized future macro and the paper says so four times.

"Vintage-aware" is **not** assumed to mean point-in-time safe — §3.4 is explicit that release lags were not certified. Correct.

Residual: horizon truncation and censoring depend on an unverified conditional-independent-censoring assumption, and CG03 remains open.

## 9. Population and transport

Survivor conditioning is disclosed thoroughly (§3.1, §3.2, §5, §6). The seen cohort is 2006–2014 originations surviving unprepaid to 2019 — a burnout-selected population — and the paper says burnout is omitted.

**"Transport" is the wrong word.** The two populations differ simultaneously in origination vintage, seasoning at entry, survivor selection, calendar exposure (a 2022-vintage facility cannot contribute intervals in 2020), cohort encoding (frozen zero-reference fallback) and duration support. Nothing can be attributed to any one of them. §4.5 concedes exactly this, which puts the title at odds with the results section. **"Population sensitivity" is defensible; "transport" is not.**

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

Economic meaningfulness is negligible either way: an M1 baseline of 0.5654 barely ranks payoff at all, and ranking calendar periods is not actionable because the calendar is observed.

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

Three problems: no calendar unit for either AUC (M2); "interval" for an 8-block percentile output (minor 4); and no uncertainty at all attached to the unseen-vintage reversal, which the paper notes but which leaves its most interesting result as a bare point estimate.

## 14. Literature and novelty

**Classification: DISTINCTIVE_COMBINATION.**

Every component is established: competing mortgage terminations (Deng1996, Deng2000, Bhattacharya2019, Bu2026); macro conditioning in credit survival (Bellotti2009, Breeden2022, Breeden2023); real-time/revision-aware macro in credit (Croushore2001, Bianchi2026); competing-risk scoring and calibration (Heyard2020, Blanche2013, Gerds2012, VanCalster2019); Freddie survival under drift with calibration (Peng2026). Pair-decomposition of a concordance statistic is itself not new.

Closest five: **Peng2026** (nearest neighbour — Freddie, survival, calibration, drift, with macro conditioning and competing termination named as future work, which makes this paper a direct answer to a stated open question); **Bu2026** (Freddie, competing risks, macro covariates, CIF — decisive cells unverified); **Breeden2023** (Freddie, APC inputs, temporal stability); **Bianchi2026** (release-aligned revision-aware macro in credit, sovereign not mortgage); **Sadhwani2021** (multistate mortgage with economic predictors, not Freddie).

§2 is markedly better than the evidence record suggests its predecessor was — the search-completeness narration is much reduced, though traces remain (§2, §6). The honest "no priority or first-study claim is made" is the right posture.

## 15. Governance and reproducibility

**Genuinely scientifically useful, not merely elaborate engineering** — and I say that having just relied on it. The claim-evidence chain let me verify 257 statements in minutes rather than trusting a prose assertion. The frozen protocols, prediction hashes, consumption ledger, hostile review and correction ledgers materially reduce the risk that reported numbers drifted from the experiment.

Two limits the paper should keep stating. First, as the integration record for the imported work honestly notes, registration-before-metrics chronology **is not independently provable** from a commit — a hash proves content, not ordering. So the freeze is weaker than prospective preregistration, and §3.6's "frozen but non-virgin" is the right framing. Second, the freeze constrained model and feature search but did **not** prevent reporting selection, which is what M1, M2 and M6 are.

§5's current treatment is about right in length. I would not expand it.

## 16. Regulatory framing

**Clean.** §5 mentions IFRS 9 and IRB only to disclaim: "implements neither a production ECL system nor regulatory PD/LGD/EAD validation and makes no compliance claim," plus explicit exclusion of decision thresholds, approval policy and loss economics. §6 and §8 repeat the boundary. §3.1 is careful that the default proxy is not a certified institution-specific definition. No implication of approval, production suitability or supervisory validation anywhere.

## 17. Abstract audit

Full sentence-level audit in `task20_abstract_audit.json`. Tally over 14 sentences: **GOOD 4, SUPPORTED 4, MISSING_QUALIFICATION 3, TOO_BROAD 2, AMBIGUOUS 1**.

The abstract survives a reader who never reaches §6 on most counts. Two exceptions: sentence 1's unscoped general claim, and the **absence of the default-discrimination result** (M1) — a material omission rather than a wording problem.

## 18. Title audit

Covered in minor 1. "Vintage-aware" is accurate and well-defended. "Regime-dependent" is inferred, not established. "Transport" conflicts with §4.5. Population generality is adequately bounded by "Mortgage Competing-Risk Models" being specification-level rather than universal.

## 19. Fatal-flaw assessment

**NO FATAL FLAW.** Searched specifically for each:

| Candidate | Verdict |
|---|---|
| Target leakage | None. Macro vintage-restricted; covariates origination-only; current state excluded from predictors |
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
| R2 | Either compute calendar-block AUC intervals or state in Table G and §4.4 that none exist and bound the claim | M2 |
| R3 | Lead with within-stratum gains; present contribution shares as resolution-conditional; state the resolution dependence | M3 |
| R4 | Add a §5 paragraph presenting both readings of the 2020 concentration without adjudicating | M4 |
| R5 | Remove all Task/CG identifiers and internal tokens from the manuscript; keep the substance | M5 |
| R6 | Add default calibration to §4.2 and §4.5, including the unseen-population slope collapse | M6 |
| R7 | Add three figures from existing frozen values | M7 |
| R8 | Retitle: replace "Regime-Dependent" and "Transport" | minor 1 |
| R9 | Add VIFs; state effective macro sample size as at least 14 macro coefficients on ~88 months; rename the 8-block output | minors 2–4 |

## 21. Recommended additional analyses

Full triage in `task20_analysis_triage.json`.

**ARXIV_BLOCKER (1):** calendar-block intervals for payoff and default AUC — or explicit disclosure of their absence.
**HIGH_VALUE_BEFORE_SUBMISSION (4):** SA04 complexity adjustment; SA05 facility-weighted rescoring; CG03 implementation review; CG06 Bu2026 full text.
**PEER_REVIEW_RESPONSE (2):** SA08 support-restricted evaluation; maturity/refinancing separation.
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
- **Major concerns:** 7
- **Minor concerns:** 5 (7 raised; 2 withdrawn after audit)
- **Claim mismatches:** 0 value errors, 0 source errors; 1 interpretation error, 2 scope errors (1 interpretation error withdrawn after audit)
- **Manuscript editing alone can resolve the issues:** YES for all seven major concerns
- **Independence:** COMPROMISED AND DISCLOSED; §2 cold read not satisfiable by this reviewer and should be repeated by one with no project exposure
