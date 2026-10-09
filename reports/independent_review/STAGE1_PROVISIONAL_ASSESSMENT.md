# STAGE 1 — PROVISIONAL ASSESSMENT (BLIND MANUSCRIPT READ)

Manuscript: "Population- and Regime-Dependent Transport of Vintage-Aware Macroeconomic Features in Mortgage Competing-Risk Models", v0.3 internal draft
Frozen commit (as stated by coordinator): 1a42d505c6652fecf384d00b2d19d737c4b8524a
File read: input/main_v0.3.md, sha256 dd7738c5726b27b703747b253f5e641bb9e3b32ba8507a18d2ba375ba765f17a (verified)
Bibliography read: input/references.bib
Nothing else was inspected: no Task 17–20 material, ledgers, claim-evidence maps, prior reviews, artifacts, repository history or web sources.

Scope note: this is a review only. Where I name a missing analysis (Q19), I am identifying what the manuscript's own claims need. I am not proposing edits.

Arithmetic I checked from the numbers in the manuscript:
- Table C intervals sum to 248,939. The interval-weighted M1 and M2 means reproduce 0.089201 and 0.105323.
- The 2020 share is 0.01440 / 0.01612 = 89.3%.
- With 2020 removed, the renormalized difference is 0.07978 − 0.07760 = +0.00217. That is 2.8% of the subgroup M1 baseline, which matches the text.
- Table D reconciles: 0.17195 × 0.00636 + 0.82805 × 0.07167 = 0.06044. The pooled M1 and M2 AUCs are also reproduced from the strata.
- Event rates: observed seen-vintage payoff is 3,823 / 248,939 = 1.536% per interval, matching the calibration text. Default is 0.112% per interval.
- 55 + 31 = 86 evaluation months, which matches 2019-01 to 2026-02. Development spans 88 months.

I found no internal arithmetic inconsistency.

---

## 1. Central research question

Does adding a national macroeconomic feature block to a monthly multinomial default/payoff model of Freddie Mac mortgages improve later out-of-time probability quality and discrimination? The block is vintage/revision-aware, with ALFRED availability dates. The paper also asks whether apparent gains survive when evaluation is broken down by:
- (a) within-period versus between-period ranking,
- (b) proper scores and calibration rather than AUC,
- (c) a population of origination vintages not seen in development,
- (d) facility-level versus calendar-level uncertainty units.

These are RQ-A to RQ-D. The underlying question is a validation question: what does a pooled metric improvement actually demonstrate?

## 2. Central empirical claim

Compared with an origination-characteristic baseline (M1), the macro-augmented model (M2) does the following:
- It improves development fit slightly (joint log loss 0.09015 to 0.08959).
- It improves pooled payoff AUC on the seasoned seen-vintage temporal evaluation (0.565 to 0.626). Almost all of that gain (98.2%) comes from between-year case-control pairs. On supported months, the within-month AUC difference is about zero or slightly negative (−0.0004).
- Its seen-vintage probability quality is worse (joint log loss +0.0161, +18%). Payoff is strongly over-predicted: mean 2.83% against 1.54% observed, with calibration slope 0.25. About 89% of the log-loss deterioration falls in calendar 2020.
- In a younger, unseen-vintage population, the joint log-loss sign reverses slightly (−0.0008), while the Brier scores and default AUC still favor M1.

The authors summarize this as "MIXED_TEMPORAL_TRANSPORT_ACROSS_POPULATIONS".

## 3. Methodological contribution

The paper itself says the contribution is not a new method. It is an empirical validation protocol that combines established pieces:
- an exact pair-weighted decomposition of pooled AUC differences into within-stratum and between-stratum parts (year and month),
- proper scores and calibration alongside AUC,
- a calendar-contribution accounting of log-loss differences,
- facility versus calendar-block bootstrap contrasts,
- conditional-entry cumulative-incidence projection,
- all under an audited information contract.

The most portable idea is the pooled-versus-stratified AUC decomposition. It shows that a macro block can raise pooled AUC by separating periods rather than ranking loans. The idea is sound and useful. It is not novel in principle, since it resembles case-mix/stratified concordance arguments, but it is well executed here.

## 4. Population the conclusions apply to

The conclusions apply narrowly to the following:
- A deterministic, unspecified sample of Freddie Mac single-family loan-performance histories.
- Seen-vintage evaluation: survivors from source vintages 2006, 2008, 2010 and 2014 still eligible at about 2019-01. 99% enter in that single month, 5,619 facilities in total.
- Unseen-vintage evaluation: the 2018, 2020 and 2022 vintages (17,352 facilities).
- Evaluation calendar: 2019-01 to 2026-02.
- One specific model: a monthly L2-penalized multinomial logit, additive, with no interactions, seven national macro transforms, origination covariates only and no current servicing state.
- Training data: one development window, 2010-09 to 2017-12, and so one macro trajectory.

The conclusions do not apply to:
- mortgages in general, or other agencies, lenders or products,
- other macro specifications (local or regional, interactions, refinance-incentive gaps, non-linear learners),
- other regimes.

The title's general phrasing ("Macroeconomic Features in Mortgage Competing-Risk Models") is wider than this population.

## 5. Statistical unit

The paper declares two units:
- The facility (loan) is the resampling unit.
- The monthly risk interval is the scoring and AUC unit. Losses weight intervals equally, so facilities with longer follow-up count more.

The macro contrast, however, varies only at the calendar-month level. The effective information about the macro effect is one realized national history:
- 88 development months and 86 evaluation months, which are serially dependent,
- 8 annual blocks, one of them partial.

The paper is candid about this mismatch (§3.4, §4.7, §6). Even so, the unit that matters for the macro claim is the calendar period, and there are very few of them.

## 6. Primary outcomes/estimands

- **Primary:** the interval-weighted joint multinomial log loss, M2 minus M1, on the seen-vintage temporal population. The paper calls the original frozen decision on this metric the "primary negative decision".
- **Secondary:**
  - cause-specific Brier scores,
  - cause-specific interval AUC (pooled),
  - payoff calibration-in-the-large and slope,
  - the year-stratified and month-stratified payoff-AUC decomposition (post hoc),
  - the calendar-year contribution to the log-loss difference (post hoc),
  - the same metrics on the unseen-vintage population,
  - the mean projected conditional-entry CIF against an Aalen–Johansen reference at 12, 24, 36 and 60 months,
  - paired fixed-model bootstrap intervals (facility and calendar-year).

Several of these are distinct estimands with different comparison populations, and the paper says so. The ex-2020 mean, the contribution residual, the full-primary pooled AUC and the eligible-month pooled AUC are explicitly kept apart.

## 7. Strongest evidence

1. **The payoff-AUC pair decomposition (Table D, §4.4).** It is an exact identity on fixed prediction arrays and reconciles arithmetically. The year-level result (between-year pairs carry 98% of the gain) is large and unambiguous. The month-level closure points the same way: within-month contribution about −0.000008, between-month +0.048. This supports the descriptive claim that the pooled payoff-AUC gain is calendar separation rather than better contemporaneous ranking. It is strong because it needs no distributional assumptions, it is a statement about the actual pairs, and the effect is large.
2. **Payoff calibration-in-the-large on the seen population.** M2's mean payoff probability is 2.83% against 1.54% observed (about 84% over-prediction), with slope 0.25 against M1's 0.51. Given the event counts (3,823 payoffs), sampling noise cannot produce this. The facility CI for the payoff Brier difference is tight. The calendar CI includes zero, but the size of the mean miscalibration is not in doubt as a description of this realized period.
3. **The calendar accounting of the log-loss difference (Table C).** It reconciles exactly and shows the deterioration is concentrated rather than diffuse.

## 8. Weakest evidence

1. **The unseen-vintage "reversal" (Table E, §4.5).** It is a point difference of −0.000819 in joint log loss with:
   - no uncertainty interval of any kind,
   - no per-year decomposition,
   - no within-period AUC decomposition, even though unseen payoff AUC jumps from 0.552 to 0.737,
   - unseen cohort categories scored through a "frozen encoding fallback",
   - a population that differs from the seen one in calendar exposure as well as vintage and seasoning.

   This is the weakest result, and yet it carries the title's "population-dependent", the MIXED headline classification and §5's "changes the study-level interpretation".
2. **The ex-2020 sensitivity.** It is an aggregate year-block resampling with only 7 blocks, one of them partial. The interval [−0.00151, +0.00671] is honestly reported as including zero, but a percentile bootstrap over 7 to 8 clusters is poorly calibrated.
3. **The CIF comparison (Table F).** The implementation compatibility of the observed reference (conditional entry, pooled censoring weights) is still open (CG03). It reuses essentially the same cohort and calendar, and it has no intervals.
4. **Default-specific results.** There are 280 seen-vintage default events and no intervals for the default AUC/Brier differences. The default AUC drop (0.694 to 0.623) is not given an uncertainty statement.

## 9. Five most important threats to validity

1. **Calendar-exposure confounding of the seen/unseen comparison.**
   - The seen population enters almost entirely at 2019-01, so 2020 dominates its early follow-up.
   - The unseen 2020 and 2022 vintages originate during or after 2020 and so carry little or no 2020 exposure. Their weight falls in 2021–2026, where Table C shows M2 is roughly neutral to mildly worse for the seen population.
   - The reversal could therefore be largely a change in calendar weights rather than a population property. The manuscript lists "calendar exposure" as one difference but does not separate it out. Without a per-year unseen breakdown, or a comparison standardized to common calendar weights, the reversal cannot be read as "population" transport at all.
2. **One realized macro history, with the macro block acting as a period proxy.**
   - The block is estimated from 88 strongly trending, collinear months: condition number 69, unemployment/HPI correlation −0.92.
   - In development it can act like a smooth calendar-trend term within an age-period-cohort structure.
   - In evaluation, 77 of 86 months are outside development support, and the correlation structure breaks (to −0.24).
   - The out-of-time behavior is therefore extrapolation of a period proxy under a single new trajectory. Every macro-level conclusion has an effective sample of about one episode. The paper acknowledges this, but the title still generalizes.
3. **Possible target contamination in 2020–2021 (not discussed).**
   - The research default is a numeric delinquency state of 3 or more, i.e. about 90+ days. In agency loan-level data, borrowers in COVID-era forbearance were generally still reported as delinquent by contractual status.
   - The observed seasoned default CIF jumps about sixfold between 12 months and 24 months (0.0061 to 0.0360). The 24-month window for 99% of facilities is 2019-01 to 2020-12.
   - The manuscript never mentions forbearance, deferral or modification. If forborne loans count as "research defaults", the default endpoint in the stress year measures a policy and accounting state, not credit default.
   - This mainly threatens the default-specific findings and the 2020 log-loss component, less so the payoff-based central claims.
4. **Model misspecification inherent to the tested "macro augmentation".** The macro block enters additively in the logit without interactions, using a national mortgage-rate level rather than a loan-specific refinance incentive (coupon minus market rate). A falling market rate therefore raises payoff probability for every loan, including seasoned survivors with no incentive or with burnout. The conclusions therefore describe this specification, not "macro features" in general. The exploratory refinance-gap result is kept separate as EXPLORATORY_ONLY, which is appropriate.
5. **Non-virgin, partly post-hoc evaluation with limited and uneven uncertainty.**
   - The evaluation outcomes had been inspected before the decompositions, the month closure and the ex-2020 sensitivity were registered.
   - Several key contrasts have no intervals: all AUC differences, the unseen differences, the CIF and the default metrics.
   - The facility bootstrap is conditional on the realized calendar, so its narrow intervals (e.g. [+0.0154, +0.0168]) are close to pseudo-replicated for any macro-level reading.
   - The paper discloses all of this, but disclosure does not supply the missing inferential support.

Secondary threats:
- Interval weighting favors long-surviving facilities.
- Payoff includes maturity. Shorter-term 2006/2008 loans would mature inside the evaluation window, and this is not quantified.
- Macro release lags are uncertified, which creates possible mild look-ahead for reporting-month macro values.
- L2 penalty selection is unspecified.
- The sampling design of the Freddie extract is unspecified.
- The month-support rule probably excludes later, payoff-sparse high-rate months disproportionately, so the within-month result describes mostly 2019–2022.
- Both baselines under-predict default heavily at the CIF level (about 0.011 against 0.036 at 24 months), and M1's default Brier is worse than M0's. The comparator is itself poorly calibrated out of time, which limits what "M2 vs M1" means.

## 10. Causal statements beyond the design

The manuscript is unusually disciplined. It repeatedly disclaims causal mechanisms (COVID effect, burnout, survivor selection, the link between payoff over-prediction and the AUC gain), and these disclaimers are appropriate. The remaining problems are framing, not explicit causal assertion:
- **The title** ("Population- and Regime-Dependent Transport"). "Dependent" implies that population and regime are the factors on which performance depends. The design can only show that performance differed between two populations that differ in many ways, and across calendar years. That is a descriptive association presented with an explanatory label.
- **§5:** "A severe joint-loss disadvantage in a seasoned surviving population does not carry unchanged into younger unseen vintages." This attributes the contrast to age/seasoning by naming it so, although calendar exposure is an equally available explanation. The next sentences hedge.
- **Abstract:** "Calendar 2020 contributes 89.34% of that deterioration." This is accounting, not causation, and is acceptable.
- **Comparisons between model specifications.** Statements such as "macro augmentation improves development fit" are legitimate in kind: comparing two fitted models on fixed data is a controlled comparison of specifications. They should not be read as causal effects of macro conditions on borrowers.

There is no outright causal claim the design cannot support. The title and the "population" framing lean causal.

## 11. Adequacy of distinctions

| Distinction | Adequately separated? | Comment |
| --- | --- | --- |
| Pooled discrimination | Yes | Clearly defined as a cross-month, cross-facility interval AUC. |
| Within-period discrimination | Yes, with caveats | Year and month decompositions are exact. The month result is support-restricted (55/86 months, 77% of intervals) and has no uncertainty or equivalence bound, which the paper admits. Not done for the unseen population or for default AUC. |
| Probability quality | Mostly | Joint log loss and Brier are reported. Interval weighting is disclosed, and a facility-weighted version is not done. |
| Calibration | Partially | Payoff calibration-in-the-large and slope are reported for the seen population only. Default calibration and unseen-population calibration are not reported. "Reliability summaries" are mentioned but not shown. |
| Temporal stability | Partially | Per-year results are reported for joint log loss on the seen population only. No per-year Brier, calibration or AUC, and nothing per-year for the unseen population. |
| Population transport | No | Population is confounded with calendar exposure, entry timing, seasoning and encoding fallback. The term "temporal transport across populations" merges two axes (time and population) that the design does not separate. |

Overall the conceptual separation is well articulated, better than in most applied papers. The empirical execution is complete for discrimination on the seen population but thin for calibration, stability and population.

## 12. Does seen- versus unseen-vintage comparison support "transport"?

Only weakly, and only in a loose sense.

In the causal-inference literature, "transportability" means carrying a result from a source population to a target under stated assumptions about which differences matter. Here the two evaluation populations differ simultaneously in:
- vintage,
- seasoning and survivor selection,
- entry timing (the seen population is synchronized at 2019-01, the unseen enters staggered across 2018–2022 originations),
- calendar exposure weights (especially 2020),
- encoding support (unseen cohorts use a fallback encoding),
- event base rates (payoff 1.54% against 1.12% per interval).

No assumption is stated about what is invariant, and nothing is done to separate these factors. The comparison shows that out-of-sample performance differences are not homogeneous across two evaluation sets, which is a heterogeneity or external-validity description. Calling it "transport" overstates the inferential structure.

The seen-vintage evaluation is itself an out-of-time ("temporal") transport. The paper's label mixes the two.

The reversal is also a −0.0008 point estimate with no interval, so even the descriptive "reversal" is not established as different from zero.

## 13. Does the evidence support "regime-dependent"?

Not convincingly.

- Regimes are not defined beforehand or measured by a regime variable.
- The evidence is one calendar year (2020) with large deterioration, followed by a single-year reversal (2021) and small positive differences afterwards.
- One episode cannot show dependence on a regime, as opposed to a single outlier period. The ex-2020 difference is small (+2.8% relative) with an interval including zero.
- The body language is appropriate: "highly concentrated in calendar 2020", "not an estimate of a COVID effect". The title's "Regime-Dependent" is stronger than the body.
- "Calendar-concentrated" or "period-heterogeneous" is supported. "Regime-dependent" implies a repeatable relationship between macro state and model error, which one realization cannot show.

## 14. Does the 2020 concentration strengthen or weaken the central argument?

It does both, depending on which argument.

- **It strengthens the methodological argument:** average-period scores can hide heterogeneity, so evaluation should report calendar contributions. A single period moving 89% of an 18% deterioration is a vivid demonstration.
- **It weakens the substantive argument that M2 has systematically worse temporal probability quality.** Outside 2020 the deterioration shrinks to +0.00217 (2.8%), with an interval including zero over very few blocks. The negative primary verdict therefore rests mostly on one year.
- **It also entangles the components.** The pooled payoff-AUC gain is between-period, and 2020 is where M2's payoff probabilities are most separated from other years. The discrimination "gain" and the probability "loss" may share a source in one period. The paper calls this link a hypothesis and does not claim it. If it were true, the two headline findings would be one finding rather than two corroborating ones.
- **It makes the seen/unseen comparison hard to read**, since the populations have different exposure to 2020 (threat 1).

Net: 2020 strengthens the validation lesson but narrows the empirical claim to "this model, in this history, failed mostly in one extraordinary year".

## 15. How should 2020 be interpreted?

A combination, described without causal claims:

- **Unusual regime (descriptively).** 2020 is far outside development support on several features: 77 of 86 evaluation months exceed the development-distance reference, and the unemployment/HPI correlation collapses. The co-movement of predictors seen in 2010–2017 does not hold. In the observed data the endpoints also behave unusually: observed default incidence rises sharply between the 12-month and 24-month horizons, and both models' 2020 log loss is far above 2019.
- **Important stress test.** Because the macro inputs are realized, vintage-aware values and not forecasts, the 2020 evaluation asks a clean question: does the fitted mapping from macro inputs to probabilities stay sensible when inputs move off-support? It does not for payoff. The CIF table shows M2 projecting about 76% 24-month payoff against 34% observed, even with the realized macro path supplied.
- **Model failure, in a limited sense.** For an out-of-time probability model, a large calibration breakdown in a period of input extrapolation is a failure of that specification to extrapolate. That is a property of the fitted model under these inputs, not a claim about why borrowers behaved as they did. M1 is also miscalibrated there (it under-predicts default), so 2020 is hard for both models. The incremental deterioration is attributable to M2's added block only in the accounting sense.

Reasonable reading: an unusual, off-support period that works as a stress test, which the macro-augmented specification failed on payoff probability quality. The 2021 sign reversal shows the failure is not a simple constant bias. The paper's own wording is consistent with this, except for the title's "regime-dependent".

## 16. Is the CIF analysis independent evidence?

No. It depends on the same calendar trajectory, and the paper says so (§4.6).

- 99% of the 5,619 landmarks enter at 2019-01. The latest entry is 2020-01.
- Every 24-month-or-longer projection window includes 2020.
- The CIF is a cumulative re-expression, through the competing-risk recursion, of the same monthly probabilities on the same facilities and period that produced the log-loss result.
- M2's 24-month payoff over-prediction (0.758 against 0.337) is essentially the 2020 monthly over-prediction compounded.

It illustrates the practical scale of the monthly miscalibration. It does not corroborate it. Its observed reference also has unresolved implementation questions (CG03), so it is the least secure of the analyses.

## 17. Is the uncertainty treatment adequate?

**Conceptually good, empirically incomplete.**

Strengths:
- The paper distinguishes facility resampling (conditional on the realized calendar) from calendar-block resampling.
- It refuses to call facility-only intervals general robustness.
- It discloses that model-refit and macro-sampling uncertainty are not estimated.

Gaps:
- **No intervals** for:
  - any AUC difference (pooled, within-year, within-month),
  - any unseen-population difference (including the "reversal"),
  - default metrics,
  - calibration statistics,
  - CIF horizons.
- **Few calendar blocks.** The calendar-block percentile bootstrap uses 8 unequal, serially dependent blocks, one partial (7 in the ex-2020 version). Percentile intervals with so few clusters have poor coverage, and the result depends heavily on whether the 2020 block is drawn. A resample of 8 blocks omits 2020 with probability (7/8)^8 ≈ 0.34. The lower endpoint (+0.00005) is therefore fragile, and the paper does treat it cautiously.
- **The facility bootstrap** is effectively pseudo-replication for any macro-level reading. The paper acknowledges this.
- **Training-side uncertainty is absent.** There is no repeat of development or refitting, and penalty selection is not described.

Bottom line: adequate for the paper's carefully hedged seen-population log-loss statements. Inadequate for the unseen "reversal" and the within-month "no gain" statements, which appear in the abstract and conclusions without any uncertainty.

## 18. Are the conclusions broader than the evidence?

In the body and the conclusion text, mostly no. The writing is heavily qualified.

In the title and the headline classification, yes:
- "Population-dependent" rests on an uninterpreted, interval-free point reversal confounded with calendar exposure.
- "Regime-dependent" rests on one year.
- "Transport" suggests a formal transportability analysis that is not done.
- "Vintage-Aware Macroeconomic Features" foregrounds vintage-awareness, but no analysis isolates its effect. There is no comparison with final-revised data, so vintage-awareness is a design property, not a finding.
- "Mortgage Competing-Risk Models" generalizes from one additive multinomial specification on one Freddie extract.

The abstract's opening sentence ("National macroeconomic inputs can distinguish calendar periods without improving risk ranking...") is an existence claim. One example supports it.

## 19. Additional analysis that appears essential (from the manuscript alone)

Essential to support the claims as currently framed:

1. **A per-calendar-year breakdown for the unseen-vintage population**, covering joint log loss and payoff AUC, ideally with the year/month pair decomposition. Alternatively, a seen-vs-unseen comparison standardized to common calendar weights. Without one of these, the "population" part of the title and of the MIXED classification cannot be told apart from calendar-exposure composition. The manuscript explicitly says this was not done.
2. **Uncertainty intervals for the unseen-vintage differences and for the AUC differences** (pooled, within-year, within-month), at least under the facility scheme and, where feasible, a calendar scheme. The abstract-level claims of a "reversal" and "no within-month gain" currently have no inferential support.
3. **A clarification of how the default proxy treats loans in COVID-era forbearance or deferral**, with a sensitivity check if they are counted as defaults. The 12-to-24-month jump in observed default incidence suggests this matters for the default-specific results and the 2020 log-loss component.
4. **A split of the 2020 log-loss difference by outcome component** (payoff, default, no-event), to show which event drives the concentration.

Important but arguably not essential:
- payoff and default calibration for the unseen population,
- a facility-weighted loss sensitivity,
- quantifying maturity within the payoff/maturity event,
- a period-trend comparator (M1 plus a smooth calendar term) to test whether the macro block behaves like a period proxy.

## 20. Explicit search for a fatal flaw

| Candidate | Finding |
| --- | --- |
| **Leakage** | None found in the scoring design. Development (2010-09 to 2017-12) and evaluation (2019-01 onward) are separated by a 2018 purge. Facility roles are disjoint. Preprocessing is training-only. The macro join is vintage-aware. Residual risk: release lags are not certified, so reporting-month macro values could include mild look-ahead. That would favor M2, and M2 still underperforms, so it would not reverse the negative seen-vintage finding. It could slightly inflate M2's pooled AUC and the unseen improvement. The CIF uses future realized macro paths, which is disclosed and is a property of a retrospective design, not leakage. **Not fatal.** |
| **Invalid temporal ordering** | None found. Landmarks follow the purge, and development precedes evaluation. **Not fatal.** |
| **Target construction error** | **Unresolved concern, not shown to be an error.** The default is a composite of delinquency state 3 or more, REO, or codes 02/03/09. COVID forbearance and deferral handling is not discussed and could reclassify 2020–2021 delinquency as "default". Payoff includes maturity, which is not quantified. Both are disclosed only partially. This could seriously affect the default-specific and 2020 results, but probably not the payoff-driven central claims. **Not fatal on present evidence. Highest-priority open question.** |
| **Competing-risk error** | None found. Monthly multinomial probabilities are coherent. The survival/CIF recursion is correct. Payoff is treated as a competing event, not censoring. The observed Aalen–Johansen reference under conditional entry with pooled censoring weights is still under author review (CG03). That affects only Table F. **Not fatal.** |
| **Pseudo-replication** | Present in the facility bootstrap with respect to macro-level questions, but disclosed. The paper does not base its conclusions on facility-only significance; it labels such results "ROBUST_FACILITY_ONLY". **Not fatal, because it is acknowledged.** |
| **Wrong statistical unit** | The effective unit for the macro contrast is the calendar period (about one history, 8 blocks). Scoring uses intervals and resampling uses facilities. The mismatch is acknowledged and no significance claim relies on the wrong unit. **Not fatal.** |
| **Invalid probability interpretation** | None found. CIFs are explicitly entry-conditional, retrospective and not prospective forecasts. Interval-level AUC is distinguished from facility-level horizon AUC. Calibration diagnostics are not used to recalibrate. **Not fatal.** |
| **Unsupported population inference** | Present at the **title and classification level**: "population-dependent transport" from one confounded, interval-free comparison, and "regime-dependent" from one year. The body is largely bounded. This is serious for framing, but it is fixable by reframing or adding analysis, and it does not invalidate the core results. **Not fatal.** |

No fatal flaw is identified on the manuscript alone. The most consequential open items are the calendar-exposure confounding of the population comparison and the unexamined treatment of forbearance in the default endpoint.

---

## Additional observations (presentation, not decisive)

- **Internal process artifacts throughout the manuscript:**
  - "Task 17 erratum", "Task 18", "Task 19", "CG03", "CG06", "Task 13A",
  - "hostile review", "literature-audit conclusion",
  - ALL-CAPS status tokens such as DISTINCTIVE COMBINATION PLAUSIBLE WITH MATERIAL LIMITATIONS, ROBUST_FACILITY_ONLY, EXPLORATORY_ONLY and DRAFT_NOT_YET_AUTHORIZED,
  - inline `Q19` provenance comments.

  An external reader cannot interpret these. The draft labels itself "not an arXiv submission". Defensive meta-commentary ("Passing software checks does not upgrade...") takes space away from substance.
- **Bibliography:**
  - Eight .bib entries are not cited in the text: Sadhwani2021, Schwartz1989, Wang2024, Fine1999, Roschewitz2025, OPSurv2024, Graf1999, Brier1950. Sadhwani2021 (deep learning on agency loan-level data with local macro and unemployment effects) is directly relevant prior work that the related-work section should engage.
  - The Bhattacharya2019 URL points to an author profile page, not the article. Its DOI is present.
  - Several 2026 references (Bu2026, Peng2026, Bianchi2026) could not be checked under the Stage 1 constraints.
  - The references list carries "PEER_REVIEWED" status tags.
  - There is no citation for the formal transportability/external-validity literature, even though "transport" is the title concept.
  - There is no citation for COVID-era forbearance and mortgage-performance reporting.
- **Model-specification details missing:**
  - how the L2 strength was chosen,
  - the sampling design of the Freddie extract,
  - how the 2006–2014 seen vintages were chosen,
  - what the "frozen encoding fallback" for unseen cohorts actually assigns.
- **Under-discussed results:**
  - The large drop in seen-vintage default AUC under M2 (0.694 to 0.623) gets little attention.
  - M1's default Brier is worse than M0's.
  - Both baselines under-predict default by roughly a factor of three at 24 to 60 months.

  Together these suggest the comparator is itself not well calibrated out of time.

---

CENTRAL CLAIM:
In a frozen Freddie Mac monthly default/payoff model, adding vintage-aware national macro features improves development fit and pooled payoff AUC. On seasoned seen-vintage loans, the AUC gain is almost entirely between-period separation, with no average within-month ranking gain on supported months. Out-of-time probability quality worsens, concentrated in calendar 2020. A small point reversal in joint log loss among younger unseen vintages makes the overall picture mixed.

CENTRAL CONTRIBUTION:
A carefully bounded empirical demonstration, using an exact pair-weighted AUC decomposition alongside proper scores, calibration and calendar accounting, that a pooled discrimination gain from shared macro inputs can reflect calendar separation coexisting with degraded probability quality. This motivates stratified validation in credit-risk model governance.

FATAL FLAW:
NO
(No disqualifying leakage, ordering, competing-risk or unit error was found. Treatment of COVID-era forbearance in the default endpoint is an unresolved target-construction question, chiefly affecting default-specific and 2020 results.)

PREPRINT DECISION:
MAJOR REVISION

PEER-REVIEW DECISION:
WEAK REJECT

TOP THREE REASONS FOR THAT DECISION:
1. The headline framing ("Population- and Regime-Dependent Transport", MIXED_TEMPORAL_TRANSPORT_ACROSS_POPULATIONS) rests on the weakest evidence. The unseen-vintage reversal is a −0.0008 point estimate with no interval and no calendar breakdown, and it is confounded with calendar exposure (especially to 2020), seasoning, entry timing and encoding fallback. "Regime-dependent" rests on one year. The robust findings (pooled AUC gain is between-period; payoff probabilities are badly over-predicted out of time, mostly in 2020) do not need these labels, and the labels are not supported.
2. The inferential base for the macro contrast is about one realized national history (88 development months, 8 evaluation blocks, one dominant episode), and uncertainty is missing for exactly the claims that carry the paper's novelty: AUC differences, within-month "no gain", the unseen reversal, calibration and CIF. The calendar-block intervals that do exist rest on too few clusters to be reliable. The default endpoint's handling of 2020–2021 forbearance is not discussed.
3. The contribution is a careful but modest combination of established methods applied to one additive multinomial specification on one Freddie extract, and the paper itself admits this. The manuscript is also cluttered with internal process artifacts (task and erratum numbers, review codes, status tokens, provenance tags) and omits relevant cited-in-bib prior work (e.g. Sadhwani et al. 2021) and transportability literature, which reduces its readiness and clarity for external readers.

STOP. Stage 1 assessment recorded. No evidence package, task material, ledger, or prior review has been requested or inspected.
