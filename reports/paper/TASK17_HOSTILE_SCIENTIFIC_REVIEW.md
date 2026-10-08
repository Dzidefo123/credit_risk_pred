# Task 17 — Hostile Scientific Review and Falsification Audit

**Manuscript under review:** `paper/main_v0.2.md`
**Base commit:** `a4479ba4be115bc9a1d4b53ab77e258b40bd57c7`
**Prior decision (Task 16):** DISTINCTIVE COMBINATION PLAUSIBLE WITH MATERIAL LIMITATIONS
**Review type:** adversarial, review-and-audit only. No model was fitted, no prediction regenerated, no frozen metric altered, no Fannie outcome inspected, no `main_v0.3.md` created.

**Arithmetic policy.** This review recombines values already frozen in Task 10/11/12 artifacts — principally an interval-weighted decomposition of an already-reported mean. The decomposition reconstructs the frozen paired delta to machine precision (`+0.01612163642779664`), which is the check that it is recombination and not a new measurement. Where a statement is an inference rather than a frozen value it is marked INFERRED.

### Deviation from the Task 17 brief — output location

The brief specifies `docs/paper/review/task17_*.json`. Those files are instead at **`reports/paper/review/task17_*.json`**.

Reason: `docs/paper/` is governed by the Task 16V frozen guard. `scripts/check_paper_evidence.py::check_paper_files` rejects any subdirectory of `docs/paper/` other than `literature/` ("Unreviewed paper directory"), the guard itself is hash-pinned in `docs/paper/paper_evidence_manifest.json`, and `tests/test_literature.py::test_approved_checker_hash_does_not_allow_further_edits` asserts that no further edit to it is accepted. Creating the specified directory therefore requires loosening a freeze that Task 16 built deliberately — a change to the project's governance model, and the brief instructs preservation of Task 14/15/16 state and forbids silent repair.

I implemented the amendment-chain version first (extending `POST_FREEZE_ALLOWED_FILES`, a `task17_compatibility_amendment.json` chaining off the Task 16 approval, and a chained `check_frozen_hash`). It worked, but it also required editing `scripts/check_literature.py::preserved_digest` and the test that exists to forbid exactly that. **I reverted it.** `scripts/check_paper_evidence.py` is back at its Task 16 approved hash `451443f7e104e25a0af6782924b81c52de356cc1ae66cef2a57351795134df72`, and no tracked file is modified by this task.

The decision is the authors': either approve a Task 17V compatibility amendment and move the four registers to the specified path, or keep them where they are. Nothing in the review's content depends on which is chosen.

---

## Executive verdict

The arithmetic is sound. Every PRIMARY number checked against its artifact is correct, the governance apparatus is better than almost anything I have seen at preprint stage, and the design decisions are defensible and documented. **The problem is not the measurements. It is that the manuscript's framing is not the framing its own frozen evidence supports, and that three pieces of evidence which cut against the headline are present in the artifacts and absent from the paper.**

Specifically:

1. **89.3% of the headline deterioration is calendar 2020.** Ex-2020 the difference is `+0.00172` rather than `+0.01612`, and 2021 reverses sign.
2. **The payoff AUC gain is between-period, not facility-level.** Within every frozen calendar year the gap is near zero or negative, and at the facility-level 24-month CIF the two models are identical to four decimal places.
3. **The unseen-vintage supplement reverses the sign of the protocol's own primary decision metric** on an evaluation population 2.6× larger than the primary one, and this is never reported.

None of these falsifies a recorded number. All three materially narrow what the recorded numbers mean. Item 3 in particular cannot stay unreported: a referee who downloads the artifacts will find it, and finding it unreported is far more damaging than reporting it.

A fourth item is a disclosure asymmetry rather than a scientific error: the frozen calendar-block interval for the primary payoff Brier difference includes zero, the manuscript gives exactly that caveat for the *exploratory* result in §8 and omits it for the *primary* result in §6.4, and the study's own frozen decision rule requires both intervals.

**Decision: MANUSCRIPT CORE SURVIVES WITH MAJOR REVISION.** The core empirical finding — a frozen macro-augmented specification that improved in-sample fit did not improve temporal probability quality in this cohort — survives. Its current scope, its stated mechanism and its headline framing do not.

---

## A. Decision

**MANUSCRIPT CORE SURVIVES WITH MAJOR REVISION**

Not *survives hostile review*: four MAJOR concerns require main-text changes, and one (SA01) requires a scoring pass on existing frozen arrays before the ranking claim can be stated in any form.

Not *requires additional empirical validation* as the primary verdict: the central negative result stands on frozen evidence once narrowed. But two submission-blocking sensitivities (SA01, SA02) are scoring passes on existing prediction arrays, not new experiments, so the distance to a defensible claim is short.

Not *central claim not adequately supported*: the claim is supported for the seen-vintage 2006–2014 survivor cohort over 2019–2026, which is a real and reportable finding.

Not *stop*: no foundational error was identified. Estimands are correctly specified, facility roles are genuinely disjoint, the ledger is real, and the governance is unusually strong.

## B. Preprint readiness

**ARXIV_PREPRINT_REQUIRES_ADDITIONAL_ANALYSIS**

CH01–CH07 are manuscript-only and could be done in a week. But SA01 is submission-blocking and is a scoring pass, not a rewrite: the claim "payoff ranking improved" appears in the title-adjacent abstract sentence, in §6.4, in §9 and in the conclusion, and the frozen CIF-level AUCs contradict the facility-level reading. Either run the within-period AUC or restate the claim as across-period ranking everywhere. If the authors choose restatement, this becomes ARXIV_PREPRINT_READY_AFTER_MANUSCRIPT_REVISION.

## C. Peer-review readiness

**PEER_REVIEW_REQUIRES_ADDITIONAL_ANALYSIS**

SA01, SA02, SA03 and SA06 are all submission-blocking. SA02 and SA06 are reporting and recombination; SA01 is a scoring pass; SA03 is reporting plus an explanation. None requires a refit. A referee at JORS, EJOR or the Journal of Credit Risk will ask all four.

## D. Strongest contribution

A fully auditable, prespecified, frozen negative result on a real calendar break, with a claim-level evidence registry, consumption ledger, hash-pinned artifacts and an explicit non-virgin-holdout disclosure. The combination of (a) competing-event probability assessment, (b) proper scores plus calibration plus CIF consequences, and (c) a governance trail that lets a reader verify what was frozen when, is genuinely uncommon. **The reproducibility and evidence-governance apparatus is the most publishable thing in this project and is currently presented as administrative background rather than as a contribution.**

The second contribution, properly stated, is the ranking-versus-probability divergence — but only once it is restated as across-period ranking.

## E. Strongest scientific concern

**The unseen-vintage reversal (F3).** The protocol names `M2-M1 paired temporal joint log loss` as the primary decision metric. On the seen-vintage population that metric is `+0.01612` (M2 worse). On the unseen-vintage population it is `-0.00082` (M2 better), on 17,352 facilities and 652,508 intervals against 5,619 and 248,939. The manuscript discloses that the supplement exists and that it was kept out of the headline, but never that it points the other way. Whatever the right explanation — young unseasoned loans, different duration support, zeroed cohort effects — the reader must be given the number.

## F. Fatal concerns

**None.** No concern meets the FATAL bar ("current evidence cannot support a central conclusion"). The central conclusion is supportable at narrower scope. I record this explicitly because the Task 17 brief invited a stop verdict and I did not find grounds for one.

## G. Major concerns

| ID | Concern | Reviewers |
|---|---|---|
| F1 | 2020 contributes 89.3% of the headline delta; ex-2020 residual `+0.00172`; 2021 reverses sign | R1, R3, R4 |
| F2 | Payoff AUC gain is between-period, not facility-level; CIF-level AUCs identical | R3, R4, R2 |
| F3 | Unseen-vintage population reverses the primary decision metric; unreported | R1, R3, R5 |
| F4 | Calendar-block interval for primary payoff Brier includes zero; caveat given for exploratory result only | R3, R5, R1 |

## H. Moderate concerns

F5 horizon selection in the CIF headline · F6 CIF cohort is effectively single-entry on one macro path (INFERRED) · F7 near-collinear development macro design with evaluation sign reversals · F8 "point-in-time" overstates frozen provenance · F9 equal interval weighting · F10 joint log loss dominated by no-event class · F11 current loan state excluded by design · F13 all models miss default CIF by ~3.4× · F15 closest-work cells remain UNKNOWN with two HIGH threats.

Minor: F12 payoff/maturity shorthand · F14 residual confirmatory register.

---

## Five reviewer reports

### R1 — Credit-risk modeller

**Strengths.** Default proxy is explicitly a research proxy, not a regulatory definition, and the manuscript never slips. Calibration is reported as diagnostic with no evaluation recalibration applied — `evaluation_recalibration: false` is enforced in the protocol and honoured in the artifacts. The §6.1 statement that discrimination coexists with material underprediction is exactly right and is the kind of thing most papers omit.

**Major.** F1, F3, F4. The headline is a 2020 result presented as a period result. A model-risk reader will immediately ask what the deterioration looks like in a non-stress year and will find `+0.00018` for 2019 and a sign reversal in 2021.

**Moderate.** F11 — neither M1 nor M2 conditions on current delinquency state. No deployed monthly mortgage model omits it. The paper is therefore answering "does national macro add to an *origination-only* monthly model", which is a legitimate research question but is not the question a practitioner will think is being asked. State it in §5.1. F13 — the baseline under-predicts 24-month default CIF by a factor of 3.4 (0.0360 observed against 0.0107), which undercuts any implied framing in which M1 is acceptable and M2 is not.

**Required evidence.** Per-year decomposition in main text; unseen-vintage number; calendar interval in §6.4; explicit statement that both models are origination-only.

**Scores.** Originality 3 · Technical quality 4 · Empirical rigor 3 · Clarity 2 · Reproducibility 5 · Significance 3 — **BORDERLINE**

### R2 — Survival / competing-risk statistician

**Strengths.** The estimand discipline is the best part of the paper. Monthly cause probabilities are correctly distinguished from continuous-time hazards. The softmax normalisation guaranteeing `h_D + h_P ≤ 1` is stated and enforced (`no independent-logit rescaling`). The refusal to report a 60-month modelled projection for want of training-duration support is a decision most authors would not make. Controls for cumulative/dynamic AUC include competing-payoff observations and this is stated — the convention is explicit, which is more than most papers manage.

**Major.** F2 as a statistical matter: the pooled monthly AUC and the one-entry-per-facility horizon AUC are different estimands and the paper reports only the one that moves. The frozen CIF-level values (12m: 0.5448 vs 0.5442; 24m: 0.5906 vs 0.5908) are the facility-level discrimination numbers and they show nothing.

**Moderate.** F6 — "historical rolling PIT paths" (plural) implies averaging over entry dates. Zero calendar truncation at the 60-month horizon with an evaluation window ending 2026-02 requires every landmark at or before 2021-02, and near-zero censoring over 24 months is consistent with concentration at the window start. If that is right, the plural is misleading and the CIF result is one path. **This is INFERRED from frozen counts; the entry-month distribution is not published and the authors must confirm it.** F5 — showing 24m and not 12m, when 12m shows no qualitative M2 failure, is horizon selection. A further point the manuscript does not make: the modelled quantity is a *mean of facility-level CIFs* and the observed quantity is a *cohort-level Aalen–Johansen estimate*; these are different functionals and the comparison needs a sentence.

**IPCW.** Weights come from a pooled marginal Kaplan–Meier censoring fit with conditional independent censoring assumed and recorded as unverified. No stabilisation or truncation is documented. Positivity is adequate for B10 (censor survival 0.99922 at 24 months) and is enforced by a `minimum_censor_survival: 0.1` gate; it is more exposed in B6, where 327 of 2,423 facilities are censored. The B6 bootstrap re-estimates censoring and Aalen–Johansen inside each draw, which is correct and better than most practice; weight-estimation uncertainty is therefore partly captured there and not at all for B10's single-pass CIF numbers. Weights do not depend on future information. **Verdict: implementation is plausible and the label is used honestly, but CG03's residual — conditional-entry/IPCW compatibility unconfirmed — is real and should be closed by author review rather than left open (SA12).**

**Censoring classification.** Default (codes 03–99, RA, 02/03/09), payoff/maturity (01), administrative (15/16/96), ambiguous same-month or mismatched-date endpoints quarantined at last confirmed active boundary, unknown states and gaps censored separately, end-of-data at 2026-02. The ambiguity rule censors rather than relabels, which is conservative and correct. Administrative exits could be informative and the paper says the assumption is unverified rather than claiming noninformativeness — appropriate.

**Left truncation.** The design is **landmark conditioning with delayed entry**, not classical left truncation on an origination clock. The protocol is explicit: entry at the first globally eligible `t0` inside the assigned calendar window conditional on still being active, one entry per facility, relative time zero at window entry, with mortgage age at entry recorded separately. `origin_A_assessment` correctly notes that an age-scale cohort would require left-truncated risk sets and that development age support does not cover it. **The current terminology is right; do not rename.** The one gap is that the manuscript uses "conditional entry" throughout without once connecting it to the landmarking literature, which costs it nothing to fix.

**Scores.** Originality 2 · Technical quality 4 · Empirical rigor 3 · Clarity 3 · Reproducibility 5 · Significance 3 — **BORDERLINE**

### R3 — Machine-learning reviewer

**Strengths.** Prespecification is real and verifiable: `prediction_generation_count: 1`, a registration hash, `virgin_holdout: false` stated in the protocol rather than discovered by a referee, and a prohibition list that includes feature search and period fixed effects. This is better pre-registration hygiene than most ML venues require.

**Major.** All four. F4 is the one I would lead with as a referee: the facility bootstrap interval `[+0.01541, +0.01678]` has width 0.0014 around a comparison whose only new inputs are seven month-constant series. The artifact itself records `conditional_on_realized_calendar: true`. Presenting that interval first, and the eight-block calendar interval `[+0.00005, +0.04150]` second, inverts the epistemic order. The calendar interval's lower bound is `5.33e-05` — the direction is preserved by a margin indistinguishable from zero, and percentile block bootstrap on eight heterogeneous blocks (one of which carries 89% of the point estimate) has poor coverage.

**Pseudoreplication.** This is the sharpest available criticism and the paper half-makes it itself. §5.4 says "a national rate or growth measure shared across many rows does not create many independent economic observations" — correct — and then the results section leads with the facility interval anyway. Effective independent macro support is 88 distinct development months, not 1,568,661 intervals. Seven macro coefficients estimated at apparent precision from 88 points, in a window where `unemployment_level` and `hpi_yoy` correlate at −0.919 and four features carry VIF 10–12, is the mechanism. The paper has the ingredients (F7) and never assembles them.

**Baseline fairness: PARTIALLY RESOLVED.** M1 and M2 share `C=1.0`, L2, lbfgs, `max_iter=3000` — identical regularisation, no per-model tuning, which removes the crudest unfairness. But identical regularisation is not *equivalent* regularisation when the added block has effective support of 88 rather than 1.5M. No complexity-adjusted development comparison exists. The development "improvement" of 0.00056 for seven parameters is not shown to exceed what degrees of freedom buy. SA04.

**Model class.** The manuscript mostly observes the right boundary but §9's "the evidence supports keeping proper scoring and calibration visible" and the conclusion's framing drift toward a general lesson. Evidence establishes *this frozen macro-augmented linear specification failed in this cohort*. With interactions and period effects prohibited and no time-varying coefficients, functional-form rigidity is an unexcluded alternative and must be named as such.

**Post-hoc handling.** Task 11's labelling is exemplary: `POST_VALIDATION_DIAGNOSTIC`, same-sample oracle explicitly optimistic, `fraction_of_frozen_excess_removed: 0.273` reported rather than spun. No complaint.

**Scores.** Originality 2 · Technical quality 3 · Empirical rigor 3 · Clarity 2 · Reproducibility 5 · Significance 3 — **WEAK_REJECT** (recoverable to WEAK_ACCEPT with CH01–CH04 and SA01)

### R4 — Mortgage economist

**Strengths.** Competing-risk treatment of payoff is correct and the refusal to call code 01 "refinancing" is right. The original-coupon gap is honestly labelled `ORIGINAL_CONTRACT_RATE_PROXY_GAP`.

**Major.** F1 and F2. On F1: 2019-01 entry, 2006–2014 vintages, still alive and unprepaid — this is a **burnout cohort by construction**. These are precisely the borrowers who did not refinance through the 2012–2016 rate troughs. Then 2020 arrives with the largest refinancing wave in the series and a model keyed to national rates over-predicts payoff by 4.5× in that year (predicted 6.93% against observed 2.38% in the 2020–2021 cell). The manuscript lists burnout omission in §10 limitations and never connects it to the mechanism, even though Stanton1995 and Deng2000 — both already cited — predict exactly this failure mode.

A related point the paper misses: the 2020 observed monthly payoff rate in this cohort (2.21% in the §7.2 cell) is *low* for a national refinancing boom, which is itself evidence of burnout selection and is the strongest single piece of support for the economic story the paper is not telling.

**Moderate.** F7. Development spans Sep 2010–Dec 2017: unemployment falling monotonically, rates range-bound, HPI steadily positive. In that window the macro block is close to a time index. In evaluation, period and the macro variables decouple. **The defensible economic statement is that macro features estimated over a monotone recovery regime act as a proxy for calendar trend and do not transport when the trend breaks** — much sharper than "temporal transport failed".

**APC.** `period = cohort + age` is acknowledged and the rank deficiency is documented. With duration bands, vintage indicators and month-constant macro in one linear predictor, the macro coefficients are not separable from calendar and seasoning. **M2 must be described throughout as a macro-augmented predictive model, never a macro-effect model.** The manuscript mostly does this; §7.2's discussion of coefficient signs drifts toward effect language and should be tightened.

**Seven-vintage design.** 2006, 2008, 2010, 2014, 2018, 2020, 2022 skip 2007, 2009, 2011–2013, 2015–2017 — so the sample straddles the GFC without covering its approach, and the primary evaluation uses only four of the seven. Calling this a seven-vintage study in §4.1 while evaluating on four in §5.2 obscures narrower support. Unequal follow-up across vintages interacts with the duration bands. Vintage selection appears to precede outcome analysis (hash-salted, documented in the recovery manifest), so I do not allege post-hoc vintage picking, but the gaps should be justified.

**Payoff/maturity.** For 2006–2014 originations on predominantly 30-year terms evaluated in 2019–2026, scheduled maturity is negligible; the pooling is empirically benign in the primary population though 2006/2008 include shorter terms. **Substantive risk: low. Terminological risk: real.** Use "payoff/maturity termination" on first use per section (CH16, SA10).

**Scores.** Originality 2 · Technical quality 3 · Empirical rigor 3 · Clarity 3 · Reproducibility 4 · Significance 3 — **BORDERLINE**

### R5 — Reproducibility / model-risk reviewer

**Strengths.** Outstanding, and I want to be unambiguous about it: hash-pinned source artifacts with `json_pointer` and `source_commit` per claim; a 258-claim evidence registry with allowed and prohibited language per claim; 123 numeric bindings tying manuscript figures to frozen values; preservation manifests across tasks; a consumption ledger with `prediction_generation_count`. The `<!-- NUM: Nxxxx -->` binding mechanism is a genuinely good idea. I have reviewed submissions from institutions with model-risk functions that document less.

**Major.** F3 and F4 are governance failures as much as scientific ones. The frozen protocol decision rule states: `positive: Both facility and calendar 95 CI upper<0`. The study's own rule treats the calendar interval as co-primary. Reporting it for the exploratory result (§8) and not the primary one (§6.4) is inconsistent with the rule, and `docs/paper/headline_claim_audit.json` records the caveat for H7 but not for H4. Similarly, a frozen supplementary population that reverses the primary metric is exactly what a model-risk register exists to surface.

**PIT verdict.** Partially defensible. ALFRED `vintagedates` and real-time `observations` with `output_type=1` were pulled for all six series, so the inputs are genuinely vintage-aware and revision-aware — better than I expected. **But `release_lags` records `count: 0` and `evidence_quality: UNMEASURED_EXACT_PROVIDER_DATES` for every series**, with the explicit reason that no certified initial provider release dates were obtained. "Point-in-time" in a credit-risk venue implies verified availability at the assessment date. Safer terminology: **vintage-aware** or **revision-aware observed macro inputs**. Alternatively keep PIT but define it in one sentence as ALFRED vintage-date availability and state that provider release timestamps were not certified. CG06 remains PARTIALLY_RESOLVED and this is why.

**Retrospective mortgage knowledge time.** `knowledge_time: UNVERIFIED` is recorded in the survival protocol and §9 states the limitation well. Static origination variables are the safest; modification flags and termination fields are the exposure. Termination fields are the event definition, not covariates, so the exposure is bounded. Severity: MODERATE, adequately disclosed.

**Reproducibility.** Code reproducibility is high. Data reproducibility is bounded by provider access and this is disclosed. An independent researcher with Freddie access could reconstruct cohort, features, splits and metrics from the frozen specifications; without access they can verify nothing empirical. That is the correct and honest position.

**Publication governance.** `[PUBLICATION TERMS REVIEW REQUIRED]` and `[ETHICS AND AUTHOR REVIEW REQUIRED]` are submission-blocking **governance** items, not scientific ones. The manuscript makes no unsupported anonymisation, PII, IRB or ethics-approval claim — it explicitly disclaims each. Author names and affiliations are still `[AUTHOR REVIEW REQUIRED]`. Fannie remains DRAFT_NOT_YET_AUTHORIZED with no results, correctly.

**Scores.** Originality 2 · Technical quality 4 · Empirical rigor 3 · Clarity 3 · Reproducibility 5 · Significance 3 — **BORDERLINE**

---

## Central claim attack

**Claim under attack:** *PIT macro augmentation improved development fit but worsened temporal probability quality relative to the mortgage baseline.*

**Verdict: SURVIVES WITH SUBSTANTIALLY NARROWED SCOPE.** The claim is true for the seen-vintage 2006–2014 survivor cohort evaluated over 2019–2026, driven predominantly by 2020. It is not true as a study-level statement, because the unseen-vintage population reverses the primary metric.

| Alternative explanation | Classification | Basis |
|---|---|---|
| Pandemic / nonstationarity | **PRIMARY CONTRIBUTOR, understated in manuscript** | 2020 = 89.3% of delta; ex-2020 `+0.00172`; 2021 reverses |
| Macro support extrapolation | **PRIMARY CONTRIBUTOR, acknowledged** | 77/86 months and 202,675/248,939 intervals outside reference; eval mean Mahalanobis² 298.5 vs dev 7.0 |
| Changing macro correlation structure | **PRIMARY CONTRIBUTOR, understated** | dev unemployment–HPI −0.919, VIF 10–12, condition 69.1; four pairs reverse sign in evaluation |
| Survivor selection / burnout | **MATERIAL, underdeveloped** | Cohort = 2006–2014 originations unprepaid to 2019; burnout omitted by design |
| Seasoning / misspecified duration | **PLAUSIBLE, not excluded** | Fixed duration bands; APC rank deficiency documented |
| Cohort composition / vintage effects | **PLAUSIBLE, not excluded** | Four of seven vintages; unseen-vintage reversal is direct evidence this matters |
| Model functional-form rigidity | **NOT EXCLUDED** | Interactions and period effects prohibited; no time-varying coefficients; only one family fitted |
| Interaction omission | **NOT EXCLUDED** | Prohibited by protocol |
| Different baseline flexibility | **PARTIALLY EXCLUDED** | Identical `C=1.0` L2 across rungs, but no complexity adjustment and unequal effective support |
| Calendar effects (non-pandemic) | **MINOR** | 2022–2026 deltas `+0.005` to `+0.008`, consistent direction, small magnitude |
| Prior evaluation exposure | **DISCLOSED, MINOR** | `virgin_holdout: false`; `prediction_generation_count: 1`; registration hash |
| Endpoint construction | **MINOR** | Payoff/maturity pooling benign for these vintages/terms |
| Censoring | **MINOR for B10** | Censor survival 0.99922 at 24m; more exposed in B6 |
| Historical data revisions | **MINOR** | ALFRED vintages retrieved; release dates uncertified (F8) |
| Structural prepayment change | **PLAUSIBLE, not separable** | Confounded with 2020 and with burnout |

The empirical result survives all of these. **What does not survive is the implied generality.** The honest statement is: *in a seasoned survivor cohort, a near-collinear national macro block estimated over a monotone post-GFC recovery window failed to transport through an unprecedented macro excursion, while improving in-sample fit.* That is a sharper, more useful and more defensible paper than the one currently written.

---

## Estimand audit

| Experiment | Population | Entry | Time origin | Horizon | Event | Competing | Censoring | Conditioning | Language matches? |
|---|---|---|---|---|---|---|---|---|---|
| B5 12-month PD | 2010 vintage, 2,423 eval facilities | Monthly landmark, active & incident-eligible | Landmark month | 12 months | Default proxy before payoff | Payoff ends eligibility | Admin/unknown/gap; horizon unascertainable | Static + duration | **YES** |
| B6 default CIF | 2010 vintage, 2,423 eval facilities | First eligible `t0` in window, conditional on active | Window entry | 12/24/36/60 | Default proxy | Payoff/maturity | Pooled KM, conditional independence | Entry-time static + deterministic duration | **YES** |
| B6 payoff CIF | as above | as above | as above | as above | Payoff/maturity | Default | as above | as above | **YES** |
| B10 monthly hazards | 4 vintages, 5,619 eval facilities | Eligible consecutive monthly interval | Interval start | 1 month | Default proxy | Payoff/maturity | as above | Static + duration + cohort + (M2) month-constant macro | **YES** |
| B10 temporal proper scores | as above | as above | as above | 1 month, pooled over 86 months | multinomial class | — | — | **Equal interval weighting, unstated** | **PARTIAL** (F9, F10) |
| B10 12/24/36/60m CIF | 5,619 facilities, first eligible entry | as above | Entry month | 12/24/36/60 | Default proxy | Payoff/maturity | Pooled KM; AJ observed reference | Rolling historical PIT path | **PARTIAL** (F6 plural "paths"; mean-of-individual vs cohort AJ) |
| B12 refinancing | Same arrays as B10 | as above | as above | 1 month + CIF | as above | as above | as above | + asymmetric coupon-gap proxy | **YES** (EXPLORATORY_ONLY honoured) |

**Estimand verdict: SOUND.** Two partials, both reporting rather than specification defects. This is the strongest technical area of the paper.

---

## Conditional entry — HIGH PRIORITY

- Loans are conditioned on surviving to an observed landmark: **YES**, explicitly.
- Older loans are necessarily selected survivors: **YES**. The primary evaluation cohort is 2006–2014 originations still active and unprepaid at the start of the 2019 window.
- Entry depends on prior performance: **YES, indirectly** — survival without default or payoff is the entry condition. It does not depend on prior performance *within* the evaluation window.
- Conditional entry induces survivor selection: **YES**, and in a direction that matters: it selects against prepayment propensity, i.e. it selects for burnout.
- Predictions are conditional on being active at entry: **YES**, correctly.
- Manuscript ever implies origination-lifetime inference: **NO.** §3, §6.2 and §10 all refuse it, repeatedly and correctly.
- Temporal differences in age-at-entry could materially influence results: **YES**, and this is the underdeveloped link. Roles are facility-disjoint by hash, so development and evaluation facilities are different loans drawn from the same vintages at different calendar times, meaning age-at-entry differs systematically between splits.

**Severity: MODERATE.** The specification is correct and honestly described. The deficiency is interpretive — survivor selection is listed as a limitation rather than used as an explanation, when it is probably the most economically coherent part of the mechanism (R4).

---

## Temporal split and facility overlap

Development 2010-09–2017-12 · purge 2018 · evaluation 2019-01–2026-02.

**Facility overlap: NONE.** Verified from the protocol, not assumed. Role assignment is `SHA256(salt + ID) mod 10`, with `role_salt: track-b-macro-eligibility-v1` for B10 and `track-b-pd-v1:` for B6 (0–6 development, 7–9 evaluation). **The split is therefore both facility-disjoint and calendar-separated**, which is stronger than most temporal-validation designs and than the manuscript claims for itself. No facility history bridges the purge, because no facility appears on both sides at all.

**Purge rationale.** With facility-disjoint roles the purge year is partly redundant for leakage — there is no shared facility to leak through. Its real function is to separate the macro and composition regimes either side of the boundary. The manuscript does not explain this and should: a referee will ask what a purge prevents when roles are already disjoint.

**Consequence worth stating:** because roles are disjoint, split differences cannot be read as matched-facility attrition. §7.1 states this correctly.

**Verdict: SOUND, UNDER-SOLD.** Report the hash-based disjointness explicitly in §5.2; it is a strength currently left implicit.

---

## Prior outcome exposure

Recorded: `virgin_holdout: false`; prior exposure state `CONSUMED` with `registration_sha256`; `prediction_generation_count: 1`; per-file code hashes at registration; Task 3/4/5 aggregate outcomes previously inspected; macro support design had documented exposure to counts; Task 11 post-hoc on inspected outcomes; Task 12 exploratory hypothesis from inspected outcomes.

**Maximum defensible language: "frozen temporal evaluation on a non-virgin holdout".** Never "confirmatory temporal validation". The manuscript mostly complies; the abstract's "demonstrate" and the conclusion's register drift (CH17).

**Verdict: WELL-GOVERNED, MINOR residual.** Task 11 and Task 12 are appropriately downgraded. No evidence of model choices changed after inspection — `prediction_generation_count: 1` and `no_retuning: true` are the right controls and are recorded.

---

## Multiple testing and researcher degrees of freedom

Prespecified and frozen before evaluation: M0/M1/M2 ladder, RATE and REDUCED sensitivities, leave-vintage-out, pandemic and GFC windows, bootstrap seeds and units, decision rule, cell suppression. Post-hoc: all of Task 11. Exploratory: all of Task 12, with its own prespecification hash but on inspected arrays.

Prohibited and apparently honoured: feature search, period fixed effects, interactions, future loan states.

**Does the final story selectively emphasise favourable results?** Not through model search — the ladder is fixed and the decision rule was set in advance. **But yes through reporting selection**, in four places: the 24-month CIF horizon (F5), the facility interval over the calendar interval (F4), the primary payoff Brier calendar interval omitted while the exploratory one is disclosed (F4), and the unseen-vintage reversal omitted entirely (F3). **Task 14 successfully prevented analytic p-hacking and did not prevent presentational selection.** That distinction is worth the authors' attention, because the governance machinery was designed for the first problem and the manuscript has the second.

---

## APC and macro identification

`period = cohort proxy + age proxy` holds exactly; rank deficiency is documented; vintage reference 2006, duration reference 0–12, unrepresented categories zeroed and flagged.

**What is identified:** a predictive mapping only. Macro coefficients are not separable from calendar, cohort and seasoning. Development VIFs of 10–12 across four of seven macro terms, condition number 69.1, and `unemployment_level`–`hpi_yoy` at −0.919 mean the macro block is internally near-collinear *as well as* confounded with the APC clock.

**Verdict: M2 must be described as a macro-augmented predictive model throughout. Never a macro-effect model.** The manuscript achieves this in most places; §7.2's coefficient-sign discussion needs tightening (CH12). The spread exclusion (`spread = mortgage rate − treasury`) is correctly handled — dropped from the primary coefficient vector, retained for eligibility — so exact redundancy is avoided. Remaining multicollinearity undermines coefficient interpretation but not predictive validity, and the manuscript does not attempt coefficient interpretation as effects.

---

## Macro pseudoreplication and uncertainty

**The single strongest methodological criticism, and the paper has the ingredients without assembling them.**

- Thousands of facilities share identical national macro values each month.
- Effective independent macro support ≈ **88 distinct development months**, not 1,568,661 intervals.
- Facility bootstrap (5,619 clusters, 1,000 draws) preserves within-loan dependence: **YES**.
- Facility bootstrap addresses calendar dependence: **NO**. The artifact records `conditional_on_realized_calendar: true`.
- Therefore `[+0.01541, +0.01678]` is **facility-composition uncertainty for a fixed pair of models on one realised calendar**, and cannot be read as uncertainty about macro transport.
- Calendar-year blocks: 8, including partial 2026, one of which (2020) carries 89.3% of the point estimate. Percentile block bootstrap on 8 heterogeneous blocks has poor coverage and the manuscript should say so.

**Verdict: the calendar interval is the one that matches the claim and must lead. The facility interval must be explicitly demoted and labelled.** The paper does not need causal macro inference, but its uncertainty language must match its design — and currently the narrow interval does rhetorical work the design cannot support.

---

## Calibration and proper scores

Computed at interval level; repeated facilities not accounted for in the point calibration summaries; `point_calibration_CI: null` and `diagnostic_refit_coefficient_CI: null` are recorded honestly rather than fabricated. Diagnostic intercept/slope only, no evaluation recalibration — enforced and honoured. Competing outcomes handled through the shared multinomial normalisation.

**Calibration slope under severe imbalance** (1.5% payoff, 0.11% default) is weakly identified and dominated by the top bin; the frozen reliability tables show this clearly. The manuscript does not overstate precision — it reports no intervals and says so — but it should note the imbalance when interpreting the 0.509 → 0.249 slope change.

**Brier audit.** Three distinct Brier objects appear: monthly cause-specific (Table T5, T6, T7), horizon-specific IPCW (Table T4, from B6), and horizon-specific IPCW within the B10 CIF block. They are never compared across types in the text, which is correct, but T4 and T5 sit close together with "Brier" unqualified in both. Label each variant (CH05/CH16 adjacent). No conceptual conflation found.

**Log-loss audit.** Three outcome categories; softmax normalisation; censored intervals excluded from the risk set rather than contributing; aggregation is an unweighted mean over intervals; no facility weighting. Long-lived mortgages contribute disproportionately (mean 44.3 intervals per facility, max ~86), and because surviving long is the complement of the modelled payoff event, **the weighting is correlated with the outcome**. This is not necessarily wrong but it makes the headline a long-duration-weighted quantity, and it is unstated. **Classification: facility-weighted sensitivity is STRONGLY_RECOMMENDED, not submission-blocking** (SA05).

---

## CIF interpretation

Observed comparator is **Aalen–Johansen** with generic `entry < t ≤ exit` risk sets and payoff treated as a competing event rather than censoring — correct. Compatible with conditional entry in construction; CG03's residual concerns implementation equivalence, not specification.

Covariates frozen at entry; duration advances deterministically; **no future loan state is used** (`prohibited: future loan states`). Macro varies along the observed path, each interval using information available at its own assessment date. **This is a retrospective sequential mapping, not a prospective forecast, and the manuscript says so clearly in the abstract, §3 and §5.5. That distinction is sufficient as stated.** No language suggesting a deployable 24-month forecast was found.

**Default-CIF propagation.** The manuscript already phrases this correctly in §3 ("an algebraic property of the model's probability recursion ... not a causal claim") and §6.5. **Verdict: correctly framed, but the abstract then presents the 24-month CIF as a third finding alongside log loss and AUC.** It is the arithmetic consequence of the monthly payoff over-prediction (2.830% predicted against 1.536% observed) compounded along a path through 2020. Demote it in the abstract to a consequence (CH05).

---

## Payoff AUC interpretation

Pooled 0.5654 → 0.6259, `+0.0604`.

| Year | M1 | M2 | Gap |
|---|---|---|---|
| 2019 | 0.539 | 0.546 | +0.007 |
| 2020 | 0.589 | 0.593 | +0.004 |
| 2021 | 0.565 | 0.561 | −0.004 |
| 2022 | 0.493 | 0.558 | +0.065 |
| 2023 | 0.494 | 0.480 | −0.014 |
| 2024 | 0.493 | 0.496 | +0.003 |
| 2025 | 0.481 | 0.477 | −0.004 |
| 2026 | 0.482 | 0.473 | −0.009 |

Facility-level 24-month CIF cumulative/dynamic payoff AUC: **M1 0.5906, M2 0.5908**. At 12 months: 0.5448 vs 0.5442.

AUC is not decomposable into within- and between-period parts, so I do not report a weighted average and I do not claim an exact decomposition. But the pattern is unambiguous: a pooled gain of +0.060 coexists with per-year gaps that are near zero or negative in seven of eight years, and with no gain whatsoever at the facility level. Since M2 adds only month-constant terms and interactions are prohibited, a facility-level ranking gain was **structurally unavailable by construction**.

**Economic meaningfulness: negligible.** An M1 baseline at 0.565 barely ranks payoff at all; the CIF-level comparison shows no improvement; and ranking calendar periods is not a capability a lender can act on, because the calendar is observed.

**Verdict: the number survives, the interpretation does not.** This is the one PRIMARY claim classified REQUIRES_NEW_ANALYSIS. SA01 (within-month or calendar-stratified AUC on frozen arrays) is submission-blocking, or the claim must be restated as across-period ranking everywhere it appears.

---

## Refinancing experiment

Honestly labelled EXPLORATORY_ONLY with its own prespecification hash; the mixed decision is reported without spin; §8 even volunteers that the payoff Brier calendar interval includes zero — the disclosure the primary section omits.

Gap distribution shift is severe: interval-weighted mean 1.066pp → 0.014pp, PSI 0.5933. **This makes the exploratory failure close to unsurprising — it is substantially an extrapolation result**, and §8's modest interpretation is appropriate.

Attack surface: original coupon is not the current contract rate; modification flags unaccounted; refinance motive unobserved; burnout omitted; calendar confounding identical to the primary design; hypothesis generated from inspected outcomes.

**Placement verdict: MOVE TO APPENDIX.** It adds a fourth mixed result to a paper already carrying a complicated message, and a hostile reviewer will read it as padding. Its one real contribution — that replacing a broad macro block with a targeted incentive proxy does not restore transport — is two sentences in §9. Keep those two sentences; move the rest, with its sensitivities, to Appendix F.

---

## Novelty

**Strongest argument against novelty.** Every component is established. Competing risks in mortgages: Deng1996, Deng2000. Macro conditioning in credit survival: Bellotti2009, Sadhwani2021, Breeden2022. Temporal stability with APC inputs on Freddie: Breeden2023, Wang2024. Freddie survival with calibration under drift: Peng2026. Real-time revision-aware macro in credit: Bianchi2026, Croushore2001. Competing-risk scoring and calibration: Heyard2020, Blanche2013, Gerds2012. The model is an unregularised-search multinomial logistic with no methodological contribution, the paper says so, and the "distinctive combination" rests on cells recorded as UNKNOWN in the authors' own matrix.

**Strongest defence on demonstrated evidence.** A prespecified, frozen, natural-calendar temporal evaluation of vintage-aware macro augmentation in a competing-risk mortgage model, assessed jointly on proper scores, calibration and cumulative incidence, with a published claim-level evidence registry and a consumption ledger. Peng2026 — the nearest neighbour — explicitly lists macro conditioning and competing termination as future work, which makes this the direct empirical answer to a stated open question.

**Most defensible framing: an empirical validation study reporting a negative result, with reproducibility governance as a secondary methodological contribution.** Not methodological novelty. Not a priority claim. The negative result is publishable because of prespecification, the frozen natural calendar break, the competing-event consequences, and the audit trail — exactly the list in the brief's §50, and all four are real here.

**Bu et al. 2026: IMPORTANT, not BLOCKING.** `threat_to_novelty: HIGH` with PIT_macro, temporal_holdout, proper_scores, calibration and distribution_shift_analysis all UNKNOWN. UNKNOWN stays UNKNOWN — I do not assume this paper differs. For an arXiv preprint, obtaining the full text is not a precondition. For peer review it is: a referee may well be an author of it. Obtain it before journal submission; drop the search-completeness narration either way (CH14).

**Peng / Lessmann.** Weakens temporal-validation novelty and calibration novelty on Freddie substantially — they have done Freddie survival, calibration and drift. What remains distinctive: competing termination rather than default-only, a natural calendar break rather than simulated drift, vintage-aware observed macro rather than none, and CIF consequences. That is a narrower but real gap, and §1 should name it (CH15).

---

## Practical significance

Temporal joint log loss 0.08920 → 0.10532 is an **18.1% relative deterioration** (derivable from frozen values). Ex-2020 it is `+0.00172`, a **1.9% relative deterioration**. Both figures should appear; quoting only the first characterises a pandemic year as a period result.

The paper discusses statistical metrics only. §9 gestures at model-risk practice but there is no economic or operational significance analysis — no loss impact, no decision impact, no monitoring-threshold implication. Given the frozen scope (`prohibited: EAD/LGD/ECL`, no funded economics) **inventing one would be worse than omitting it**, and the manuscript is right not to. But §9 should state plainly that economic significance was not assessed, rather than leaving the reader to infer materiality from a log-loss difference.

---

## Required sensitivities

**SUBMISSION_BLOCKING** — SA01 within-period payoff AUC (scoring pass) · SA02 ex-2020 difference with calendar blocks (recombination + resampling) · SA03 unseen-vintage full reporting (reporting) · SA06 landmark entry-month distribution (reporting).

**STRONGLY_RECOMMENDED** — SA04 complexity-adjusted or calendar-block-regularised comparison · SA05 facility-weighted loss · SA08 support-restricted evaluation · SA12 CG03 implementation review.

**OPTIONAL** — SA07 current-state conditioning · SA09 nonlinear/interaction baseline · SA10 maturity separation.

**NOT_NEEDED** for a Freddie-scoped claim — SA11 Fannie replication.

Note that all four blocking items are **reporting, recombination or scoring on existing frozen arrays**. None requires a refit. This is a favourable position: the gap between the current manuscript and a defensible one is largely disclosure.

---

## Minimum sufficient revision

**Manuscript-only (no new analysis):**

1. CH01 — per-year decomposition table and ex-2020 residual in main text; qualify abstract and conclusion.
2. CH02 — report the unseen-vintage reversal with its caveats.
3. CH04 — report the calendar-block interval for the primary payoff Brier; lead with calendar intervals; demote and relabel the facility interval.
4. CH05 — show all four CIF horizons; demote the CIF in the abstract to a consequence.
5. CH03 (restatement route) — restate the payoff AUC claim as across-period ranking and report the frozen CIF-level AUCs.
6. CH07 — replace "point-in-time" with "vintage-aware", or define PIT explicitly and note uncertified release dates.
7. CH10, CH12, CH13, CH14, CH17, CH18, CH19.

**Requires new work (none a refit):** SA06 entry-month distribution (reporting); SA01 if the authors prefer the measurement route over restatement; SA02 ex-2020 interval.

**Deliberately not recommended:** SA07 and SA09 are good follow-on papers, not preconditions. Demanding every possible sensitivity would be the wrong review.

---

## Figure and table plan

| Item | Verdict | Note |
|---|---|---|
| F1 architecture | KEEP | Must show hash-based role disjointness, currently invisible |
| F2 state/estimand diagram | KEEP | Strongest pedagogical asset |
| F3 development vs temporal scores | **MODIFY** | Lead with calendar intervals; mark the facility interval as conditional |
| F4 macro support diagnostics | KEEP | Diagnostic labelling already correct |
| F5 payoff ranking and calibration | **MODIFY** | Must show per-year AUC, not pooled only |
| F6 conditional CIF | **MODIFY** | All four horizons; label the calendar span of the path |
| F7 refinancing | **MOVE_TO_APPENDIX** | With §8 |
| F8 proposed Fannie | **DROP** | A figure for work with no results invites the obvious question |
| **NEW_FIGURE_NEEDED** | — | Per-year joint log-loss delta with interval weights — the single most informative plot available and currently absent |

| Table | Verdict |
|---|---|
| T1 populations | KEEP; add unseen-vintage row |
| T2 specifications | KEEP; add effective macro support (88 months) |
| T3 PD temporal | **MOVE_TO_SUPPLEMENT** (CH20) |
| T4 structural horizons | KEEP; label Brier variant |
| T5 macro comparison | KEEP; add M0 row commentary and per-year companion |
| T6 payoff calibration | KEEP; note imbalance effect on slope |
| T7 refinancing | **MOVE_TO_APPENDIX** |
| T8 interpretation boundaries | KEEP — best table in the paper |
| **MISSING** | Per-year decomposition; all-horizon CIF comparison; seen vs unseen vintage |

---

## Abstract audit (sentence level)

| Sentence | Classification |
|---|---|
| "Longitudinal mortgage risk requires calibrated probabilities..." | SUPPORTED |
| "An economically plausible predictor can improve ranking without improving the probabilities..." | **OVERSTATED** — the ranking improvement is across-period (F2) |
| "We investigate whether a frozen point-in-time macroeconomic feature set..." | **SUPPORTED_WITH_QUALIFICATION** — "point-in-time" (F8) |
| "The study uses monthly mortgage histories, a research default proxy..." | SUPPORTED |
| "Evaluation separates development, a purge period, and a later temporal period..." | SUPPORTED |
| "Macro augmentation improved development joint log loss from 0.09015 to 0.08959..." | **SUPPORTED_WITH_QUALIFICATION** — "improved" unadjusted for complexity |
| "...but temporal log loss deteriorated from 0.08920 to 0.10532." | **SUPPORTED_WITH_QUALIFICATION** — 89.3% from 2020; reverses on unseen vintages |
| "Temporal payoff AUC nevertheless increased from 0.565 to 0.626..." | **OVERSTATED** — F2 |
| "...while payoff Brier score worsened." | **SUPPORTED_WITH_QUALIFICATION** — calendar interval includes zero |
| "In the frozen conditional-entry comparison using historical rolling macro paths..." | **AMBIGUOUS** — "paths" plural (F6) |
| "...observed 33.7%, baseline 25.9%, macro 75.8%." | **SUPPORTED_WITH_QUALIFICATION** — one horizon of four (F5) |
| "These results demonstrate a design-specific divergence..." | **OVERSTATED** — "demonstrate" on a non-virgin holdout (F14) |
| "Interpretation is restricted by conditional entry, retrospective mortgage knowledge time..." | SUPPORTED |
| "Historical rolling-path incidence comparisons are not prospective macro forecasts." | SUPPORTED |
| "The evidence supports joint assessment of competing-event probabilities..." | SUPPORTED |
| "...it does not establish a causal explanation or cross-provider transportability." | SUPPORTED |

Three OVERSTATED, one AMBIGUOUS, six SUPPORTED_WITH_QUALIFICATION out of sixteen.

---

## Title

"Temporal Transport of Mortgage Competing-Risk Probabilities with Point-in-Time Macroeconomic Information" has two problems: "temporal transport" names a topic rather than a result, and "point-in-time" overstates the provenance evidence (F8).

**Recommended:** *Vintage-aware macroeconomic features improve in-sample fit but not temporal probability quality in a seasoned Freddie Mac competing-risk cohort.*

States the finding, names the scope, drops the overclaimed label. "Out-of-Time Evaluation..." would be safer than the current title but still does not state the result.

---

## Meta-review

**Strongest contribution.** A prespecified, frozen, fully auditable negative result on a real calendar break, with reproducibility governance that is itself worth publishing.

**Strongest weakness.** The manuscript's framing is not the framing its own evidence supports. Three findings that cut against the headline — the 2020 concentration, the between-period nature of the AUC gain, and the unseen-vintage reversal — are present in the frozen artifacts and absent from the paper. None was hidden; all are one query away in files the authors published. That is the saving grace and also the problem: a referee will find them, and will ask why they had to.

**Is the central result believable?** **Yes.** The arithmetic is correct, the design is sound, the governance is real. I attacked the numbers and they held.

**Is the interpretation appropriately bounded?** **Not yet.** The paper is scrupulous about the boundaries it has identified — causality, lifetime risk, regulatory use, cross-provider transport — and silent about three it has not: calendar concentration, between-period versus within-period ranking, and the vintage-population boundary.

There is an irony worth naming. This manuscript hedges more than any I have reviewed, and it is nonetheless overclaiming — because the hedges address abstract threats while the concrete ones go unmentioned. Removing half the caveats and adding these three numbers would produce a paper that is simultaneously shorter, bolder and more defensible.

**Before preprint:** CH01–CH05, CH07, CH17; SA06 reporting.
**Before peer review:** additionally SA01, SA02, SA03; CH08–CH16; obtain Bu2026 full text; close CG03 by author review; resolve the publication-terms and author-review flags.

---

## Reviewer scores

| | R1 Credit risk | R2 Survival | R3 ML | R4 Mortgage econ | R5 Reproducibility |
|---|---|---|---|---|---|
| Originality | 3 | 2 | 2 | 2 | 2 |
| Technical quality | 4 | 4 | 3 | 3 | 4 |
| Empirical rigor | 3 | 3 | 3 | 3 | 3 |
| Clarity | 2 | 3 | 2 | 3 | 3 |
| Reproducibility | 5 | 5 | 5 | 4 | 5 |
| Significance | 3 | 3 | 3 | 3 | 3 |
| **Recommendation** | BORDERLINE | BORDERLINE | WEAK_REJECT | BORDERLINE | BORDERLINE |

Internal stress test, not a prediction of actual peer review.

---

## Recommended next task

**Task 18: manuscript revision to v0.3** implementing CH01–CH05, CH07 and CH17, plus the SA06 entry-month reporting. These are disclosure changes against already-frozen evidence and require no new experiment.

Defer SA01/SA02/SA04 to a separately registered Task 19 so that any scoring pass on frozen arrays is itself frozen and logged, rather than folded into a revision pass — the same discipline that has served this project well so far.
