"""Task 17 register builder.

Outputs live under reports/paper/review/ rather than the docs/paper/review/ path named
in the Task 17 brief: docs/paper/ is governed by the Task 16V frozen guard
(scripts/check_paper_evidence.py, hash-pinned and asserted immutable by
tests/test_literature.py::test_approved_checker_hash_does_not_allow_further_edits),
and creating a new subdirectory there would require loosening that freeze. See the
Deviations section of reports/paper/TASK17_HOSTILE_SCIENTIFIC_REVIEW.md.

Emits the three Task 17 JSON registers from a single declarative source so that
claim identifiers, severities and survival verdicts cannot drift between files.

This script performs NO empirical computation beyond arithmetic recombination of
values already frozen in Task 10/11/12 artifacts (interval-weighted
decomposition of an already-reported mean). It fits no models, reads no private
data and mutates no frozen artifact.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
REVIEW = ROOT / "reports" / "paper" / "review"
REGISTRY = ROOT / "docs" / "paper" / "track_b_claim_evidence_registry.json"
MACRO = ROOT / "reports" / "track_b" / "macro_competing_risk_validation.json"

BASE_COMMIT = "a4479ba4be115bc9a1d4b53ab77e258b40bd57c7"
TASK16_DECISION = "DISTINCTIVE COMBINATION PLAUSIBLE WITH MATERIAL LIMITATIONS"

# --------------------------------------------------------------------------
# Cross-cutting reviewer findings. Each is traced to a frozen artifact.
# --------------------------------------------------------------------------

FINDINGS = {
    "F1_CALENDAR_CONCENTRATION": {
        "title": "2020 contributes 89.3% of the headline temporal deterioration",
        "severity": "MAJOR",
        "reviewers": ["R1", "R3", "R4"],
        "evidence": "reports/track_b/macro_competing_risk_validation.json#/stability/calendar",
        "derivation": (
            "Interval-weighted mean of the frozen per-year joint log losses reconstructs the "
            "frozen paired delta exactly (+0.01612163642779664). 2020 weight 51809/248939 = 0.2081 "
            "and per-year delta +0.06920 contribute +0.01440, i.e. 89.3% of the total. "
            "Excluding 2020 the residual delta is +0.00172 (~1.9% of the M1 temporal loss 0.08920 "
            "rather than ~18.1%). 2021 reverses sign (M1 0.13513 vs M2 0.12905)."
        ),
        "manuscript_conflict": (
            "Section 7.3 states the frozen hypothesis assessment 'does not support pandemic-only ... "
            "explanations' and section 7.2 cautions 'against attributing the entire result to one acute "
            "calendar episode'. Both statements are defensible as stated (H7 is NOT_SUPPORTED because "
            "direction persists post-2022), but the manuscript never discloses the magnitude "
            "concentration, and the abstract/conclusion present the aggregate as characterising "
            "2019-2026 as a whole."
        ),
        "arithmetic_only": True,
    },
    "F2_BETWEEN_PERIOD_AUC": {
        "title": "The payoff AUC gain is between-period, not facility-level discrimination",
        "severity": "MAJOR",
        "reviewers": ["R3", "R4", "R2"],
        "evidence": (
            "reports/track_b/macro_competing_risk_validation.json#/stability/calendar ; "
            "#/cif/horizons/24/models ; docs/track_b/macro_competing_risk_protocol.json#/ladder ; "
            "#/prohibited"
        ),
        "derivation": (
            "M2 differs from M1 by exactly seven national macro terms that are constant within a "
            "calendar month, and the protocol prohibits interactions and period fixed effects, so "
            "within-month facility ordering can change only through refitted non-macro coefficients. "
            "Frozen per-year payoff AUC gaps are 2019 +0.007, 2020 +0.004, 2021 -0.004, 2022 +0.065, "
            "2023 -0.014, 2024 +0.003, 2025 -0.004, 2026 -0.009, against a pooled gain of +0.060. At "
            "the one-entry-per-facility 24-month CIF level the two models are indistinguishable: "
            "M1 cumulative/dynamic payoff AUC 0.5906 vs M2 0.5908 (12-month: 0.5448 vs 0.5442)."
        ),
        "manuscript_conflict": (
            "Sections 6.4 and 9 describe the gain as ordering 'payoff-prone observations' more "
            "effectively, which reads as facility-level discrimination. Table T5 reports only the "
            "pooled value. The CIF-level payoff AUCs are never reported."
        ),
        "arithmetic_only": True,
    },
    "F3_UNSEEN_VINTAGE_REVERSAL": {
        "title": "The primary decision metric reverses sign on the unseen-vintage population",
        "severity": "MAJOR",
        "reviewers": ["R1", "R3", "R5"],
        "evidence": "reports/track_b/macro_competing_risk_validation.json#/unseen_vintage",
        "derivation": (
            "On 17,352 facilities and 652,508 intervals (2.6x the primary evaluation) M2 joint log "
            "loss is 0.069811 against M1 0.070631, i.e. M2 is better on the protocol's own primary "
            "decision metric. Payoff AUC moves 0.5522 to 0.7373. Cause Brier scores still favour M1 "
            "marginally (payoff 0.011217 vs 0.011682), so the supplement is mixed rather than a clean "
            "contradiction, and the populations differ in seasoning and cohort encoding."
        ),
        "manuscript_conflict": (
            "Section 4.3 states unseen vintages are 'a separate supplementary evaluation' not added "
            "'simply to enlarge the event count', and section 5.2 repeats the restriction. Neither "
            "discloses that the headline metric changes direction there. A reader cannot tell from the "
            "manuscript that a larger evaluation population in the same study points the other way."
        ),
        "arithmetic_only": False,
    },
    "F4_UNCERTAINTY_ASYMMETRY": {
        "title": "Calendar-block uncertainty is disclosed for the exploratory result but not the primary one",
        "severity": "MAJOR",
        "reviewers": ["R3", "R5", "R1"],
        "evidence": (
            "reports/track_b/macro_competing_risk_validation.json#/paired_calendar/intervals ; "
            "docs/paper/statistical_unit_audit.json ; "
            "docs/track_b/macro_competing_risk_protocol.json#/decision"
        ),
        "derivation": (
            "The frozen paired calendar-year interval for the primary payoff Brier difference is "
            "[-4.738e-05, +0.013215] and therefore includes zero. The joint log-loss calendar interval "
            "lower bound is +5.33e-05, effectively touching zero on eight annual blocks. The frozen "
            "protocol decision rule requires BOTH facility and calendar 95% intervals, so the calendar "
            "interval is not an optional sensitivity under the study's own rules."
        ),
        "manuscript_conflict": (
            "Section 8 explicitly notes that the exploratory refinancing payoff Brier 'calendar-year "
            "interval includes zero'. Section 6.4, which carries the central ranking-versus-probability "
            "contrast, reports no calendar interval for the primary payoff Brier at all. "
            "docs/paper/headline_claim_audit.json records the caveat for headline H7 but not H4."
        ),
        "arithmetic_only": False,
    },
    "F5_HORIZON_SELECTION": {
        "title": "The headline CIF horizon is the one at which M2 fails worst",
        "severity": "MODERATE",
        "reviewers": ["R2", "R3"],
        "evidence": "reports/track_b/macro_competing_risk_validation.json#/cif/horizons",
        "derivation": (
            "Frozen mean predicted payoff CIF against observed: 12m observed 0.1330, M1 0.1385, "
            "M2 0.1222 (M2 absolute error 0.0108 vs M1 0.0055, same order of magnitude); 24m observed "
            "0.3367, M1 0.2587, M2 0.7581; 36m observed 0.5056, M1 0.3536, M2 0.8079; 60m observed "
            "0.6094, M1 0.4860, M2 0.8187. The 12-month horizon shows no qualitative M2 failure."
        ),
        "manuscript_conflict": (
            "The abstract and section 6.5 report only the 24-month comparison. MACRO_CIF_12_* claims "
            "exist in the frozen registry and are not shown. Selecting the horizon at which the "
            "discrepancy is largest, without showing the horizon at which it is absent, is a "
            "presentation choice a reviewer will read as favourable selection."
        ),
        "arithmetic_only": False,
    },
    "F6_CIF_COHORT_IS_SINGLE_PATH": {
        "title": "The CIF cohort is effectively a single-entry cohort on one macro path",
        "severity": "MODERATE",
        "reviewers": ["R2", "R4"],
        "evidence": (
            "reports/track_b/macro_competing_risk_validation.json#/cif/horizons ; "
            "docs/track_b/macro_competing_risk_protocol.json#/cif"
        ),
        "derivation": (
            "All four horizons record facilities = landmarks = 5,619 with "
            "calendar_truncated_landmarks = 0, including the 60-month horizon. With an evaluation "
            "window ending 2026-02 this requires every landmark to fall on or before 2021-02, and the "
            "frozen observed table shows essentially no censoring over 24 months "
            "(censor_survival 0.99922, zero censored at month 24). The entry distribution is therefore "
            "concentrated at the start of the evaluation window and the 24-month 'historical rolling "
            "PIT path' spans approximately 2019-01 to 2021-01."
        ),
        "manuscript_conflict": (
            "Section 3 and section 5.5 describe 'historical rolling PIT paths' in the plural, which "
            "implies averaging over many entry dates and macro trajectories. If entry is concentrated, "
            "the 24-month CIF result is one realised macro path covering the pandemic, which makes it "
            "the same evidence as F1 rather than independent corroboration."
        ),
        "arithmetic_only": False,
        "inference_flag": (
            "Entry concentration is INFERRED from zero calendar truncation at 60 months plus the "
            "frozen censoring profile. The landmark entry-month distribution is not itself published. "
            "Authors must confirm it from the frozen arrays; if entry is in fact dispersed, this "
            "finding is withdrawn and F6 reduces to a reporting request."
        ),
    },
    "F7_COLLINEARITY_BREAKDOWN": {
        "title": "Development macro design is near-collinear and the correlation structure inverts",
        "severity": "MODERATE",
        "reviewers": ["R3", "R4"],
        "evidence": "reports/track_b/macro_signal_attribution_stability.json#/macro_shift/correlations",
        "derivation": (
            "Distinct-month development correlations: unemployment_level vs hpi_yoy -0.919; VIF 10.2, "
            "11.6, 11.6 and 11.2 for unemployment, treasury, mortgage rate and HPI growth; condition "
            "number 69.1. In evaluation months the same pairs change sign: unemployment vs treasury "
            "+0.332 to -0.544, unemployment vs mortgage rate +0.392 to -0.490, unemployment vs CPI "
            "+0.440 to -0.400, and hpi_yoy vs cpi_yoy -0.568 to +0.924. Development spans 88 distinct "
            "months with mean Mahalanobis distance 7.0 (equal to the dimension); evaluation months have "
            "mean 298.5 and maximum 2,893.9."
        ),
        "manuscript_conflict": (
            "Section 7.1 and section 7.2 report support shift and coefficient instability as separate "
            "observations. The sharper and more defensible statement available from the frozen evidence "
            "is that a near-collinear macro design estimated over a monotone post-GFC recovery window "
            "cannot identify its components, and that the fitted combination breaks when the "
            "collinearity breaks. This is a stronger framing than the manuscript currently uses."
        ),
        "arithmetic_only": False,
    },
    "F8_PIT_LABEL": {
        "title": "'Point-in-time' overstates what the frozen provenance evidence establishes",
        "severity": "MODERATE",
        "reviewers": ["R5", "R1"],
        "evidence": "reports/track_b/pit_macro_data_audit_task9_api_v5.json#/release_lags",
        "derivation": (
            "ALFRED vintage dates and real-time observation pulls are recorded for all six series, so "
            "the inputs are genuinely vintage-aware and revision-aware. However release_lags records "
            "count = 0 and evidence_quality = UNMEASURED_EXACT_PROVIDER_DATES for every one of UNRATE, "
            "DGS10, MORTGAGE30US, USSTHPI, GDPC1 and CPIAUCSL, with the explicit reason that no "
            "certified initial provider release dates were obtained."
        ),
        "manuscript_conflict": (
            "The title, abstract and section 4.4 use 'point-in-time', which a credit-risk or real-time-"
            "data reader will take to mean verified availability at the assessment date. The defensible "
            "label is vintage-aware or revision-aware. Section 9 already draws the right distinction for "
            "the mortgage panel; the same discipline is not applied to the macro label itself."
        ),
        "arithmetic_only": False,
    },
    "F9_INTERVAL_WEIGHTING": {
        "title": "Equal interval weighting makes the estimand a long-duration-weighted quantity",
        "severity": "MODERATE",
        "reviewers": ["R2", "R3"],
        "evidence": (
            "docs/paper/statistical_unit_audit.json ; "
            "reports/track_b/macro_competing_risk_validation.json#/split_counts"
        ),
        "derivation": (
            "The primary evaluation averages 248,939 monthly intervals over 5,619 facilities, a mean of "
            "44.3 intervals per facility. Under equal interval weighting a facility that survives the "
            "whole window contributes up to 86 terms while an early exit contributes a handful. Because "
            "surviving long is itself the complement of the payoff event being modelled, the weighting "
            "is correlated with the outcome and the pooled score is not a facility-average risk."
        ),
        "manuscript_conflict": (
            "Section 5.3 defines joint log loss as an average over the observed class without stating "
            "the weighting unit or its consequence. Section 10 notes repeated observations only in the "
            "context of resampling. No facility-weighted sensitivity exists."
        ),
        "arithmetic_only": True,
    },
    "F10_JOINT_LOSS_INSENSITIVITY": {
        "title": "Joint log loss is dominated by the no-event class and is a weak primary endpoint",
        "severity": "MODERATE",
        "reviewers": ["R3", "R1"],
        "evidence": "reports/track_b/macro_competing_risk_validation.json#/primary ; #/split_counts",
        "derivation": (
            "In the primary evaluation 3,823 payoffs and 280 defaults occur across 248,939 intervals, so "
            "roughly 98.4% of terms are no-event. M0 (duration and vintage only) records temporal joint "
            "log loss 0.08926 against M1 0.08920, a difference of 6e-05, while temporal default AUC "
            "moves 0.607 to 0.694. The chosen primary metric is close to blind to a large "
            "discrimination change."
        ),
        "manuscript_conflict": (
            "Section 5.3 presents joint log loss as the headline proper score and section 7.2 notes that "
            "no-event intervals drive aggregate deterioration without drawing the conclusion that the "
            "metric is a poor primary endpoint for this event rate."
        ),
        "arithmetic_only": True,
    },
    "F11_DISTRESS_STATE_OMITTED": {
        "title": "Current loan state is excluded by design, which bounds the question being answered",
        "severity": "MODERATE",
        "reviewers": ["R1", "R4"],
        "evidence": "docs/track_b/macro_competing_risk_protocol.json#/sensitivities/updated_state",
        "derivation": (
            "The frozen protocol records updated_state = false with the reason 'Optional; omitted to "
            "isolate structural/macro research'. M1 and M2 therefore condition only on origination "
            "characteristics, duration band and cohort. Task 5 separately documents strong distress-state "
            "dependence (PD_DISTRESS, and a near-saturated hazard AUC on current-state recognition)."
        ),
        "manuscript_conflict": (
            "Nothing in the manuscript is false here, but the reader is not told that the comparison is "
            "between two origination-only models. A credit-risk referee will note that no deployed "
            "monthly mortgage model omits current delinquency state, and that macro increment "
            "conditional on current state is the question of practical interest and is untested."
        ),
        "arithmetic_only": False,
    },
    "F12_PAYOFF_MATURITY_CONFLATION": {
        "title": "Payoff and maturity are pooled and the shorthand leaks into interpretation",
        "severity": "MINOR",
        "reviewers": ["R4", "R2"],
        "evidence": (
            "docs/track_b/survival_competing_risk_protocol.json#/events ; "
            "docs/track_b/macro_competing_risk_protocol.json"
        ),
        "derivation": (
            "Source termination code 01 pools voluntary payoff with scheduled maturity. For the primary "
            "vintages 2006-2014 evaluated in 2019-2026 on 30-year terms, scheduled maturity is "
            "negligible, so the pooling is empirically benign in the primary population. The 2006 and "
            "2008 vintages include shorter original terms where maturity is possible."
        ),
        "manuscript_conflict": (
            "Section 4.2 states the shorthand correctly once; sections 6.4, 6.5, 8 and 9 then use "
            "'payoff' unqualified in sentences about refinancing-like behaviour. The exposure is "
            "largely terminological rather than substantive."
        ),
        "arithmetic_only": False,
    },
    "F13_DEFAULT_CIF_UNIVERSAL_MISS": {
        "title": "Every model badly under-predicts default CIF, including the baseline",
        "severity": "MODERATE",
        "reviewers": ["R1", "R2"],
        "evidence": "reports/track_b/macro_competing_risk_validation.json#/cif/horizons/24/models",
        "derivation": (
            "At 24 months observed default CIF is 0.0360 while mean predicted is 0.0113 (M0), 0.0107 "
            "(M1) and 0.0127 (M2). The baseline under-predicts observed default incidence by a factor "
            "of roughly 3.4. Default cumulative/dynamic AUC is essentially identical across models "
            "(0.6803 M1 vs 0.6804 M2)."
        ),
        "manuscript_conflict": (
            "Section 6.5 discusses offsetting cause errors in the abstract but never reports that the "
            "mortgage baseline itself fails the observed default CIF comparison at the same horizon "
            "used for the headline payoff result. This weakens any implied contrast in which M1 is the "
            "acceptable model and M2 the failing one."
        ),
        "arithmetic_only": False,
    },
    "F14_PRIOR_EXPOSURE": {
        "title": "Prior exposure is disclosed but the surviving language is still too strong in places",
        "severity": "MINOR",
        "reviewers": ["R5"],
        "evidence": (
            "docs/paper/experiment_design_freeze.json#/experiments ; "
            "docs/track_b/macro_competing_risk_protocol.json#/splits/virgin_holdout"
        ),
        "derivation": (
            "The freeze records virgin_holdout = false, prior_exposure state CONSUMED, "
            "prediction_generation_count = 1 and a registration hash, with Task 11 post-hoc and Task 12 "
            "exploratory on the same inspected arrays. Governance here is unusually strong and the "
            "manuscript discloses it in sections 5.2 and 10."
        ),
        "manuscript_conflict": (
            "The abstract's 'demonstrate a design-specific divergence' and the conclusion's framing "
            "still read as confirmatory. Maximum defensible language is 'frozen temporal evaluation on "
            "a non-virgin holdout', never 'confirmatory temporal validation'."
        ),
        "arithmetic_only": False,
    },
    "F15_NOVELTY_UNRESOLVED": {
        "title": "Closest-work comparison leaves the decisive cells UNKNOWN",
        "severity": "MODERATE",
        "reviewers": ["R5", "R3"],
        "evidence": (
            "docs/paper/closest_paper_matrix.json ; "
            "docs/paper/literature/citation_gap_resolution.json"
        ),
        "derivation": (
            "Bu2026 is recorded as Freddie data, competing risk, macro covariates and CIF with "
            "threat_to_novelty HIGH and PIT_macro, temporal_holdout, proper_scores, calibration and "
            "distribution_shift_analysis all UNKNOWN. Peng2026 is recorded as Freddie, calibration YES, "
            "Brier, distribution shift YES, threat_to_novelty HIGH, with macro as FUTURE_EXTENSION. "
            "CG03 and CG06 remain PARTIALLY_RESOLVED."
        ),
        "manuscript_conflict": (
            "Section 2 and section 10 narrate the incompleteness of the search at length. That narration "
            "does not resolve the threat and reads as a process artifact. Two HIGH threats with "
            "unresolved cells is a referee-visible gap."
        ),
        "arithmetic_only": False,
    },
}

# --------------------------------------------------------------------------
# PRIMARY claim attacks
# --------------------------------------------------------------------------

MEASUREMENT_SURVIVES = (
    "The recorded number is a verified property of the frozen artifact and survives as a "
    "measurement. What it is taken to mean does not survive unchanged."
)

CLAIMS = [
    {
        "claim_id": "SURV_60_NAIVE_NET_DEFAULT",
        "experiment_id": "B6",
        "claim_text": (
            "At 60 months after conditional evaluation entry the frozen descriptive naive net default "
            "risk with payoff censored is 5.21%."
        ),
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R2: the quantity is a Kaplan-Meier estimate under a censoring assumption the study itself "
            "records as unverified, computed on a conditional-entry cohort. Presented beside the "
            "competing-risk CIF it invites the reading that net risk is an error, when it is a different "
            "and sometimes appropriate estimand."
        ),
        "alternative_explanation": (
            "The gap between 5.21% and 3.10% is the arithmetic consequence of a high competing payoff "
            "rate in this cohort, not evidence about estimator quality."
        ),
        "estimand_concern": (
            "Net risk under hypothetical removal of payoff versus observed-world CIF. Manuscript "
            "section 6.2 states this correctly."
        ),
        "selection_concern": (
            "Conditional entry at an eligible calendar boundary; the cohort is a survivor population, "
            "not an origination cohort."
        ),
        "uncertainty_concern": (
            "B6 facility bootstrap with 400 draws reestimates censoring and Aalen-Johansen within each "
            "draw, which is appropriate. No interval is reported in the manuscript for this pair."
        ),
        "literature_concern": "CG03 PARTIALLY_RESOLVED; conditional-entry/IPCW implementation compatibility unconfirmed.",
        "severity": "MINOR",
        "survives_attack": True,
        "findings": [],
        "required_remedy": (
            "Report the frozen bootstrap interval alongside the two point estimates and keep the "
            "estimand sentence adjacent to the numbers rather than in the following paragraph."
        ),
        "remedy_requires_new_experiment": False,
        "manuscript_action": "RETAIN_WITH_INTERVAL",
    },
    {
        "claim_id": "SURV_60_DEFAULT_CIF",
        "experiment_id": "B6",
        "claim_text": (
            "At 60 months after conditional evaluation entry the frozen descriptive competing-risk "
            "default CIF is 3.10%."
        ),
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R2: same conditional-entry and censoring caveats. The manuscript correctly refuses to "
            "pair this with a modelled 60-month projection for want of training-duration support, "
            "which is the right call."
        ),
        "alternative_explanation": "None material; this is a descriptive nonparametric estimate.",
        "estimand_concern": "Observed-world cumulative incidence after conditional entry. Correctly stated.",
        "selection_concern": "Survivor cohort as above.",
        "uncertainty_concern": "Frozen interval exists in the statistical unit audit and is not quoted in the manuscript.",
        "literature_concern": "CG03 PARTIALLY_RESOLVED.",
        "severity": "MINOR",
        "survives_attack": True,
        "findings": [],
        "required_remedy": "Quote the frozen interval.",
        "remedy_requires_new_experiment": False,
        "manuscript_action": "RETAIN_WITH_INTERVAL",
    },
    {
        "claim_id": "MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS",
        "experiment_id": "B10",
        "claim_text": "M1 development joint log loss is 0.09015.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R3: a development fit statistic carries no evidential weight on its own and the "
            "manuscript says so. The only risk is the ladder narrative implying that each rung adds "
            "real signal."
        ),
        "alternative_explanation": "In-sample improvement from added parameters.",
        "estimand_concern": "Interval-weighted development average; see F9 and F10 on the weighting and the metric.",
        "selection_concern": "Development facilities assigned by hash salt; roles disjoint from evaluation.",
        "uncertainty_concern": "No development interval; none needed for a fit statistic.",
        "literature_concern": "None.",
        "severity": "MINOR",
        "survives_attack": True,
        "findings": ["F9_INTERVAL_WEIGHTING", "F10_JOINT_LOSS_INSENSITIVITY"],
        "required_remedy": "State the weighting unit where the metric is defined.",
        "remedy_requires_new_experiment": False,
        "manuscript_action": "RETAIN_WITH_DEFINITION",
    },
    {
        "claim_id": "MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS",
        "experiment_id": "B10",
        "claim_text": "M2 development joint log loss is 0.08959.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R3: the improvement over M1 is 0.00056 for seven added parameters on 88 distinct macro "
            "months. No complexity-adjusted comparison is reported, so the development 'gain' is not "
            "shown to exceed what added degrees of freedom buy."
        ),
        "alternative_explanation": (
            "Ordinary in-sample gain from seven parameters whose effective independent support is the "
            "number of distinct development months (88), not the number of intervals (1,568,661)."
        ),
        "estimand_concern": "As above.",
        "selection_concern": "As above.",
        "uncertainty_concern": (
            "No development-side interval and no information criterion. The manuscript's phrase "
            "'development improvement' is not supported as a statistically meaningful improvement."
        ),
        "literature_concern": "None.",
        "severity": "MODERATE",
        "survives_attack": True,
        "findings": ["F7_COLLINEARITY_BREAKDOWN", "F10_JOINT_LOSS_INSENSITIVITY"],
        "required_remedy": (
            "Describe this as in-sample fit change, not improvement, or report a complexity-adjusted "
            "comparison. State the 88-distinct-month effective support beside the interval count."
        ),
        "remedy_requires_new_experiment": False,
        "manuscript_action": "NARROW_LANGUAGE",
    },
    {
        "claim_id": "MACRO_PRIMARY_M1_JOINT_LOG_LOSS",
        "experiment_id": "B10",
        "claim_text": "M1 temporal joint log loss is 0.08920.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R3: the baseline is barely distinguishable from M0 (0.08926), so the manuscript's "
            "'mortgage baseline' is close to a duration-and-vintage baseline on this metric."
        ),
        "alternative_explanation": "No-event domination of the metric (F10).",
        "estimand_concern": "Interval-weighted; long-duration facilities dominate (F9).",
        "selection_concern": (
            "Evaluation facilities are 2006-2014 originations surviving to 2019 without payoff or "
            "default, i.e. a negatively selected non-prepayer population. Burnout is not represented."
        ),
        "uncertainty_concern": "Reported only through the paired difference.",
        "literature_concern": "Stanton1995 and Deng2000 establish burnout; the design omits it.",
        "severity": "MODERATE",
        "survives_attack": True,
        "findings": ["F9_INTERVAL_WEIGHTING", "F10_JOINT_LOSS_INSENSITIVITY", "F11_DISTRESS_STATE_OMITTED"],
        "required_remedy": (
            "State explicitly that the temporal population is a seasoned survivor cohort and that "
            "M0 and M1 are nearly indistinguishable on joint loss."
        ),
        "remedy_requires_new_experiment": False,
        "manuscript_action": "RETAIN_WITH_POPULATION_STATEMENT",
    },
    {
        "claim_id": "MACRO_PRIMARY_M2_JOINT_LOG_LOSS",
        "experiment_id": "B10",
        "claim_text": "M2 temporal joint log loss is 0.10532.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R1/R3/R4: the aggregate is driven by one calendar year. 2020 supplies 89.3% of the "
            "difference from M1; ex-2020 the gap is +0.00172 and 2021 reverses. The same metric "
            "reverses sign entirely on the unseen-vintage population."
        ),
        "alternative_explanation": (
            "Out-of-support extrapolation during an unprecedented macro excursion, amplified by a "
            "near-collinear development design, in a burned-out survivor cohort. All three are "
            "supported by frozen diagnostics and none is identified."
        ),
        "estimand_concern": "As above.",
        "selection_concern": "Survivor cohort; four of seven vintages.",
        "uncertainty_concern": (
            "Calendar-block interval on eight annual blocks has a lower bound of +5.33e-05. The "
            "quantity is close to being indistinguishable from zero under the resampling unit that "
            "matches the shared macro path."
        ),
        "literature_concern": "None beyond F15.",
        "severity": "MAJOR",
        "survives_attack": True,
        "findings": [
            "F1_CALENDAR_CONCENTRATION",
            "F3_UNSEEN_VINTAGE_REVERSAL",
            "F4_UNCERTAINTY_ASYMMETRY",
            "F7_COLLINEARITY_BREAKDOWN",
        ],
        "required_remedy": (
            "Report the per-year decomposition and the ex-2020 residual in the main text; report the "
            "unseen-vintage direction reversal; lead with the calendar-block interval."
        ),
        "remedy_requires_new_experiment": False,
        "manuscript_action": "NARROW_LANGUAGE_AND_DISCLOSE_DECOMPOSITION",
    },
    {
        "claim_id": "MACRO_DELTA_FACILITY_JOINT_LOG_LOSS",
        "experiment_id": "B10",
        "claim_text": (
            "The paired M2-minus-M1 temporal joint log-loss difference is +0.01612 with facility "
            "bootstrap interval [+0.01541, +0.01678]."
        ),
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R3: this is the pseudoreplication exposure. Resampling 5,619 facilities conditions on the "
            "single realised macro path shared by every facility in a month. The artifact itself "
            "records conditional_on_realized_calendar = true. An interval of width 0.0014 around a "
            "comparison whose only new inputs are seven month-constant series cannot be read as "
            "uncertainty about macro transport."
        ),
        "alternative_explanation": (
            "The narrowness is a property of the resampling unit, not evidence of a precisely "
            "determined effect."
        ),
        "estimand_concern": "Facility-composition uncertainty for a fixed pair of models on a fixed calendar.",
        "selection_concern": "None additional.",
        "uncertainty_concern": "Understates uncertainty about the quantity the paper is actually claiming.",
        "literature_concern": "None.",
        "severity": "MAJOR",
        "survives_attack": True,
        "findings": ["F4_UNCERTAINTY_ASYMMETRY"],
        "required_remedy": (
            "Retain the number but demote it: state in the same sentence that it is conditional "
            "facility-composition uncertainty and is not uncertainty about macro transport."
        ),
        "remedy_requires_new_experiment": False,
        "manuscript_action": "NARROW_LANGUAGE_AND_DEMOTE",
    },
    {
        "claim_id": "MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS",
        "experiment_id": "B10",
        "claim_text": (
            "The calendar-year block sensitivity interval for the same difference is "
            "[+0.00005, +0.04150]."
        ),
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R3: eight annual blocks including a partial 2026, with one block (2020) carrying 89% of "
            "the point estimate. The lower bound of 5.33e-05 means the direction is preserved by a "
            "margin that is numerically indistinguishable from zero. Block bootstrap on eight "
            "highly heterogeneous units has poor coverage properties."
        ),
        "alternative_explanation": (
            "The interval is wide and nearly touches zero precisely because the effect is a "
            "single-year phenomenon."
        ),
        "estimand_concern": "Appropriate unit for a macro claim; this is the interval that should lead.",
        "selection_concern": "None additional.",
        "uncertainty_concern": (
            "Eight blocks is too few for percentile block bootstrap coverage; the manuscript should "
            "say so rather than presenting the interval as a routine sensitivity."
        ),
        "literature_concern": "None.",
        "severity": "MAJOR",
        "survives_attack": True,
        "findings": ["F1_CALENDAR_CONCENTRATION", "F4_UNCERTAINTY_ASYMMETRY"],
        "required_remedy": (
            "Promote to the primary interval, state the eight-block coverage limitation, and note "
            "that one block dominates the point estimate."
        ),
        "remedy_requires_new_experiment": False,
        "manuscript_action": "PROMOTE_AND_QUALIFY",
    },
    {
        "claim_id": "MACRO_DEVELOPMENT_M1_PAYOFF_AUC",
        "experiment_id": "B10",
        "claim_text": "M1 development payoff AUC as recorded.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": "R3: development discrimination; no transport weight. Same pooling concern as F2.",
        "alternative_explanation": "Pooled across development months; between-month component not separated.",
        "estimand_concern": "Pooled interval-level ranking, not within-month ranking.",
        "selection_concern": "None additional.",
        "uncertainty_concern": "None reported; none needed in development.",
        "literature_concern": "None.",
        "severity": "MINOR",
        "survives_attack": True,
        "findings": ["F2_BETWEEN_PERIOD_AUC"],
        "required_remedy": "Define the pooling explicitly where AUC is introduced.",
        "remedy_requires_new_experiment": False,
        "manuscript_action": "RETAIN_WITH_DEFINITION",
    },
    {
        "claim_id": "MACRO_DEVELOPMENT_M2_PAYOFF_AUC",
        "experiment_id": "B10",
        "claim_text": "M2 development payoff AUC as recorded.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": "R3: as above.",
        "alternative_explanation": "As above.",
        "estimand_concern": "As above.",
        "selection_concern": "None additional.",
        "uncertainty_concern": "None reported.",
        "literature_concern": "None.",
        "severity": "MINOR",
        "survives_attack": True,
        "findings": ["F2_BETWEEN_PERIOD_AUC"],
        "required_remedy": "As above.",
        "remedy_requires_new_experiment": False,
        "manuscript_action": "RETAIN_WITH_DEFINITION",
    },
    {
        "claim_id": "MACRO_PRIMARY_M1_PAYOFF_AUC",
        "experiment_id": "B10",
        "claim_text": "M1 temporal pooled payoff AUC is 0.565.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R3: 0.565 is barely above chance, and the manuscript builds a ranking-versus-probability "
            "contrast on a baseline that essentially does not rank payoff."
        ),
        "alternative_explanation": "Origination characteristics carry little monthly payoff signal in a burned-out cohort.",
        "estimand_concern": "Pooled across 86 months.",
        "selection_concern": "Survivor cohort; burnout omitted.",
        "uncertainty_concern": "Only the paired difference has an interval.",
        "literature_concern": "Burnout literature predicts exactly this.",
        "severity": "MODERATE",
        "survives_attack": True,
        "findings": ["F2_BETWEEN_PERIOD_AUC"],
        "required_remedy": "State that the baseline payoff ranking is near chance so the contrast is read correctly.",
        "remedy_requires_new_experiment": False,
        "manuscript_action": "RETAIN_WITH_CONTEXT",
    },
    {
        "claim_id": "MACRO_PRIMARY_M2_PAYOFF_AUC",
        "experiment_id": "B10",
        "claim_text": "M2 temporal pooled payoff AUC is 0.626.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R3/R4: the gain is between-period. Within every frozen calendar year the gap is near zero "
            "or negative, and at the facility-level 24-month CIF the two models are identical to four "
            "decimal places (0.5906 vs 0.5908). M2 adds only month-constant terms and interactions are "
            "prohibited, so a facility-level ranking gain was not available by construction."
        ),
        "alternative_explanation": (
            "M2 ranks calendar periods, which raises pooled AUC without improving the ordering of "
            "mortgages within any period."
        ),
        "estimand_concern": (
            "The quantity measured is pooled across-period ranking; the quantity the text implies is "
            "facility-level discrimination. These are different estimands."
        ),
        "selection_concern": "None additional.",
        "uncertainty_concern": "Facility bootstrap on a between-month quantity is the wrong unit.",
        "literature_concern": "None.",
        "severity": "MAJOR",
        "survives_attack": False,
        "findings": ["F2_BETWEEN_PERIOD_AUC"],
        "required_remedy": (
            "Either report a within-calendar-period or calendar-stratified payoff AUC, or restate the "
            "claim as across-period ranking and report the frozen CIF-level AUCs showing no "
            "facility-level gain. The measurement survives; the interpretation does not."
        ),
        "remedy_requires_new_experiment": True,
        "manuscript_action": "RESTATE_AS_ACROSS_PERIOD_OR_ADD_SENSITIVITY",
    },
    {
        "claim_id": "MACRO_DEVELOPMENT_M1_PAYOFF_BRIER",
        "experiment_id": "B10",
        "claim_text": "M1 development payoff Brier as recorded.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": "R3: development score; no transport weight.",
        "alternative_explanation": "None material.",
        "estimand_concern": "Monthly cause-specific Brier on risk intervals; distinct from the IPCW horizon Brier in Table T4.",
        "selection_concern": "None additional.",
        "uncertainty_concern": "None reported.",
        "literature_concern": "None.",
        "severity": "MINOR",
        "survives_attack": True,
        "findings": [],
        "required_remedy": "Label the Brier variant explicitly wherever it appears beside the IPCW horizon Brier.",
        "remedy_requires_new_experiment": False,
        "manuscript_action": "RETAIN_WITH_LABEL",
    },
    {
        "claim_id": "MACRO_DEVELOPMENT_M2_PAYOFF_BRIER",
        "experiment_id": "B10",
        "claim_text": "M2 development payoff Brier as recorded.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": "R3: as above.",
        "alternative_explanation": "None material.",
        "estimand_concern": "As above.",
        "selection_concern": "None additional.",
        "uncertainty_concern": "None reported.",
        "literature_concern": "None.",
        "severity": "MINOR",
        "survives_attack": True,
        "findings": [],
        "required_remedy": "As above.",
        "remedy_requires_new_experiment": False,
        "manuscript_action": "RETAIN_WITH_LABEL",
    },
    {
        "claim_id": "MACRO_PRIMARY_M1_PAYOFF_BRIER",
        "experiment_id": "B10",
        "claim_text": "M1 temporal monthly payoff Brier is 0.015127.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R3: with an observed monthly payoff rate of 1.536% a constant-rate predictor attains "
            "approximately 0.01512. M1's 0.015127 is therefore indistinguishable from predicting the "
            "base rate for everyone, which is consistent with its near-chance AUC."
        ),
        "alternative_explanation": "The baseline is effectively a base-rate predictor for payoff.",
        "estimand_concern": "Monthly cause-specific Brier.",
        "selection_concern": "None additional.",
        "uncertainty_concern": "Only paired differences carry intervals.",
        "literature_concern": "None.",
        "severity": "MODERATE",
        "survives_attack": True,
        "findings": ["F10_JOINT_LOSS_INSENSITIVITY"],
        "required_remedy": (
            "Report a base-rate reference Brier so the reader can see how little room either model "
            "occupies on this scale."
        ),
        "remedy_requires_new_experiment": False,
        "manuscript_action": "RETAIN_WITH_REFERENCE",
    },
    {
        "claim_id": "MACRO_PRIMARY_M2_PAYOFF_BRIER",
        "experiment_id": "B10",
        "claim_text": "M2 temporal monthly payoff Brier is 0.019825.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R3/R5: this is the second half of the central contrast and its calendar-block interval "
            "[-4.738e-05, +0.013215] includes zero. The manuscript discloses exactly this caveat for "
            "the exploratory refinancing payoff Brier in section 8 and omits it here. The frozen "
            "protocol decision rule requires both intervals, so the omission conflicts with the "
            "study's own rule."
        ),
        "alternative_explanation": (
            "Driven by 2020, where payoff Brier is 0.02168 (M1) against 0.04429 (M2); in 2019 and in "
            "every year from 2022 the two are equal to four decimal places."
        ),
        "estimand_concern": "Monthly cause-specific Brier.",
        "selection_concern": "None additional.",
        "uncertainty_concern": "Calendar-block interval includes zero and is not reported.",
        "literature_concern": "None.",
        "severity": "MAJOR",
        "survives_attack": False,
        "findings": ["F1_CALENDAR_CONCENTRATION", "F4_UNCERTAINTY_ASYMMETRY"],
        "required_remedy": (
            "Report the calendar-block interval in section 6.4 and state that the payoff Brier "
            "deterioration is not separable from zero under calendar resampling. The number survives; "
            "the word 'worsened' as an unqualified finding does not."
        ),
        "remedy_requires_new_experiment": False,
        "manuscript_action": "DISCLOSE_CALENDAR_INTERVAL_AND_NARROW",
    },
    {
        "claim_id": "MACRO_CIF_24_OBSERVED_PAYOFF",
        "experiment_id": "B10",
        "claim_text": "Observed 24-month Aalen-Johansen payoff CIF is 33.7%.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R2: the observed reference is sound, but it describes a cohort whose entry is "
            "concentrated at the start of the evaluation window, so the 24-month window is "
            "approximately 2019-01 to 2021-01 and contains the refinancing wave."
        ),
        "alternative_explanation": "None; this is the observed quantity.",
        "estimand_concern": "Conditional-entry observed CIF. Correctly labelled.",
        "selection_concern": "Entry concentration is inferred, not published (see F6).",
        "uncertainty_concern": "No interval quoted in the manuscript.",
        "literature_concern": "CG03 PARTIALLY_RESOLVED.",
        "severity": "MODERATE",
        "survives_attack": True,
        "findings": ["F6_CIF_COHORT_IS_SINGLE_PATH"],
        "required_remedy": "Publish the landmark entry-month distribution and name the calendar span of the path.",
        "remedy_requires_new_experiment": False,
        "manuscript_action": "DISCLOSE_ENTRY_DISTRIBUTION",
    },
    {
        "claim_id": "MACRO_CIF_24_M1_PAYOFF",
        "experiment_id": "B10",
        "claim_text": "M1 mean 24-month projected payoff CIF is 25.9%.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R2: M1 under-predicts by 7.8 percentage points, a 23% relative shortfall. The manuscript "
            "frames the CIF section around M2's failure while M1 also misses materially, and misses "
            "default CIF by a factor of 3.4."
        ),
        "alternative_explanation": "Burnout cohort meets an unmodelled refinancing wave.",
        "estimand_concern": "Mean of facility-level projected CIFs, compared with a cohort-level AJ estimate. These are not the same functional and the manuscript does not note it.",
        "selection_concern": "As above.",
        "uncertainty_concern": "No interval.",
        "literature_concern": "None.",
        "severity": "MODERATE",
        "survives_attack": True,
        "findings": ["F6_CIF_COHORT_IS_SINGLE_PATH", "F13_DEFAULT_CIF_UNIVERSAL_MISS"],
        "required_remedy": "Report the default CIF comparison in the same table and note the mean-of-individual versus cohort-estimator distinction.",
        "remedy_requires_new_experiment": False,
        "manuscript_action": "RETAIN_WITH_COMPANION_NUMBERS",
    },
    {
        "claim_id": "MACRO_CIF_24_M2_PAYOFF",
        "experiment_id": "B10",
        "claim_text": "M2 mean 24-month projected payoff CIF is 75.8%.",
        "evidence_status": "VERIFIED_FROZEN_ARTIFACT",
        "reviewer_attack": (
            "R2/R3: the figure is the compounding of a monthly payoff over-prediction already reported "
            "(2.830% predicted against 1.536% observed) over a path that runs through 2020, so it is "
            "not independent evidence. At 12 months the same construction shows no qualitative "
            "failure (M2 0.1222 against observed 0.1330, closer in absolute terms than M1 is at 24 "
            "months). Reporting only the 24-month figure selects the horizon at which the quantity is "
            "most dramatic."
        ),
        "alternative_explanation": (
            "Deterministic algebraic propagation of the 2020 monthly over-prediction through the "
            "survival recursion."
        ),
        "estimand_concern": (
            "Retrospective sequential mapping along one realised macro path, not a forecast. The "
            "manuscript says this; the abstract still presents it as a third finding."
        ),
        "selection_concern": "Horizon selection (F5); entry concentration (F6).",
        "uncertainty_concern": "No interval for any CIF quantity.",
        "literature_concern": "None.",
        "severity": "MAJOR",
        "survives_attack": True,
        "findings": [
            "F1_CALENDAR_CONCENTRATION",
            "F5_HORIZON_SELECTION",
            "F6_CIF_COHORT_IS_SINGLE_PATH",
        ],
        "required_remedy": (
            "Present the 12, 24, 36 and 60-month comparisons together; present the CIF as the "
            "consequence of the monthly calibration result rather than as separate evidence; remove it "
            "from the abstract as an independent finding."
        ),
        "remedy_requires_new_experiment": False,
        "manuscript_action": "REFRAME_AS_CONSEQUENCE_AND_SHOW_ALL_HORIZONS",
    },
]


def build_attack_registry() -> dict:
    return {
        "version": "task17-v1",
        "base_commit": BASE_COMMIT,
        "task16_prior_decision": TASK16_DECISION,
        "scope": "Adversarial review of PRIMARY Task 14 claims. Review and audit only.",
        "no_new_experiments": True,
        "no_frozen_values_changed": True,
        "arithmetic_policy": (
            "Interval-weighted recombination of already-frozen per-year aggregates is used to "
            "decompose an already-reported mean. No model was fitted, no prediction regenerated and no "
            "frozen metric altered."
        ),
        "unknown_policy": "UNKNOWN means not verified, not absent.",
        "cross_cutting_findings": FINDINGS,
        "claims": CLAIMS,
    }


SURVIVAL = {
    "SURV_60_NAIVE_NET_DEFAULT": ("SURVIVES", "Descriptive estimand correctly stated; add the frozen interval."),
    "SURV_60_DEFAULT_CIF": ("SURVIVES", "Descriptive estimand correctly stated; add the frozen interval."),
    "MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS": ("SURVIVES", "Fit statistic; define the weighting unit."),
    "MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS": (
        "SURVIVES_WITH_NARROWER_LANGUAGE",
        "'Improved development fit' must become 'changed in-sample fit' absent a complexity-adjusted comparison.",
    ),
    "MACRO_DEVELOPMENT_M1_PAYOFF_AUC": ("SURVIVES", "Development statistic; define pooling."),
    "MACRO_DEVELOPMENT_M2_PAYOFF_AUC": ("SURVIVES", "Development statistic; define pooling."),
    "MACRO_DEVELOPMENT_M1_PAYOFF_BRIER": ("SURVIVES", "Label the Brier variant."),
    "MACRO_DEVELOPMENT_M2_PAYOFF_BRIER": ("SURVIVES", "Label the Brier variant."),
    "MACRO_PRIMARY_M1_JOINT_LOG_LOSS": (
        "SURVIVES_WITH_NARROWER_LANGUAGE",
        "Must be accompanied by the statement that M0 and M1 are near-identical on this metric and that the cohort is a seasoned survivor population.",
    ),
    "MACRO_PRIMARY_M2_JOINT_LOG_LOSS": (
        "SURVIVES_WITH_NARROWER_LANGUAGE",
        "Survives as a measurement. The generalisation to 2019-2026 does not: 89.3% of the gap is 2020, 2021 reverses, and the unseen-vintage population reverses the sign of the primary metric.",
    ),
    "MACRO_PRIMARY_M1_PAYOFF_AUC": (
        "SURVIVES_WITH_NARROWER_LANGUAGE",
        "Must be described as pooled across-period ranking and noted as near chance.",
    ),
    "MACRO_PRIMARY_M2_PAYOFF_AUC": (
        "REQUIRES_NEW_ANALYSIS",
        "The number survives; 'improved payoff ranking' as facility-level discrimination is not supported. Frozen per-year and CIF-level AUCs contradict it. Needs a within-calendar-period or calendar-stratified AUC, or restatement as across-period ranking.",
    ),
    "MACRO_PRIMARY_M1_PAYOFF_BRIER": (
        "SURVIVES_WITH_NARROWER_LANGUAGE",
        "Needs a base-rate reference so the scale is interpretable.",
    ),
    "MACRO_PRIMARY_M2_PAYOFF_BRIER": (
        "SURVIVES_WITH_NARROWER_LANGUAGE",
        "The value survives; 'payoff Brier worsened' as a robust finding does not, because the frozen calendar-block interval includes zero and the study's own decision rule requires that interval.",
    ),
    "MACRO_DELTA_FACILITY_JOINT_LOG_LOSS": (
        "SURVIVES_WITH_NARROWER_LANGUAGE",
        "Must be labelled conditional facility-composition uncertainty, not uncertainty about macro transport.",
    ),
    "MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS": (
        "SURVIVES_WITH_NARROWER_LANGUAGE",
        "Promote to primary; state the eight-block coverage limitation and the single-block dominance.",
    ),
    "MACRO_CIF_24_OBSERVED_PAYOFF": (
        "SURVIVES_WITH_NARROWER_LANGUAGE",
        "Requires publication of the landmark entry-month distribution and the calendar span of the path.",
    ),
    "MACRO_CIF_24_M1_PAYOFF": (
        "SURVIVES_WITH_NARROWER_LANGUAGE",
        "Must be reported with the companion default CIF miss and the mean-of-individual versus cohort-estimator distinction.",
    ),
    "MACRO_CIF_24_M2_PAYOFF": (
        "SURVIVES_WITH_NARROWER_LANGUAGE",
        "Survives as a measurement but must be reframed as the consequence of the monthly payoff calibration result and shown alongside the 12, 36 and 60-month horizons.",
    ),
}


def build_survival_matrix() -> dict:
    rows = []
    for c in CLAIMS:
        verdict, rationale = SURVIVAL[c["claim_id"]]
        rows.append(
            {
                "claim_id": c["claim_id"],
                "experiment_id": c["experiment_id"],
                "severity": c["severity"],
                "verdict": verdict,
                "rationale": rationale,
                "findings": c["findings"],
                "requires_new_experiment": c["remedy_requires_new_experiment"],
            }
        )
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return {
        "version": "task17-v1",
        "base_commit": BASE_COMMIT,
        "legend": [
            "SURVIVES",
            "SURVIVES_WITH_NARROWER_LANGUAGE",
            "REQUIRES_NEW_ANALYSIS",
            "NOT_SUPPORTED_AFTER_REVIEW",
        ],
        "note": (
            "No PRIMARY claim is NOT_SUPPORTED: every recorded number is a verified property of its "
            "frozen artifact. The review attacks interpretation, scope and selective reporting, not "
            "arithmetic."
        ),
        "counts": counts,
        "claims": rows,
    }


CHANGES = [
    ("CH01", "MAJOR", "6.3, 7.2, 7.3, Abstract, Conclusion",
     "The headline temporal deterioration is presented as a property of the 2019-2026 period; 89.3% of it comes from calendar 2020 and 2021 reverses sign.",
     "reports/track_b/macro_competing_risk_validation.json#/stability/calendar",
     "Add the per-year joint log-loss table with interval weights and contributions; state the ex-2020 residual (+0.00172) in the main text; qualify the abstract accordingly.",
     False, True, True),
    ("CH02", "MAJOR", "4.3, 5.2, 6.3, new subsection",
     "The unseen-vintage supplement reverses the sign of the protocol's primary decision metric and this is never reported.",
     "reports/track_b/macro_competing_risk_validation.json#/unseen_vintage",
     "Report M1 0.070631 vs M2 0.069811 on 17,352 facilities with the cause-Brier counterpoint and the population differences; discuss what the reversal implies for the scope of the conclusion.",
     False, True, True),
    ("CH03", "MAJOR", "6.4, 9, Table T5",
     "The payoff AUC gain is described as facility-level ranking; frozen per-year and CIF-level AUCs show it is between-period.",
     "reports/track_b/macro_competing_risk_validation.json#/stability/calendar ; #/cif/horizons",
     "Restate as across-period ranking; report the frozen 12 and 24-month CIF-level payoff AUCs (M1 0.5448/0.5906 vs M2 0.5442/0.5908); note that month-constant terms and prohibited interactions make a within-month gain structurally unavailable.",
     False, True, True),
    ("CH04", "MAJOR", "6.3, 6.4",
     "The calendar-block interval for the primary payoff Brier includes zero and is not reported, while the identical caveat is given for the exploratory result in section 8.",
     "reports/track_b/macro_competing_risk_validation.json#/paired_calendar/intervals ; docs/track_b/macro_competing_risk_protocol.json#/decision",
     "Report [-4.738e-05, +0.013215] in section 6.4; lead with calendar-block intervals throughout; state that the frozen decision rule requires both intervals.",
     False, True, True),
    ("CH05", "MODERATE", "Abstract, 6.5, Figure F6",
     "Only the 24-month CIF horizon is shown, which is the horizon at which M2 is worst; the 12-month horizon shows no qualitative failure.",
     "reports/track_b/macro_competing_risk_validation.json#/cif/horizons",
     "Show all four frozen horizons in one table; remove the CIF from the abstract as an independent finding and present it as the consequence of the monthly calibration result.",
     False, True, True),
    ("CH06", "MODERATE", "3, 5.5, 6.5",
     "'Historical rolling PIT paths' implies many entry dates; zero calendar truncation at 60 months implies entry is concentrated at the window start.",
     "reports/track_b/macro_competing_risk_validation.json#/cif/horizons",
     "Publish the landmark entry-month distribution; if concentrated, name the calendar span of the path and state that the CIF comparison and the 2020 calendar diagnostic are the same evidence.",
     False, True, True),
    ("CH07", "MODERATE", "Title, Abstract, 4.4, 9",
     "'Point-in-time' is used where frozen evidence records release lags as UNMEASURED for all six series.",
     "reports/track_b/pit_macro_data_audit_task9_api_v5.json#/release_lags",
     "Replace with 'vintage-aware' or 'revision-aware', or define PIT explicitly as ALFRED vintage-date availability and state that provider release timestamps were not certified.",
     False, True, True),
    ("CH08", "MODERATE", "5.3, 5.4, 10",
     "The weighting unit of the joint loss is never stated and no facility-weighted sensitivity exists.",
     "docs/paper/statistical_unit_audit.json",
     "State equal interval weighting and its consequence (mean 44.3 intervals per facility; weighting correlated with the payoff outcome). Add a facility-weighted sensitivity or flag its absence as a limitation.",
     True, False, True),
    ("CH09", "MODERATE", "5.3, 6.3, 9",
     "Joint log loss is the headline proper score but is ~98.4% no-event terms and barely separates M0 from M1.",
     "reports/track_b/macro_competing_risk_validation.json#/primary ; #/split_counts",
     "Demote joint log loss to one of several reported scores; promote cause-specific Brier and calibration; report the M0 comparison so the reader can see the metric's sensitivity.",
     False, True, True),
    ("CH10", "MODERATE", "5.1, 9, 10",
     "Current delinquency state is excluded by design and the reader is not told that both models are origination-only.",
     "docs/track_b/macro_competing_risk_protocol.json#/sensitivities/updated_state",
     "State in section 5.1 that no current-state covariate enters either model and that the macro increment conditional on current state is untested.",
     False, True, True),
    ("CH11", "MODERATE", "6.5, Table, 9",
     "Every model under-predicts default CIF by roughly a factor of 3.4 at 24 months; only the payoff comparison is shown.",
     "reports/track_b/macro_competing_risk_validation.json#/cif/horizons/24/models",
     "Report the default CIF comparison beside the payoff one so the baseline's own failure is visible.",
     False, True, True),
    ("CH12", "MODERATE", "7.1, 7.2, 9",
     "Support shift and coefficient instability are reported as separate observations; the sharper available statement is collinearity breakdown.",
     "reports/track_b/macro_signal_attribution_stability.json#/macro_shift/correlations",
     "State the development correlation structure (unemployment vs HPI growth -0.919, VIF 10-12, condition number 69.1, 88 distinct months) and the evaluation sign reversals as a single mechanism paragraph.",
     False, False, True),
    ("CH13", "MODERATE", "Title",
     "'Temporal Transport' states the topic, not the finding, and overstates what a single provider and one macro excursion establish.",
     "Whole-manuscript",
     "Adopt a title naming the result and its scope, e.g. 'Vintage-aware macro features improve in-sample fit but not temporal probability quality in a seasoned Freddie Mac competing-risk cohort'.",
     False, False, True),
    ("CH14", "MODERATE", "2.D, 10 (literature-informed limitations)",
     "Two paragraphs narrate the incompleteness of the authors' own literature search.",
     "docs/paper/closest_paper_matrix.json",
     "Delete the search-completeness narration. Either obtain Bu2026 full text and fill the UNKNOWN cells, or state in one sentence that the closest comparison is unresolved.",
     False, False, True),
    ("CH15", "MODERATE", "1, 2.C",
     "Peng2026 lists macro conditioning and competing termination as future work and is the natural anchor; this is not used.",
     "docs/paper/closest_paper_matrix.json",
     "Position the study in section 1 as the direct empirical answer to that stated open question.",
     False, False, True),
    ("CH16", "MINOR", "4.2, 6.4, 6.5, 8, 9",
     "'Payoff' shorthand is defined once and then used unqualified in refinancing-flavoured sentences.",
     "docs/track_b/survival_competing_risk_protocol.json#/events",
     "Use 'payoff/maturity termination' on first use in each section, or add a one-line note that scheduled maturity is negligible for these vintages and terms.",
     False, False, True),
    ("CH17", "MINOR", "Abstract, 13",
     "'Demonstrate' and the confirmatory register overstate a non-virgin frozen evaluation.",
     "docs/track_b/macro_competing_risk_protocol.json#/splits/virgin_holdout",
     "Replace 'demonstrate' with 'record' or 'observe'; use 'frozen temporal evaluation', never 'confirmatory temporal validation'.",
     False, True, True),
    ("CH18", "EDITORIAL", "Throughout",
     "Caveat density buries the finding; most paragraphs retract the preceding sentence.",
     "paper/main_v0.2.md",
     "Consolidate repeated caveats into section 10 and Table T8; state results and move.",
     False, False, True),
    ("CH19", "EDITORIAL", "11, Appendices A-H",
     "A proposed-replication section and eight placeholder appendices read as unfinished.",
     "paper/main_v0.2.md",
     "Reduce section 11 to two sentences in future work; remove appendix headers until content exists.",
     False, True, True),
    ("CH20", "MINOR", "6.1, Table T3",
     "The XGBoost challenger result (AP 0.0175 vs 0.1763) is extreme enough to read as a configuration failure rather than a model-class finding.",
     "reports/track_b/expanded_pd_validation.json",
     "State the bounded configuration explicitly and avoid any implied comparison of model families; consider moving the comparison to a supplement.",
     False, False, True),
]


def build_change_register() -> dict:
    return {
        "version": "task17-v1",
        "base_commit": BASE_COMMIT,
        "applied": False,
        "note": "Task 17 does not rewrite the manuscript. No v0.3 was created.",
        "changes": [
            {
                "change_id": cid,
                "severity": sev,
                "section": sec,
                "problem": prob,
                "evidence": ev,
                "proposed_change": prop,
                "requires_new_analysis": rna,
                "blocking_preprint": bp,
                "blocking_peer_review": bpr,
                "status": "PROPOSED",
            }
            for cid, sev, sec, prob, ev, prop, rna, bp, bpr in CHANGES
        ],
    }


SENSITIVITIES = [
    ("SA01", "Is the payoff AUC gain facility-level or between-period?",
     "F2_BETWEEN_PERIOD_AUC",
     "Frozen per-year payoff AUC gaps near zero; frozen CIF-level AUC identical (0.5906 vs 0.5908). Pooled gain +0.060 unexplained by either.",
     "Recompute temporal payoff AUC within each calendar month and aggregate (and/or a calendar-stratified concordance) on the existing frozen prediction arrays. No refit; scoring only.",
     False, "LOW: scoring an existing frozen prediction array under a prespecified stratification.",
     "SUBMISSION_BLOCKING", True,
     "If within-month AUC gain is near zero, the ranking claim is restated as across-period and the paper is stronger and more precise.",
     "If within-month AUC gain is material, the current claim stands as written and the concern is withdrawn."),
    ("SA02", "Does the conclusion survive exclusion of calendar 2020?",
     "F1_CALENDAR_CONCENTRATION",
     "Interval-weighted decomposition gives 2020 contribution +0.01440 of +0.01612; ex-2020 residual +0.00172; 2021 reverses.",
     "Report the frozen per-year table and an ex-2020 paired difference with calendar-block resampling over the remaining seven blocks. Scoring only on frozen arrays.",
     False, "LOW: the per-year aggregates are already frozen; this is recombination plus a resampling pass.",
     "SUBMISSION_BLOCKING", True,
     "If the ex-2020 difference remains positive and separable from zero, the conclusion generalises beyond the pandemic and is substantially strengthened.",
     "If it is not separable from zero, the paper's finding narrows to an out-of-support macro excursion result, which is still publishable but is a different claim."),
    ("SA03", "Why does the unseen-vintage population reverse the primary metric?",
     "F3_UNSEEN_VINTAGE_REVERSAL",
     "M2 0.069811 vs M1 0.070631 on 17,352 facilities; payoff AUC 0.552 to 0.737; cause Brier still favours M1.",
     "Report the unseen-vintage result in full with its per-year decomposition and duration/cohort encoding caveats. No new fitting.",
     False, "LOW for reporting; MODERATE if used to generate a new explanatory hypothesis on inspected outcomes.",
     "SUBMISSION_BLOCKING", True,
     "A coherent account (young unseasoned loans, different duration support, cohort reference encoding) turns the reversal into a scope statement about seasoning.",
     "Without an account, the headline conclusion cannot be stated at study level and must be stated at seen-vintage-cohort level."),
    ("SA04", "Is the comparison fair given M2's seven extra parameters?",
     "M0/M1/M2 baseline fairness",
     "Fixed L2 C=1.0 for all rungs; no complexity adjustment; development gain 0.00056 on 88 distinct macro months.",
     "Report information criteria or a development-side complexity-adjusted comparison on frozen fits; alternatively refit M2 under macro-specific regularisation chosen by calendar-block cross-validation as a clearly labelled new experiment.",
     True, "MODERATE: a refit after outcomes are inspected must be registered as a new experiment with its own freeze.",
     "STRONGLY_RECOMMENDED", False,
     "If a calendar-block-regularised M2 transports, the lesson becomes a regularisation and effective-sample-size lesson rather than a macro-information lesson.",
     "If it still fails, the conclusion is substantially strengthened against the functional-form objection."),
    ("SA05", "Does the estimand change under facility weighting?",
     "F9_INTERVAL_WEIGHTING",
     "Mean 44.3 intervals per facility; weighting correlated with survival, which is the complement of the modelled event.",
     "Recompute the paired difference with per-facility normalised weights on frozen prediction arrays. Scoring only.",
     False, "LOW.", "STRONGLY_RECOMMENDED", False,
     "Agreement between weightings removes a standard referee objection cheaply.",
     "Disagreement means the headline is a long-duration-weighted quantity and must be labelled as such."),
    ("SA06", "Is the CIF result an artifact of concentrated entry on one macro path?",
     "F6_CIF_COHORT_IS_SINGLE_PATH",
     "Zero calendar truncation at all horizons including 60 months; censor survival 0.99922 at 24 months.",
     "Publish the landmark entry-month distribution; if dispersion permits, stratify the CIF comparison by entry year.",
     False, "LOW for the distribution; MODERATE for stratified reporting on inspected outcomes.",
     "SUBMISSION_BLOCKING", True,
     "Dispersed entry supports the plural 'paths' language and makes the CIF partly independent evidence.",
     "Concentrated entry means the CIF and the 2020 calendar diagnostic are one finding and must be presented as such."),
    ("SA07", "Does the macro increment survive conditioning on current loan state?",
     "F11_DISTRESS_STATE_OMITTED",
     "updated_state = false by design; Task 5 documents strong distress-state dependence.",
     "A new prespecified experiment adding current delinquency state to M1 and M2, with its own freeze and registration.",
     True, "HIGH: new experiment on a population whose outcomes are inspected; requires a fresh holdout or an explicit exploratory label.",
     "OPTIONAL", False,
     "A state-conditional macro increment would be the practically relevant result and a strong follow-on paper.",
     "Absence of increment conditional on state would sharpen the negative result considerably."),
    ("SA08", "Does macro restricted to supported months behave differently?",
     "F7_COLLINEARITY_BREAKDOWN, support extrapolation",
     "77 of 86 evaluation months outside the development distance reference; 202,675 of 248,939 intervals.",
     "Evaluate the frozen models on the 9 in-support months only; scoring on frozen arrays.",
     False, "LOW, but the in-support subset is small and dominated by 2019.",
     "STRONGLY_RECOMMENDED", False,
     "Comparable performance in-support localises the failure to extrapolation and supports the mechanism narrative.",
     "Failure in-support too would falsify the extrapolation explanation and require a different account."),
    ("SA09", "Would a nonlinear or interaction-bearing baseline change the conclusion?",
     "Model-class attack",
     "Protocol prohibits interactions and period effects; only multinomial logistic was fitted.",
     "New prespecified experiment with a flexible challenger and/or macro-by-duration interactions, separately frozen.",
     True, "HIGH.", "OPTIONAL", False,
     "Would broaden the conclusion from one specification to a model class.",
     "Would confirm that the result is specification-bound, which the manuscript must then state plainly."),
    ("SA10", "Are maturity and voluntary payoff separable?",
     "F12_PAYOFF_MATURITY_CONFLATION",
     "Source code 01 pools both; original_loan_term is in the feature set so separation is partly derivable.",
     "Descriptive count of terminations at or near scheduled maturity by vintage and term on frozen panel.",
     False, "LOW.", "OPTIONAL", False,
     "Negligible maturity counts retire the objection in one sentence.",
     "Material maturity counts require the terminology change in CH16 to be substantive rather than editorial."),
    ("SA11", "External cross-provider replication (Fannie).",
     "Generalisation",
     "Protocol DRAFT_NOT_YET_AUTHORIZED; no results; provenance pending.",
     "As specified in the frozen Fannie protocol once authorised.",
     False, "LOW once authorised and frozen.", "NOT_NEEDED", False,
     "Would convert a provider-specific finding into a cross-provider one.",
     "Not required for a Freddie-scoped claim; its absence is adequately disclosed."),
    ("SA12", "Is the observed CIF comparator compatible with conditional entry?",
     "CG03 residual",
     "Aalen-Johansen with generic entry<t<=exit risk sets; pooled censoring KM; compatibility flagged as requiring author review.",
     "Author-level implementation review against Aalen1978/Austin2016/Heyard2020; no computation.",
     False, "NONE.", "STRONGLY_RECOMMENDED", False,
     "Closes CG03 and removes an easy referee target.",
     "An incompatibility would affect every CIF number in the paper."),
]


def build_sensitivity_register() -> dict:
    return {
        "version": "task17-v1",
        "base_commit": BASE_COMMIT,
        "executed": False,
        "note": "Task 17 specifies, it does not execute. No analysis below was run.",
        "analyses": [
            {
                "analysis_id": aid,
                "scientific_question": q,
                "reviewer_concern": rc,
                "current_evidence": ce,
                "proposed_design": pd_,
                "would_use_holdout": wh,
                "risk_of_post_hoc_bias": risk,
                "priority": pri,
                "submission_blocking": sb,
                "expected_interpretation_if_positive": pos,
                "expected_interpretation_if_negative": neg,
            }
            for aid, q, rc, ce, pd_, wh, risk, pri, sb, pos, neg in SENSITIVITIES
        ],
    }


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    outputs = {
        "task17_claim_attack_registry.json": build_attack_registry(),
        "task17_claim_survival_matrix.json": build_survival_matrix(),
        "task17_manuscript_change_register.json": build_change_register(),
        "task17_sensitivity_register.json": build_sensitivity_register(),
    }
    for name, payload in outputs.items():
        path = REVIEW / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")

    # Consistency guards.
    primary = {
        c["claim_id"]
        for c in json.loads(REGISTRY.read_text(encoding="utf-8"))["claims"]
        if c.get("claim_strength") == "PRIMARY"
    }
    attacked = {c["claim_id"] for c in CLAIMS}
    assert attacked == primary, f"PRIMARY coverage mismatch: {primary ^ attacked}"
    assert set(SURVIVAL) == primary, "survival matrix coverage mismatch"
    for c in CLAIMS:
        for f in c["findings"]:
            assert f in FINDINGS, f"unknown finding {f}"
    macro = json.loads(MACRO.read_text(encoding="utf-8"))
    cal = macro["stability"]["calendar"]
    total = sum(cal[y]["M1"]["intervals"] for y in cal)
    recon = sum(
        cal[y]["M1"]["intervals"]
        / total
        * (cal[y]["M2"]["joint_log_loss"] - cal[y]["M1"]["joint_log_loss"])
        for y in cal
    )
    frozen = macro["paired_facility"]["intervals"]["joint_log_loss"]["delta"]
    assert abs(recon - frozen) < 1e-12, f"decomposition does not reconstruct frozen delta: {recon} vs {frozen}"
    print(f"guards passed: {len(primary)} PRIMARY claims; decomposition reconstructs {frozen!r}")


if __name__ == "__main__":
    main()
