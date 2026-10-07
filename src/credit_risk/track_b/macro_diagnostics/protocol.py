"""Explicit post-validation diagnostic choices, frozen before computation."""

from credit_risk.track_b.macro_support.study import immutable_json

BASE = "3108a9134fc17c3b359999f11b5767cf1332b35f"
LABEL = "POST_VALIDATION_DIAGNOSTIC"
PRIVATE = "data/track_b/models/macro_diagnostics_v1"
HYPOTHESES = {
    "H1": "Macro support extrapolation drives failure",
    "H2": "Payoff intercept/base-rate drift dominates",
    "H3": "Payoff macro coefficients are temporally unstable",
    "H4": "Default macro relationships are temporally unstable",
    "H5": "Changing macro correlation structure contributes",
    "H6": "Mortgage composition/survivor shift contributes",
    "H7": "Pandemic alone explains failure",
}
WINDOWS = [(2010, 2013), (2014, 2017), (2019, 2021), (2022, 2025)]


def freeze(root):
    spec = dict(
        version="macro-diagnostics-v1.0.0",
        base_commit=BASE,
        analysis_label=LABEL,
        task10_conclusion="NO RELIABLE TEMPORAL MACRO INCREMENT DEMONSTRATED",
        population="Task10 primary development and seen-vintage temporal arrays unchanged",
        supplementary_population="Task10 unseen-vintage arrays described separately only",
        macro_features="Exact seven primary Task10 terms; no alternatives added",
        distributions="Risk-interval weighted plus distinct-month macro summaries",
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
        bins="Development risk-row deciles, duplicate edges collapsed; infinite tail bins",
        psi="Development-frozen bins; add 1e-6 to each bin frequency then renormalize",
        support="Development min/max and 5th/95th percentiles; no invalidity thresholds",
        multivariate=(
            "Distinct-month development-fitted standardized PCA; Mahalanobis "
            "pseudoinverse rcond1e-10; development95 distance descriptive "
            "reference"
        ),
        primary_attribution=(
            "Frozen cause-vs-none M2 coefficients on exact development standardization"
        ),
        excess_logit=(
            "Macro terms plus explicitly retained nonmacro/intercept coefficient-change residual"
        ),
        probability_attribution=(
            "POST_HOC_PROBABILITY_ATTRIBUTION: M1 logit baseline plus one M2 "
            "macro contribution, others zero at development means; nonadditive, "
            "not unique"
        ),
        ablation=(
            "FROZEN_COEFFICIENT_DIAGNOSTIC_ABLATION: subtract one feature's two "
            "cause-vs-none contributions from M2 logits without refit"
        ),
        component_substitution=(
            "POST_HOC_COMPONENT_SUBSTITUTION_DIAGNOSTIC: swap raw cause hazards; "
            "fail if hazard sum exceeds1, never renormalize"
        ),
        oracle=(
            "POST_HOC_ORACLE_DIAGNOSTIC: two joint multinomial intercept offsets "
            "on frozen M2 logits, same evaluation outcomes; optimistic in-sample "
            "diagnosis only"
        ),
        optional_slope_oracle=False,
        diagnostic_refits=dict(
            label="DIAGNOSTIC_REFIT_ONLY",
            windows=WINDOWS,
            specification=(
                "Task10 M2 family, C1, tol1e-8, max3000, seed61010; within-window encoder"
            ),
            evaluation_role_copy=(
                "Explicit isolated post-validation training view; original role arrays read-only"
            ),
            minimum_defaults=20,
            minimum_payoffs=100,
            minimum_months=24,
            failures="Report unsupported/nonconverged windows; no retry/tuning",
            coefficient_scales=(
                "Both window SD and common Task10 development SD; native unit contrast"
            ),
            uncertainty=(
                "No naive row-level SE; identification and window/LVO dispersion, "
                "not confidence intervals"
            ),
            no_replacement_scoring=True,
        ),
        calibration=dict(minimum_events=20, minimum_facilities=100, applied_to_models=False),
        composition=(
            "Interval-weighted and one first row per facility; latter means "
            "surviving into each split, not original entire cohorts"
        ),
        cif=dict(
            horizons=[12, 24, 36, 60],
            one_landmark_per_facility=True,
            paths="Same frozen historical rolling PIT macro path; not prospective forecasts",
        ),
        hypotheses=HYPOTHESES,
        hypothesis_interpretation=(
            "Prelisted diagnostic mechanisms; descriptive association cannot "
            "identify causal contributions/APC"
        ),
        root_cause_interpretation=(
            "Rank frozen-contribution and error-decomposition evidence, not feature selection"
        ),
        no_task10_ledger_api=True,
        no_model_promotion=True,
        no_task12_implementation=True,
    )
    immutable_json(root / "docs/track_b/macro_signal_diagnostic_protocol.json", spec)
    return spec
