"""Freeze scientific choices before fitting; no outcome-driven amendment."""

from credit_risk.track_b.macro_support.eligibility import REDUCED
from credit_risk.track_b.macro_support.study import immutable_json
from credit_risk.track_b.multivintage.study import lf_hash

PRIMARY = (
    "unemployment_level",
    "unemployment_change_3m",
    "treasury_10y_level",
    "mortgage_30y_level",
    "hpi_yoy",
    "cpi_yoy",
    "gdp_qoq",
)
RATE = tuple("mortgage_treasury_spread" if n == "mortgage_30y_level" else n for n in PRIMARY)
NUMERIC = (
    "orig_credit_score",
    "orig_ltv",
    "orig_dti",
    "orig_upb",
    "orig_interest_rate",
    "original_loan_term",
)
CATEGORICAL = ("loan_purpose", "occupancy_status")
STRUCTURAL = ("duration_band", "cohort")


def freeze(root):
    spec = dict(
        version="macro-hazard-v1.0.0",
        parent_commit="1978ec4655a585f4ab0d3bcc2bd6a8442b1c0090",
        eligibility_sha256_lf="f1bd2f1082574b26f3ca969764e51d7228445c247fa89e80c2ded4bac74a95fe",
        task9a_validation_sha256_lf=lf_hash(
            root / "docs/track_b/macro_support_validation_design.json"
        ),
        family="Discrete monthly multinomial logistic, classes 0 none / 1 default / 2 payoff",
        ladder={
            "M0": list(STRUCTURAL),
            "M1": [*STRUCTURAL, *NUMERIC, *CATEGORICAL],
            "M2": [*STRUCTURAL, *NUMERIC, *CATEGORICAL, *PRIMARY],
        },
        primary_macro=list(PRIMARY),
        rate_sensitivity=list(RATE),
        reduced_macro=list(REDUCED),
        rate_identification=(
            "Exclude spread from primary coefficient vector; retain full8 eligibility"
        ),
        preprocessing=dict(
            numeric_imputation="Development risk-row median; one missing indicator per numeric",
            all_missing_numeric="STOP",
            original_upb="log1p after nonnegative check",
            numeric_scaling="Development risk-row mean / population SD; constants scale1",
            categorical="Development-only sorted vocabulary; missing explicit token; drop first",
            unknown_category="All-zero reference encoding, separately counted, no evaluation fit",
            duration="Task9A bands, reference0-12; no knot/band changes",
            cohort=(
                "Task9A vintage indicators, reference2006; unrepresented effects zero and flagged"
            ),
        ),
        estimator=dict(
            solver="lbfgs",
            C=1.0,
            penalty="L2",
            max_iter=3000,
            tol=1e-8,
            random_state=61010,
            class_weights=None,
            thread_limit=1,
        ),
        splits=dict(
            development=["2010-09", "2017-12"],
            reduced_development=["2006-02", "2017-12"],
            purge=["2018-01", "2018-12"],
            evaluation=["2019-01", "2026-02"],
            role_salt="track-b-macro-eligibility-v1",
            primary_vintages=[2006, 2008, 2010, 2014],
            unseen_vintages=[2018, 2020, 2022],
            virgin_holdout=False,
        ),
        bootstrap=dict(
            unit="facility",
            replicates=1000,
            seed=61035,
            minimum_valid=950,
            interval="Percentile95 paired fixed-model differences",
            auc_min_events=20,
            calendar_sensitivity="1000 calendar-year block resamples, seed61036",
            no_model_refitting=True,
        ),
        calibration=dict(
            bins=10,
            logits_clip=1e-12,
            minimum_events=20,
            diagnostic_intercept_slope_only=True,
            evaluation_recalibration=False,
        ),
        cif=dict(
            horizons=[12, 24, 36, 60],
            landmarks="First eligible evaluation interval per facility",
            future_macro="Historical rolling PIT paths, NOT entry-time known forecasts",
            late_landmarks="Exclude per-horizon solely if full calendar path exceeds2026-02",
            observed_reference="Aalen-Johansen, payoff competing event not censoring",
            ipcw="Task6 pooled censor KM, conditional independent censoring assumed",
            minimum_at_risk=200,
            minimum_censor_survival=0.1,
            minimum_default_events=20,
            prediction_tolerance=1e-10,
        ),
        cell_suppression=dict(minimum_cause_events=20, minimum_facilities=100),
        sensitivities=dict(
            reduced=True,
            rate=True,
            updated_state=False,
            updated_state_reason="Optional; omitted to isolate structural/macro research",
            leave_vintage_out=[2006, 2008, 2010, 2014],
            leave_vintage_out_minimum_defaults=100,
            leave_vintage_out_minimum_payoffs=500,
            heldout_effect="Zero relative training reference; if2006 heldout use2008",
            pandemic=[2020, 2021],
            gfc_reduced=[2007, 2008, 2009],
        ),
        decision=dict(
            primary="M2-M1 paired temporal joint log loss, smaller better",
            positive=(
                "Both facility and calendar95 CI upper<0; cause Brier and absolute "
                "mean calibration error degradation<=1e-4"
            ),
            mixed="Any paired proper-score improvement but positive criterion not satisfied",
            negative="No paired proper-score improvement demonstrated",
            no_retuning=True,
        ),
        ledger=dict(
            namespace="TASK10_MACRO_HAZARD",
            register_before_fit=True,
            consume_once="All prespecified final temporal predictions in one sealed session",
            deterministic_replay=(
                "Verify-only replay of same frozen inputs/artifacts; no new fit/decision"
            ),
        ),
        prohibited=[
            "future loan states",
            "feature search",
            "period fixed effects",
            "interactions",
            "causal effects",
            "regulatory PD",
            "EAD/LGD/ECL",
            "fairness claim",
        ],
    )
    immutable_json(root / "docs/track_b/macro_competing_risk_protocol.json", spec)
    return spec
