"""Post-validation score accounting and prelisted hypothesis assessment, not selection."""

import joblib
import numpy as np

from credit_risk.track_b.macro_hazard.metrics import losses, scores
from credit_risk.track_b.macro_hazard.protocol import PRIMARY
from credit_risk.track_b.macro_hazard.study import population
from credit_risk.track_b.macro_support.study import read_json

from .distributions import correlation, distinct_months
from .math import ablated, brier_accounting, components
from .protocol import HYPOTHESES


def supplemental(root, result):
    old = root / "data/track_b/models/macro_hazard_v1"
    spec = read_json(root / "docs/track_b/macro_competing_risk_protocol.json")
    _, development, all_evaluation, seen = population(root, spec)
    rows = all_evaluation[seen]
    p = {n: np.load(old / (n + "_evaluation.npy"), mmap_mode="r")[seen] for n in ["M1", "M2"]}
    y = rows["event"]
    difference = losses(y, p["M2"]) - losses(y, p["M1"])
    accounting = dict(
        label="POST_VALIDATION_DIAGNOSTIC",
        brier={
            cause: {n: brier_accounting(y == code, q[:, code]) for n, q in p.items()}
            for code, cause in [(1, "default"), (2, "payoff")]
        },
        joint_loss_by_realized_event={
            name: dict(
                intervals=int((y == code).sum()),
                mean_delta=float(difference[y == code, 0].mean()),
                weighted_contribution=float(difference[y == code, 0].sum() / len(y)),
            )
            for code, name in [(0, "no_event"), (1, "default"), (2, "payoff")]
        },
    )
    assert np.isclose(
        sum(
            v["weighted_contribution"] for v in accounting["joint_loss_by_realized_event"].values()
        ),
        result["frozen_task10"]["paired_facility"]["intervals"]["joint_log_loss"]["delta"],
    )
    logits, parts = components(joblib.load(old / "M2.joblib"), rows)
    accounting["frozen_ablation_ranking"] = {
        name: dict(
            label="FROZEN_COEFFICIENT_DIAGNOSTIC_ABLATION",
            scores=scores(rows, ablated(logits, parts["macro"], i)),
        )
        for i, name in enumerate(PRIMARY)
    }
    refits = result["diagnostic_refits"]
    for key, r in refits.items():
        start, end = map(int, key.split("_"))
        source = development if end <= 2017 else rows
        window = source[(source["month"] // 12 >= start) & (source["month"] // 12 <= end)]
        _, values = distinct_months(window)
        r["macro_correlation_geometry"] = correlation(values)
    stability = {}
    for cause in ["default", "payoff"]:
        stability[cause] = {}
        for feature in PRIMARY:
            values = {
                key: next(
                    c["common_task10_sd_log_odds"]
                    for c in r["macro_coefficients"]
                    if c["cause"] == cause and c["feature"] == feature
                )
                for key, r in refits.items()
                if r["status"] == "CONVERGED"
            }
            x = np.array(list(values.values()))
            stability[cause][feature] = dict(
                common_task10_sd_log_odds=values,
                sign_reversal=bool(x.min() < 0 < x.max()),
                minimum=float(x.min()),
                maximum=float(x.max()),
                uncertainty=(
                    "Window estimates only; no independent-row CI. Late default "
                    "window has sparse events."
                ),
            )
    return accounting, stability


def assess(result):
    """Statuses evaluate the prelisted diagnostic hypotheses, not model validity anew."""
    fraction = result["oracle"]["fraction_of_frozen_excess_removed"]
    stability = result["coefficient_stability"]
    registers = {
        "H1": (
            "SUPPORTED_BY_DIAGNOSTIC_EVIDENCE",
            (
                "Outcome-free range/PCA distance shifts plus frozen "
                "contribution/error diagnostics align; this is "
                "predictive/mechanical evidence, not an identified causal "
                "effect."
            ),
        ),
        "H2": (
            "SUPPORTED_BY_DIAGNOSTIC_EVIDENCE" if fraction > 0.5 else "NOT_SUPPORTED",
            f"Oracle level correction removes {fraction:.1%} of excess joint loss; "
            "level drift exists but does not explain most failure.",
        ),
        "H3": (
            "SUPPORTED_BY_DIAGNOSTIC_EVIDENCE"
            if any(r["sign_reversal"] for r in stability["payoff"].values())
            else "INCONCLUSIVE",
            (
                "Coarse-window payoff signs/magnitudes change, including "
                "rate/HPI associations. Different populations and correlated "
                "predictors limit interpretation."
            ),
        ),
        "H4": (
            "PARTIALLY_SUPPORTED",
            (
                "Default coefficient changes and 2020 ranking deterioration "
                "are visible; the late diagnostic window has only 43 "
                "defaults and no coefficient confidence intervals."
            ),
        ),
        "H5": (
            "PARTIALLY_SUPPORTED",
            (
                "Distinct-month correlation geometry and VIF change, "
                "consistent with unstable mappings; its separate "
                "contribution to failure is not identified."
            ),
        ),
        "H6": (
            "PARTIALLY_SUPPORTED",
            (
                "Age/vintage composition and nonmacro coefficient allocation "
                "shift; disjoint roles prevent interpreting split "
                "differences as matched-facility attrition or causal APC "
                "identification."
            ),
        ),
        "H7": (
            "NOT_SUPPORTED",
            (
                "2020 drives much of the aggregate error, but frozen "
                "proper-score deterioration and opposite-direction payoff "
                "underprediction persist after 2022."
            ),
        ),
    }
    register = {
        key: dict(
            hypothesis=HYPOTHESES[key],
            status=status,
            evidence=reason,
            label="POST_VALIDATION_DIAGNOSTIC",
        )
        for key, (status, reason) in registers.items()
    }
    ranking = [
        dict(
            rank=1,
            mechanism="Out-of-support macro mapping with nonlinear payoff-tail amplification",
            category="PRIMARY CONTRIBUTOR",
            evidence=(
                "Pandemic unemployment/rate contributions and frozen "
                "ablations align with the 2020 payoff excess and cumulative "
                "exit overshoot."
            ),
        ),
        dict(
            rank=2,
            mechanism="Regime-dependent coefficient mapping",
            category="PRIMARY CONTRIBUTOR",
            evidence=(
                "Window sign reversals and overprediction in 2020 followed "
                "by underprediction after 2022; not an identified causal "
                "attribution."
            ),
        ),
        dict(
            rank=3,
            mechanism="Calibration-level drift",
            category="SECONDARY CONTRIBUTOR",
            evidence=(
                f"Joint intercept oracle removes {fraction:.1%} of excess loss "
                "and leaves substantial residual deterioration."
            ),
        ),
        dict(
            rank=4,
            mechanism="Duration/cohort allocation and survivor composition",
            category="SECONDARY CONTRIBUTOR",
            evidence=(
                "Exact frozen logit decomposition exposes structural "
                "coefficient redistribution, but APC components remain "
                "unidentified."
            ),
        ),
        dict(
            rank=5,
            mechanism="Changing macro collinearity",
            category="LIMITED EVIDENCE",
            evidence=(
                "Predictor geometry changes; direct attribution of score "
                "deterioration to correlation changes is unresolved."
            ),
        ),
        dict(
            rank=6,
            mechanism="Pandemic alone or global intercept error alone",
            category="NOT SUPPORTED",
            evidence=(
                "Later-period underprediction persists and oracle correction "
                "removes a minority of excess loss."
            ),
        ),
    ]
    return dict(
        label="POST_VALIDATION_DIAGNOSTIC",
        hypothesis_register=register,
        root_cause_ranking=ranking,
        decision="MACRO FAILURE MECHANISMS PARTIALLY IDENTIFIED",
        reason=(
            "Mechanical payoff amplification is clear; causal/APC "
            "separation and sparse-default coefficient transport remain "
            "unresolved."
        ),
        task10_conclusion_unchanged="NO RELIABLE TEMPORAL MACRO INCREMENT DEMONSTRATED",
        next_task="Track B Task 12 — Prespecified Refinancing-Incentive Payoff Hazard Research",
        future_research_hypotheses=[
            dict(
                label="FUTURE_HYPOTHESIS_NOT_IMPLEMENTED",
                hypothesis=(
                    "A prespecified original-coupon versus PIT mortgage-rate "
                    "incentive representation could improve payoff probability "
                    "transport, tested against frozen mortgage-only baselines "
                    "with new independent evidence."
                ),
            ),
            dict(
                label="FUTURE_HYPOTHESIS_NOT_IMPLEMENTED",
                hypothesis=(
                    "A later separately prespecified dynamic baseline or "
                    "time-varying coefficient experiment may be needed if "
                    "incentive structure alone does not transport."
                ),
            ),
        ],
        task12_started=False,
        model_promoted=False,
    )
