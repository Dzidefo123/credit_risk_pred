"""Two jointly fitted cause logits: coherent discrete multinomial hazards."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .math import curves

STATIC = (
    "orig_credit_score",
    "orig_ltv",
    "orig_dti",
    "orig_interest_rate",
    "original_loan_term",
    "number_of_borrowers",
    "loan_purpose",
    "occupancy_status",
)
DYNAMIC = (
    "delinquency_state",
    "loan_age",
    "current_principal_balance",
    "current_interest_rate",
    "remaining_legal_months",
)


def frame(f, registry, view):
    names = registry[view]
    expected = set(STATIC) | {"duration_band"} | (set(DYNAMIC) if view == "dynamic" else set())
    if set(names) != expected:
        raise ValueError("Survival feature firewall")
    for name in names:
        if registry["fields"][name]["classification"] not in [
            "STATIC_AT_ORIGINATION",
            "TIME_VARYING_KNOWN_AT_T0",
        ]:
            raise ValueError("Forbidden predictor")
    x = f.loc[:, [n for n in names if n != "duration_band"]].copy()
    x["duration_band"] = pd.cut(
        f.duration,
        [0, 12, 24, 36, 60, 120, np.inf],
        labels=["01_12", "13_24", "25_36", "37_60", "61_120", "121_plus"],
    ).astype(object)
    cats = ["loan_purpose", "occupancy_status", "duration_band"] + (
        ["delinquency_state"] if view == "dynamic" else []
    )
    for name in names:
        if name in cats:
            x[name] = x[name].astype(object).where(x[name].notna(), np.nan)
        else:
            x[name] = pd.to_numeric(x[name], errors="raise")
    if view == "dynamic":
        if x.current_principal_balance.lt(0).any():
            raise ValueError("Negative current balance")
        x["current_principal_balance"] = np.log1p(x.current_principal_balance)
    return x


def fit(f, registry, view):
    x = frame(f, registry, view)
    cats = ["loan_purpose", "occupancy_status", "duration_band"] + (
        ["delinquency_state"] if view == "dynamic" else []
    )
    nums = [n for n in x if n not in cats]
    if x.isna().all().any() or set(f.event_code) != set([0, 1, 2]):
        raise ValueError("Invalid development support")
    pre = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                ),
                nums,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                ),
                cats,
            ),
        ],
        verbose_feature_names_out=False,
    )
    model = make_pipeline(
        pre, LogisticRegression(C=1, solver="lbfgs", max_iter=3000, tol=1e-8, random_state=61006)
    )
    model.fit(x, f.event_code)
    if model[-1].n_iter_.max() >= 3000:
        raise ValueError("Cause logits failed convergence")
    return model


def probabilities(model, f, registry, view):
    p = model.predict_proba(frame(f, registry, view))
    if model[-1].classes_.tolist() != [0, 1, 2] or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Cause probability coherence failed")
    return p


def forecast(model, subjects, registry, horizon):
    d = []
    p = []
    for t in range(1, horizon + 1):
        f = subjects.copy()
        f["duration"] = t
        q = probabilities(model, f, registry, "structural")
        d.append(q[:, 1])
        p.append(q[:, 2])
    return curves(np.column_stack(d), np.column_stack(p))


def likelihood(risk, p, weights=None):
    w = np.ones(len(risk)) if weights is None else np.asarray(weights, float)
    y = risk.event_code.to_numpy(int)
    return dict(
        joint_log_loss=float(-w @ np.log(np.maximum(p[np.arange(len(p)), y], 1e-15)) / w.sum()),
        default_brier=float(w @ ((p[:, 1] - (y == 1)) ** 2) / w.sum()),
        payoff_brier=float(w @ ((p[:, 2] - (y == 2)) ** 2) / w.sum()),
    )


def coefficients(model):
    names = model[0].get_feature_names_out()
    out = []
    for cause, index in [("default", 1), ("payoff_maturity", 2)]:
        for name, value in zip(names, model[-1].coef_[index] - model[-1].coef_[0], strict=True):
            out.append(
                dict(
                    cause=cause,
                    feature=str(name),
                    log_odds=float(value),
                    odds_ratio=float(np.exp(value)),
                    interpretation=(
                        "Conditional cause vs no-event odds; not proportional-hazard ratio or "
                        "causal CIF effect"
                    ),
                )
            )
    return out
