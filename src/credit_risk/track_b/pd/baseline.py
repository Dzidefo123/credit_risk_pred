"""Prespecified mortgage cohort and loan-cluster evaluation; no Track A fitting."""

import hashlib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.validation.calibration import calibration_diagnostics

FEATURES = ("orig_credit_score", "orig_ltv", "loan_age", "delinquency_state")
SALT = "track-b-pd-v1:"
DEV_END = "2014-12"
EVAL_START = "2016-01"
SEED = 31003
MIN_DEV_EVENTS = 10
STABLE_EVAL_EVENTS = 20
KNOWN = {"positive_default": 1, "negative_survived_horizon": 0, "competing_payoff": 0}


def counts(f):
    pos = f.binary_default_12m.eq(1)
    return dict(
        landmarks=int(len(f)),
        loans=int(f.loan_id.nunique()),
        positive_landmarks=int(pos.sum()),
        default_loans=int(f.loc[pos, "loan_id"].nunique()),
    )


def cohort(panel, *, payoff=True, quarterly=False, physical=False):
    if panel.duplicated(["loan_id", "t0"]).any():
        raise ValueError("Duplicate loan/landmark")
    e = panel.loc[panel.eligible.eq(True)].copy()
    state = pd.to_numeric(e.delinquency_state, errors="coerce")
    if state.isna().any() or not state.between(0, 2).all():
        raise ValueError("Eligible prevalent default/unknown state contradicts protocol")
    known = e.outcome_status.isin(KNOWN)
    for status, label in KNOWN.items():
        if not e.loc[e.outcome_status.eq(status), "binary_default_12m"].eq(label).all():
            raise ValueError("Outcome label contradicts protocol")
    if e.loc[~known, "binary_default_12m"].notna().any():
        raise ValueError("Unknown outcome has imputed label")
    f = e.loc[known].copy()
    if not payoff:
        f = f.loc[f.outcome_status.ne("competing_payoff")].copy()
    if quarterly:
        f = f.loc[f.t0.str[-2:].astype(int).isin([3, 6, 9, 12])].copy()
    if physical:
        keys = set(zip(panel.loan_id, pd.PeriodIndex(panel.t0, freq="M"), strict=True))
        keep = [
            all((loan, month + k) in keys for k in range(1, 13))
            for loan, month in zip(f.loan_id, pd.PeriodIndex(f.t0, freq="M"), strict=True)
        ]
        f = f.loc[keep].copy()
    return f.sort_values(["loan_id", "t0"]).reset_index(drop=True)


def cohort_hash(f):
    c = f[["loan_id", "t0", "outcome_status", "binary_default_12m"]].copy()
    c["binary_default_12m"] = c.binary_default_12m.astype("Int64")
    return hashlib.sha256(
        c.sort_values(["loan_id", "t0"]).to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()


def split(f):
    a = f.loan_id.map(lambda s: int(hashlib.sha256((SALT + s).encode()).hexdigest(), 16) % 10 < 7)
    dev = f.loc[a & f.t0.le(DEV_END)].copy()
    ev = f.loc[~a & f.t0.ge(EVAL_START)].copy()
    if set(dev.loan_id) & set(ev.loan_id):
        raise ValueError("Loan overlap")
    if (
        len(dev)
        and len(ev)
        and pd.Period(dev.t0.max(), freq="M") + 12 >= pd.Period(ev.t0.min(), freq="M")
    ):
        raise ValueError("Outcome horizon not purged")
    return dev, ev


def gate(dev, ev):
    if (
        counts(dev)["default_loans"] < MIN_DEV_EVENTS
        or dev.binary_default_12m.nunique() != 2
        or counts(ev)["default_loans"] == 0
        or ev.binary_default_12m.nunique() != 2
    ):
        return "INSUFFICIENT EVENT SUPPORT"
    return (
        "BASELINE ESTABLISHED — EXPLORATORY ONLY"
        if counts(ev)["default_loans"] < STABLE_EVAL_EVENTS
        else "BASELINE ESTABLISHED"
    )


def features(f, registry, requested=FEATURES):
    if tuple(requested) != FEATURES or registry["selected"] != list(FEATURES):
        raise ValueError("Prespecified feature firewall")
    for name in requested:
        item = registry["fields"].get(name, {})
        if item.get("classification") not in {
            "STATIC_AT_ORIGINATION",
            "TIME_VARYING_KNOWN_AT_T0",
        } or not item.get("selected"):
            raise ValueError("Forbidden feature")
    return f.loc[:, list(requested)].apply(pd.to_numeric, errors="raise")


def fit_baselines(dev, ev, registry):
    if gate(dev, ev) == "INSUFFICIENT EVENT SUPPORT":
        raise ValueError("Pre-fit event-support gate failed")
    x = features(dev, registry)
    if x.isna().all().any():
        raise ValueError("All-missing development predictor")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000, tol=1e-8, class_weight=None, random_state=SEED
        ),
    )
    model.fit(x, dev.binary_default_12m.astype(int))
    if model[-1].n_iter_[0] >= 2000:
        raise ValueError("Logistic convergence limit exhausted")
    return model, {
        "null": np.full(len(ev), float(dev.binary_default_12m.mean())),
        "logistic": model.predict_proba(features(ev, registry))[:, 1],
    }


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if (
        len(y) == 0
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or not ((p >= 0) & (p <= 1)).all()
    ):
        raise ValueError("Invalid metric inputs")
    m = dict(
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        observed_rate=float(y.mean()),
        mean_probability=float(p.mean()),
        roc_auc=None,
        gini=None,
        pr_auc_trapezoid=None,
        average_precision=None,
    )
    if len(np.unique(y)) < 2:
        m["discrimination_status"] = "UNDEFINED — SINGLE CLASS"
        return m
    precision, recall, _ = precision_recall_curve(y, p)
    m.update(
        roc_auc=float(roc_auc_score(y, p)),
        gini=float(2 * roc_auc_score(y, p) - 1),
        pr_auc_trapezoid=float(auc(recall, precision)),
        average_precision=float(average_precision_score(y, p)),
        discrimination_status="estimated",
    )
    return m


def cluster_indices(loans, rng):
    groups = pd.Series(np.arange(len(loans))).groupby(np.asarray(loans), sort=True).apply(list)
    chosen = rng.integers(0, len(groups), len(groups))
    return np.concatenate([np.asarray(groups.iloc[i]) for i in chosen])


def bootstrap(f, p, draws=500, seed=SEED):
    rng = np.random.default_rng(seed)
    values = {k: [] for k in metrics(f.binary_default_12m, p) if k != "discrimination_status"}
    single = 0
    for _ in range(draws):
        idx = cluster_indices(f.loan_id, rng)
        m = metrics(f.binary_default_12m.to_numpy()[idx], np.asarray(p)[idx])
        single += m["roc_auc"] is None
        for k in values:
            if m[k] is not None:
                values[k].append(m[k])
    return dict(
        unit="loan_id",
        draws=draws,
        seed=seed,
        single_class_draws=int(single),
        scope=(
            "Evaluation sampling only; fitted model fixed; no training uncertainty "
            "or borrower clustering"
        ),
        intervals={
            k: dict(
                lower=float(np.quantile(v, 0.025)),
                upper=float(np.quantile(v, 0.975)),
                valid_draws=len(v),
            )
            if v
            else None
            for k, v in values.items()
        },
    )


def diagnostics(f, p):
    n = counts(f)
    c = calibration_diagnostics(f.binary_default_12m.astype(int), p)
    c["interval_reason"] = (
        "No calibration interval estimated: sparse event-loan support; "
        "no iid landmark inference justified"
    )
    c["support_status"] = (
        "UNSTABLE / INSUFFICIENT EVENT SUPPORT"
        if n["default_loans"] < STABLE_EVAL_EVENTS
        else c["slope_status"]
    )
    bins = pd.cut(p, [-1e-12, 0.005, 0.02, 1.0], include_lowest=True)
    table = []
    for level in bins.categories:
        mask = bins == level
        if mask.any():
            table.append(
                dict(
                    bin=str(level),
                    **counts(f.loc[mask]),
                    observed_rate=float(f.loc[mask, "binary_default_12m"].mean()),
                    mean_probability=float(np.asarray(p)[mask].mean()),
                    sparse=True,
                )
            )
    return dict(
        effective_sample=n,
        metrics=metrics(f.binary_default_12m, p),
        calibration=c,
        reliability=table,
        uncertainty=bootstrap(f, p),
    )
