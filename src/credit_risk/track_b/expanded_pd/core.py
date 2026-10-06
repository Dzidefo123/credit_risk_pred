"""Task 5 fixed features, grouped partitions, hazard and paired cluster metrics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from credit_risk.track_b.pd.baseline import counts, metrics
from credit_risk.validation.calibration import calibration_diagnostics

NUMERIC = (
    "orig_credit_score",
    "orig_ltv",
    "orig_dti",
    "orig_interest_rate",
    "original_loan_term",
    "number_of_borrowers",
    "loan_age",
    "delinquency_state",
)
CATEGORICAL = ("loan_purpose", "occupancy_status")
SEED = 51005


def internal_split(f):
    group = f.loan_id.map(
        lambda i: int(hashlib.sha256(("task5-development-v1:" + i).encode()).hexdigest(), 16) % 10
    )
    return f.loc[group < 6].copy(), f.loc[group.between(6, 7)].copy(), f.loc[group >= 8].copy()


def check_time(name, observed, t0):
    if pd.Period(observed, freq="M") > pd.Period(t0, freq="M"):
        raise ValueError("Future feature observation prohibited: " + name)


def frame(f, registry, static=False):
    names = [
        n for n in registry["selected"] if not static or n not in ["loan_age", "delinquency_state"]
    ]
    if set(names) != set([*NUMERIC, *CATEGORICAL]) - (
        set(["loan_age", "delinquency_state"]) if static else set()
    ):
        raise ValueError("Prespecified feature firewall")
    for name in names:
        rule = registry["fields"].get(name, {})
        if not rule.get("selected") or rule.get("classification") not in [
            "STATIC_AT_ORIGINATION",
            "TIME_VARYING_KNOWN_AT_T0",
        ]:
            raise ValueError("Prohibited feature")
    x = f.loc[:, names].copy()
    for name in names:
        if name in NUMERIC:
            x[name] = pd.to_numeric(x[name], errors="raise")
        else:
            x[name] = x[name].astype(object).where(x[name].notna(), np.nan)
    return x


def pipeline(kind, config, static=False):
    nums = [n for n in NUMERIC if not static or n not in ["loan_age", "delinquency_state"]]
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
                list(CATEGORICAL),
            ),
        ],
        verbose_feature_names_out=False,
    )
    estimator = (
        LogisticRegression(**config, random_state=SEED)
        if kind == "logistic"
        else XGBClassifier(**config)
    )
    return make_pipeline(pre, estimator)


def fit_model(kind, f, registry, config, static=False):
    x = frame(f, registry, static)
    if x.isna().all().any():
        raise ValueError("All-missing development feature")
    model = pipeline(kind, config, static)
    model.fit(x, f.binary_default_12m.astype(int))
    if kind == "logistic" and model[-1].n_iter_[0] >= config["max_iter"]:
        raise ValueError("Nonconverged logistic fit")
    return model


def calibrate(model, cal, selection, registry):
    pc = model.predict_proba(frame(cal, registry))[:, 1]
    raw = model.predict_proba(frame(selection, registry))[:, 1]
    calibrator = LogisticRegression(C=1.0, max_iter=2000, tol=1e-8, random_state=SEED)
    calibrator.fit(
        logit(np.clip(pc, 1e-12, 1 - 1e-12)).reshape(-1, 1), cal.binary_default_12m.astype(int)
    )
    adjusted = apply_calibrator(calibrator, raw)
    a, b = (
        metrics(selection.binary_default_12m, raw),
        metrics(selection.binary_default_12m, adjusted),
    )
    retained = (
        calibrator.coef_[0, 0] > 0
        and b["brier"] <= 0.99 * a["brier"]
        and b["log_loss"] <= 0.99 * a["log_loss"]
        and b["roc_auc"] >= a["roc_auc"] - 0.002
    )
    return calibrator, dict(
        decision="CALIBRATION RETAINED" if retained else "RAW RETAINED",
        effective_sample=counts(selection),
        raw=a,
        sigmoid=b,
        sigmoid_intercept=float(calibrator.intercept_[0]),
        sigmoid_slope=float(calibrator.coef_[0, 0]),
    )


def apply_calibrator(calibrator, p):
    return calibrator.predict_proba(logit(np.clip(p, 1e-12, 1 - 1e-12)).reshape(-1, 1))[:, 1]


def hazard_periods(f):
    risk = f.loc[f.eligible.eq(True) & f.observed_followup_months.ge(1)].copy()
    risk["hazard_event"] = (
        risk.outcome_status.eq("positive_default") & risk.event_offset.eq(1)
    ).astype(int)
    if risk.groupby("loan_id").hazard_event.sum().gt(1).any():
        raise ValueError("Duplicated hazard default event")
    risk["binary_default_12m"] = risk.hazard_event
    return risk


def survival_pd(h):
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[1] != 12 or not np.isfinite(h).all() or ((h < 0) | (h > 1)).any():
        raise ValueError("Expected twelve valid monthly hazards")
    with np.errstate(divide="ignore"):
        return -np.expm1(np.log1p(-h).sum(axis=1))


def hazard_forecast(model, f, registry):
    h = []
    x = frame(f, registry)
    for month in range(12):
        future = x.copy()
        future["loan_age"] = x.loan_age + month
        h.append(model.predict_proba(future)[:, 1])
    return survival_pd(np.column_stack(h))


class WeightedMetrics:
    """Cache score order/ties; weights implement complete loan resampling exactly."""

    def __init__(self, y, p):
        self.y = np.asarray(y, dtype=float)
        self.p = np.asarray(p, dtype=float)
        metrics(self.y, self.p)
        self.order = np.argsort(-self.p, kind="stable")
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.p[self.order])) + 1]
        safe = np.clip(self.p, np.finfo(float).eps, 1 - np.finfo(float).eps)
        self.loss = -(self.y * np.log(safe) + (1 - self.y) * np.log1p(-safe))

    def evaluate(self, w):
        w = np.asarray(w, dtype=float)
        total = w.sum()
        pos = float(w @ self.y)
        neg = total - pos
        if total <= 0 or (w < 0).any():
            raise ValueError("Invalid bootstrap weights")
        m = dict(
            brier=float(w @ ((self.p - self.y) ** 2) / total),
            log_loss=float(w @ self.loss / total),
            observed_rate=pos / total,
            mean_probability=float(w @ self.p / total),
            roc_auc=None,
            gini=None,
            average_precision=None,
            pr_auc_trapezoid=None,
        )
        if not pos or not neg:
            return m
        ws = w[self.order]
        ys = self.y[self.order]
        tp = np.cumsum(np.add.reduceat(ws * ys, self.starts))
        fp = np.cumsum(np.add.reduceat(ws * (1 - ys), self.starts))
        recall = tp / pos
        precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
        area = float(np.trapezoid(np.r_[0, recall], np.r_[0, fp / neg]))
        m.update(
            roc_auc=area,
            gini=2 * area - 1,
            average_precision=float(np.diff(np.r_[0, recall]) @ precision),
            pr_auc_trapezoid=float(np.trapezoid(np.r_[1, precision], np.r_[0, recall])),
        )
        return m


def clustered(f, predictions, draws=1000, seed=SEED):
    ids, indices = np.unique(f.loan_id.astype(str), return_inverse=True)
    caches = {k: WeightedMetrics(f.binary_default_12m, p) for k, p in predictions.items()}
    samples = {k: {metric: [] for metric in caches[k].evaluate(np.ones(len(f)))} for k in caches}
    rng = np.random.default_rng(seed)
    fingerprint = hashlib.sha256()
    invalid = 0
    for _ in range(draws):
        mult = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
        fingerprint.update(mult.astype("<i4").tobytes())
        w = mult[indices]
        for name, cache in caches.items():
            m = cache.evaluate(w)
            for key, value in m.items():
                samples[name][key].append(value)
        invalid += m["roc_auc"] is None
    intervals = {}
    for name, ms in samples.items():
        intervals[name] = {}
        for key, values in ms.items():
            valid = [v for v in values if v is not None]
            intervals[name][key] = (
                dict(
                    lower=float(np.quantile(valid, 0.025)),
                    upper=float(np.quantile(valid, 0.975)),
                    valid_draws=len(valid),
                )
                if valid
                else None
            )
    paired = {}
    for a, b in [("logistic", "xgboost"), ("null", "logistic"), ("null", "xgboost")]:
        if a not in samples or b not in samples:
            continue
        pointa = caches[a].evaluate(np.ones(len(f)))
        pointb = caches[b].evaluate(np.ones(len(f)))
        paired[b + "-minus-" + a] = {}
        for metric in ["roc_auc", "average_precision", "brier", "log_loss"]:
            values = [
                vb - va
                for va, vb in zip(samples[a][metric], samples[b][metric], strict=True)
                if va is not None and vb is not None
            ]
            paired[b + "-minus-" + a][metric] = dict(
                point=pointb[metric] - pointa[metric],
                lower=float(np.quantile(values, 0.025)),
                upper=float(np.quantile(values, 0.975)),
                valid_draws=len(values),
            )
    return dict(
        draws=draws,
        seed=seed,
        unit="loan_id",
        invalid_single_class_draws=int(invalid),
        loan_sample_sequence_sha256=fingerprint.hexdigest(),
        fixed_fit_only=True,
        intervals=intervals,
        paired=paired,
    )


def diagnostic(f, p):
    sample = counts(f)
    c = calibration_diagnostics(f.binary_default_12m.astype(int), p)
    c["confidence_intervals"] = None
    c["interval_reason"] = (
        "Point calibration diagnostics only; no iid/Wald interval; loan-cluster "
        "intervals supplied for core predictive metrics"
    )
    c["support_status"] = (
        "UNSTABLE / INSUFFICIENT EVENT SUPPORT"
        if sample["default_loans"] < 20
        else c["slope_status"]
    )
    edges = [-1e-12, 0.005, 0.02, 0.1, 1.0]
    bins = pd.cut(p, edges, include_lowest=True)
    reliability = []
    for level in bins.categories:
        mask = bins == level
        if mask.any():
            reliability.append(
                dict(
                    bin=str(level),
                    **counts(f.loc[mask]),
                    observed_rate=float(f.loc[mask, "binary_default_12m"].mean()),
                    mean_probability=float(np.asarray(p)[mask].mean()),
                    support="sparse"
                    if counts(f.loc[mask])["default_loans"] < 20
                    else "descriptive",
                )
            )
    return dict(
        effective_sample=sample,
        metrics=metrics(f.binary_default_12m, p),
        calibration=c,
        reliability=reliability,
    )


class Ledger:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.path = self.folder / "evaluation_ledger.json"

    def freeze(self, specification):
        if self.path.exists():
            raise ValueError("Silent reevaluation prohibited")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.data = dict(
            specification_sha256=hashlib.sha256(
                json.dumps(specification, sort_keys=True).encode()
            ).hexdigest(),
            state="FROZEN",
            specification=specification,
            post_evaluation_model_change=False,
            prior_access=(
                "Tasks3/4 aggregate outcomes and nested Task3 scores already inspected; "
                "Task5 model metrics sealed"
            ),
        )
        self._write()

    def open(self):
        expected = self.data["specification_sha256"]
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        actual = hashlib.sha256(
            json.dumps(self.data["specification"], sort_keys=True).encode()
        ).hexdigest()
        if actual != expected or self.data["specification_sha256"] != expected:
            raise ValueError("Ledger specification integrity failed")
        if self.data["state"] != "FROZEN":
            raise ValueError("Evaluation is not unopened/frozen")
        from datetime import UTC, datetime

        self.data.update(state="ACCESS_STARTED", accessed_at=datetime.now(UTC).isoformat())
        self._write()

    def complete(self, metrics_computed):
        if self.data["state"] != "ACCESS_STARTED":
            raise ValueError("Invalid evaluation ledger state")
        self.data.update(
            state="CONSUMED", metrics_computed=metrics_computed, prediction_generation_count=1
        )
        self._write()

    def _write(self):
        self.path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
