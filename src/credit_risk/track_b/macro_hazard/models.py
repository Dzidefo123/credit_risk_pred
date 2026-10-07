"""Training-only encoders and weak-L2 multinomial hazards with frozen feature order."""

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .data import MACRO
from .protocol import CATEGORICAL, NUMERIC

BANDS = ["0-12", "13-24", "25-36", "37-60", "61-84", "85-120", "121-180", "181-240", "241+"]


@dataclass
class Encoder:
    mortgage: bool
    macro: tuple = ()
    reference: int = 2006
    parameters: dict = field(default_factory=dict)
    names: list = field(default_factory=list)

    def fit(self, data):
        if self.parameters or np.any(data["role"] != 0):
            raise ValueError("Preprocessing may fit development only, exactly once")
        if {"mortgage_30y_level", "mortgage_treasury_spread"} <= set(self.macro):
            raise ValueError("Algebraically dependent rate coefficient vector")
        numeric = self.raw_numeric(data)
        if numeric.shape[1] and np.isnan(numeric).all(axis=0).any():
            raise ValueError("All-missing development predictor")
        medians = np.nanmedian(numeric, axis=0) if numeric.shape[1] else np.array([])
        imputed = np.where(np.isnan(numeric), medians, numeric)
        means = imputed.mean(axis=0) if numeric.shape[1] else np.array([])
        scales = imputed.std(axis=0) if numeric.shape[1] else np.array([])
        scales[scales == 0] = 1
        categories = (
            {
                n: sorted(np.unique(data["categories"][:, i]).tolist())
                for i, n in enumerate(CATEGORICAL)
            }
            if self.mortgage
            else {}
        )
        self.parameters = dict(
            medians=medians.tolist(),
            means=means.tolist(),
            scales=scales.tolist(),
            categories=categories,
            training_rows=len(data),
            reference=self.reference,
        )
        nums = [*NUMERIC] if self.mortgage else []
        self.names = [*nums, *self.macro]
        self.names += [n + ":missing" for n in nums]
        self.names += ["duration:" + b for b in BANDS[1:]]
        self.names += [
            f"vintage:{y}"
            for y in [2006, 2008, 2010, 2014, 2018, 2020, 2022]
            if y != self.reference
        ]
        for name, values in categories.items():
            self.names += [name + ":" + str(v) for v in values[1:]]
        return self

    def raw_numeric(self, data):
        pieces = [data["numeric"]] if self.mortgage else []
        if self.macro:
            selected = data["macro"][:, [MACRO.index(n) for n in self.macro]]
            if not np.isfinite(selected).all():
                raise ValueError("Required PIT macro term unavailable")
            pieces.append(selected)
        return np.column_stack(pieces) if pieces else np.empty((len(data), 0))

    def transform(self, data):
        if not self.parameters:
            raise ValueError("Encoder is not development-fitted")
        raw = self.raw_numeric(data)
        imputed = np.where(np.isnan(raw), self.parameters["medians"], raw)
        numeric = (imputed - self.parameters["means"]) / self.parameters["scales"]
        cols = [sparse.csr_matrix(numeric)]
        if self.mortgage:
            cols.append(sparse.csr_matrix(np.isnan(data["numeric"]).astype(float)))
        # Exact Task9A bands; vectorize without outcome-dependent category selection.
        band = np.searchsorted([12, 24, 36, 60, 84, 120, 180, 240], data["duration"], side="left")
        if np.any(data["duration"] < 0):
            raise ValueError("Negative proxy duration")
        cols.extend(sparse.csr_matrix((band == i).astype(float)[:, None]) for i in range(1, 9))
        known = [2006, 2008, 2010, 2014, 2018, 2020, 2022]
        trained = self.parameters.get("trained_vintages", known)
        cols.extend(
            sparse.csr_matrix(((data["vintage"] == y) & (y in trained)).astype(float)[:, None])
            for y in known
            if y != self.reference
        )
        for i, name in enumerate(CATEGORICAL if self.mortgage else []):
            cols.extend(
                sparse.csr_matrix((data["categories"][:, i] == v).astype(float)[:, None])
                for v in self.parameters["categories"][name][1:]
            )
        result = sparse.hstack(cols, format="csr")
        if result.shape[1] != len(self.names) or not np.isfinite(result.data).all():
            raise ValueError("Design matrix identity/stability failed")
        return result


def fit(data, spec, mortgage, macro=(), reference=2006):
    if set(np.unique(data["event"])) != {0, 1, 2} or np.any(data["role"] != 0):
        raise ValueError("Invalid development events/roles")
    encoder = Encoder(mortgage, tuple(macro), reference).fit(data)
    encoder.parameters["trained_vintages"] = np.unique(data["vintage"]).astype(int).tolist()
    x = encoder.transform(data)
    energy = np.asarray(x.power(2).sum(axis=0)).ravel()
    encoder.parameters["zero_design_columns"] = [
        n for n, e in zip(encoder.names, energy, strict=True) if e < 1e-20
    ]
    if macro:
        _, first = np.unique(data["month"], return_index=True)
        values = data["macro"][first][:, [MACRO.index(n) for n in macro]]
        rank = int(np.linalg.matrix_rank(np.column_stack([np.ones(len(values)), values])))
        if rank != len(macro) + 1:
            raise ValueError("Development macro coefficient block not identifiable")
        encoder.parameters["macro_intercept_rank"] = rank
    model = LogisticRegression(
        C=spec["estimator"]["C"], solver="lbfgs", max_iter=3000, tol=1e-8, random_state=61010
    )
    with threadpool_limits(limits=1):
        model.fit(x, data["event"])
    if model.n_iter_.max() >= 3000 or not np.isfinite(model.coef_).all():
        raise ValueError("STOP — development convergence/stability invalid")
    return encoder, model


def predict(bundle, data):
    encoder, model = bundle
    with threadpool_limits(limits=1):
        p = model.predict_proba(encoder.transform(data))
    if (
        model.classes_.tolist() != [0, 1, 2]
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0))
    ):
        raise ValueError("Multinomial probability conservation failed")
    return p


def artifact(bundle):
    encoder, model = bundle
    result = dict(
        feature_order=encoder.names,
        preprocessing=encoder.parameters,
        intercept=model.intercept_.tolist(),
        coefficients=model.coef_.tolist(),
        classes=model.classes_.tolist(),
        iterations=model.n_iter_.tolist(),
        macro=list(encoder.macro),
        reference=encoder.reference,
    )
    result["cause_vs_none_coefficients"] = [
        dict(
            cause=cause,
            feature=name,
            log_odds=float(value),
            odds_ratio=float(np.exp(value)),
            interpretation="Conditional cause-vs-no-event odds, not causal or hazard ratio",
        )
        for code, cause in [(1, "default"), (2, "payoff")]
        for name, value in zip(encoder.names, model.coef_[code] - model.coef_[0], strict=True)
    ]
    return result
