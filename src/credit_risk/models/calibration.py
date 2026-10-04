"""Monotone held-out probability calibration, separate from base model training."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from credit_risk.models.pd import predict_probability


class ProbabilityCalibrator(BaseEstimator):
    def __init__(self, method: str = "raw", epsilon: float = 1e-6):
        self.method = method
        self.epsilon = epsilon

    def _probabilities(self, probabilities) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Probabilities must be a nonempty finite vector in [0,1]")
        if not 0 < self.epsilon < 0.5:
            raise ValueError("epsilon must be in (0,0.5)")
        return np.clip(p, self.epsilon, 1 - self.epsilon)

    def fit(self, probabilities, labels):
        p = self._probabilities(probabilities)
        y = np.asarray(labels)
        if y.shape != p.shape or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("Calibration requires aligned binary labels with both classes")
        if self.method == "sigmoid":
            x = logit(p)

            def objective(parameters):
                z = parameters[0] * x + parameters[1]
                residual = expit(z) - y
                return float(np.mean(np.logaddexp(0, z) - y * z)), np.array(
                    [np.mean(residual * x), np.mean(residual)]
                )

            result = minimize(
                objective,
                [1.0, 0.0],
                jac=True,
                method="L-BFGS-B",
                bounds=[(1e-8, None), (None, None)],
                options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if not result.success or not np.isfinite(result.x).all():
                raise ValueError(f"Sigmoid calibration did not converge: {result.message}")
            self.slope_, self.intercept_ = map(float, result.x)
        elif self.method == "isotonic":
            self.estimator_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        elif self.method != "raw":
            raise ValueError(f"Unsupported calibration method: {self.method}")
        self.sample_count_ = len(y)
        self.is_fitted_ = True
        return self

    def transform(self, probabilities) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        p = self._probabilities(probabilities)
        if self.method == "sigmoid":
            p = expit(self.slope_ * logit(p) + self.intercept_)
        elif self.method == "isotonic":
            p = self.estimator_.predict(p)
        return np.clip(p, self.epsilon, 1 - self.epsilon)

    def parameters(self) -> dict:
        check_is_fitted(self, "is_fitted_")
        result = {"method": self.method, "epsilon": self.epsilon, "fit_rows": self.sample_count_}
        if self.method == "sigmoid":
            result.update(slope=self.slope_, intercept=self.intercept_)
        elif self.method == "isotonic":
            result["breakpoints"] = len(self.estimator_.X_thresholds_)
        return result


@dataclass
class CalibratedPDModel:
    pipeline: Pipeline
    calibrator: ProbabilityCalibrator

    def predict_proba(self, predictors):
        p = self.calibrator.transform(predict_probability(self.pipeline, predictors))
        return np.column_stack([1 - p, p])
