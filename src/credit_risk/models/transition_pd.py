"""Empirical Markov transition PD benchmark for longitudinal portfolio accounts."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from credit_risk.data.validation import STATES
from credit_risk.portfolio.roll_rates import roll_rates


class TransitionPDModel(BaseEstimator):
    def __init__(self, horizon_months: int = 12, minimum_state_pairs: int = 20):
        self.horizon_months = horizon_months
        self.minimum_state_pairs = minimum_state_pairs

    def fit(self, history, accounts=None, as_of=None):
        if not isinstance(self.horizon_months, int) or not 1 <= self.horizon_months <= 60:
            raise ValueError("horizon_months must be an integer in [1,60]")
        if not isinstance(self.minimum_state_pairs, int) or self.minimum_state_pairs < 1:
            raise ValueError("minimum_state_pairs must be positive")
        result = roll_rates(history, as_of=as_of, accounts=accounts)
        support = result.count_matrix.sum(axis=1)
        sparse = support.iloc[:-1] < self.minimum_state_pairs
        if sparse.any():
            raise ValueError(
                f"Insufficient observed transition support: {support.iloc[:-1][sparse].to_dict()}"
            )
        matrix = result.probability_matrix.to_numpy(dtype=float, copy=True)
        # Absorption follows the recorded-default contract, even if no default
        # persistence pair has yet been observed. No other unsupported row is filled.
        matrix[-1] = [0, 0, 0, 0, 0, 1]
        if not np.isfinite(matrix).all() or not np.allclose(matrix.sum(axis=1), 1):
            raise ValueError("Transition matrix is not stochastic")
        self.transition_matrix_ = matrix
        self.support_ = support.to_dict()
        self.diagnostics_ = result.diagnostics
        self.horizon_pd_ = np.clip(np.linalg.matrix_power(matrix, self.horizon_months)[:, -1], 0, 1)
        self.is_fitted_ = True
        return self

    def predict_pd(self, states):
        check_is_fitted(self, "is_fitted_")
        values = pd.Series(states)
        if values.empty or values.isna().any() or not values.isin(STATES).all():
            raise ValueError("Predicted states must be a nonempty vector of known states")
        return self.horizon_pd_[values.map({state: i for i, state in enumerate(STATES)}).to_numpy()]

    def metadata(self):
        check_is_fitted(self, "is_fitted_")
        return {
            "model": "empirical time-homogeneous count-weighted Markov benchmark",
            "horizon_months": self.horizon_months,
            "fit_as_of": self.diagnostics_["as_of"],
            "state_order": list(STATES),
            "state_pair_counts": self.support_,
            "transition_matrix": self.transition_matrix_.tolist(),
            "horizon_pd_by_state": dict(zip(STATES, self.horizon_pd_.tolist(), strict=True)),
            "fit_coverage": self.diagnostics_,
            "probability_status": "uncalibrated benchmark",
            "target": "recorded absorbing DEFAULT by the end of the forward horizon",
            "default_row": "structural absorption under validated source contract",
            "limitations": [
                "Not prospectively validated or calibrated",
                "Time homogeneity and state-only risk; no macroeconomic or account covariates",
                "Repeated account observations are dependent; no uncertainty intervals",
                "Observed-pair selection bias possible under missing follow-up",
            ],
        }
