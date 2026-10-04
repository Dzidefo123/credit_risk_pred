"""Label-free, training-fitted capping and monotonic feature transforms."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from credit_risk.data.validation import (
    ORIGINATION_FEATURES,
    DataContractError,
    validate_origination,
)


class OriginationFeatures(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        lower_quantile: float = 0.001,
        upper_quantile: float = 0.999,
        log_transform: bool = True,
    ):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.log_transform = log_transform

    @staticmethod
    def _frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame) or set(X.columns) != set(ORIGINATION_FEATURES):
            raise DataContractError("Scoring requires exactly the ten declared predictors")
        validate_origination(X, require_target=False)
        return X.loc[:, ORIGINATION_FEATURES].astype(float)

    def fit(self, X: pd.DataFrame, y=None):
        frame = self._frame(X)
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("Invalid clipping quantiles")
        self.feature_names_in_ = np.asarray(ORIGINATION_FEATURES, dtype=object)
        self.n_features_in_ = len(ORIGINATION_FEATURES)
        bounds = []
        for name in ORIGINATION_FEATURES:
            values = frame[name].dropna()
            bounds.append(
                values.quantile([self.lower_quantile, self.upper_quantile]).to_numpy()
                if not values.empty
                else np.array([0.0, 0.0])
            )
        self.lower_bounds_ = pd.Series([b[0] for b in bounds], index=ORIGINATION_FEATURES)
        self.upper_bounds_ = pd.Series([b[1] for b in bounds], index=ORIGINATION_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "upper_bounds_")
        frame = self._frame(X).clip(self.lower_bounds_, self.upper_bounds_, axis=1)
        if self.log_transform:
            columns = [name for name in ORIGINATION_FEATURES if name != "age"]
            frame.loc[:, columns] = np.log1p(frame.loc[:, columns])
        return frame

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_.copy()
