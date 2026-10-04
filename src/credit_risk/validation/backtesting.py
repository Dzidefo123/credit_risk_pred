"""Segment diagnostics and mature dated-outcome backtesting without fabricated dates."""

import numpy as np
import pandas as pd

from credit_risk.validation.metrics import binary_metrics
from credit_risk.validation.settings import ValidationConfig


def segment_diagnostics(
    predictors: pd.DataFrame,
    labels,
    probabilities,
    config: ValidationConfig,
    threshold: float = 0.1,
) -> list[dict]:
    binary_metrics(labels, probabilities, threshold)
    if len(predictors) != len(labels):
        raise ValueError("Segment predictors and outcomes are not aligned")
    frame = predictors.reset_index(drop=True)
    y, p = np.asarray(labels), np.asarray(probabilities)
    definitions = {
        "age": np.select(
            [frame.age.isna(), frame.age < 35, frame.age < 55],
            ["missing", "under_35", "35_to_54"],
            default="55_plus",
        ),
        "income_missing": np.where(frame.MonthlyIncome.isna(), "missing", "observed"),
        "utilization": np.where(
            frame.RevolvingUtilizationOfUnsecuredLines > 1, "over_one", "not_over_one"
        ),
    }
    results = []
    for dimension, assignments in definitions.items():
        for segment in np.unique(assignments):
            mask = assignments == segment
            bad, count = int(y[mask].sum()), int(mask.sum())
            results.append(
                {
                    "dimension": dimension,
                    "segment": str(segment),
                    "rows": count,
                    "bad_count": bad,
                    "low_support": count < config.minimum_segment_rows
                    or min(bad, count - bad) < config.minimum_segment_events,
                    "metrics": binary_metrics(y[mask], p[mask], threshold),
                }
            )
    return results


def backtest_predictions(frame: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.DataFrame:
    required = {"observation_date", "performance_end", "label", "probability"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Dated predictions and performance-window ends are required")
    data = frame.copy(deep=True)
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff.tzinfo is not None:
        raise ValueError("as_of must be a valid timezone-naive date")
    for name in ("observation_date", "performance_end"):
        data[name] = pd.to_datetime(data[name], errors="raise")
        if data[name].isna().any() or data[name].dt.tz is not None:
            raise ValueError("Dates must be nonmissing and timezone-naive")
    if (data.performance_end < data.observation_date).any():
        raise ValueError("Performance end precedes observation")
    if not data.label.dropna().isin([0, 1]).all():
        raise ValueError("Observed labels must be binary")
    if not np.isfinite(data.probability).all() or not data.probability.between(0, 1).all():
        raise ValueError("Invalid predicted probabilities")
    eligible = data.loc[(data.performance_end <= cutoff) & data.label.notna()].copy()
    rows = []
    for month, cohort in eligible.groupby(eligible.observation_date.dt.to_period("M"), sort=True):
        rows.append(
            {
                "observation_month": str(month),
                "rows": len(cohort),
                **binary_metrics(cohort.label.astype(int), cohort.probability),
            }
        )
    return pd.DataFrame(rows)
