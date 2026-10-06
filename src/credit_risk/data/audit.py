"""Read-only, JSON-safe descriptive audits; caller rules are not lending policy."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NumericRule:
    """Inclusive bounds/whole-number checks with an explicit caller justification."""

    explanation: str
    minimum: float | None = None
    maximum: float | None = None
    integer: bool = False
    classification: str = "review"

    def __post_init__(self):
        if not self.explanation.strip() or self.classification not in {"review", "impossible"}:
            raise ValueError("Rules need an explanation and review/impossible classification")
        for bound in (self.minimum, self.maximum):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("Rule bounds must be finite")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("Rule minimum exceeds maximum")


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def audit_frame(
    frame: pd.DataFrame,
    *,
    target: str | None = None,
    required_columns=(),
    duplicate_columns=None,
    rules=None,
    tail_quantiles=(0.01, 0.99),
) -> dict:
    """Summarize supplied rows without mutation, imputation, scoring or fitting.

    Duplicate counts exclude the first occurrence and ignore the dataframe index.
    Use duplicate_columns to exclude a source row ID from profile duplication.
    Binary prevalence excludes missing labels, with its denominator reported.
    Quantile tails are descriptive, not impossible-value decisions. Infinities
    are counted separately and excluded from finite numerical summaries.
    Schema/domain errors fail explicitly; no invalid binary target is coerced.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Audit input must be a pandas DataFrame")
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Audit needs nonempty data with unique column names")
    if not all(isinstance(c, str) and c for c in frame.columns):
        raise ValueError("Audit column names must be nonempty strings")
    if isinstance(required_columns, str) or isinstance(duplicate_columns, str):
        raise TypeError("Column selections must be sequences, not strings")
    selected = list(frame.columns) if duplicate_columns is None else list(duplicate_columns)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Duplicate-profile columns must be nonempty and unique")
    rules = {} if rules is None else dict(rules)
    needed = set(required_columns) | set(selected) | set(rules)
    if target is not None:
        needed.add(target)
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ValueError(f"Missing audit columns: {missing}")
    if len(tail_quantiles) != 2 or not 0 <= tail_quantiles[0] < tail_quantiles[1] <= 1:
        raise ValueError("Tail quantiles need 0 <= lower < upper <= 1")

    columns = {}
    findings = {}
    for name in frame.columns:
        series = frame[name]
        observed = series.dropna()
        unique = int(observed.nunique())
        summary = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_fraction": float(series.isna().mean()),
            "unique_count": unique,
            "unique_values": (
                sorted(
                    (_scalar(v) for v in observed.unique()), key=lambda v: (str(type(v)), str(v))
                )
                if unique <= 20
                else None
            ),
        }
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        if pd.api.types.is_complex_dtype(series):
            raise ValueError(f"Complex numerical column is unsupported: {name}")
        if numeric:
            values = observed.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            # Scale descriptive calculations to avoid overflow on finite extreme values.
            scale = float(np.max(np.abs(finite))) if len(finite) else 0.0
            scaled = finite / scale if scale else finite
            lower, upper = (
                np.quantile(scaled, tail_quantiles) * scale if len(finite) else (None, None)
            )
            summary["numeric"] = {
                "finite_count": len(finite),
                "nonfinite_count": int((~np.isfinite(values)).sum()),
                "minimum": float(finite.min()) if len(finite) else None,
                "maximum": float(finite.max()) if len(finite) else None,
                "mean": float(scaled.mean() * scale) if len(finite) else None,
                "median": float(np.median(scaled) * scale) if len(finite) else None,
                "lower_tail_boundary": float(lower) if lower is not None else None,
                "upper_tail_boundary": float(upper) if upper is not None else None,
                "below_lower_tail": int((finite < lower).sum()) if len(finite) else 0,
                "above_upper_tail": int((finite > upper).sum()) if len(finite) else 0,
            }
        columns[name] = summary
        if name in rules:
            rule = rules[name]
            if not isinstance(rule, NumericRule) or not numeric:
                raise ValueError(f"NumericRule requires a numeric column: {name}")
            below = (
                finite < rule.minimum if rule.minimum is not None else np.zeros(len(finite), bool)
            )
            above = (
                finite > rule.maximum if rule.maximum is not None else np.zeros(len(finite), bool)
            )
            fractional = finite % 1 != 0 if rule.integer else np.zeros(len(finite), bool)
            findings[name] = {
                "classification": rule.classification,
                "explanation": rule.explanation,
                "minimum": rule.minimum,
                "maximum": rule.maximum,
                "integer": rule.integer,
                "below_minimum": int(below.sum()),
                "above_maximum": int(above.sum()),
                "fractional": int(fractional.sum()),
                "violating_finite_rows": int((below | above | fractional).sum()),
            }

    target_summary = None
    if target is not None:
        labels = frame[target].dropna()
        if not labels.isin([0, 1]).all():
            raise ValueError("Target must contain binary 0/1 labels or missing values")
        target_summary = {
            "name": target,
            "observed_count": len(labels),
            "missing_count": int(frame[target].isna().sum()),
            "positive_count": int((labels == 1).sum()),
            "negative_count": int((labels == 0).sum()),
            "positive_fraction": float((labels == 1).mean()) if len(labels) else None,
        }
    return {
        "rows": len(frame),
        "column_count": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_profile_columns": selected,
        "duplicate_profiles": int(frame.duplicated(selected).sum()),
        "tail_quantiles": list(tail_quantiles),
        "columns": columns,
        "target": target_summary,
        "rule_findings": findings,
    }
