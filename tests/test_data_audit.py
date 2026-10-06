"""Descriptive audit correctness, explicit rules and source non-mutation."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from credit_risk.data.audit import NumericRule, audit_frame


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4],
            "income": [10.0, 10.0, np.nan, 30.0],
            "age": [20, 20, 0, 21],
            "target": [1, 1, 0, np.nan],
            "category": ["a", "a", "b", None],
        }
    )


def test_deterministic_json_safe_and_non_mutating(frame):
    original = frame.copy(deep=True)
    first = audit_frame(frame, target="target", duplicate_columns=["income", "age", "target"])
    assert first == audit_frame(
        frame, target="target", duplicate_columns=["income", "age", "target"]
    )
    json.dumps(first, allow_nan=False)
    assert_frame_equal(frame, original)
    assert first["rows"] == 4 and first["column_count"] == 5
    assert first["duplicate_rows"] == 0
    assert first["duplicate_profiles"] == 1
    assert first["columns"]["category"]["unique_values"] == ["a", "b"]


def test_missingness_and_binary_prevalence_denominator(frame):
    result = audit_frame(frame, target="target")
    assert result["columns"]["income"]["missing_count"] == 1
    assert result["columns"]["income"]["missing_fraction"] == 0.25
    assert result["target"] == {
        "name": "target",
        "observed_count": 3,
        "missing_count": 1,
        "positive_count": 2,
        "negative_count": 1,
        "positive_fraction": 2 / 3,
    }


def test_full_row_duplicates_ignore_dataframe_index(frame):
    repeated = pd.concat([frame.iloc[:1], frame.iloc[:1]], ignore_index=True)
    repeated.index = ["first", "second"]
    assert audit_frame(repeated)["duplicate_rows"] == 1


def test_ranges_and_configurable_tails():
    result = audit_frame(
        pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 100.0]}), tail_quantiles=(0.2, 0.8)
    )
    numeric = result["columns"]["x"]["numeric"]
    assert numeric["minimum"] == 0 and numeric["maximum"] == 100
    assert numeric["median"] == 2
    assert numeric["mean"] == pytest.approx(21.2)
    assert numeric["below_lower_tail"] == numeric["above_upper_tail"] == 1
    assert result["rule_findings"] == {}


def test_explicit_rules_count_union_and_keep_source_unchanged():
    frame = pd.DataFrame({"x": [-1.5, 0.0, 1.0, 20.0, np.nan, np.inf]})
    original = frame.copy(deep=True)
    rule = NumericRule(
        "Caller-specified test bounds, not lending policy",
        minimum=0,
        maximum=10,
        integer=True,
        classification="review",
    )
    report = audit_frame(frame, rules={"x": rule})
    finding = report["rule_findings"]["x"]
    assert finding["below_minimum"] == 1 and finding["above_maximum"] == 1
    assert finding["fractional"] == 1 and finding["violating_finite_rows"] == 2
    assert report["columns"]["x"]["numeric"]["nonfinite_count"] == 1
    json.dumps(report, allow_nan=False)
    assert_frame_equal(frame, original)


def test_all_missing_nullable_columns_and_missing_target():
    frame = pd.DataFrame(
        {
            "x": pd.Series([None, None], dtype="Float64"),
            "target": pd.Series([None, None], dtype="Int64"),
        }
    )
    result = audit_frame(frame, target="target")
    numeric = result["columns"]["x"]["numeric"]
    assert numeric["minimum"] is None and numeric["upper_tail_boundary"] is None
    assert numeric["finite_count"] == 0
    assert result["columns"]["x"]["unique_values"] == []
    assert result["target"]["positive_fraction"] is None
    json.dumps(result, allow_nan=False)


def test_infinities_reported_not_silently_imputed():
    result = audit_frame(pd.DataFrame({"x": [np.inf, -np.inf, np.nan]}))
    assert result["columns"]["x"]["numeric"]["nonfinite_count"] == 2
    assert result["columns"]["x"]["missing_count"] == 1
    assert result["columns"]["x"]["numeric"]["minimum"] is None
    json.dumps(result, allow_nan=False)


def test_high_cardinality_values_are_not_exposed():
    result = audit_frame(pd.DataFrame({"identifier": range(25)}))
    assert result["columns"]["identifier"]["unique_count"] == 25
    assert result["columns"]["identifier"]["unique_values"] is None


def test_non_numeric_and_boolean_schema():
    frame = pd.DataFrame(
        {
            "text": ["x", None],
            "flag": [True, False],
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        }
    )
    result = audit_frame(frame, required_columns=["date"])
    assert all("numeric" not in c for c in result["columns"].values())
    assert result["columns"]["date"]["unique_count"] == 2
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(),
        pd.DataFrame({"x": []}),
        pd.DataFrame([[1, 2]], columns=["x", "x"]),
        pd.DataFrame({1: [1]}),
        pd.DataFrame({"": [1]}),
    ],
)
def test_empty_or_invalid_columns_rejected(frame):
    with pytest.raises(ValueError):
        audit_frame(frame)


def test_non_dataframe_rejected():
    with pytest.raises(TypeError, match="DataFrame"):
        audit_frame([[1]])


@pytest.mark.parametrize(
    "options",
    [
        {"required_columns": ["absent"]},
        {"target": "absent"},
        {"duplicate_columns": ["absent"]},
        {"rules": {"absent": NumericRule("test")}},
    ],
)
def test_missing_requested_schema_rejected(frame, options):
    with pytest.raises(ValueError, match="Missing audit columns"):
        audit_frame(frame, **options)


@pytest.mark.parametrize(
    "options",
    [
        {"duplicate_columns": []},
        {"duplicate_columns": ["age", "age"]},
        {"tail_quantiles": (0.9, 0.1)},
        {"tail_quantiles": (-1, 0.9)},
        {"tail_quantiles": (0.1, 2)},
        {"tail_quantiles": (0.1,)},
    ],
)
def test_invalid_audit_options_rejected(frame, options):
    with pytest.raises(ValueError):
        audit_frame(frame, **options)


@pytest.mark.parametrize("selection", ["required_columns", "duplicate_columns"])
def test_string_schema_selection_rejected(frame, selection):
    with pytest.raises(TypeError, match="sequences"):
        audit_frame(frame, **{selection: "age"})


@pytest.mark.parametrize("labels", [[0, 2], ["0", "1"], [0, np.inf]])
def test_invalid_binary_target_is_not_coerced(labels):
    with pytest.raises(ValueError, match="binary"):
        audit_frame(pd.DataFrame({"target": labels}), target="target")


@pytest.mark.parametrize(
    "options",
    [
        {"explanation": ""},
        {"explanation": "test", "minimum": 3, "maximum": 2},
        {"explanation": "test", "minimum": np.inf},
        {"explanation": "test", "classification": "bank policy"},
    ],
)
def test_invalid_rules_rejected(options):
    with pytest.raises(ValueError):
        NumericRule(**options)


def test_rule_requires_numeric_column(frame):
    with pytest.raises(ValueError, match="numeric column"):
        audit_frame(frame, rules={"category": NumericRule("test")})


def test_complex_values_rejected():
    with pytest.raises(ValueError, match="Complex"):
        audit_frame(pd.DataFrame({"complex": [1 + 2j]}))


def test_finite_extremes_do_not_overflow_json_statistics():
    result = audit_frame(pd.DataFrame({"x": [1e308, 1e308], "y": [-1e308, 1e308]}))
    json.dumps(result, allow_nan=False)
    assert result["columns"]["x"]["numeric"]["mean"] == 1e308
    assert result["columns"]["x"]["numeric"]["median"] == 1e308
    assert result["columns"]["y"]["numeric"]["median"] == 0
