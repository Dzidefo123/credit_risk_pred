"""Target-window boundaries, censoring, prior eligibility and information cutoff."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from credit_risk.data.targets import TargetConfig, build_forward_targets
from credit_risk.data.validation import DataContractError, validate_history


def history(dpd_values, flags=None):
    flags = flags or [False] * len(dpd_values)
    rows = []
    for mob, (dpd, default) in enumerate(zip(dpd_values, flags, strict=True)):
        state = (
            "DEFAULT"
            if default
            else "CURRENT"
            if dpd == 0
            else "DPD_1_29"
            if dpd < 30
            else "DPD_30_59"
            if dpd < 60
            else "DPD_60_89"
            if dpd < 90
            else "DPD_90_PLUS"
        )
        rows.append(
            {
                "account_id": "A",
                "origination_date": pd.Timestamp("2022-01-01"),
                "observation_date": pd.Timestamp("2022-01-01") + pd.offsets.MonthEnd(mob + 1),
                "months_on_book": mob,
                "dpd": dpd,
                "state": state,
                "default_flag": default,
                "opening_balance": 100.0,
                "draws": 0.0,
                "interest": 0.0,
                "payment": 0.0,
                "scheduled_payment": 3.0,
                "write_off": 0.0,
                "balance": 100.0,
                "credit_limit": 1000.0,
                "utilization": 0.1,
                "is_synthetic": False,
            }
        )
    return pd.DataFrame(rows)


def test_horizon_boundary_and_future_exclusion():
    frame = history([0, 0, 0, 90])
    original = frame.copy(deep=True)
    targets = build_forward_targets(frame, TargetConfig(horizon_months=2))
    assert targets.label.tolist()[:2] == [0, 1]
    assert targets.performance_end.iloc[0] == pd.Timestamp("2022-03-31")
    assert targets.status.iloc[-1] == "preexisting_default"
    assert_frame_equal(frame, original)
    assert "dpd" not in targets and "balance" not in targets


def test_bad_observed_before_maturity_and_right_censoring():
    targets = build_forward_targets(history([0, 90]), TargetConfig(horizon_months=12))
    assert targets.status.iloc[0] == "bad"
    assert targets.label.iloc[0] == 1
    targets = build_forward_targets(history([0, 0]), TargetConfig(horizon_months=12))
    assert targets.status.eq("censored").all()
    assert targets.label.isna().all()


def test_indeterminate_definition_is_configurable():
    frame = history([0, 45, 0])
    targets = build_forward_targets(frame, TargetConfig(horizon_months=2))
    assert targets.status.iloc[0] == "indeterminate"
    assert pd.isna(targets.label.iloc[0])
    targets = build_forward_targets(
        frame, TargetConfig(horizon_months=2, indeterminate_dpd_threshold=None)
    )
    assert targets.label.iloc[0] == 0


def test_missing_months_and_left_truncated_histories_are_unlabeled():
    frame = history([0, 0, 0, 0]).drop(index=1)
    targets = build_forward_targets(frame, TargetConfig(horizon_months=2))
    assert targets.status.tolist() == ["censored", "history_incomplete", "history_incomplete"]
    assert targets.label.isna().all()
    targets = build_forward_targets(history([0, 0, 90]).iloc[1:])
    assert targets.status.iloc[0] == "history_incomplete"
    assert pd.isna(targets.label.iloc[0])


def test_preexisting_default_excluded_even_after_dpd_cure():
    targets = build_forward_targets(history([90, 0, 0]))
    assert targets.status.eq("preexisting_default").all()
    assert targets.label.isna().all()


def test_recorded_default_overrides_dpd_threshold():
    targets = build_forward_targets(history([0, 0], [False, True]))
    assert targets.label.iloc[0] == 1
    with pytest.raises(DataContractError, match="absorbing"):
        validate_history(history([0, 0, 0], [False, True, False]))


def test_as_of_cutoff_does_not_use_later_performance():
    frame = history([0, 90])
    targets = build_forward_targets(frame, as_of="2022-01-31")
    assert len(targets) == 1
    assert targets.status.iloc[0] == "censored"
    assert pd.isna(targets.first_default_date.iloc[0])
    assert build_forward_targets(frame).label.iloc[0] == 1
    assert build_forward_targets(frame, as_of="2021-12-31").empty


def test_bad_observed_with_missing_future_month_still_known():
    targets = build_forward_targets(history([0, 0, 90]).drop(index=1))
    assert targets.label.iloc[0] == 1


def test_invalid_target_configuration():
    with pytest.raises(ValueError):
        TargetConfig(default_dpd_threshold=30, indeterminate_dpd_threshold=30)
