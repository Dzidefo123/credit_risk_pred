"""Canonical origination and monthly-account contracts, without silent cleaning."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

ORIGINATION_FEATURES = (
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30_59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60_89DaysPastDueNotWorse",
    "NumberOfDependents",
)
ORIGINATION_TARGET = "SeriousDlqin2yrs"
STATES = ("CURRENT", "DPD_1_29", "DPD_30_59", "DPD_60_89", "DPD_90_PLUS", "DEFAULT")
HISTORY_COLUMNS = (
    "account_id",
    "origination_date",
    "observation_date",
    "months_on_book",
    "dpd",
    "state",
    "default_flag",
    "opening_balance",
    "draws",
    "interest",
    "payment",
    "scheduled_payment",
    "write_off",
    "balance",
    "credit_limit",
    "utilization",
    "is_synthetic",
)
ACCOUNT_COLUMNS = (
    "account_id",
    "origination_date",
    "age",
    "monthly_income",
    "origination_limit",
    "is_synthetic",
)


class DataContractError(ValueError):
    """The data cannot safely be interpreted under the declared contract."""


@dataclass(frozen=True)
class QualityReport:
    rows: int
    missing: dict[str, int]
    duplicate_records: int
    suspicious: dict[str, int]


def _require(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing or frame.empty or frame.columns.duplicated().any():
        raise DataContractError(f"Empty data, duplicate columns, or missing fields: {missing}")


def _numbers(frame: pd.DataFrame, columns: list[str], allow_missing: bool = False) -> None:
    for name in columns:
        series = frame[name]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise DataContractError(f"{name} must be numeric")
        values = series.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any() or (not allow_missing and np.isnan(values).any()):
            raise DataContractError(f"{name} has missing or non-finite values")
        if (series.dropna() < 0).any():
            raise DataContractError(f"{name} must be nonnegative")


def _integers(frame: pd.DataFrame, columns: list[str]) -> None:
    for name in columns:
        if (frame[name].dropna() % 1 != 0).any():
            raise DataContractError(f"{name} must contain whole numbers")


def _dates(frame: pd.DataFrame, name: str, month_end: bool = False) -> None:
    values = frame[name]
    if not pd.api.types.is_datetime64_dtype(values) or values.isna().any():
        raise DataContractError(f"{name} must contain timezone-naive dates")
    if (values != values.dt.normalize()).any():
        raise DataContractError(f"{name} must have no time component")
    if month_end and not values.dt.is_month_end.all():
        raise DataContractError(f"{name} must be month-end snapshots")


def _identifiers(frame: pd.DataFrame) -> None:
    if (
        frame.account_id.isna().any()
        or not frame.account_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all()
    ):
        raise DataContractError("account_id must be a nonempty string")
    if not pd.api.types.is_bool_dtype(frame.is_synthetic) or frame.is_synthetic.isna().any():
        raise DataContractError("is_synthetic must be explicitly boolean")


def validate_origination(frame: pd.DataFrame, require_target: bool = True) -> QualityReport:
    columns = ORIGINATION_FEATURES + ((ORIGINATION_TARGET,) if require_target else ())
    _require(frame, columns)
    allowed = set(ORIGINATION_FEATURES) | {ORIGINATION_TARGET, "source_row_id"}
    extra = set(frame.columns) - allowed
    if extra:
        raise DataContractError(
            f"Unexpected origination fields (potential leakage): {sorted(extra)}"
        )
    _numbers(frame, list(ORIGINATION_FEATURES), allow_missing=True)
    counts = [c for c in ORIGINATION_FEATURES if c.startswith("Number") or c == "age"]
    _integers(frame, counts)
    if ORIGINATION_TARGET in frame:
        _numbers(frame, [ORIGINATION_TARGET])
        if (
            frame[ORIGINATION_TARGET].isna().any()
            or not frame[ORIGINATION_TARGET].isin([0, 1]).all()
        ):
            raise DataContractError("SeriousDlqin2yrs must be an observed binary label")
    if "source_row_id" in frame and (
        frame.source_row_id.isna().any() or frame.source_row_id.duplicated().any()
    ):
        raise DataContractError("source_row_id must be nonmissing and unique")
    delinquency = [c for c in counts if "PastDue" in c or c == "NumberOfTimes90DaysLate"]
    return QualityReport(
        rows=len(frame),
        missing={c: int(frame[c].isna().sum()) for c in columns},
        duplicate_records=int(frame[list(columns)].duplicated().sum()),
        suspicious={
            "age_zero_or_over_110": int(((frame.age == 0) | (frame.age > 110)).sum()),
            "utilization_over_one": int((frame.RevolvingUtilizationOfUnsecuredLines > 1).sum()),
            "delinquency_count_90_or_more": int((frame[delinquency] >= 90).any(axis=1).sum()),
        },
    )


def validate_accounts(accounts: pd.DataFrame) -> None:
    _require(accounts, ACCOUNT_COLUMNS)
    _identifiers(accounts)
    _dates(accounts, "origination_date")
    _numbers(accounts, ["age", "monthly_income", "origination_limit"])
    _integers(accounts, ["age"])
    if accounts.account_id.duplicated().any() or (accounts.origination_limit <= 0).any():
        raise DataContractError("Accounts need unique IDs and positive origination limits")


def validate_history(history: pd.DataFrame, accounts: pd.DataFrame | None = None) -> None:
    _require(history, HISTORY_COLUMNS)
    _identifiers(history)
    _dates(history, "origination_date")
    _dates(history, "observation_date", month_end=True)
    amounts = [
        "opening_balance",
        "draws",
        "interest",
        "payment",
        "scheduled_payment",
        "write_off",
        "balance",
        "credit_limit",
        "utilization",
        "months_on_book",
        "dpd",
    ]
    _numbers(history, amounts)
    _integers(history, ["months_on_book", "dpd"])
    if history.duplicated(["account_id", "observation_date"]).any():
        raise DataContractError("Duplicate account-month snapshot")
    if not pd.api.types.is_bool_dtype(history.default_flag) or history.default_flag.isna().any():
        raise DataContractError("default_flag must be boolean")
    if not history.state.isin(STATES).all():
        raise DataContractError("Unknown delinquency state")
    months = (
        (history.observation_date.dt.year - history.origination_date.dt.year) * 12
        + history.observation_date.dt.month
        - history.origination_date.dt.month
    )
    if (history.observation_date < history.origination_date).any() or not months.eq(
        history.months_on_book
    ).all():
        raise DataContractError("Dates and months_on_book disagree")
    if (history.credit_limit <= 0).any():
        raise DataContractError("credit_limit must be positive")
    if not np.allclose(
        history.balance / history.credit_limit, history.utilization, atol=1e-6, rtol=1e-6
    ):
        raise DataContractError("Utilization does not reconcile to balance / limit")
    reconciled = history.opening_balance + history.draws + history.interest
    reconciled = reconciled - history.payment - history.write_off
    if not np.allclose(reconciled, history.balance, atol=0.011, rtol=0):
        raise DataContractError("Balance movements do not reconcile")
    # Recorded default is distinct from a configurable analytic DPD threshold.
    expected = pd.Series(
        np.select(
            [
                history.default_flag,
                history.dpd == 0,
                history.dpd < 30,
                history.dpd < 60,
                history.dpd < 90,
            ],
            ["DEFAULT", "CURRENT", "DPD_1_29", "DPD_30_59", "DPD_60_89"],
            default="DPD_90_PLUS",
        ),
        index=history.index,
    )
    if not history.state.eq(expected).all():
        raise DataContractError("DPD, state and recorded default disagree")
    ordered = history.sort_values(["account_id", "observation_date"]).reset_index(drop=True)
    grouped = ordered.groupby("account_id", sort=False)
    if (grouped.origination_date.nunique() != 1).any() or (
        grouped.is_synthetic.nunique() != 1
    ).any():
        raise DataContractError("Account provenance or origination date changes over time")
    prior_default = grouped.default_flag.cummax()
    if (prior_default & ~ordered.default_flag).any():
        raise DataContractError("Recorded default must be absorbing")
    prior_balance = grouped.balance.shift()
    consecutive = grouped.months_on_book.diff().eq(1)
    if not np.allclose(
        ordered.loc[consecutive, "opening_balance"], prior_balance[consecutive], atol=0.011, rtol=0
    ):
        raise DataContractError("Opening balance does not match prior closing balance")
    if accounts is not None:
        validate_accounts(accounts)
        joined = history.merge(
            accounts[["account_id", "origination_date", "is_synthetic"]],
            on="account_id",
            how="left",
            validate="many_to_one",
            suffixes=("", "_account"),
        )
        if (
            joined.origination_date_account.isna().any()
            or not joined.origination_date.eq(joined.origination_date_account).all()
            or not joined.is_synthetic.eq(joined.is_synthetic_account).all()
        ):
            raise DataContractError("History does not match its account contract")
