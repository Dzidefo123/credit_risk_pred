"""Explicit source adapters; they never impute, discard rows or load models."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from credit_risk.data.validation import (
    DataContractError,
    QualityReport,
    validate_accounts,
    validate_history,
    validate_origination,
)


@dataclass(frozen=True)
class OriginationData:
    frame: pd.DataFrame
    quality: QualityReport
    target_semantics: str = "Inherited SeriousDlqin2yrs; source two-year delinquency label"


@dataclass(frozen=True)
class PortfolioData:
    accounts: pd.DataFrame
    history: pd.DataFrame


def load_origination_csv(path: str | Path, require_target: bool = True) -> OriginationData:
    frame = pd.read_csv(path)
    aliases = {
        "NumberOfTime30-59DaysPastDueNotWorse": "NumberOfTime30_59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse": "NumberOfTime60_89DaysPastDueNotWorse",
        "Unnamed: 0": "source_row_id",
        "": "source_row_id",
    }
    frame = frame.rename(columns=aliases)
    if frame.columns.duplicated().any():
        raise DataContractError("Ambiguous aliases in source columns")
    return OriginationData(frame, validate_origination(frame, require_target))


def load_portfolio_csv(accounts_path: str | Path, history_path: str | Path) -> PortfolioData:
    accounts = pd.read_csv(accounts_path, dtype={"account_id": str})
    history = pd.read_csv(history_path, dtype={"account_id": str})
    try:
        accounts["origination_date"] = pd.to_datetime(accounts["origination_date"], errors="raise")
        for field in ("origination_date", "observation_date"):
            history[field] = pd.to_datetime(history[field], errors="raise")
    except (KeyError, ValueError) as exc:
        raise DataContractError(f"Invalid or missing portfolio date fields: {exc}") from exc
    validate_accounts(accounts)
    validate_history(history, accounts)
    return PortfolioData(accounts, history)
