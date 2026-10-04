"""Reproducible, explicitly synthetic monthly histories with reconciled balances."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from credit_risk import __version__
from credit_risk.data.loaders import PortfolioData
from credit_risk.data.validation import STATES, validate_history
from credit_risk.utils.config import ConfigModel


class SyntheticPortfolioConfig(ConfigModel):
    n_accounts: int = Field(default=500, ge=1, le=100000)
    calendar_months: int = Field(default=36, ge=2, le=120)
    origination_months: int = Field(default=12, ge=1, le=120)
    start_month: str = "2022-01-01"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    annual_interest_rate: float = Field(default=0.18, ge=0, le=1)
    transition_matrix: list[list[float]] = Field(
        default_factory=lambda: [
            [0.94, 0.045, 0.01, 0.003, 0.001, 0.001],
            [0.40, 0.35, 0.18, 0.04, 0.02, 0.01],
            [0.15, 0.20, 0.35, 0.20, 0.07, 0.03],
            [0.07, 0.08, 0.15, 0.35, 0.25, 0.10],
            [0.02, 0.03, 0.05, 0.10, 0.40, 0.40],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    @model_validator(mode="after")
    def check_generator(self) -> "SyntheticPortfolioConfig":
        if self.origination_months > self.calendar_months:
            raise ValueError("Origination window must fit the observation calendar")
        start = pd.Timestamp(self.start_month)
        if start.tzinfo is not None or start.day != 1 or start != start.normalize():
            raise ValueError("start_month must be a timezone-naive first-of-month date")
        matrix = np.asarray(self.transition_matrix)
        if matrix.shape != (6, 6) or not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("Transition matrix must be finite nonnegative 6x6")
        if not np.allclose(matrix.sum(axis=1), 1, atol=1e-10, rtol=0):
            raise ValueError("Transition rows must sum to one")
        if not np.array_equal(matrix[-1], [0, 0, 0, 0, 0, 1]):
            raise ValueError("Recorded DEFAULT must be absorbing")
        return self


def generate_portfolio(config: SyntheticPortfolioConfig | None = None) -> PortfolioData:
    config = config or SyntheticPortfolioConfig()
    rng = np.random.default_rng(config.seed)
    start = pd.Timestamp(config.start_month)
    matrix = np.asarray(config.transition_matrix)
    accounts, histories = [], []
    dpd_ranges = [(0, 1), (1, 30), (30, 60), (60, 90), (90, 120), (120, 181)]
    payment_fractions = [1.6, 0.8, 0.4, 0.15, 0.02, 0.0]
    for number in range(config.n_accounts):
        account_id = f"SYN-{number + 1:07d}"
        cohort = int(rng.integers(config.origination_months))
        originated = start + pd.DateOffset(months=cohort)
        income = round(float(rng.lognormal(np.log(3500), 0.5)), 2)
        limit = round(float(np.clip(income * rng.uniform(0.5, 2.0), 500, 25000)), 2)
        latent_risk = float(rng.beta(2, 8))  # Internal simulation parameter; never a feature.
        balance = round(limit * float(rng.uniform(0.1, 0.8)), 2)
        accounts.append(
            {
                "account_id": account_id,
                "origination_date": originated,
                "age": int(rng.integers(21, 76)),
                "monthly_income": income,
                "origination_limit": limit,
                "is_synthetic": True,
            }
        )
        state = 0
        for mob in range(config.calendar_months - cohort):
            opening = balance
            prior_state = state
            if mob:
                probabilities = matrix[state].copy()
                probabilities[np.arange(6) > state] *= 1 + 3 * latent_risk
                probabilities /= probabilities.sum()
                state = int(rng.choice(6, p=probabilities))
                if state >= 2 and prior_state < 2:
                    limit = round(max(500, 0.9 * limit), 2)
            dpd = int(rng.integers(*dpd_ranges[state]))
            interest = (
                round(opening * config.annual_interest_rate / 12, 2) if mob and state < 5 else 0.0
            )
            draws = round(limit * float(rng.uniform(0, 0.05)), 2) if mob and state < 2 else 0.0
            due = (
                round(min(opening + interest, max(25, opening * 0.03 + interest)), 2)
                if mob
                else 0.0
            )
            payment = round(min(opening + interest + draws, due * payment_fractions[state]), 2)
            write_off = round(opening * 0.2, 2) if state == 5 and prior_state != 5 else 0.0
            balance = round(opening + draws + interest - payment - write_off, 2)
            histories.append(
                {
                    "account_id": account_id,
                    "origination_date": originated,
                    "observation_date": originated + pd.offsets.MonthEnd(mob + 1),
                    "months_on_book": mob,
                    "dpd": dpd,
                    "state": STATES[state],
                    "default_flag": state == 5,
                    "opening_balance": opening,
                    "draws": draws,
                    "interest": interest,
                    "payment": payment,
                    "scheduled_payment": due,
                    "write_off": write_off,
                    "balance": balance,
                    "credit_limit": limit,
                    "utilization": round(balance / limit, 8),
                    "is_synthetic": True,
                }
            )
    portfolio = PortfolioData(pd.DataFrame(accounts), pd.DataFrame(histories))
    validate_history(portfolio.history, portfolio.accounts)
    return portfolio


def write_portfolio(
    portfolio: PortfolioData, output_dir: str | Path, config: SyntheticPortfolioConfig
) -> dict:
    """Write CSVs and a provenance manifest; refuse to overwrite previous runs."""
    validate_history(portfolio.history, portfolio.accounts)
    if not portfolio.accounts.is_synthetic.all() or not portfolio.history.is_synthetic.all():
        raise ValueError("Synthetic export requires explicit synthetic provenance")
    directory = Path(output_dir)
    paths = {name: directory / name for name in ("accounts.csv", "history.csv", "manifest.json")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Output already exists; choose a new run directory")
    directory.mkdir(parents=True, exist_ok=True)
    portfolio.accounts.to_csv(paths["accounts.csv"], index=False, mode="x", date_format="%Y-%m-%d")
    portfolio.history.to_csv(paths["history.csv"], index=False, mode="x", date_format="%Y-%m-%d")
    metadata = {
        "is_synthetic": True,
        "generator_version": __version__,
        "config": config.model_dump(),
        "accounts": len(portfolio.accounts),
        "snapshots": len(portfolio.history),
        "state_counts": {k: int(v) for k, v in portfolio.history.state.value_counts().items()},
        "files_sha256": {
            name: sha256(paths[name].read_bytes()).hexdigest()
            for name in ("accounts.csv", "history.csv")
        },
    }
    paths["manifest.json"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
