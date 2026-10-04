"""Contracts must reject unsafe data without silently altering populations."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from credit_risk.cli import main
from credit_risk.data.loaders import load_origination_csv, load_portfolio_csv
from credit_risk.data.synthetic_portfolio import (
    SyntheticPortfolioConfig,
    generate_portfolio,
    write_portfolio,
)
from credit_risk.data.validation import (
    ORIGINATION_FEATURES,
    DataContractError,
    validate_history,
    validate_origination,
)


@pytest.fixture
def portfolio():
    return generate_portfolio(
        SyntheticPortfolioConfig(n_accounts=12, calendar_months=18, origination_months=6, seed=7)
    )


def test_origination_normalizes_source_names_and_preserves_missing_rows(tmp_path):
    frame = pd.DataFrame([{name: 1.0 for name in ORIGINATION_FEATURES}] * 2)
    frame["SeriousDlqin2yrs"] = [0, 0]
    frame["MonthlyIncome"] = np.nan
    frame.insert(0, "Unnamed: 0", [1, 2])
    frame = frame.rename(
        columns={"NumberOfTime30_59DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse"}
    )
    path = tmp_path / "source.csv"
    frame.to_csv(path, index=False)
    loaded = load_origination_csv(path)
    assert len(loaded.frame) == 2
    assert loaded.quality.missing["MonthlyIncome"] == 2
    assert loaded.quality.duplicate_records == 1
    assert loaded.frame.source_row_id.tolist() == [1, 2]
    assert loaded.frame.MonthlyIncome.isna().all()


def test_origination_rejects_leakage_and_invalid_labels():
    frame = pd.DataFrame([{name: 1.0 for name in ORIGINATION_FEATURES}])
    frame["SeriousDlqin2yrs"] = 0
    original = frame.copy(deep=True)
    validate_origination(frame)
    assert_frame_equal(frame, original)
    with pytest.raises(DataContractError, match="Unexpected"):
        validate_origination(frame.assign(future_dpd=90))
    with pytest.raises(DataContractError, match="binary"):
        validate_origination(frame.assign(SeriousDlqin2yrs=2))
    with pytest.raises(DataContractError, match="whole"):
        validate_origination(frame.assign(age=20.5))
    with pytest.raises(DataContractError, match="non-finite"):
        validate_origination(frame.assign(DebtRatio=np.inf))


def test_synthetic_determinism_and_provenance(portfolio):
    again = generate_portfolio(
        SyntheticPortfolioConfig(n_accounts=12, calendar_months=18, origination_months=6, seed=7)
    )
    assert_frame_equal(portfolio.accounts, again.accounts)
    assert_frame_equal(portfolio.history, again.history)
    other = generate_portfolio(
        SyntheticPortfolioConfig(n_accounts=12, calendar_months=18, origination_months=6, seed=8)
    )
    assert not portfolio.history.equals(other.history)
    assert portfolio.history.is_synthetic.all()
    assert portfolio.accounts.is_synthetic.all()
    assert "latent_risk" not in portfolio.accounts
    for _, group in portfolio.history.groupby("account_id"):
        assert group.months_on_book.tolist() == list(range(len(group)))
    pd.testing.assert_series_equal(
        portfolio.history.groupby("account_id").observation_date.max(),
        pd.Series(
            pd.Timestamp("2023-06-30"), index=portfolio.accounts.account_id, name="observation_date"
        ).sort_index(),
        check_names=False,
    )


def test_forced_deterioration_and_absorbing_default():
    matrix = [[0.0] * 6 for _ in range(6)]
    for index in range(6):
        matrix[index][min(index + 1, 5)] = 1.0
    data = generate_portfolio(
        SyntheticPortfolioConfig(
            n_accounts=1, calendar_months=8, origination_months=1, transition_matrix=matrix
        )
    )
    assert data.history.state.tolist() == [
        "CURRENT",
        "DPD_1_29",
        "DPD_30_59",
        "DPD_60_89",
        "DPD_90_PLUS",
        "DEFAULT",
        "DEFAULT",
        "DEFAULT",
    ]
    assert data.history.default_flag.tolist() == [False] * 5 + [True] * 3
    assert data.history.write_off.iloc[5] > 0
    assert data.history.write_off.iloc[6] == 0
    validate_history(data.history, data.accounts)


def test_forced_cure():
    matrix = np.eye(6).tolist()
    matrix[0] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    matrix[1] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    data = generate_portfolio(
        SyntheticPortfolioConfig(
            n_accounts=1, calendar_months=4, origination_months=1, transition_matrix=matrix
        )
    )
    assert data.history.dpd.tolist()[2] == 0
    assert data.history.state.tolist() == ["CURRENT", "DPD_1_29", "CURRENT", "DPD_1_29"]


@pytest.mark.parametrize(
    "settings",
    [
        {"origination_months": 40},
        {"start_month": "2022-01-15"},
        {"transition_matrix": [[1.0]]},
        {"transition_matrix": [[0.1] * 6] * 6},
        {"transition_matrix": np.eye(6).tolist()[:-1] + [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]},
    ],
)
def test_invalid_generator_parameters(settings):
    with pytest.raises((ValidationError, ValueError)):
        SyntheticPortfolioConfig(**settings)


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate",
        "balance",
        "utilization",
        "mob",
        "state",
        "missing",
        "unknown_account",
        "provenance",
    ],
)
def test_history_contract_rejects_corruption(portfolio, corruption):
    history = portfolio.history.copy(deep=True)
    if corruption == "duplicate":
        history = pd.concat([history, history.iloc[[0]]], ignore_index=True)
    elif corruption == "balance":
        history.loc[0, "payment"] += 100
    elif corruption == "utilization":
        history.loc[0, "utilization"] = 9.0
    elif corruption == "mob":
        history.loc[0, "months_on_book"] += 1
    elif corruption == "state":
        history.loc[0, "state"] = "DEFAULT"
    elif corruption == "missing":
        history.loc[0, "credit_limit"] = np.nan
    elif corruption == "unknown_account":
        history.loc[0, "account_id"] = "not-in-accounts"
    else:
        history.loc[0, "is_synthetic"] = False
    with pytest.raises(DataContractError):
        validate_history(history, portfolio.accounts)


def test_csv_round_trip_manifest_and_overwrite_protection(portfolio, tmp_path):
    config = SyntheticPortfolioConfig(
        n_accounts=12, calendar_months=18, origination_months=6, seed=7
    )
    manifest = write_portfolio(portfolio, tmp_path, config)
    assert manifest["is_synthetic"] is True
    assert manifest["snapshots"] == len(portfolio.history)
    loaded = load_portfolio_csv(tmp_path / "accounts.csv", tmp_path / "history.csv")
    assert_frame_equal(loaded.history, portfolio.history, check_dtype=False, check_exact=False)
    before = (tmp_path / "history.csv").read_bytes()
    with pytest.raises(FileExistsError):
        write_portfolio(portfolio, tmp_path, config)
    assert (tmp_path / "history.csv").read_bytes() == before


def test_generate_cli_and_invalid_configuration(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("n_accounts: 3\ncalendar_months: 4\norigination_months: 1\n")
    args = ["generate-portfolio", "--config", str(config), "--output-dir", str(tmp_path / "run")]
    assert main(args) == 0
    capsys.readouterr()
    assert main(args) == 2
    assert "already exists" in capsys.readouterr().err


def test_boolean_target_is_not_a_numeric_outcome():
    frame = pd.DataFrame([{name: 1.0 for name in ORIGINATION_FEATURES}])
    frame["SeriousDlqin2yrs"] = True
    with pytest.raises(DataContractError, match="numeric"):
        validate_origination(frame)


def test_source_alias_collision_is_rejected(tmp_path):
    frame = pd.DataFrame([{name: 1.0 for name in ORIGINATION_FEATURES}])
    frame["SeriousDlqin2yrs"] = 0
    frame["NumberOfTime30-59DaysPastDueNotWorse"] = 1
    path = tmp_path / "ambiguous.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(DataContractError, match="Ambiguous"):
        load_origination_csv(path)


def test_history_validation_does_not_depend_on_dataframe_index(portfolio):
    history = portfolio.history.sample(frac=1, random_state=1)
    history.index = [0] * len(history)
    validate_history(history, portfolio.accounts)


def test_target_cli_round_trip_and_overwrite_protection(portfolio, tmp_path, capsys):
    config = SyntheticPortfolioConfig(
        n_accounts=12, calendar_months=18, origination_months=6, seed=7
    )
    write_portfolio(portfolio, tmp_path, config)
    target_config = tmp_path / "target.yaml"
    target_config.write_text("horizon_months: 2\n")
    args = [
        "build-targets",
        "--accounts",
        str(tmp_path / "accounts.csv"),
        "--history",
        str(tmp_path / "history.csv"),
        "--config",
        str(target_config),
        "--as-of",
        "2022-03-31",
        "--output",
        str(tmp_path / "targets.csv"),
    ]
    assert main(args) == 0
    capsys.readouterr()
    targets = pd.read_csv(tmp_path / "targets.csv")
    assert (pd.to_datetime(targets.observation_date) <= pd.Timestamp("2022-03-31")).all()
    assert (tmp_path / "targets.manifest.json").is_file()
    assert main(args) == 2
    assert "already exist" in capsys.readouterr().err
