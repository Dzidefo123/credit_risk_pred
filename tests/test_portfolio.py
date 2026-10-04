"""Hand-calculated vintages, migration denominators, gaps and cutoff behavior."""

import json
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from credit_risk.cli import main
from credit_risk.data.validation import STATES, DataContractError, validate_history
from credit_risk.portfolio.roll_rates import roll_rates
from credit_risk.portfolio.runner import run_portfolio_analytics
from credit_risk.portfolio.settings import PortfolioAnalyticsConfig
from credit_risk.portfolio.vintage import vintage_table


@pytest.fixture
def toy_portfolio():
    specifications = {
        "A": ("2022-01-01", 100, [0, 2, 0, 5]),
        "B": ("2022-01-01", 200, [0, 3, 5, 5]),
        "C": ("2022-02-01", 0, [0, 1, 0]),
    }
    accounts, history = [], []
    for account, (start, balance, states) in specifications.items():
        date = pd.Timestamp(start)
        accounts.append(
            {
                "account_id": account,
                "origination_date": date,
                "age": 30,
                "monthly_income": 1000,
                "origination_limit": 300,
                "is_synthetic": True,
            }
        )
        for mob, state in enumerate(states):
            history.append(
                {
                    "account_id": account,
                    "origination_date": date,
                    "observation_date": date + pd.offsets.MonthEnd(mob + 1),
                    "months_on_book": mob,
                    "state": STATES[state],
                    "dpd": [0, 10, 40, 70, 100, 120][state],
                    "default_flag": state == 5,
                    "opening_balance": balance,
                    "draws": 0,
                    "interest": 0,
                    "payment": 0,
                    "scheduled_payment": 0,
                    "write_off": 0,
                    "balance": balance,
                    "credit_limit": 300,
                    "utilization": balance / 300,
                    "is_synthetic": True,
                }
            )
    accounts, history = pd.DataFrame(accounts), pd.DataFrame(history)
    validate_history(history, accounts)
    return accounts, history


def cell(table, cohort, mob):
    return table.loc[(table.origination_month == cohort) & (table.months_on_book == mob)].iloc[0]


def test_hand_calculated_vintage_denominators_curves_and_exposure(toy_portfolio):
    accounts, history = toy_portfolio
    before = history.copy(deep=True)
    table = vintage_table(
        accounts, history, PortfolioAnalyticsConfig(max_months_on_book=3, minimum_cohort_size=2)
    )
    assert len(table) == 8
    feb = cell(table, "2022-01", 1)
    assert feb.observed_accounts == 2 and feb.cohort_accounts == 2
    assert feb.balance_exposure == 300
    assert feb.bad_rate == 0 and feb.delinquency_rate == 1
    march = cell(table, "2022-01", 2)
    assert march.bad_rate == march.cumulative_default_rate == 0.5
    april = cell(table, "2022-01", 3)
    assert april.cumulative_observed_default_count == 2 and april.cumulative_default_rate == 1
    future = cell(table, "2022-02", 3)
    assert not future.matured and pd.isna(future.cumulative_default_rate)
    assert pd.isna(future.balance_exposure) and pd.isna(future.observed_accounts)
    assert cell(table, "2022-02", 0).low_support
    assert not cell(table, "2022-01", 0).low_support
    for _, cohort in table.groupby("origination_month"):
        assert (cohort.cumulative_default_rate.dropna().diff().dropna() >= 0).all()
    assert_frame_equal(history, before)


@pytest.mark.parametrize("missing_mob", [0, 1, 2, 3])
def test_missing_history_never_shrinks_original_cohort_or_becomes_good(toy_portfolio, missing_mob):
    accounts, history = toy_portfolio
    history = history.loc[~((history.account_id == "A") & (history.months_on_book == missing_mob))]
    table = vintage_table(accounts, history, PortfolioAnalyticsConfig(max_months_on_book=3))
    missing = cell(table, "2022-01", missing_mob)
    assert missing.cohort_accounts == 2 and missing.observed_accounts == 1
    assert missing.snapshot_coverage == 0.5
    assert not missing.fully_observed and pd.isna(missing.cumulative_default_rate)
    april = cell(table, "2022-01", 3)
    assert pd.isna(april.cumulative_default_rate)
    assert april.cumulative_default_lower_bound <= 1


def test_account_with_no_history_stays_in_vintage_denominator(toy_portfolio):
    accounts, history = toy_portfolio
    history = history.loc[history.account_id != "A"]
    table = vintage_table(accounts, history, PortfolioAnalyticsConfig(max_months_on_book=3))
    april = cell(table, "2022-01", 3)
    assert april.cohort_accounts == 2 and april.cumulative_observed_default_count == 1
    assert april.cumulative_default_lower_bound == 0.5
    assert pd.isna(april.cumulative_default_rate)


def test_recorded_default_and_analytic_bad_are_different(toy_portfolio):
    accounts, history = toy_portfolio
    history.loc[(history.account_id == "A") & (history.months_on_book == 1), ["dpd", "state"]] = [
        100,
        "DPD_90_PLUS",
    ]
    table = vintage_table(accounts, history, PortfolioAnalyticsConfig(max_months_on_book=3))
    row = cell(table, "2022-01", 1)
    assert row.bad_rate == 0.5 and row.cumulative_default_rate == 0


def test_hand_calculated_count_and_balance_migrations_and_cures(toy_portfolio):
    accounts, history = toy_portfolio
    before = history.copy(deep=True)
    result = roll_rates(history, accounts=accounts)
    assert result.diagnostics["consecutive_pairs"] == 8
    assert result.diagnostics["not_yet_due_origins"] == 3
    assert result.diagnostics["missing_next"] == 0
    assert result.count_matrix.loc["CURRENT"].sum() == 4
    assert result.probability_matrix.loc["CURRENT", "DPD_1_29"] == 0.25
    assert result.balance_probability_matrix.loc["CURRENT", "DPD_1_29"] == 0
    assert result.balance_probability_matrix.loc["CURRENT", "DPD_60_89"] == 0.5
    assert result.probability_matrix.loc["DEFAULT", "DEFAULT"] == 1
    assert result.probability_matrix.loc["DPD_90_PLUS"].isna().all()
    assert result.balance_probability_matrix.loc["DPD_1_29"].isna().all()
    rows = result.state_summary.set_index("from_state")
    assert rows.loc["DEFAULT", "new_default_rate"] == 0
    assert rows.loc["DEFAULT", "stay_rate"] == 1
    assert rows.loc["DPD_30_59", "cure_rate"] == 1
    assert rows.loc["DPD_60_89", "roll_forward_rate"] == 1
    supported = result.count_matrix.sum(axis=1) > 0
    assert np.allclose(result.probability_matrix.loc[supported].sum(axis=1), 1)
    assert int(result.count_matrix.to_numpy().sum()) == len(result.pairs)
    assert result.balance_matrix.to_numpy().sum() == result.pairs.origin_balance.sum()
    assert not result.pairs.duplicated(["account_id", "origin_date"]).any()
    assert_frame_equal(history, before)


def test_gaps_not_bridged_and_missing_next_denominator_explicit(toy_portfolio):
    accounts, history = toy_portfolio
    history = history.loc[~((history.account_id == "A") & (history.months_on_book == 1))]
    result = roll_rates(history, accounts=accounts)
    assert len(result.pairs) == 6
    assert result.diagnostics["gap_links_excluded"] == 1
    assert result.diagnostics["eligible_origins"] == 7
    assert result.diagnostics["missing_next"] == 1
    assert result.diagnostics["pair_coverage"] == pytest.approx(6 / 7)
    january = result.monthly_summary.iloc[0]
    assert january.eligible_origins == 2 and january.paired_origins == 1
    assert january.pair_coverage == 0.5


def test_cutoff_excludes_future_and_after_calendar_end_does_not_infer_exits(toy_portfolio):
    accounts, history = toy_portfolio
    result = roll_rates(history, "2022-03-15", accounts)
    assert result.pairs.destination_date.max() == pd.Timestamp("2022-02-28")
    assert len(result.pairs) == 2
    vintage = vintage_table(
        accounts, history, PortfolioAnalyticsConfig(max_months_on_book=3), "2022-03-15"
    )
    assert pd.isna(cell(vintage, "2022-01", 2).cumulative_default_rate)
    later = roll_rates(history, "2022-05-31", accounts)
    assert later.diagnostics["missing_next"] == 3
    assert later.diagnostics["not_yet_due_origins"] == 0
    assert len(later.pairs) == 8


def test_no_eligible_pairs_gives_unknown_rates_not_identity(toy_portfolio):
    accounts, history = toy_portfolio
    result = roll_rates(history, "2022-01-31", accounts)
    assert result.pairs.empty and result.probability_matrix.isna().all().all()
    assert result.count_matrix.to_numpy().sum() == 0
    assert result.diagnostics["pair_coverage"] is None


def test_mixed_provenance_invalid_cutoff_and_configuration_rejected(toy_portfolio):
    accounts, history = toy_portfolio
    with pytest.raises(ValueError, match="as_of"):
        roll_rates(history, "2022-04-30T10:00:00", accounts)
    with pytest.raises(ValidationError):
        PortfolioAnalyticsConfig(delinquency_dpd_threshold=100, bad_dpd_threshold=90)
    accounts.loc[accounts.account_id == "A", "is_synthetic"] = False
    history.loc[history.account_id == "A", "is_synthetic"] = False
    with pytest.raises(DataContractError, match="separately"):
        vintage_table(accounts, history)


def test_runner_export_reproduction_manifest_integrity_and_no_overwrite(toy_portfolio, tmp_path):
    accounts, history = toy_portfolio
    a, h = tmp_path / "accounts.csv", tmp_path / "history.csv"
    accounts.to_csv(a, index=False)
    history.to_csv(h, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "is_synthetic": True,
                "files_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in [a, h]},
            }
        )
    )
    config = PortfolioAnalyticsConfig(max_months_on_book=3, minimum_cohort_size=2)
    first = run_portfolio_analytics(a, h, tmp_path / "run", config, source_manifest=manifest)
    second = run_portfolio_analytics(a, h, tmp_path / "repeat", config, source_manifest=manifest)
    assert first["vintage_checkpoints"] == second["vintage_checkpoints"]
    assert first["artifacts_sha256"] == second["artifacts_sha256"]
    assert first["is_synthetic"] and first["roll_diagnostics"]["consecutive_pairs"] == 8
    matrices = pd.read_csv(tmp_path / "run" / "monthly_transition_matrices.csv")
    assert len(matrices) == 3 * 36 and matrices.pairs.sum() == 8
    with pytest.raises(FileExistsError):
        run_portfolio_analytics(a, h, tmp_path / "run", config)
    h.write_text(h.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        run_portfolio_analytics(a, h, tmp_path / "tampered", config, source_manifest=manifest)


def test_portfolio_cli_uses_declared_cutoff_and_reports_errors(toy_portfolio, tmp_path, capsys):
    accounts, history = toy_portfolio
    a, h = tmp_path / "accounts.csv", tmp_path / "history.csv"
    accounts.to_csv(a, index=False)
    history.to_csv(h, index=False)
    config = tmp_path / "portfolio.yaml"
    config.write_text("max_months_on_book: 3\n")
    args = [
        "analyze-portfolio",
        "--accounts",
        str(a),
        "--history",
        str(h),
        "--config",
        str(config),
        "--as-of",
        "2022-03-15",
        "--output-dir",
        str(tmp_path / "run"),
    ]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["as_of"] == "2022-03-15" and result["roll_diagnostics"]["consecutive_pairs"] == 2
    assert main(args) == 2
    assert json.loads(capsys.readouterr().err)["level"] == "ERROR"


@pytest.mark.parametrize("scenario", ["complete", "gap", "left_truncated", "early_cutoff"])
def test_sql_matches_python_vintages_and_roll_matrices(toy_portfolio, scenario):
    from pathlib import Path

    import duckdb

    accounts, history = toy_portfolio
    cutoff = "2022-03-15" if scenario == "early_cutoff" else "2022-04-30"
    if scenario in {"gap", "left_truncated"}:
        mob = 1 if scenario == "gap" else 0
        history = history.loc[~((history.account_id == "A") & (history.months_on_book == mob))]
    config = PortfolioAnalyticsConfig(max_months_on_book=3)
    parameters = pd.DataFrame(
        {
            "as_of": [pd.Timestamp(cutoff)],
            **{key: [value] for key, value in config.model_dump().items()},
        }
    )
    calendar = pd.DataFrame({"months_on_book": range(4)})
    states = pd.DataFrame({"state": STATES, "ordinal": range(6)})
    root = Path(__file__).resolve().parents[1]
    with duckdb.connect() as database:
        for name, frame in [
            ("portfolio_accounts", accounts),
            ("portfolio_history", history),
            ("analytics_parameters", parameters),
            ("mob_calendar", calendar),
            ("delinquency_states", states),
        ]:
            database.register(name, frame)
        sql_vintage = database.execute((root / "sql/vintage_analysis.sql").read_text()).df()
        sql_rolls = database.execute((root / "sql/roll_rates.sql").read_text()).df()
    python_vintage = vintage_table(accounts, history, config, cutoff)
    sql_vintage["origination_month"] = sql_vintage.origination_month.dt.strftime("%Y-%m")
    columns = list(sql_vintage.columns)
    assert_frame_equal(sql_vintage[columns], python_vintage[columns], check_dtype=False)
    rolls = roll_rates(history, cutoff, accounts)
    for row in sql_rolls.itertuples():
        assert row.pairs == rolls.count_matrix.loc[row.from_state, row.to_state]
        expected = rolls.probability_matrix.loc[row.from_state, row.to_state]
        assert (
            pd.isna(row.probability)
            if pd.isna(expected)
            else row.probability == pytest.approx(expected)
        )
        expected = rolls.balance_probability_matrix.loc[row.from_state, row.to_state]
        assert (
            pd.isna(row.balance_probability)
            if pd.isna(expected)
            else row.balance_probability == pytest.approx(expected)
        )


def test_partial_improvement_is_roll_back_but_not_full_cure(toy_portfolio):
    accounts, history = toy_portfolio
    history.loc[(history.account_id == "A") & (history.months_on_book == 2), ["dpd", "state"]] = [
        10,
        "DPD_1_29",
    ]
    result = roll_rates(history, accounts=accounts)
    row = result.state_summary.set_index("from_state").loc["DPD_30_59"]
    assert row.roll_back_rate == 1 and row.cure_rate == 0


def test_cutoff_before_first_snapshot_keeps_immature_cells_and_empty_headers(toy_portfolio):
    accounts, history = toy_portfolio
    vintages = vintage_table(
        accounts, history, PortfolioAnalyticsConfig(max_months_on_book=3), "2022-01-15"
    )
    assert len(vintages) == 4 and not vintages.matured.any()
    assert vintages.cumulative_default_rate.isna().all()
    result = roll_rates(history, "2022-01-15", accounts)
    assert result.diagnostics["snapshots_as_of"] == 0
    assert "pair_coverage" in result.monthly_summary.columns


def test_homogeneous_real_source_provenance_is_supported(toy_portfolio):
    accounts, history = toy_portfolio
    accounts = accounts.assign(is_synthetic=False)
    history = history.assign(is_synthetic=False)
    vintages = vintage_table(accounts, history, PortfolioAnalyticsConfig(max_months_on_book=3))
    assert not vintages.is_synthetic.any()
    assert len(roll_rates(history, accounts=accounts).pairs) == 8
