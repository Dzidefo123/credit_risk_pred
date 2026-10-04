"""Loss arithmetic, modeled PD horizon, snapshot alignment and portfolio reconciliation."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError
from sklearn.exceptions import NotFittedError

from credit_risk.cli import main
from credit_risk.data.validation import STATES, validate_history
from credit_risk.models.transition_pd import TransitionPDModel
from credit_risk.portfolio.expected_loss import (
    calculate_expected_loss,
    exposure_at_default,
    loss_table,
    portfolio_loss_summary,
    segment_loss_summary,
    stress_probability,
)
from credit_risk.portfolio.loss_runner import portfolio_snapshot, run_expected_loss
from credit_risk.portfolio.loss_settings import ExpectedLossConfig, LossScenario


@pytest.fixture
def transition_portfolio():
    accounts, history = [], []
    for number, initial in enumerate([0, 0, 1, 2, 3, 4, 5]):
        account = f"A{number}"
        accounts.append(
            {
                "account_id": account,
                "origination_date": pd.Timestamp("2022-01-01"),
                "age": 30,
                "monthly_income": 1000,
                "origination_limit": 200,
                "is_synthetic": True,
            }
        )
        for mob, state in enumerate([initial, 0 if number == 0 else 5]):
            history.append(
                {
                    "account_id": account,
                    "origination_date": pd.Timestamp("2022-01-01"),
                    "observation_date": pd.Timestamp("2022-01-01") + pd.offsets.MonthEnd(mob + 1),
                    "months_on_book": mob,
                    "dpd": [0, 10, 40, 70, 100, 120][state],
                    "state": STATES[state],
                    "default_flag": state == 5,
                    "opening_balance": 100,
                    "draws": 0,
                    "interest": 0,
                    "payment": 0,
                    "scheduled_payment": 0,
                    "write_off": 0,
                    "balance": 100,
                    "credit_limit": 200,
                    "utilization": 0.5,
                    "is_synthetic": True,
                }
            )
    accounts, history = pd.DataFrame(accounts), pd.DataFrame(history)
    validate_history(history, accounts)
    return accounts, history


@pytest.fixture
def loss_inputs():
    snapshot = pd.DataFrame(
        {
            "account_id": ["a", "b", "c", "d"],
            "observation_date": pd.to_datetime(["2022-01-31"] * 4),
            "balance": [100.0, 300.0, 0.0, 100.0],
            "credit_limit": [200.0, 200.0, 0.0, 1000.0],
            "default_flag": [False, False, False, True],
            "state": ["CURRENT", "DPD_1_29", "CURRENT", "DEFAULT"],
            "origination_month": ["2022-01"] * 4,
        }
    )
    scores = snapshot[["account_id", "observation_date"]].copy()
    scores["pd"] = [0.2, 0.5, 0, 1]
    return snapshot, scores


def test_scalar_vector_loss_and_undrawn_conversion_hand_calculated():
    assert calculate_expected_loss(0.2, 0.45, 150) == pytest.approx(13.5)
    assert np.allclose(calculate_expected_loss([0, 0.5, 1], 0.4, [100, 100, 100]), [0, 20, 40])
    assert np.array_equal(exposure_at_default([100, 300, 0], [200, 200, 0], 0.5), [150, 300, 0])
    assert np.array_equal(stress_probability([0, 0.2, 1], 2), [0, 1 / 3, 1])


@pytest.mark.parametrize(
    "p,lgd,ead",
    [
        (-0.1, 0.4, 100),
        (1.1, 0.4, 100),
        (0.1, 1.1, 100),
        (0.1, 0.4, -1),
        (np.nan, 0.4, 100),
        (0.1, 0.4, np.inf),
        (True, 0.4, 100),
        ("0.1", 0.4, 100),
        ([], 0.4, 100),
        ([[0.1]], 0.4, 100),
        ([0.1, 0.2], 0.4, [100, 200, 300]),
    ],
)
def test_invalid_loss_inputs_never_become_zero(p, lgd, ead):
    with pytest.raises(ValueError):
        calculate_expected_loss(p, lgd, ead)


def test_markov_horizon_and_structural_default_hand_calculated(transition_portfolio):
    accounts, history = transition_portfolio
    model = TransitionPDModel(horizon_months=2, minimum_state_pairs=1).fit(history, accounts)
    assert model.predict_pd(["CURRENT", "DPD_1_29", "DEFAULT"]).tolist() == [0.75, 1, 1]
    assert np.allclose(model.transition_matrix_.sum(axis=1), 1)
    assert model.metadata()["probability_status"] == "uncalibrated benchmark"
    one = TransitionPDModel(1, 1).fit(history, accounts)
    assert one.predict_pd(["CURRENT"])[0] == 0.5
    # Structural absorption remains defined without observed DEFAULT origins.
    no_default_origins = history.loc[history.account_id != "A6"]
    assert TransitionPDModel(2, 1).fit(no_default_origins, accounts).predict_pd(["DEFAULT"])[0] == 1
    with pytest.raises(ValueError, match="support"):
        TransitionPDModel(2, 2).fit(history, accounts)
    with pytest.raises(ValueError, match="known"):
        model.predict_pd(["UNKNOWN"])
    with pytest.raises(NotFittedError):
        TransitionPDModel().predict_pd(["CURRENT"])


def test_fit_cutoff_does_not_use_future_transitions(transition_portfolio):
    accounts, history = transition_portfolio
    future = history.loc[history.months_on_book == 1].copy()
    future["observation_date"] = pd.Timestamp("2022-03-31")
    future["months_on_book"] = 2
    future["state"], future["dpd"], future["default_flag"] = "DEFAULT", 120, True
    augmented = pd.concat([history, future], ignore_index=True)
    baseline = TransitionPDModel(2, 1).fit(history, accounts, as_of="2022-02-28")
    restricted = TransitionPDModel(2, 1).fit(augmented, accounts, as_of="2022-02-28")
    assert np.array_equal(baseline.transition_matrix_, restricted.transition_matrix_)
    expanded = TransitionPDModel(2, 1).fit(augmented, accounts)
    assert expanded.predict_pd(["CURRENT"])[0] == pytest.approx(8 / 9)


def test_snapshot_exact_month_coverage_and_no_forward_fill(transition_portfolio):
    accounts, history = transition_portfolio
    snapshot, missing, coverage = portfolio_snapshot(accounts, history, "2022-03-15")
    assert coverage["snapshot_date"] == "2022-02-28" and missing.empty
    assert len(snapshot) == 7
    with pytest.raises(ValueError, match="Incomplete"):
        portfolio_snapshot(accounts, history, "2022-03-31")
    partial = history.loc[~((history.account_id == "A0") & (history.months_on_book == 1))]
    with pytest.raises(ValueError, match="Incomplete"):
        portfolio_snapshot(accounts, partial)
    snapshot, missing, coverage = portfolio_snapshot(accounts, partial, require_complete=False)
    assert coverage["coverage"] == 6 / 7 and missing.account_id.tolist() == ["A0"]
    assert "A0" not in snapshot.account_id.to_list()


def test_keyed_scores_defaulted_stock_and_portfolio_conservation(loss_inputs):
    snapshot, scores = loss_inputs
    before = snapshot.copy(deep=True)
    rows = loss_table(snapshot, scores.sample(frac=1, random_state=1))
    base = rows.loc[rows.scenario == "base"].set_index("account_id")
    assert base.loc["a", "forward_expected_loss"] == pytest.approx(13.5)
    assert base.loc["b", "ead"] == 300
    assert base.loc["d", "ead"] == 100 and base.loc["d", "forward_expected_loss"] == 0
    assert base.loc["d", "defaulted_loss_assumption"] == 45
    assert base.loc["d", "credit_conversion_factor"] == 0
    assert base.loc["d", "available_undrawn"] == 0
    summary = portfolio_loss_summary(rows, top_n=1).set_index("scenario")
    assert summary.loc["base", "forward_expected_loss"] == 81
    assert summary.loc["base", "defaulted_loss_assumption"] == 45
    assert summary.loc["base", "combined_loss_proxy"] == 126
    assert summary.loc["base", "ead_weighted_nondefault_pd"] == pytest.approx(0.4)
    assert summary.loc["base", "top_n_ead_share"] == pytest.approx(300 / 550)
    assert summary.loc["base", "top_n_forward_el_share"] == pytest.approx(67.5 / 81)
    segments = segment_loss_summary(rows, top_n=1)
    cohort = segments.loc[
        (segments.dimension == "origination_month") & (segments.scenario == "base")
    ].iloc[0]
    assert cohort.top_n_ead_share == pytest.approx(summary.loc["base", "top_n_ead_share"])
    states = segments.loc[segments.dimension == "state"].groupby("scenario")
    for scenario, group in states:
        assert group.combined_loss_proxy.sum() == pytest.approx(
            summary.loc[scenario, "combined_loss_proxy"]
        )
    assert (summary.forward_expected_loss.diff().dropna() > 0).all()
    assert_frame_equal(snapshot, before)


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "missing_pd", "default_pd"])
def test_score_population_or_invalid_pd_rejected(loss_inputs, problem):
    snapshot, scores = loss_inputs
    if problem == "missing":
        scores = scores.iloc[:-1]
    if problem == "extra":
        scores = pd.concat([scores, scores.iloc[[0]].assign(account_id="extra")])
    if problem == "duplicate":
        scores = pd.concat([scores, scores.iloc[[0]]])
    if problem == "missing_pd":
        scores.loc[0, "pd"] = np.nan
    if problem == "default_pd":
        scores.loc[3, "pd"] = 0.8
    with pytest.raises(ValueError):
        loss_table(snapshot, scores)


def test_zero_exposure_and_all_default_portfolios_have_explicit_unavailable_rates(loss_inputs):
    snapshot, scores = loss_inputs
    rows = loss_table(snapshot.iloc[[2]], scores.iloc[[2]])
    summary = portfolio_loss_summary(rows)
    assert summary.forward_expected_loss.eq(0).all()
    assert summary.forward_el_rate.isna().all() and summary.ead_hhi.isna().all()
    default = portfolio_loss_summary(loss_table(snapshot.iloc[[3]], scores.iloc[[3]]))
    assert default.forward_expected_loss.eq(0).all() and default.mean_nondefault_pd.isna().all()


def test_scenario_config_rejects_ambiguous_or_invalid_assumptions():
    for settings in [
        {"scenarios": [LossScenario(name="base"), LossScenario(name="base")]},
        {"scenarios": [LossScenario(name="other")]},
        {"scenarios": [LossScenario(name="base", pd_odds_multiplier=2.0)]},
        {"horizon_months": 0},
    ]:
        with pytest.raises(ValidationError):
            ExpectedLossConfig(**settings)
    with pytest.raises(ValidationError):
        LossScenario(name="bad", lgd=1.1)
    with pytest.raises(ValueError):
        exposure_at_default(10, 100, 1.1)


def test_runner_reproducible_outputs_and_model_scores_not_constant(transition_portfolio, tmp_path):
    accounts, history = transition_portfolio
    a, h = tmp_path / "accounts.csv", tmp_path / "history.csv"
    accounts.to_csv(a, index=False)
    history.to_csv(h, index=False)
    config = ExpectedLossConfig(horizon_months=2, minimum_state_pairs=1)
    first = run_expected_loss(a, h, tmp_path / "first", config)
    second = run_expected_loss(a, h, tmp_path / "repeat", config)
    assert first["scenario_summary"] == second["scenario_summary"]
    assert first["artifacts_sha256"] == second["artifacts_sha256"]
    scores = pd.read_csv(tmp_path / "first" / "pd_scores.csv")
    assert sorted(scores.pd.unique()) == [0.75, 1]
    assert first["coverage"]["coverage"] == 1 and first["is_synthetic"]
    with pytest.raises(FileExistsError):
        run_expected_loss(a, h, tmp_path / "first", config)
    malformed = tmp_path / "manifest.json"
    malformed.write_text("{}")
    with pytest.raises(ValueError, match="mapping"):
        run_expected_loss(a, h, tmp_path / "invalid", config, source_manifest=malformed)


def test_loss_cli_and_configuration_error(transition_portfolio, tmp_path, capsys):
    accounts, history = transition_portfolio
    a, h = tmp_path / "accounts.csv", tmp_path / "history.csv"
    accounts.to_csv(a, index=False)
    history.to_csv(h, index=False)
    config = tmp_path / "loss.yaml"
    config.write_text("horizon_months: 2\nminimum_state_pairs: 1\n")
    args = [
        "expected-loss",
        "--accounts",
        str(a),
        "--history",
        str(h),
        "--config",
        str(config),
        "--output-dir",
        str(tmp_path / "run"),
    ]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["scenario_summary"][0]["nondefault_accounts"] == 1
    assert main(args) == 2
    assert json.loads(capsys.readouterr().err)["level"] == "ERROR"


@pytest.mark.parametrize("partial", [False, True])
def test_snapshot_and_delinquency_sql_preserve_coverage(transition_portfolio, partial):
    from pathlib import Path

    import duckdb

    accounts, history = transition_portfolio
    if partial:
        history = history.loc[~((history.account_id == "A0") & (history.months_on_book == 1))]
    root = Path(__file__).resolve().parents[1]
    parameters = pd.DataFrame(
        {
            "as_of": [pd.Timestamp("2022-03-15")],
            "bad_dpd_threshold": [90],
            "delinquency_dpd_threshold": [30],
        }
    )
    with duckdb.connect() as database:
        database.register("portfolio_accounts", accounts)
        database.register("portfolio_history", history)
        database.register("analytics_parameters", parameters)
        sql_snapshot = database.execute((root / "sql/portfolio_snapshot.sql").read_text()).df()
        database.register("portfolio_snapshot", sql_snapshot)
        distribution = database.execute((root / "sql/delinquency_analysis.sql").read_text()).df()
    python_snapshot, missing, coverage = portfolio_snapshot(
        accounts, history, "2022-03-15", require_complete=False
    )
    assert len(sql_snapshot) == coverage["expected_accounts"]
    assert sql_snapshot.snapshot_observed.sum() == coverage["observed_accounts"]
    observed = sql_snapshot.loc[sql_snapshot.snapshot_observed]
    columns = ["account_id", "observation_date", "balance", "credit_limit", "state", "default_flag"]
    assert_frame_equal(
        observed[columns].reset_index(drop=True),
        python_snapshot[columns].sort_values("account_id").reset_index(drop=True),
        check_dtype=False,
    )
    assert distribution.observed_balance.sum() == python_snapshot.balance.sum()
    assert distribution.observed_accounts.sum() == len(python_snapshot)
    if partial:
        unknown = distribution.loc[distribution.state == "UNOBSERVED"].iloc[0]
        assert pd.isna(unknown.observed_bad_rate) and unknown.expected_accounts == len(missing)
