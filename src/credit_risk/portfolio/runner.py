"""Reproducible portfolio analytics exports, kept separate from origination PD."""

import inspect
import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from credit_risk import __version__
from credit_risk.data.loaders import load_portfolio_csv
from credit_risk.portfolio.common import prepare_history
from credit_risk.portfolio.plots import plot_portfolio
from credit_risk.portfolio.roll_rates import roll_rates, transition_matrices
from credit_risk.portfolio.settings import PortfolioAnalyticsConfig
from credit_risk.portfolio.vintage import vintage_table


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def records(frame):
    # pandas converts missing floating rates into JSON null, never NaN.
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def run_portfolio_analytics(
    accounts_path, history_path, output_dir, config=None, as_of=None, source_manifest=None
):
    config = config or PortfolioAnalyticsConfig()
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Portfolio output is not empty; choose a new directory")
    hashes = {"accounts": file_hash(accounts_path), "history": file_hash(history_path)}
    provenance = None
    if source_manifest is not None:
        provenance = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
        for name, path in (("accounts", accounts_path), ("history", history_path)):
            if provenance["files_sha256"][Path(path).name] != hashes[name]:
                raise ValueError(f"Source manifest checksum mismatch: {name}")
    portfolio = load_portfolio_csv(accounts_path, history_path)
    data, cutoff = prepare_history(portfolio.history, portfolio.accounts, as_of)
    vintages = vintage_table(portfolio.accounts, portfolio.history, config, cutoff)
    if vintages.empty:
        raise ValueError("No booked cohorts at the requested cutoff")
    rolls = roll_rates(portfolio.history, cutoff, portfolio.accounts)
    synthetic = bool(portfolio.accounts.is_synthetic.all())
    if provenance is not None and provenance.get("is_synthetic") != synthetic:
        raise ValueError("Source manifest provenance mismatch")
    directory.mkdir(parents=True, exist_ok=True)
    exports = {
        "vintages": vintages,
        "transition_pairs": rolls.pairs,
        "transition_counts": rolls.count_matrix,
        "transition_probabilities": rolls.probability_matrix,
        "transition_balances": rolls.balance_matrix,
        "transition_balance_probabilities": rolls.balance_probability_matrix,
        "roll_state_summary": rolls.state_summary,
        "roll_monthly_summary": rolls.monthly_summary,
    }
    for name, table in exports.items():
        table.to_csv(
            directory / f"{name}.csv",
            index=name.startswith("transition_") and name != "transition_pairs",
            date_format="%Y-%m-%d",
        )
    monthly_matrices = []
    for date, pairs in rolls.pairs.groupby("origin_date", sort=True):
        counts, probabilities, balances, balance_probabilities = transition_matrices(pairs)
        for origin in counts.index:
            for destination in counts.columns:
                monthly_matrices.append(
                    {
                        "origin_date": str(date.date()),
                        "from_state": origin,
                        "to_state": destination,
                        "pairs": int(counts.loc[origin, destination]),
                        "origin_balance": float(balances.loc[origin, destination]),
                        "probability": probabilities.loc[origin, destination],
                        "balance_probability": balance_probabilities.loc[origin, destination],
                    }
                )
    import pandas as pd

    monthly_table = pd.DataFrame(
        monthly_matrices,
        columns=[
            "origin_date",
            "from_state",
            "to_state",
            "pairs",
            "origin_balance",
            "probability",
            "balance_probability",
        ],
    )
    monthly_table.to_csv(directory / "monthly_transition_matrices.csv", index=False)
    checkpoints = []
    for mob in sorted({0, 6, 12, 24, config.max_months_on_book}):
        if mob > config.max_months_on_book:
            continue
        cells = vintages.loc[vintages.months_on_book == mob]
        complete = cells.loc[cells.fully_observed]
        denominator = int(complete.cohort_accounts.sum())
        defaults = int(complete.cumulative_observed_default_count.sum())
        checkpoints.append(
            {
                "months_on_book": mob,
                "cohorts": len(cells),
                "matured_cohorts": int(cells.matured.sum()),
                "complete_cohorts": len(complete),
                "complete_cohort_accounts": denominator,
                "recorded_defaults": defaults,
                "cumulative_default_rate": defaults / denominator if denominator else None,
            }
        )
    final = data.loc[data.observation_date == data.observation_date.max()]
    nondefault = rolls.pairs.loc[rolls.pairs.from_state != "DEFAULT"]
    result = {
        "package_version": __version__,
        "is_synthetic": synthetic,
        "source_sha256": hashes,
        "source_manifest_sha256": file_hash(source_manifest) if source_manifest else None,
        "generator_provenance": provenance,
        "config": config.model_dump(),
        "as_of": str(cutoff.date()),
        "vintage_cells": len(vintages),
        "immature_cells": int((~vintages.matured).sum()),
        "mature_incomplete_cells": int((vintages.matured & ~vintages.fully_observed).sum()),
        "low_support_cohorts": int(
            vintages.loc[vintages.low_support, "origination_month"].nunique()
        ),
        "vintage_checkpoints": checkpoints,
        "roll_diagnostics": rolls.diagnostics,
        "count_matrix": records(rolls.count_matrix.rename_axis("from_state").reset_index()),
        "probability_matrix": records(
            rolls.probability_matrix.rename_axis("from_state").reset_index()
        ),
        "balance_probability_matrix": records(
            rolls.balance_probability_matrix.rename_axis("from_state").reset_index()
        ),
        "roll_state_summary": records(rolls.state_summary),
        "overall_nondefault_pairs": len(nondefault),
        "overall_new_default_rate": float(nondefault.new_default.mean())
        if len(nondefault)
        else None,
        "latest_observed_snapshot": {
            "date": str(final.observation_date.iloc[0].date()) if len(final) else None,
            "accounts": len(final),
            "recorded_defaults": int(final.default_flag.sum()),
            "balance_exposure": float(final.balance.sum()),
            "state_counts": {str(k): int(v) for k, v in final.state.value_counts().items()},
        },
        "definitions": {
            "snapshot_bad": "recorded default OR dpd >= configured bad threshold",
            "snapshot_delinquency": "recorded default OR dpd >= configured delinquency threshold",
            "cumulative_default": (
                "ever recorded default / original cohort; missing unless every account has MOB 0..h"
            ),
            "cumulative_default_lower_bound": (
                "observed ever-default accounts / original cohort; "
                "not an estimate under missingness"
            ),
            "transition": "same account, consecutive calendar month-ends, both within cutoff",
            "roll_forward": ("higher state, including DEFAULT; deterioration synonym"),
            "roll_back": "any lower ordinal state; partial improvement counts",
            "cure": "delinquent nondefault origin to CURRENT; subset of roll-back",
            "new_default": "nondefault origin to DEFAULT; default persistence excluded",
            "monthly_denominators": (
                "forward/default among nondefault pairs; "
                "back/cure among delinquent nondefault pairs"
            ),
            "exposure": (
                "observed closing balance in source currency units; not EAD or expected loss"
            ),
        },
        "versions": {name: version(name) for name in ("numpy", "pandas", "matplotlib")},
        "source_code_sha256": {
            Path(path).name: file_hash(path)
            for path in (
                __file__,
                inspect.getfile(vintage_table),
                inspect.getfile(roll_rates),
                inspect.getfile(prepare_history),
                inspect.getfile(PortfolioAnalyticsConfig),
            )
        },
        "limitations": [
            "Synthetic simulation demonstrates mechanics, not real portfolio performance"
            if synthetic
            else "Real source definitions require domain review",
            "No closure status: missing accounts are not assumed closed or good",
            "Observed-pair transition estimates can be biased by missing follow-up",
            "Pooled transition rates combine ages and calendar months, not causal estimates",
            "Sparse cohorts/states: no smoothing or uncertainty intervals in Phase 6",
        ],
    }
    plot_portfolio(vintages, rolls, directory, synthetic)
    result["artifacts_sha256"] = {
        path.name: file_hash(path) for path in sorted(directory.iterdir()) if path.is_file()
    }
    (directory / "analytics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result
