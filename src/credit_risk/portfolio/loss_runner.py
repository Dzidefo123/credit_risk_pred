"""Model-driven forward loss and separate defaulted-stock assumptions by snapshot."""

import inspect
import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from credit_risk import __version__
from credit_risk.data.loaders import load_portfolio_csv
from credit_risk.models.transition_pd import TransitionPDModel
from credit_risk.portfolio.common import prepare_history
from credit_risk.portfolio.expected_loss import (
    loss_table,
    portfolio_loss_summary,
    segment_loss_summary,
)
from credit_risk.portfolio.loss_settings import ExpectedLossConfig


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def portfolio_snapshot(accounts, history, as_of=None, require_complete=True):
    data, cutoff = prepare_history(history, accounts, as_of)
    date = cutoff if cutoff.is_month_end else cutoff - pd.offsets.MonthEnd(1)
    master = accounts.loc[accounts.origination_date <= date]
    if master.empty:
        raise ValueError("No booked accounts at snapshot month-end")
    snapshot = data.loc[data.observation_date == date].copy()
    missing = master.loc[~master.account_id.isin(snapshot.account_id), ["account_id"]].copy()
    missing["observation_date"] = date
    missing["reason"] = "missing_exact_snapshot; not assumed closed or performing"
    if require_complete and len(missing):
        raise ValueError(
            f"Incomplete snapshot: {len(missing)} of {len(master)} booked accounts missing"
        )
    if snapshot.empty:
        raise ValueError("No observed exposures at snapshot month-end")
    snapshot = snapshot.merge(
        master[["account_id", "age", "monthly_income"]], on="account_id", validate="one_to_one"
    )
    snapshot["origination_month"] = snapshot.origination_date.dt.strftime("%Y-%m")
    coverage = {
        "information_cutoff": str(cutoff.date()),
        "snapshot_date": str(date.date()),
        "expected_accounts": len(master),
        "observed_accounts": len(snapshot),
        "missing_accounts": len(missing),
        "coverage": len(snapshot) / len(master),
        "require_complete_snapshot": require_complete,
    }
    return snapshot, missing, coverage


def run_expected_loss(
    accounts_path, history_path, output_dir, config=None, as_of=None, source_manifest=None
):
    config = config or ExpectedLossConfig()
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Expected-loss output is not empty; choose a new directory")
    hashes = {"accounts": file_hash(accounts_path), "history": file_hash(history_path)}
    provenance = None
    if source_manifest is not None:
        provenance = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
        if not isinstance(provenance, dict) or not isinstance(provenance.get("files_sha256"), dict):
            raise ValueError("Source manifest must contain a files_sha256 mapping")
        for name, path in (("accounts", accounts_path), ("history", history_path)):
            if provenance["files_sha256"].get(Path(path).name) != hashes[name]:
                raise ValueError(f"Source manifest checksum mismatch: {name}")
    portfolio = load_portfolio_csv(accounts_path, history_path)
    snapshot, missing, coverage = portfolio_snapshot(
        portfolio.accounts, portfolio.history, as_of, config.require_complete_snapshot
    )
    synthetic = bool(snapshot.is_synthetic.all())
    if provenance is not None and provenance.get("is_synthetic") is not synthetic:
        raise ValueError("Source manifest provenance mismatch")
    model = TransitionPDModel(config.horizon_months, config.minimum_state_pairs).fit(
        portfolio.history, accounts=portfolio.accounts, as_of=coverage["snapshot_date"]
    )
    scores = snapshot[["account_id", "observation_date"]].copy()
    scores["pd"] = model.predict_pd(snapshot.state)
    rows = loss_table(snapshot, scores, config)
    summary = portfolio_loss_summary(rows, config.top_n_concentration)
    segments = segment_loss_summary(rows, top_n=config.top_n_concentration)
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in {
        "snapshot": snapshot,
        "missing_snapshot_accounts": missing,
        "pd_scores": scores,
        "account_losses": rows,
        "scenario_summary": summary,
        "segment_summary": segments,
    }.items():
        table.to_csv(directory / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    metadata = model.metadata()
    metadata["source_sha256"] = hashes
    (directory / "pd_model.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    result = {
        "package_version": __version__,
        "is_synthetic": synthetic,
        "coverage": coverage,
        "source_sha256": hashes,
        "source_manifest_sha256": file_hash(source_manifest) if source_manifest else None,
        "generator_provenance": provenance,
        "config": config.model_dump(),
        "pd_model": metadata,
        "scenario_summary": records(summary),
        "segment_summary": records(segments),
        "definitions": {
            "formula": "forward nondefault EL = scenario model PD x LGD x EAD",
            "ead": "balance + CCF x max(limit - balance, 0); existing default EAD = balance",
            "scenario_pd": "p / (p + (1-p)/odds_multiplier); endpoints preserved",
            "defaulted_stock": "Default PD 1; no undrawn availability; loss proxy = LGD x balance",
            "combined_loss_proxy": (
                "forward nondefault EL + separate defaulted-stock residual loss assumption"
            ),
            "concentration": "account EAD HHI and top-n shares; no connected-borrower identities",
        },
        "versions": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "matplotlib")
        },
        "source_code_sha256": {
            Path(path).name: file_hash(path)
            for path in (
                __file__,
                inspect.getfile(TransitionPDModel),
                inspect.getfile(loss_table),
                inspect.getfile(ExpectedLossConfig),
                inspect.getfile(prepare_history),
            )
        },
        "limitations": [
            "Educational analytical framework; not regulatory IFRS 9/ECL compliance",
            "Synthetic simulation, not observed borrower performance"
            if synthetic
            else "Source semantics require domain validation",
            "PD benchmark uncalibrated and without prospective backtest; no accuracy claim",
            "Scenarios are deterministic sensitivity assumptions, not macroeconomic forecasts",
            "LGD/CCF are configured assumptions, not fitted recovery/utilization models",
            "No cash-flow timing, discounting, staging, maturity schedules or scenario weights",
            "Defaulted-stock loss proxy excludes prior write-offs; not a provision estimate",
            "Missing exposures are unknown; partial reports require explicit opt-in",
        ],
    }
    from credit_risk.portfolio.loss_plots import plot_loss

    plot_loss(summary, segments, config.horizon_months, synthetic, directory)
    result["artifacts_sha256"] = {
        p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file()
    }
    (directory / "expected_loss.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result
