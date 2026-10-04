"""Run separate simulations; oracle labels serve evaluation, never IPW training."""

import inspect
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk import __version__
from credit_risk.decisioning.reject_inference import (
    FEATURES,
    SCENARIOS,
    fit_observed_models,
    generate_selection,
    logistic_model,
    population_risk_bounds,
    reject_sensitivity,
)
from credit_risk.decisioning.reject_settings import RejectInferenceConfig
from credit_risk.validation.metrics import binary_metrics
from credit_risk.validation.runner import digest, write_json


def run_reject_experiment(output_dir, config=None):
    config = config or RejectInferenceConfig()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Reject inference output is not empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    metric_rows = []
    for seed in config.seeds:
        for scenario in SCENARIOS:
            sample = generate_selection(config, seed, scenario)
            train, test = sample.train_positions, sample.holdout_positions
            x_train = sample.features.iloc[train].reset_index(drop=True)
            x_test = sample.features.iloc[test].reset_index(drop=True)
            models, diagnostics, propensity = fit_observed_models(
                x_train,
                sample.accepted[train],
                sample.observed_outcome[train],
                config,
                seed,
                scenario != "deterministic_no_overlap",
            )
            # Explicit unattainable benchmark, isolated from accepted-only and IPW learners.
            oracle = logistic_model(config.logistic_c)
            oracle.fit(x_train, sample.oracle_outcome[train])
            models["oracle_all_labels_benchmark"] = oracle
            diagnostic = dict(
                seed=seed,
                scenario=scenario,
                train_rows=len(train),
                holdout_rows=len(test),
                train_acceptance_rate=float(sample.accepted[train].mean()),
                true_zero_support_fraction=float((sample.true_propensity[train] == 0).mean()),
                true_min_propensity=float(sample.true_propensity[train].min()),
                population_risk_bounds=population_risk_bounds(
                    sample.accepted[test], sample.observed_outcome[test]
                ),
                **diagnostics,
            )
            if propensity is not None:
                diagnostic["oof_propensity_metrics"] = binary_metrics(
                    sample.accepted[train], propensity
                )
            predictions = pd.DataFrame(
                dict(
                    row_position=test,
                    accepted=sample.accepted[test],
                    observed_outcome=sample.observed_outcome[test],
                    synthetic_oracle_outcome=sample.oracle_outcome[test],
                )
            )
            evaluations = {}
            for name, model in models.items():
                probabilities = model.predict_proba(x_test.loc[:, FEATURES])[:, 1]
                predictions[name] = probabilities
                evaluations[name] = {}
                for segment, mask in [
                    ("all", np.ones(len(test), dtype=bool)),
                    ("accepted", sample.accepted[test] == 1),
                    ("rejected", sample.accepted[test] == 0),
                ]:
                    if not mask.any():
                        evaluations[name][segment] = None
                        continue
                    metrics = binary_metrics(sample.oracle_outcome[test][mask], probabilities[mask])
                    evaluations[name][segment] = metrics
                    metric_rows.append(
                        dict(
                            seed=seed,
                            scenario=scenario,
                            model=name,
                            segment=segment,
                            rows=int(mask.sum()),
                            **metrics,
                        )
                    )
            sensitivities = reject_sensitivity(
                sample.accepted[test],
                sample.observed_outcome[test],
                predictions.accepted_only.to_numpy(),
                config.reject_odds_multipliers,
            )
            predictions.to_csv(output / f"{scenario}_{seed}_holdout.csv", index=False)
            # Training CSV exposes only legitimately observed labels plus diagnostic propensity.
            training = x_train.copy()
            training.insert(0, "row_position", train)
            training["accepted"] = sample.accepted[train]
            training["observed_outcome"] = sample.observed_outcome[train]
            if propensity is not None:
                training["oof_propensity"] = propensity
            training.to_csv(output / f"{scenario}_{seed}_training.csv", index=False)
            results.append(
                dict(**diagnostic, evaluation=evaluations, reject_sensitivity=sensitivities)
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    aggregates = []
    for (scenario, model), group in metrics.loc[metrics.segment == "all"].groupby(
        ["scenario", "model"]
    ):
        aggregates.append(
            dict(
                scenario=scenario,
                model=model,
                repetitions=len(group),
                brier_mean=float(group.brier.mean()),
                log_loss_mean=float(group.log_loss.mean()),
                log_loss_seed_sd=float(group.log_loss.std(ddof=1)) if len(group) > 1 else None,
                mean_pd=float(group.mean_probability.mean()),
                true_synthetic_bad_rate=float(group.observed_bad_rate.mean()),
            )
        )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        rows = [r for r in aggregates if r["scenario"] == scenario]
        ax.bar(
            [r["model"].replace("oracle_all_labels_benchmark", "oracle") for r in rows],
            [r["log_loss_mean"] for r in rows],
            color="#286b8c",
        )
        ax.tick_params(axis="x", rotation=25)
        ax.set_title(scenario)
        ax.set_ylabel("All-applicant synthetic holdout log loss")
    fig.suptitle("Reject inference simulation — independent of origination model holdout")
    fig.savefig(output / "reject_inference.png", dpi=160)
    plt.close(fig)
    manifest = dict(
        package_version=__version__,
        is_synthetic=True,
        original_final_test_accessed=False,
        config=config.model_dump(),
        features=list(FEATURES),
        source_code_sha256={
            Path(p).name: digest(p)
            for p in [
                inspect.getfile(generate_selection),
                inspect.getfile(RejectInferenceConfig),
                __file__,
            ]
        },
        versions={name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy"]},
        results=results,
        aggregate_metrics=aggregates,
        limitations=[
            "Synthetic potential outcomes are not real rejected-applicant outcomes",
            "IPW requires selection conditional on recorded X and positive acceptance support",
            "MNAR hidden factors violate conditional selection exchangeability",
            "Clipping trades variance for bias and cannot restore structural zero support",
            "Oracle labels evaluate the simulation; the oracle model is unattainable",
            "Seed standard deviations describe Monte Carlo variation, not confidence intervals",
            "No real-data reject inference model is promoted or Phase 8 policy changed",
        ],
        artifacts_sha256={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    write_json(output / "reject_inference.json", manifest)
    return manifest
