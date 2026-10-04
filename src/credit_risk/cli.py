"""CLI for reproducible data preparation and configuration checks."""

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from credit_risk import __version__
from credit_risk.data.loaders import load_origination_csv, load_portfolio_csv
from credit_risk.data.synthetic_portfolio import (
    SyntheticPortfolioConfig,
    generate_portfolio,
    write_portfolio,
)
from credit_risk.data.targets import TargetConfig, build_forward_targets
from credit_risk.decisioning.reject_settings import RejectInferenceConfig
from credit_risk.decisioning.settings import PolicyComparisonConfig
from credit_risk.monitoring.settings import MonitoringConfig
from credit_risk.portfolio.loss_settings import ExpectedLossConfig
from credit_risk.portfolio.settings import PortfolioAnalyticsConfig
from credit_risk.utils.config import (
    DecisionPolicyConfig,
    DevelopmentConfig,
    ModelConfig,
    load_config,
)
from credit_risk.utils.logging import configure_logging
from credit_risk.validation.settings import ValidationConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="credit-risk-lab")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-config", help="Validate all example YAML files")
    check.add_argument("--config-dir", type=Path, default=Path("configs"))
    generate = commands.add_parser("generate-portfolio", help="Generate explicitly synthetic CSVs")
    generate.add_argument("--config", type=Path, default=Path("configs/synthetic_portfolio.yaml"))
    generate.add_argument("--output-dir", type=Path, default=Path("data/raw/synthetic-demo"))
    original = commands.add_parser("validate-origination", help="Inspect an origination CSV")
    original.add_argument("--csv", type=Path, required=True)
    target = commands.add_parser("build-targets", help="Build censored forward-window labels")
    target.add_argument("--accounts", type=Path, required=True)
    target.add_argument("--history", type=Path, required=True)
    target.add_argument("--config", type=Path, default=Path("configs/target.yaml"))
    target.add_argument("--as-of", default=None)
    target.add_argument("--output", type=Path, required=True)
    train = commands.add_parser(
        "train", help="Fit origination candidates, reserve calibration/test"
    )
    train.add_argument("--csv", type=Path, required=True)
    train.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    train.add_argument("--development-config", type=Path, default=Path("configs/development.yaml"))
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--output-dir", type=Path, required=True)
    validate = commands.add_parser(
        "validate", help="Calibrate frozen models and evaluate locked final holdout"
    )
    validate.add_argument("--csv", type=Path, required=True)
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--config", type=Path, default=Path("configs/validation.yaml"))
    portfolio = commands.add_parser(
        "analyze-portfolio", help="Vintage curves and consecutive-month roll rates"
    )
    portfolio.add_argument("--accounts", type=Path, required=True)
    portfolio.add_argument("--history", type=Path, required=True)
    portfolio.add_argument("--config", type=Path, default=Path("configs/portfolio.yaml"))
    portfolio.add_argument("--as-of", default=None)
    portfolio.add_argument("--source-manifest", type=Path, default=None)
    portfolio.add_argument("--output-dir", type=Path, required=True)
    loss = commands.add_parser(
        "expected-loss", help="Model-derived portfolio PD, expected loss and scenarios"
    )
    loss.add_argument("--accounts", type=Path, required=True)
    loss.add_argument("--history", type=Path, required=True)
    loss.add_argument("--config", type=Path, default=Path("configs/expected_loss.yaml"))
    loss.add_argument("--as-of", default=None)
    loss.add_argument("--source-manifest", type=Path, default=None)
    loss.add_argument("--output-dir", type=Path, required=True)
    policy = commands.add_parser(
        "compare-policies", help="Compare illustrative development credit strategies"
    )
    policy.add_argument("--csv", type=Path, required=True)
    policy.add_argument("--run-dir", type=Path, required=True)
    policy.add_argument("--validation-dir", type=Path, required=True)
    policy.add_argument("--config", type=Path, default=Path("configs/credit_strategy.yaml"))
    policy.add_argument("--output-dir", type=Path, required=True)
    reject = commands.add_parser(
        "reject-inference", help="Run explicitly synthetic selection-bias experiments"
    )
    reject.add_argument("--config", type=Path, default=Path("configs/reject_inference.yaml"))
    reject.add_argument("--output-dir", type=Path, required=True)
    reference = commands.add_parser(
        "freeze-monitor-reference", help="Freeze development monitoring bins"
    )
    for name in ("csv", "run-dir", "validation-dir", "output-dir"):
        reference.add_argument("--" + name, type=Path, required=True)
    reference.add_argument("--config", type=Path, default=Path("configs/monitoring.yaml"))
    monitor = commands.add_parser(
        "monitor", help="Compare label-free population against frozen reference"
    )
    for name in (
        "source-csv",
        "run-dir",
        "validation-dir",
        "reference-dir",
        "current-csv",
        "output-dir",
    ):
        monitor.add_argument("--" + name, type=Path, required=True)
    monitor.add_argument("--config", type=Path, default=Path("configs/monitoring.yaml"))
    monitor.add_argument("--current-manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    logger = configure_logging()
    try:
        if args.command == "check-config":
            directory = args.config_dir.resolve()
            development = load_config(directory / "development.yaml", DevelopmentConfig)
            load_config(directory / "model.yaml", ModelConfig)
            load_config(directory / "decision_policy.yaml", DecisionPolicyConfig)
            load_config(directory / "synthetic_portfolio.yaml", SyntheticPortfolioConfig)
            load_config(directory / "target.yaml", TargetConfig)
            load_config(directory / "validation.yaml", ValidationConfig)
            load_config(directory / "portfolio.yaml", PortfolioAnalyticsConfig)
            load_config(directory / "expected_loss.yaml", ExpectedLossConfig)
            load_config(directory / "credit_strategy.yaml", PolicyComparisonConfig)
            load_config(directory / "reject_inference.yaml", RejectInferenceConfig)
            load_config(directory / "monitoring.yaml", MonitoringConfig)
            logger = configure_logging(development.log_level)
            result = {
                "status": "valid",
                "version": __version__,
                "paths": {
                    name: str(path)
                    for name, path in development.paths.resolve(directory.parent).items()
                },
            }
        elif args.command == "generate-portfolio":
            config = load_config(args.config, SyntheticPortfolioConfig)
            result = write_portfolio(generate_portfolio(config), args.output_dir, config)
        elif args.command == "validate-origination":
            data = load_origination_csv(args.csv)
            result = {
                "status": "valid",
                "target_semantics": data.target_semantics,
                "quality": asdict(data.quality),
            }
        elif args.command == "freeze-monitor-reference":
            from credit_risk.monitoring.runner import freeze_monitor_reference

            config = load_config(args.config, MonitoringConfig)
            result = freeze_monitor_reference(
                args.csv, args.run_dir, args.validation_dir, args.output_dir, config
            )
        elif args.command == "monitor":
            from credit_risk.monitoring.runner import run_monitoring

            config = load_config(args.config, MonitoringConfig)
            manifest = run_monitoring(
                args.source_csv,
                args.run_dir,
                args.validation_dir,
                args.reference_dir,
                args.current_csv,
                args.output_dir,
                config,
                args.current_manifest,
            )
            result = {
                "status": manifest["status"],
                "reference_rows": manifest["reference_rows"],
                "current_rows": manifest["current_rows"],
                "alerts": manifest["alerts"],
                "pd_psi": manifest["metrics"]["pd"]["psi"],
                "score_psi": manifest["metrics"]["score"]["psi"],
                "final_test_scored": False,
                "output_dir": str(args.output_dir),
            }
        elif args.command == "reject-inference":
            from credit_risk.decisioning.reject_runner import run_reject_experiment

            config = load_config(args.config, RejectInferenceConfig)
            manifest = run_reject_experiment(args.output_dir, config)
            result = {
                "status": "simulated",
                "is_synthetic": True,
                "original_final_test_accessed": False,
                "aggregate_metrics": manifest["aggregate_metrics"],
                "output_dir": str(args.output_dir),
            }
        elif args.command == "compare-policies":
            from credit_risk.decisioning.runner import run_policy_comparison

            config = load_config(args.config, PolicyComparisonConfig)
            result = run_policy_comparison(
                args.csv, args.run_dir, args.validation_dir, args.output_dir, config
            )
        elif args.command == "expected-loss":
            from credit_risk.portfolio.loss_runner import run_expected_loss

            config = load_config(args.config, ExpectedLossConfig)
            loss = run_expected_loss(
                args.accounts,
                args.history,
                args.output_dir,
                config,
                args.as_of,
                args.source_manifest,
            )
            result = {
                "status": "calculated",
                "is_synthetic": loss["is_synthetic"],
                "coverage": loss["coverage"],
                "scenario_summary": loss["scenario_summary"],
                "output_dir": str(args.output_dir),
            }
        elif args.command == "analyze-portfolio":
            from credit_risk.portfolio.runner import run_portfolio_analytics

            config = load_config(args.config, PortfolioAnalyticsConfig)
            analytics = run_portfolio_analytics(
                args.accounts,
                args.history,
                args.output_dir,
                config,
                args.as_of,
                args.source_manifest,
            )
            result = {
                "status": "analyzed",
                "is_synthetic": analytics["is_synthetic"],
                "as_of": analytics["as_of"],
                "vintage_checkpoints": analytics["vintage_checkpoints"],
                "roll_diagnostics": analytics["roll_diagnostics"],
                "output_dir": str(args.output_dir),
            }
        elif args.command == "validate":
            from credit_risk.validation.runner import run_validation

            config = load_config(args.config, ValidationConfig)
            validation = run_validation(args.csv, args.run_dir, args.output_dir, config)
            result = {
                "status": "validated",
                "selected_methods": validation["selection"]["selected_methods"],
                "preferred_candidate": validation["selection"]["preferred_candidate"],
                "final_metrics": validation["final_metrics"],
                "output_dir": str(args.output_dir),
            }
        elif args.command == "train":
            from credit_risk.models.pd import run_origination_experiment

            config = load_config(args.config, ModelConfig)
            development = load_config(args.development_config, DevelopmentConfig)
            logger = configure_logging(development.log_level)
            seed = development.seed if args.seed is None else args.seed
            result = run_origination_experiment(args.csv, args.output_dir, config, seed)
        else:
            portfolio = load_portfolio_csv(args.accounts, args.history)
            config = load_config(args.config, TargetConfig)
            targets = build_forward_targets(portfolio.history, config, args.as_of)
            metadata_path = args.output.with_suffix(".manifest.json")
            if args.output.exists() or metadata_path.exists():
                raise FileExistsError("Targets already exist; choose a new output path")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            targets.to_csv(args.output, index=False, mode="x", date_format="%Y-%m-%d")
            result = {
                "version": __version__,
                "config": config.model_dump(),
                "as_of": args.as_of or str(portfolio.history.observation_date.max().date()),
                "rows": len(targets),
                "status_counts": {
                    name: int(count) for name, count in targets.status.value_counts().items()
                },
                "source_sha256": {
                    "accounts": sha256(args.accounts.read_bytes()).hexdigest(),
                    "history": sha256(args.history.read_bytes()).hexdigest(),
                },
                "target_sha256": sha256(args.output.read_bytes()).hexdigest(),
                "is_synthetic": bool(portfolio.history.is_synthetic.all()),
            }
            metadata_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        logger.error("Command failed", extra={"details": {"error": str(exc)}})
        return 2
    logger.info("Command completed", extra={"details": {"command": args.command}})
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
