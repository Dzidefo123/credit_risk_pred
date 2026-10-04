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
from credit_risk.utils.config import (
    DecisionPolicyConfig,
    DevelopmentConfig,
    ModelConfig,
    load_config,
)
from credit_risk.utils.logging import configure_logging


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
