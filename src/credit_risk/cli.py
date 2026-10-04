"""Phase 2 CLI for validating configuration without executing legacy code."""

import argparse
import json
from pathlib import Path

from credit_risk import __version__
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
    check = commands.add_parser("check-config", help="Validate all three YAML configuration files")
    check.add_argument("--config-dir", type=Path, default=Path("configs"))
    args = parser.parse_args(argv)
    logger = configure_logging()
    try:
        config_dir = args.config_dir.resolve()
        development = load_config(config_dir / "development.yaml", DevelopmentConfig)
        load_config(config_dir / "model.yaml", ModelConfig)
        load_config(config_dir / "decision_policy.yaml", DecisionPolicyConfig)
        logger = configure_logging(development.log_level)
        paths = development.paths.resolve(config_dir.parent)
    except (OSError, ValueError) as exc:
        logger.error("Configuration validation failed", extra={"details": {"error": str(exc)}})
        return 2
    logger.info("Configuration validated", extra={"details": {"seed": development.seed}})
    print(
        json.dumps(
            {
                "status": "valid",
                "version": __version__,
                "paths": {name: str(path) for name, path in paths.items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
