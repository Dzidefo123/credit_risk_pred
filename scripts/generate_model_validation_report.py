"""Generate the MDVR from committed aggregates; no raw data or model access."""

import argparse
from pathlib import Path

from credit_risk.reporting.mdvr import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--generated-at", help="Optional fixed timezone-aware ISO timestamp")
    args = parser.parse_args()
    try:
        _, manifest = generate(args.root, generated_at=args.generated_at)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"MDVR generation failed: {exc}\n")
    print(
        f"Generated {manifest['report_path']}; evidence checks passed; "
        "no raw data access, model loading or training."
    )


if __name__ == "__main__":
    main()
