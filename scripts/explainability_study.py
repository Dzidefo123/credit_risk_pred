"""Generate Task 6 training-only explanations without saving models or row matrices."""

import argparse
import json
from pathlib import Path

from credit_risk.explainability.reporting import generate_figures, render_report
from credit_risk.explainability.study import run_explainability_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    result = run_explainability_study(
        args.root, args.source, progress=lambda message: print(message, flush=True)
    )
    output = args.root / "reports/model_validation"
    output.mkdir(parents=True, exist_ok=True)
    generate_figures(result, output)
    (output / "explainability_stability.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "EXPLAINABILITY_STABILITY.md").write_text(render_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "rank_agreement": result["xgboost_rank_agreement"]["mean_rho"],
                "sign_flips": [
                    r["feature"] for r in result["logistic_stability"] if r["sign_flip"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
