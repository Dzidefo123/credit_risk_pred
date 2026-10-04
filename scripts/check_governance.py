"""Check governance claims against committed evidence without fitting/loading/scoring models."""

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

from credit_risk.decisioning.settings import PolicyComparisonConfig
from credit_risk.monitoring.settings import MonitoringConfig
from credit_risk.utils.config import load_config

EVIDENCE = (
    "reports/phase4_experiment.json",
    "reports/phase5_validation_summary.json",
    "reports/phase7_expected_loss_summary.json",
    "reports/phase8_policy_summary.json",
    "reports/phase9_reject_summary.json",
    "reports/phase10_monitoring_summary.json",
    "reports/phase11_integration_summary.json",
)
DOCUMENTS = (
    "docs/architecture.md",
    "reports/model_card.md",
    "reports/model_risk_register.md",
    "reports/validation_report.md",
    "reports/monitoring_report.md",
    "reports/credit_policy.md",
    "docs/model_governance.md",
    "reports/phase13_governance_report.md",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_snapshot(root):
    evidence = {name: read_json(root / name) for name in EVIDENCE}
    training = evidence[EVIDENCE[0]]["experiment"]
    validation = evidence[EVIDENCE[1]]
    portfolio = evidence[EVIDENCE[2]]
    policy_run = evidence[EVIDENCE[3]]
    monitoring = evidence[EVIDENCE[5]]
    integration = evidence[EVIDENCE[6]]
    candidate = validation["selection"]["preferred_candidate"]
    method = validation["selection"]["selected_methods"][candidate]
    model_sha = validation["artifacts_sha256"][f"{candidate}_selected.joblib"]
    if not (
        training["source_sha256"]
        == validation["source_sha256"]
        == policy_run["source_sha256"]
        == monitoring["reference"]["source_sha256"]
    ):
        raise ValueError("Origination source identities disagree")
    if not (
        model_sha
        == policy_run["model_sha256"]
        == monitoring["reference"]["model_identity"]["model_sha256"]
        == integration["responses"]["score"]["model_sha256"]
    ):
        raise ValueError("Selected model identities disagree")
    policies = load_config(root / "configs/credit_strategy.yaml", PolicyComparisonConfig)
    baseline = next(p for p in policies.policies if p.name == "baseline").model_dump()
    thresholds = load_config(root / "configs/monitoring.yaml", MonitoringConfig).model_dump()
    if thresholds != monitoring["reference"]["config"]:
        raise ValueError("Current monitoring configuration differs from frozen reference evidence")
    if baseline != next(p for p in policy_run["config"]["policies"] if p["name"] == "baseline"):
        raise ValueError("Current baseline policy differs from recorded policy evidence")
    return {
        "schema_version": 1,
        "documentation_revision": "phase13",
        "review_date": "2026-10-04",
        "software_package_version": tomllib.loads(
            (root / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["version"],
        "use_status": "research_only",
        "production_approval": {
            "status": "not_approved",
            "model_owner": None,
            "independent_validator": None,
            "credit_policy_approver": None,
            "deployment_approver": None,
        },
        "origination": {
            "candidate": candidate,
            "calibration_method": method,
            "model_sha256": model_sha,
            "benchmark_model_sha256": validation["artifacts_sha256"][
                "logistic_regression_selected.joblib"
            ],
            "selection_artifact_sha256": validation["artifacts_sha256"]["selection.json"],
            "selection_lock_sha256": validation["selection_sha256"],
            "source_sha256": validation["source_sha256"],
            "target_semantics": validation["target_semantics"],
            "validation_type": validation["validation_type"],
            "partitions": validation["partitions"],
            "selected_methods": validation["selection"]["selected_methods"],
            "final_metrics": {
                name: validation["final_metrics"][name][calibration]
                for name, calibration in validation["selection"]["selected_methods"].items()
            },
            "reliability": {
                name: validation["reliability_summary"][name][calibration]
                for name, calibration in validation["selection"]["selected_methods"].items()
            },
            "uncertainty": validation["uncertainty"],
            "missingness_by_partition": validation["missingness_by_partition"],
            "feature_names": training["feature_names"],
            "frozen_training_versions": training["versions"],
        },
        "portfolio_benchmark": {
            "is_synthetic": portfolio["is_synthetic"],
            "pd_model": portfolio["pd_model"],
            "config": portfolio["config"],
            "production_approved": False,
        },
        "policy": {"baseline": baseline, "demonstration_only": True},
        "monitoring": {
            "config": thresholds,
            "reference_partition": monitoring["reference"]["reference_partition"],
            "temporal_reference": monitoring["reference"]["temporal_reference"],
            "control_status": monitoring["control"]["status"],
            "perturbation_status": monitoring["controlled_perturbation"]["status"],
            "perturbation_alert_count": len(monitoring["controlled_perturbation"]["alerts"]),
        },
        "open_findings": [f"GOV-{i:03d}" for i in range(1, 10)],
        "evidence": [
            {"path": name, "canonical_json_sha256": canonical_digest(value)}
            for name, value in evidence.items()
        ],
    }


def partitions_table(snapshot):
    rows = ["| Partition | Rows | Events | Purpose |", "| --- | ---: | ---: | --- |"]
    purposes = {
        "train": "Base model / preprocessing fit",
        "development": "Candidate / calibration selection",
        "calibration": "Calibrator fit",
        "test": "Consumed final evaluation",
    }
    for name, item in snapshot["origination"]["partitions"].items():
        rows.append(f"| {name} | {item['rows']:,} | {item['bad_count']:,} | {purposes[name]} |")
    return "\n".join(rows)


def metrics_table(snapshot):
    rows = [
        "| Selected model | AUC | Gini | KS | Average precision | Brier | Log loss | ECE | O/E |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    origin = snapshot["origination"]
    keys = ("roc_auc", "gini", "ks", "average_precision", "brier", "log_loss")
    for name, method in origin["selected_methods"].items():
        values = [origin["final_metrics"][name][k] for k in keys]
        values += [
            origin["reliability"][name]["ece"],
            origin["reliability"][name]["observed_expected_ratio"],
        ]
        row = " | ".join(f"{v:.6f}" for v in values)
        rows.append(f"| {name} / {method} | {row} |")
    return "\n".join(rows)


def monitoring_table(snapshot):
    rows = ["| Alert channel | Warning | Critical | Unit |", "| --- | ---: | ---: | --- |"]
    units = {
        "psi": "PSI / feature CSI equivalent",
        "missingness": "Absolute proportion change",
        "ks": "Numeric KS distance",
        "out_of_range": "Fraction outside reference support",
        "pd_mean": "Absolute PD proportion change",
        "score_mean": "Absolute score points change",
    }
    for key, unit in units.items():
        threshold = snapshot["monitoring"]["config"][key]
        rows.append(f"| {key} | {threshold['warning']:g} | {threshold['critical']:g} | {unit} |")
    return "\n".join(rows)


def check_block(path, marker, expected):
    text = path.read_text(encoding="utf-8")
    pattern = rf"<!-- evidence:{marker} -->\s*(.*?)\s*<!-- /evidence:{marker} -->"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches != [expected]:
        raise ValueError(f"Stale or missing evidence block: {path.name}/{marker}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--register", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    expected = build_snapshot(root)
    register = args.register or root / "reports/governance_register.json"
    if read_json(register) != expected:
        raise ValueError("Governance register differs from recorded evidence/current configuration")
    check_block(root / "reports/model_card.md", "partitions", partitions_table(expected))
    check_block(root / "reports/model_card.md", "final_metrics", metrics_table(expected))
    check_block(root / "reports/monitoring_report.md", "thresholds", monitoring_table(expected))
    risks = (root / "reports/model_risk_register.md").read_text(encoding="utf-8")
    for finding in expected["open_findings"]:
        if finding not in risks:
            raise ValueError(f"Missing documented finding: {finding}")
    links = 0
    for name in DOCUMENTS:
        path = root / name
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            url = urlsplit(target)
            if url.scheme or not url.path:
                continue
            destination = (path.parent / unquote(url.path)).resolve()
            if not destination.is_relative_to(root) or not destination.is_file():
                raise ValueError(f"Broken/outside-project evidence link: {name}: {target}")
            links += 1
    print(
        json.dumps(
            {
                "status": "passed",
                "evidence_files": len(EVIDENCE),
                "documents": len(DOCUMENTS),
                "local_links": links,
                "use_status": expected["use_status"],
                "fitting_performed": False,
                "final_test_scored": False,
            }
        )
    )


if __name__ == "__main__":
    main()
