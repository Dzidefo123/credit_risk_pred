"""Fail-closed loading of an explicit, repository-relative evidence allowlist."""

import hashlib
import json
import math
from pathlib import Path

import yaml

from credit_risk.validation.holdout_registry import verify_repository_registry

TARGET = "two-year serious-delinquency outcome"
FROZEN = {
    "src/credit_risk/models/pd.py",
    "src/credit_risk/features/origination.py",
    "src/credit_risk/utils/config.py",
    "src/credit_risk/validation/metrics.py",
    "src/credit_risk/models/calibration.py",
}
SOURCES = {
    "reports/model_validation/pd_diagnostics.json": "TRAINING_ONLY",
    "reports/model_validation/calibration_study.json": "TRAINING_ONLY",
    "reports/model_validation/explainability_stability.json": "TRAINING_ONLY",
    "reports/model_validation/PD_DIAGNOSTICS.md": "TRAINING_ONLY",
    "reports/model_validation/CALIBRATION_STUDY.md": "TRAINING_ONLY",
    "reports/model_validation/EXPLAINABILITY_STABILITY.md": "TRAINING_ONLY",
    "reports/model_validation/pd_diagnostics.png": "TRAINING_ONLY",
    "reports/model_validation/calibration_comparison.png": "TRAINING_ONLY",
    "reports/model_validation/logistic_coefficients.png": "TRAINING_ONLY",
    "reports/model_validation/logistic_coefficient_stability.png": "TRAINING_ONLY",
    "reports/model_validation/xgboost_shap_summary.png": "TRAINING_ONLY",
    "reports/model_validation/xgboost_shap_stability.png": "TRAINING_ONLY",
    "reports/model_validation/cross_model_ranks.png": "TRAINING_ONLY",
    "reports/model_validation/shap_dependence.png": "TRAINING_ONLY",
    "reports/data/TRAINING_DATA_AUDIT.json": "DESCRIPTIVE",
    "reports/data/DATA_AUDIT.md": "DESCRIPTIVE",
    "docs/DATA_DICTIONARY.md": "DESCRIPTIVE",
    "docs/TARGET_DEFINITION.md": "DESCRIPTIVE",
    "reports/data/DATASET_SUITABILITY.md": "DESCRIPTIVE",
    "reports/data/LEAKAGE_REVIEW.md": "DESCRIPTIVE",
    "reports/phase4_experiment.json": "GOVERNANCE",
    "reports/phase5_validation_summary.json": "HISTORICAL_LOCKED_HOLDOUT",
    "reports/holdout_registry.json": "GOVERNANCE",
    "docs/HOLDOUT_REGISTRY.md": "GOVERNANCE",
    "scripts/smoke_container.py": "GOVERNANCE",
    "tests/test_container_smoke.py": "GOVERNANCE",
    "configs/model.yaml": "GOVERNANCE",
    "pyproject.toml": "GOVERNANCE",
    "uv.lock": "GOVERNANCE",
    ".github/workflows/ci.yml": "GOVERNANCE",
    ".gitattributes": "GOVERNANCE",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "Non-finite quantitative evidence")


def close(a, b, name, tolerance=1e-10):
    require(
        isinstance(a, (int, float))
        and not isinstance(a, bool)
        and isinstance(b, (int, float))
        and not isinstance(b, bool)
        and math.isfinite(a)
        and math.isfinite(b)
        and math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance),
        f"Conflicting evidence: {name}",
    )


def ranks(values):
    return {
        key: 1
        + sum(v > value for v in values.values())
        + (sum(v == value for v in values.values()) - 1) / 2
        for key, value in values.items()
    }


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.payloads = {}
        self.data = {}
        self.categories = dict(SOURCES)
        for name in SOURCES:
            payload = self.read(name)
            if name.endswith(".json"):
                try:
                    self.data[name] = json.loads(payload, object_pairs_hook=unique_object)
                    require(isinstance(self.data[name], dict), "Evidence JSON must be an object")
                    finite_tree(self.data[name])
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"Malformed evidence {name}: {exc}") from exc
        self.task4 = self.data["reports/model_validation/pd_diagnostics.json"]
        self.task5 = self.data["reports/model_validation/calibration_study.json"]
        self.task6 = self.data["reports/model_validation/explainability_stability.json"]
        self.audit = self.data["reports/data/TRAINING_DATA_AUDIT.json"]
        self.historical = self.data["reports/phase5_validation_summary.json"]
        try:
            self.experiment = self.data["reports/phase4_experiment.json"]["experiment"]
            self.validate()
        except (
            KeyError,
            TypeError,
            IndexError,
            OverflowError,
            StopIteration,
            ZeroDivisionError,
            yaml.YAMLError,
        ) as exc:
            raise ValueError(f"Malformed/incomplete evidence schema: {exc}") from exc

    def read(self, name):
        path = self.root / name
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts,
            f"Unsafe evidence path: {name}",
        )
        require(path.resolve().is_relative_to(self.root), f"Evidence escapes repository: {name}")
        require(
            path.resolve().suffix == path.suffix
            and not path.resolve().is_relative_to(self.root / "artifacts")
            and not path.resolve().is_relative_to(self.root / "data"),
            f"Forbidden evidence payload: {name}",
        )
        require(path.is_file(), f"Missing required evidence: {name}")
        payload = path.read_bytes()
        self.payloads[name] = payload
        return payload

    def verify_hash(self, name, expected):
        # Only declared aggregates or Python sources can enter a digest check.
        require(
            name in SOURCES
            or (name.startswith(("src/credit_risk/", "scripts/")) and name.endswith(".py")),
            f"Forbidden evidence reference: {name}",
        )
        payload = self.payloads.get(name)
        if payload is None:
            payload = self.read(name)
            self.categories[name] = "GOVERNANCE"
        candidates = [payload]
        if name not in FROZEN and Path(name).suffix in {".py", ".md", ".json"}:
            candidates.append(payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        require(
            expected in [hashlib.sha256(v).hexdigest() for v in candidates],
            f"Evidence hash mismatch: {name}",
        )

    def validate(self):
        t4, t5, t6 = self.task4, self.task5, self.task6
        for item in (t4, t5, t6):
            require(
                item["schema_version"] == 1 and item["target"] == TARGET,
                "Conflicting schema/target definition",
            )
            require(
                item["provenance"]["partition"] == "original train only"
                and item["provenance"]["original_holdout_access"]
                == "excluded at CSV parsing; no frozen model/prediction loading",
                "Training-only evidence reports holdout access",
            )
            require(
                item["rows"] == t4["rows"] and item["provenance"] == t4["provenance"],
                "Conflicting training population/provenance",
            )
        require(
            t4["config"] == t5["model_config"] == t6["model_config"],
            "Conflicting candidate configuration",
        )
        config = yaml.safe_load(self.payloads["configs/model.yaml"])
        # Pydantic names/defaults are fully expanded in the evidence JSON.
        for key, value in config.items():
            require(t4["config"][key] == value, f"Configuration differs from evidence: {key}")
        require(
            self.historical["target_semantics"] == self.experiment["target_semantics"]
            and "two-year" in self.historical["target_semantics"],
            "Conflicting historical target definition",
        )
        target_doc = self.payloads["docs/TARGET_DEFINITION.md"].decode("utf-8")
        require(
            "SeriousDlqin2yrs" in target_doc
            and "next-two-years" in target_doc
            and "12-month PD" in target_doc
            and "Unsupported" in target_doc,
            "Conflicting documentary target definition",
        )
        require(
            self.audit["target"]["name"] == "SeriousDlqin2yrs"
            and self.audit["rows"] == t4["rows"]
            and self.audit["target"]["positive_count"] == t4["events"],
            "Conflicting audit population/target",
        )
        for item in (self.historical, self.experiment):
            require(
                item["source_sha256"] == t4["provenance"]["source_sha256"],
                "Conflicting historical source identity",
            )
        require(
            self.audit["provenance"]["source_sha256"] == self.experiment["source_sha256"]
            and self.audit["provenance"]["split_assignments_sha256"]
            == t4["provenance"]["assignments_sha256"],
            "Conflicting audit source/split",
        )
        retained = self.historical["final_metrics"]["xgboost"]["sigmoid"]
        for item, key in (
            (t4, "historical_locked_holdout_result"),
            (t5, "historical_metrics"),
            (t6, "historical_metrics"),
        ):
            for metric in ("roc_auc", "brier", "log_loss"):
                # These older summaries retain six decimal places.
                close(item[key][metric], round(retained[metric], 6), "historical " + metric)
        require(
            t5["recommendations"]["xgboost"]["recommendation"] == "RAW"
            and t5["recommendations"]["logistic_regression"]["recommendation"] == "ISOTONIC",
            "Development conclusion no longer matches calibration evidence",
        )
        for model in ("logistic_regression", "xgboost"):
            c = t4["candidates"][model]
            for metric in c["oof"]["metrics"]:
                if metric in c["cv_summary"]:
                    vals = c["cv_summary"][metric]["fold_values"]
                    close(c["cv_summary"][metric]["mean"], sum(vals) / len(vals), model + metric)
            for method in ("sigmoid", "isotonic"):
                paired = t5["paired_comparisons"][model][method]
                for metric, difference in paired["point_differences"].items():
                    a = t5["models"][model][method]["pooled_oof"]["metrics"][metric]
                    b = t5["models"][model]["raw"]["pooled_oof"]["metrics"][metric]
                    close(difference, a - b, f"Task 5 {model}/{method}/{metric}")
        lr = t4["candidates"]["logistic_regression"]["oof"]["metrics"]
        xgb = t4["candidates"]["xgboost"]["oof"]["metrics"]
        require(
            xgb["roc_auc"] > lr["roc_auc"]
            and xgb["brier"] < lr["brier"]
            and xgb["log_loss"] < lr["log_loss"],
            "Champion conclusion conflicts with Task 4 evidence",
        )
        require(t4["fold_count"] == len(t4["fold_design"]) == 5, "Unsupported fold design")
        close(t4["observed_event_rate"], t4["events"] / t4["rows"], "event fraction")
        for folder, metadata in (("phase4", self.experiment), ("phase5", self.historical)):
            for name, digest in metadata["artifacts_sha256"].items():
                require(
                    len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                    f"Malformed recorded frozen digest: {folder}/{name}",
                )
        locations = {
            "pd.py": "models/pd.py",
            "origination.py": "features/origination.py",
            "config.py": "utils/config.py",
            "metrics.py": "validation/metrics.py",
        }
        for name, digest in self.experiment["source_code_sha256"].items():
            self.verify_hash("src/credit_risk/" + locations[name], digest)
        for other in (t5, t6):
            for a, b in zip(t4["fold_design"], other["fold_design"], strict=True):
                for key in (
                    "fold",
                    "evaluation_rows",
                    "evaluation_events",
                    "evaluation_positions_sha256",
                    "group_overlap",
                ):
                    alias = {"fold": "outer_fold", "group_overlap": "outer_group_overlap"}
                    other_key = alias.get(key, key) if other is t5 else key
                    require(a[key] == b[other_key], "Conflicting evaluation fold identity")
        features = self.experiment["feature_names"]
        require(len(features) == 10 and len(set(features)) == 10, "Invalid predictor allowlist")
        tree = {r["feature"]: r for r in t6["xgboost_stability"]}
        for name, row in tree.items():
            means = [
                next(f["mean_abs_shap"] for f in fold["features"] if f["feature"] == name)
                for fold in t6["xgboost_folds"]
            ]
            counts = [f["evaluation_rows"] for f in t6["xgboost_folds"]]
            close(
                row["pooled_mean_abs_shap"],
                sum(v * n for v, n in zip(means, counts, strict=True)) / sum(counts),
                "Task 6 SHAP magnitude " + name,
            )
        for i, fold in enumerate(t6["xgboost_folds"]):
            feature_ranks = ranks({r["feature"]: r["mean_abs_shap"] for r in fold["features"]})
            for name, row in tree.items():
                close(row["fold_ranks"][i], feature_ranks[name], "Task 6 fold rank " + name)
        for name, column in self.audit["columns"].items():
            close(
                column["missing_fraction"],
                column["missing_count"] / self.audit["rows"],
                "Audit missing fraction " + name,
            )
        tree_ranks = ranks({n: r["pooled_mean_abs_shap"] for n, r in tree.items()})
        for name, row in tree.items():
            close(row["global_rank"], tree_ranks[name], "Task 6 rank " + name)
        logistic = {r["feature"]: r for r in t6["logistic_stability"]}
        for name, row in logistic.items():
            values = [v for v in row["fold_coefficients"] if v is not None]
            close(
                row["coefficient"]["mean"], sum(values) / len(values), "Task 6 coefficient " + name
            )
        lr_ranks = ranks(
            {n: sum(abs(v) for v in logistic[n]["fold_coefficients"]) / 5 for n in features}
        )
        original_tree_ranks = ranks({n: tree[n]["pooled_mean_abs_shap"] for n in features})
        require(
            {r["feature"] for r in t6["cross_model_comparison"]} == set(features),
            "Conflicting cross-model predictor list",
        )
        for row in t6["cross_model_comparison"]:
            name = row["feature"]
            close(row["logistic_standardized_rank"], lr_ranks[name], "Logistic rank " + name)
            close(row["xgboost_shap_rank"], original_tree_ranks[name], "XGB rank " + name)
        pairs = t6["xgboost_rank_agreement"]["pairs"]
        close(
            t6["xgboost_rank_agreement"]["mean_rho"],
            sum(p["spearman_rho"] for p in pairs) / len(pairs),
            "Task 6 rank agreement",
        )
        for row in logistic.values():
            positive = any(
                v is not None and v > row["sign_tolerance"] for v in row["fold_coefficients"]
            )
            negative = any(
                v is not None and v < -row["sign_tolerance"] for v in row["fold_coefficients"]
            )
            require(row["sign_flip"] == (positive and negative), "Conflicting logistic sign flip")
        for item, key in (
            (t4, "reused_source_sha256"),
            (t5, "source_code_sha256"),
            (t6, "source_code_sha256"),
            (t5, "preserved_task4_and_registry_sha256"),
            (t6, "preserved_evidence_sha256"),
        ):
            for name, expected in item[key].items():
                self.verify_hash(name, expected)
        for name, expected in t4["diagnostic_source_sha256"].items():
            self.verify_hash("src/credit_risk/validation/" + name, expected)
        name = "src/credit_risk/data/audit.py"
        payload = self.read(name)
        self.categories[name] = "GOVERNANCE"
        require(
            hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()
            == self.audit["provenance"]["audit_module_source_sha256"],
            "Audit module hash mismatch",
        )
        # Registry verifier reads the ledger only, never applicant rows or predictions.
        self.ledger = verify_repository_registry(self.root / "reports/holdout_registry.json")
        self.verify_hash("reports/holdout_registry.json", t4["provenance"]["registry_sha256"])
        self.verify_hash(
            "src/credit_risk/models/calibration.py",
            t5["source_code_sha256"]["src/credit_risk/models/calibration.py"],
        )

    def manifest_sources(self):
        return [
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "evidence_category": self.categories[name],
                "sha256_lf": hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest(),
            }
            for name, payload in sorted(self.payloads.items())
        ]

    def assert_unchanged(self):
        for name, payload in self.payloads.items():
            require(
                (self.root / name).read_bytes() == payload,
                f"Evidence mutated during generation: {name}",
            )
