"""MDVR traceability, failure modes and explicit no-data/no-model/no-fit guards."""

import hashlib
import json
import re
from pathlib import Path

import joblib
import pandas as pd
import pytest

from credit_risk.models import pd as models
from credit_risk.reporting import mdvr
from credit_risk.reporting.evidence import FROZEN, SOURCES, Evidence

ROOT = Path(__file__).resolve().parents[1]
STAMP = "2026-10-06T12:00:00+00:00"


@pytest.fixture
def evidence_root(tmp_path):
    # Only committed aggregate evidence and source bytes, no datasets/bundles.
    evidence = Evidence(ROOT)
    names = set(evidence.payloads) | {
        "scripts/generate_model_validation_report.py",
        "src/credit_risk/reporting/__init__.py",
        "src/credit_risk/reporting/evidence.py",
        "src/credit_risk/reporting/mdvr.py",
    }
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    return tmp_path


def change(root, name, fn):
    path = root / name
    value = json.loads(path.read_text(encoding="utf-8"))
    fn(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_deterministic_generation_manifest_and_no_mutation(evidence_root):
    before = {p: p.read_bytes() for p in evidence_root.rglob("*") if p.is_file()}
    a, ma = mdvr.generate(evidence_root, generated_at=STAMP, commit="evidence-baseline")
    b, mb = mdvr.generate(evidence_root, generated_at=STAMP, commit="evidence-baseline")
    assert a == b and ma == mb
    assert all(p.read_bytes() == value for p, value in before.items())
    assert (evidence_root / mdvr.REPORT).read_bytes() == a.encode("utf-8")
    assert ma["report_sha256"] == hashlib.sha256(a.encode("utf-8")).hexdigest()
    assert json.loads((evidence_root / mdvr.MANIFEST).read_text(encoding="utf-8")) == ma
    assert ma["verification"]["holdout_access"] is False
    assert ma["registry"]["historical_anchor_verified"]
    assert ma["registry"]["entries"][0]["status"] == "consumed"
    assert ma["quantitative_evidence"]["historical"]["regenerated"] is False
    for source in ma["source_evidence"]:
        assert (
            source["sha256"]
            == hashlib.sha256((evidence_root / source["path"]).read_bytes()).hexdigest()
        )


def test_expected_timestamp_variation_only(evidence_root):
    a, ma = mdvr.build(evidence_root, generated_at=STAMP, commit="baseline")
    b, mb = mdvr.build(evidence_root, generated_at="2026-10-07T12:00:00+00:00", commit="baseline")
    assert a == b
    assert ma.pop("generated_at") != mb.pop("generated_at")
    assert ma == mb


def test_categories_target_historical_wording_and_relative_links(evidence_root):
    report, manifest = mdvr.build(evidence_root, generated_at=STAMP)
    categories = {v["evidence_category"] for v in manifest["source_evidence"]}
    assert categories == {"TRAINING_ONLY", "HISTORICAL_LOCKED_HOLDOUT", "DESCRIPTIVE", "GOVERNANCE"}
    for n in range(1, 26):
        assert f"## {n}. " in report
    assert "two-year serious-delinquency outcome" in report
    assert "should not be interpreted as a regulatory 12-month or lifetime" in report
    assert "already-consumed holdout" in report and "not regenerated" in report
    assert "0.868152" in report and "0.048545" in report and "0.176030" in report
    assert "Visible IFRS 9 boundary" in report and "SICR" in report
    assert "pristine" in report.lower() and "unverified" in report
    assert "Raw XGBoost is the development champion" in report
    assert "C:\\" not in report and str(evidence_root) not in report
    for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", report):
        assert not Path(link).is_absolute() and not link.startswith("file:")
        assert (
            (evidence_root / Path(mdvr.REPORT).parent / link)
            .resolve()
            .is_relative_to(evidence_root)
        )
        if link != "model_validation_manifest.json":
            assert (evidence_root / Path(mdvr.REPORT).parent / link).is_file(), link
    for claim in manifest["claims"].values():
        assert claim["path"] in SOURCES
        if "pointer" in claim:
            value = json.loads((evidence_root / claim["path"]).read_text(encoding="utf-8"))
            for key in claim["pointer"].split("/")[1:]:
                value = value[key]


def test_metrics_calibration_and_ranks_are_loaded_not_hardcoded(evidence_root):
    e = Evidence(evidence_root)
    report, manifest = mdvr.build(evidence_root, generated_at=STAMP)
    for model, candidate in e.task4["candidates"].items():
        assert manifest["quantitative_evidence"]["task4"]["results"][model] == candidate["oof"]
        for key in ("roc_auc", "brier", "log_loss"):
            assert mdvr.num(candidate["oof"]["metrics"][key]) in report
    assert (
        manifest["quantitative_evidence"]["task5"]["recommendations"] == e.task5["recommendations"]
    )
    for model, methods in e.task5["models"].items():
        for method, value in methods.items():
            assert (
                manifest["quantitative_evidence"]["task5"]["results"][model][method]
                == value["pooled_oof"]
            )
    assert (
        manifest["quantitative_evidence"]["task6"]["xgboost_stability"]
        == e.task6["xgboost_stability"]
    )
    assert (
        manifest["quantitative_evidence"]["task6"]["rank_agreement"]
        == e.task6["xgboost_rank_agreement"]
    )


@pytest.mark.parametrize(
    "name", ["reports/model_validation/pd_diagnostics.json", "docs/TARGET_DEFINITION.md", "uv.lock"]
)
def test_missing_evidence_fails_before_output(evidence_root, name):
    (evidence_root / name).unlink()
    with pytest.raises(ValueError, match="Missing required evidence"):
        mdvr.generate(evidence_root)
    assert not (evidence_root / mdvr.REPORT).exists()


@pytest.mark.parametrize(
    "payload", ["{", "[]", '{"schema_version":1,"schema_version":2}', '{"value":NaN}', "{}"]
)
def test_malformed_evidence_fails(evidence_root, payload):
    (evidence_root / "reports/model_validation/pd_diagnostics.json").write_text(
        payload, encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Malformed|Duplicate|Non-finite|object"):
        mdvr.generate(evidence_root)


@pytest.mark.parametrize(
    "case", ["target", "historical", "calibration", "rank", "population", "config", "fold"]
)
def test_conflicting_evidence_fails(evidence_root, case):
    paths = {
        "target": "pd_diagnostics",
        "historical": "pd_diagnostics",
        "calibration": "calibration_study",
        "rank": "explainability_stability",
        "population": "explainability_stability",
        "config": "calibration_study",
        "fold": "calibration_study",
    }

    def mutate(d):
        if case == "target":
            d["target"] = "12-month regulatory PD"
        elif case == "historical":
            d["historical_locked_holdout_result"]["roc_auc"] = 0.5
        elif case == "calibration":
            d["models"]["logistic_regression"]["isotonic"]["pooled_oof"]["metrics"]["brier"] += 0.01
        elif case == "rank":
            d["xgboost_stability"][0]["global_rank"] += 1
        elif case == "population":
            d["rows"] += 1
        elif case == "config":
            d["model_config"]["classification_threshold"] += 0.01
        elif case == "fold":
            d["fold_design"][0]["evaluation_positions_sha256"] = "different"

    change(evidence_root, "reports/model_validation/" + paths[case] + ".json", mutate)
    with pytest.raises(ValueError, match="Conflicting"):
        mdvr.generate(evidence_root)


def test_frozen_source_hash_requires_exact_bytes(evidence_root):
    name = "src/credit_risk/models/pd.py"
    (evidence_root / name).write_bytes((evidence_root / name).read_bytes().replace(b"\r\n", b"\n"))
    with pytest.raises(ValueError, match="hash mismatch"):
        mdvr.generate(evidence_root)


def test_git_line_ending_conversion_of_non_frozen_evidence(evidence_root):
    for path in evidence_root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".json"}:
            name = path.relative_to(evidence_root).as_posix()
            if name not in FROZEN:
                path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    report, _ = mdvr.generate(evidence_root, generated_at=STAMP)
    assert "Final Development Conclusion" in report


def test_reset_registry_is_blocked(evidence_root):
    change(evidence_root, "reports/holdout_registry.json", lambda d: d.update(entries=[]))
    with pytest.raises(ValueError, match="hash mismatch|anchor"):
        mdvr.generate(evidence_root)


def test_hash_reference_cannot_request_raw_holdout(evidence_root):
    change(
        evidence_root,
        "reports/model_validation/pd_diagnostics.json",
        lambda d: d["reused_source_sha256"].update(
            {"artifacts/phase5-validation-001/test_predictions.csv": "0" * 64}
        ),
    )
    with pytest.raises(ValueError, match="Forbidden evidence reference"):
        mdvr.generate(evidence_root)


def test_no_raw_consumed_holdout_no_model_loading_and_no_training(evidence_root, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Report attempted raw-data/model/training access")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(models, "build_pd_model", forbidden)
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(XGBClassifier, "fit", forbidden)
    real_open = Path.open
    opened = []

    def guarded_open(path, *args, **kwargs):
        assert path.suffix not in {".csv", ".joblib", ".pkl", ".pickle"}
        assert not path.resolve().is_relative_to(evidence_root / "artifacts")
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    mdvr.generate(evidence_root, generated_at=STAMP)
    assert opened and not (evidence_root / "artifacts").exists()


def test_evidence_mutation_is_detected_before_output(evidence_root, monkeypatch):
    original = mdvr.render

    def mutate(e, metadata):
        report = original(e, metadata)
        (e.root / "docs/DATA_DICTIONARY.md").write_text("mutated", encoding="utf-8")
        return report

    monkeypatch.setattr(mdvr, "render", mutate)
    with pytest.raises(ValueError, match="mutated"):
        mdvr.generate(evidence_root)
    assert not (evidence_root / mdvr.REPORT).exists()


def test_timestamp_requires_timezone(evidence_root):
    with pytest.raises(ValueError, match="timezone-aware"):
        mdvr.generate(evidence_root, generated_at="2026-10-06T12:00:00")


def test_unexpected_readable_schema_failure_is_clear(evidence_root):
    change(
        evidence_root,
        "reports/model_validation/explainability_stability.json",
        lambda d: d["xgboost_stability"][0].pop("fold_ranks"),
    )
    with pytest.raises(ValueError, match="Malformed/incomplete"):
        mdvr.generate(evidence_root)


def test_fold_ranks_cannot_be_changed_independently(evidence_root):
    change(
        evidence_root,
        "reports/model_validation/explainability_stability.json",
        lambda d: d["xgboost_stability"][0]["fold_ranks"].__setitem__(0, 12),
    )
    with pytest.raises(ValueError, match="Conflicting evidence: Task 6 fold rank"):
        mdvr.generate(evidence_root)
