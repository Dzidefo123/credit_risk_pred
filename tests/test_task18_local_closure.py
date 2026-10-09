"""Synthetic-fixture validation of scripts/task18_local_closure.py (v2).

The real frozen arrays are private and absent from a fresh checkout, so the script is
validated against a synthetic repository whose hash chain is constructed to match the
conventions the production code uses:

  - prediction arrays and evaluation.npy hashed with raw bytes (track_b.data.schemas.digest)
  - the ledger hashed with raw bytes and pinned by the public report's ledger_sha256
  - the ledger's registration carrying risk_array_sha256 for evaluation.npy

Each of the eight audited v1 defects has a test here.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/task18_local_closure.py"
REGISTRATION = ROOT / "reports/paper/task18_analysis_registration.json"

PAYOFF = 2
CUTOFF_ORDINAL = 2026 * 12 + 2 - 1  # "2026-02"


def raw_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(tmp_path: Path, *, n_facilities=60, months_each=30, start="2019-01",
                  seed=7, with_registration=True, break_continuity=False,
                  corrupt_prediction=False, shift_predictions=False) -> Path:
    """Create a synthetic repository with a self-consistent hash chain."""
    rng = np.random.default_rng(seed)
    root = tmp_path / "repo"
    (root / "src/credit_risk").mkdir(parents=True)
    (root / "reports/track_b").mkdir(parents=True)
    (root / "reports/paper").mkdir(parents=True)
    (root / "docs/track_b").mkdir(parents=True)
    private = root / "data/track_b/models/macro_hazard_v1"
    private.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")

    start_ordinal = int(start[:4]) * 12 + int(start[5:7]) - 1
    rows = []
    for facility in range(n_facilities):
        for step in range(months_each):
            month = start_ordinal + step
            if break_continuity and facility == 0 and step >= 5:
                month += 3  # shift the tail so end - start + 1 no longer equals the row count
            rows.append((facility, month, 2006 if facility % 2 else 2010, 0))
    dtype = [("facility", "i8"), ("month", "i8"), ("vintage", "i8"), ("event", "i8")]
    evaluation = np.array(rows, dtype=dtype)

    # Weak latent signal so AUCs are realistic rather than degenerate, with a payoff rate
    # that clears the 20-event support rule early and falls short later, exercising both
    # the eligible and the sparse-excluded paths.
    latent = rng.normal(0.0, 1.0, len(evaluation))
    steps = evaluation["month"] - start_ordinal
    intercept = np.where(steps < 12, -0.6, -3.2)
    probability = 1.0 / (1.0 + np.exp(-(intercept + 0.35 * latent)))
    evaluation["event"] = np.where(rng.random(len(evaluation)) < probability, PAYOFF, 0)

    # M1 scores on the latent alone. M2 adds a month-constant shift, which reproduces the
    # real study's structure: a between-month component with no within-month gain.
    def softmax_payoff(logit):
        payoff_odds = np.exp(logit)
        total = 1.0 + 0.01 + payoff_odds
        return np.column_stack([1.0 / total, np.full(len(logit), 0.01) / total, payoff_odds / total])

    m1 = softmax_payoff(-2.0 + 0.35 * latent)
    month_shift = np.where(steps < 12, 1.4, -1.1)
    if shift_predictions:
        month_shift = month_shift * 2.0
    m2 = softmax_payoff(-2.0 + 0.35 * latent + month_shift)
    np.save(private / "evaluation.npy", evaluation)
    np.save(private / "M1_evaluation.npy", m1)
    np.save(private / "M2_evaluation.npy", m2)

    # Ledger carries the risk-array hash; the report pins the ledger.
    ledger = {
        "state": "CONSUMED",
        "prediction_generation_count": 1,
        "virgin_holdout": False,
        "registration": {
            "namespace": "TASK10_MACRO_HAZARD",
            "risk_array_sha256": raw_sha(private / "evaluation.npy"),
        },
    }
    ledger_path = private / "task10_evaluation_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    if corrupt_prediction:
        np.save(private / "M2_evaluation.npy", m2 * 0.5)  # after hashing below

    from sklearn.metrics import roc_auc_score

    seen = np.isin(evaluation["vintage"], [2006, 2010])
    pooled1 = float(roc_auc_score(evaluation["event"][seen] == PAYOFF, m1[seen][:, PAYOFF]))
    pooled2 = float(roc_auc_score(evaluation["event"][seen] == PAYOFF, m2[seen][:, PAYOFF]))
    ids, starts, counts = np.unique(evaluation["facility"][seen], return_index=True, return_counts=True)
    entry = evaluation["month"][seen][starts]
    horizons = [12, 24, 36, 60]
    report = {
        "ledger_sha256": raw_sha(ledger_path),
        "prediction_hashes": {
            "M1_evaluation.npy": raw_sha(private / "M1_evaluation.npy"),
            "M2_evaluation.npy": raw_sha(private / "M2_evaluation.npy"),
        },
        "primary": {
            "M1": {"scores": {"payoff_auc": pooled1}},
            "M2": {"scores": {"payoff_auc": pooled2}},
        },
        "split_counts": {"evaluation_seen": {"intervals": int(seen.sum())}},
        "cif": {
            "landmarks": len(ids),
            "horizons": {
                str(h): {
                    "calendar_truncated_landmarks": int(np.count_nonzero(entry + h - 1 > CUTOFF_ORDINAL))
                }
                for h in horizons
            },
        },
    }
    (root / "reports/track_b/macro_competing_risk_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (root / "docs/track_b/macro_competing_risk_protocol.json").write_text(
        json.dumps({"splits": {"primary_vintages": [2006, 2010]}, "cif": {"horizons": horizons}}, indent=2),
        encoding="utf-8",
    )
    if with_registration:
        (root / "reports/paper/task18_analysis_registration.json").write_bytes(
            REGISTRATION.read_bytes()
        )
    return root


def run(root: Path, *extra, cwd=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *extra],
        capture_output=True, text=True, cwd=str(cwd or root),
    )


def output(root: Path) -> dict:
    return json.loads((root / "reports/paper/task18_local_closure_output.json").read_text())


# ------------------------------------------------------------------ happy path


def test_runs_clean_on_a_consistent_fixture(tmp_path):
    root = build_fixture(tmp_path)
    result = run(root)
    assert result.returncode == 0, result.stderr
    data = output(root)
    assert data["version"] == "task18-local-closure-v2"
    assert data["reconciliation_clean"] is True
    assert data["SA01_month"]["status"] == "EXECUTED"
    assert data["SA06_entry"]["status"] == "EXECUTED"


def test_full_hash_chain_is_verified(tmp_path):
    """Defect 2: evaluation.npy and the ledger are verified, not just predictions."""
    root = build_fixture(tmp_path)
    assert run(root).returncode == 0
    chain = output(root)["verification_chain"]
    assert chain["ledger"]["matches"] is True
    assert chain["risk_array"]["matches"] is True
    assert set(chain["predictions"]) == {"M1_evaluation.npy", "M2_evaluation.npy"}
    assert all(v["matches"] for v in chain["predictions"].values())


def test_registration_is_verified_not_asserted(tmp_path):
    """Defect 6."""
    root = build_fixture(tmp_path)
    assert run(root).returncode == 0
    assert output(root)["verification_chain"]["registration"]["matches"] is True


# -------------------------------------------------------------- fail-closed


def test_missing_registration_refuses_by_default(tmp_path):
    root = build_fixture(tmp_path, with_registration=False)
    result = run(root)
    assert result.returncode != 0
    assert "registration not found" in result.stderr


def test_missing_registration_can_be_explicitly_waived(tmp_path):
    root = build_fixture(tmp_path, with_registration=False)
    assert run(root, "--no-registration").returncode == 0
    assert output(root)["verification_chain"]["registration"]["status"] == "ABSENT_AND_WAIVED"


def test_tampered_registration_refuses(tmp_path):
    root = build_fixture(tmp_path)
    path = root / "reports/paper/task18_analysis_registration.json"
    path.write_bytes(path.read_bytes() + b"\n")
    result = run(root)
    assert result.returncode != 0
    assert "registration hash mismatch" in result.stderr


def test_tampered_prediction_array_refuses(tmp_path):
    root = build_fixture(tmp_path)
    private = root / "data/track_b/models/macro_hazard_v1"
    array = np.load(private / "M2_evaluation.npy")
    np.save(private / "M2_evaluation.npy", array * 0.5)
    result = run(root)
    assert result.returncode != 0
    assert "prediction array hash mismatch" in result.stderr


def test_tampered_risk_array_refuses(tmp_path):
    root = build_fixture(tmp_path)
    private = root / "data/track_b/models/macro_hazard_v1"
    array = np.load(private / "evaluation.npy")
    array["event"][0] = PAYOFF if array["event"][0] == 0 else 0
    np.save(private / "evaluation.npy", array)
    result = run(root)
    assert result.returncode != 0
    assert "evaluation.npy hash does not match" in result.stderr


def test_reconciliation_mismatch_fails_closed(tmp_path):
    """Defect 5: a pooled-AUC mismatch must not yield an EXECUTED output."""
    root = build_fixture(tmp_path)
    report_path = root / "reports/track_b/macro_competing_risk_validation.json"
    report = json.loads(report_path.read_text())
    report["primary"]["M2"]["scores"]["payoff_auc"] += 0.05
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    result = run(root)
    assert result.returncode != 0
    assert "RECONCILIATION FAILURE" in result.stderr
    assert not (root / "reports/paper/task18_local_closure_output.json").exists()


def test_reconciliation_mismatch_can_be_investigated_but_is_flagged(tmp_path):
    root = build_fixture(tmp_path)
    report_path = root / "reports/track_b/macro_competing_risk_validation.json"
    report = json.loads(report_path.read_text())
    report["primary"]["M2"]["scores"]["payoff_auc"] += 0.05
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert run(root, "--allow-mismatch").returncode == 0
    data = output(root)
    assert data["reconciliation_clean"] is False
    assert data["mismatch_waived"] is True
    assert data["reconciliation_failures"]


# ------------------------------------------------------------ root resolution


def test_wrong_root_is_refused(tmp_path):
    """Defect 1: a non-repository root must be rejected, not silently used."""
    root = build_fixture(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    result = run(root, cwd=outside)
    assert result.returncode == 0  # explicit --root still works from anywhere
    bad = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(outside)],
        capture_output=True, text=True, cwd=str(outside),
    )
    assert bad.returncode != 0
    assert "not the repository root" in bad.stderr


def test_auto_location_refuses_when_copied_outside_a_repository(tmp_path):
    """The audited scenario: the script sitting in Downloads, run from Downloads."""
    outside = tmp_path / "downloads"
    outside.mkdir()
    copied = outside / "task18_local_closure.py"
    copied.write_bytes(SCRIPT.read_bytes())
    result = subprocess.run(
        [sys.executable, str(copied)], capture_output=True, text=True, cwd=str(outside)
    )
    assert result.returncode != 0
    assert "could not locate the repository root" in result.stderr
    # and it must not have invented a root from its own depth
    assert str(tmp_path) not in (result.stdout or "")


def test_script_in_repo_locates_root_from_any_cwd(tmp_path):
    """Running the in-repo script from an unrelated cwd resolves the real repository."""
    outside = tmp_path / "unrelated"
    outside.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(ROOT)],
        capture_output=True, text=True, cwd=str(outside),
    )
    # the real private directory is absent here, so it must refuse for that reason
    assert result.returncode != 0
    assert "frozen model directory not found" in result.stderr


# --------------------------------------------------- decomposition correctness


def test_three_pair_buckets_are_kept_separate(tmp_path):
    """Defect 3: excluded months' within-pairs must not land in the between bucket."""
    root = build_fixture(tmp_path)
    assert run(root).returncode == 0
    sa01 = output(root)["SA01_month"]
    pairs = sa01["pair_structure"]
    assert sa01["excluded_month_count"] > 0, "fixture must exercise the exclusion path"
    assert pairs["within_month_pairs_excluded"] > 0
    assert (
        pairs["within_month_pairs_eligible"]
        + pairs["within_month_pairs_excluded"]
        + pairs["between_month_pairs"]
        == pairs["total_pairs"]
    )
    assert "NOT part of the between-month bucket" in pairs["note"]


def test_contribution_share_is_pair_weighted(tmp_path):
    """Defect 4: contribution must carry the pair weight."""
    root = build_fixture(tmp_path)
    assert run(root).returncode == 0
    sa01 = output(root)["SA01_month"]
    gains, pairs = sa01["gains"], sa01["pair_structure"]
    weight = pairs["within_month_pairs_eligible"] / pairs["total_pairs"]
    expected = weight * gains["within_month_stratum_gain_pair_weighted"]
    assert gains["within_month_contribution_to_pooled_gain"] == pytest.approx(expected, rel=1e-9)
    assert gains["within_month_contribution_share"] == pytest.approx(
        expected / gains["pooled"], rel=1e-9
    )
    assert gains["eligible_within_pair_weight"] == pytest.approx(weight, rel=1e-9)
    # and the magnitude ratio must be reported under its own distinct name
    assert gains["within_stratum_gain_as_fraction_of_pooled_gain"] == pytest.approx(
        gains["within_month_stratum_gain_pair_weighted"] / gains["pooled"], rel=1e-9
    )


def test_no_auc_is_imputed_for_sparse_months(tmp_path):
    root = build_fixture(tmp_path)
    assert run(root).returncode == 0
    sa01 = output(root)["SA01_month"]
    assert sa01["imputed_auc_count"] == 0
    for record in sa01["excluded_months"]:
        assert "M1_payoff_auc" not in record
        assert "reason" in record
    for record in sa01["per_month"]:
        assert record["cases"] >= 20 and record["controls"] >= 20


def test_no_between_month_auc_is_solved(tmp_path):
    """With a mixed residual, solving a 'between-month AUC' would mislabel it."""
    root = build_fixture(tmp_path)
    assert run(root).returncode == 0
    blob = json.dumps(output(root)["SA01_month"])
    assert "between_month_solved" not in blob


# -------------------------------------------------------- landmark continuity


def test_monthly_continuity_is_checked(tmp_path):
    """Defect 7: replicate the frozen continuity assertion, not just contiguity."""
    root = build_fixture(tmp_path, break_continuity=True)
    result = run(root)
    assert result.returncode != 0
    assert "monthly-continuous" in result.stderr


def test_entry_distribution_reconciles_with_truncation_counts(tmp_path):
    root = build_fixture(tmp_path)
    assert run(root).returncode == 0
    sa06 = output(root)["SA06_entry"]
    assert sa06["landmark_reconciliation"]["matches"] is True
    for record in sa06["per_horizon_truncation_check"].values():
        assert record["matches"] is True
    assert sum(sa06["entry_month_distribution"].values()) == sa06["landmark_reconciliation"]["computed_landmarks"]
    assert sum(sa06["entry_year_distribution"].values()) == sa06["landmark_reconciliation"]["computed_landmarks"]


# ----------------------------------------------------------------- integrity


def test_script_is_read_only_and_honest_about_what_it_reads():
    """Defect 8: the docstring must not claim it never reads loan-level values."""
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("np.save", ".fit(", "joblib.dump", "to_csv", "shutil", "os.remove", "unlink"):
        assert forbidden not in source, forbidden
    assert "reads_loan_level_arrays" in source
    assert "emits_loan_level_values" in source
    assert "It reads loan-level data and emits aggregates only." in source
    assert "facility_keys" not in source


def test_script_does_not_mutate_the_private_directory(tmp_path):
    root = build_fixture(tmp_path)
    private = root / "data/track_b/models/macro_hazard_v1"
    before = {p.name: raw_sha(p) for p in sorted(private.iterdir())}
    assert run(root).returncode == 0
    after = {p.name: raw_sha(p) for p in sorted(private.iterdir())}
    assert before == after


def test_output_records_provenance_and_guarantees(tmp_path):
    root = build_fixture(tmp_path)
    assert run(root).returncode == 0
    data = output(root)
    for flag in ("no_model_fitted", "no_prediction_regenerated", "no_calibration_fitted"):
        assert data[flag] is True, flag
    assert data["reads_loan_level_arrays"] is True
    assert data["emits_loan_level_values"] is False
    assert data["v2_defects_addressed"] == 8
