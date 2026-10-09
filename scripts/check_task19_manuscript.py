"""Validate v0.3 evidence bindings and preservation; no scoring or model imports."""

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def preservation(private=False):
    manifest = read("reports/paper/task19_preservation_manifest.json")
    allowed_code = {"tests/test_hostile_review.py", "tests/test_frozen_array_sensitivity.py"}
    if not set(manifest.get("engineering_compatibility_changes", {})) <= allowed_code:
        raise ValueError("Scientific evidence cannot receive an engineering exception")
    for name, expected in manifest["public_lf_hashes"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        approved = manifest.get("engineering_compatibility_changes", {}).get(name)
        if actual != expected:
            if (
                not approved
                or expected != approved["original_sha256_lf"]
                or actual != approved["amended_sha256_lf"]
            ):
                raise ValueError("Prior scientific/public artifact changed: " + name)
    for name, expected in manifest["git_refs"].items():
        actual = subprocess.check_output(["git", "rev-parse", name], cwd=ROOT, text=True).strip()
        if actual != expected:
            raise ValueError("Frozen tag changed")
    count = 0
    if private:
        for name, expected in manifest["private_byte_hashes"].items():
            value = hashlib.sha256()
            with (ROOT / name).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != expected:
                raise ValueError("Private frozen hash changed: " + name)
            count += 1
    return dict(
        public=len(manifest["public_lf_hashes"]),
        private=count,
        refs=len(manifest["git_refs"]),
        engineering_exceptions=len(manifest.get("engineering_compatibility_changes", {})),
    )


def admission():
    folder = ROOT / "reports/paper/task19_evidence"
    record = read("reports/paper/task19_evidence/admission.json")
    for name, expected in record["admitted_files_sha256"].items():
        if (
            Path(name).name != name
            or hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Local evidence admission hash mismatch")
    reg = read("reports/paper/task19_evidence/local_closure_registration.json")
    verification = read("reports/paper/task19_evidence/local_closure_verification.json")
    output = read("reports/paper/task19_evidence/local_closure_output.json")
    if reg["code_sha256"] != record["admitted_files_sha256"]["local_closure_script.py"]:
        raise ValueError("Registered local script differs")
    if (
        verification["output_sha256"]
        != record["admitted_files_sha256"]["local_closure_output.json"]
    ):
        raise ValueError("Local output differs from verification")
    if (folder / "local_closure_registration.sha256").read_text().strip() != record[
        "admitted_files_sha256"
    ]["local_closure_registration.json"]:
        raise ValueError("Local registration seal differs")
    frozen = read("reports/track_b/macro_competing_risk_validation.json")
    old = read("docs/paper/literature/task16_preservation_manifest.json")
    for name, expected in output["inputs_verified"].items():
        if (
            expected != old["private_byte_hashes"][name]
            or reg["private_input_hashes"][name] != expected
        ):
            raise ValueError("Local inputs do not bind frozen artifact hashes")
    for model in ("M1", "M2"):
        if (
            output["SA01_month"]["full_primary_auc_reconciled"][model]
            != frozen["primary"][model]["scores"]["payoff_auc"]
        ):
            raise ValueError("Local pooled AUC reconciliation changed")
    if (
        not output["SA06_entry"]["reconciliations_passed"]
        or record["new_analysis_performed_in_task19"]
    ):
        raise ValueError("Local closure provenance boundary changed")
    return dict(admitted_files=len(record["admitted_files_sha256"]), rerun=False)


def validate_bindings(text, bindings):
    builder = module("build_task19_manuscript")
    ids = [b["binding_id"] for b in bindings]
    if len(ids) != len(set(ids)) or set(ids) != set(re.findall(r"<!-- Q19: (Q\d+) -->", text)):
        raise ValueError("Quantitative marker/map mismatch")
    stripped = text.split("## References")[0]
    for binding in bindings:
        source = binding["source_artifact"]
        if not source.startswith(("reports/", "docs/")) or ".." in Path(source).parts:
            raise ValueError("Quantitative source outside public evidence")
        value = builder.pointer(read(source), binding["source_field"])
        if value != binding["exact_value"]:
            raise ValueError("Frozen field/value mismatch")
        if builder.digest(ROOT / source) != binding["source_sha256_lf"]:
            raise ValueError("Bound source hash mismatch")
        display = builder.display(value, binding["display_format"])
        marker = display + "<!-- Q19: " + binding["binding_id"] + " -->"
        if marker not in text or display != binding["displayed_value"]:
            raise ValueError("Quantitative display differs from evidence")
        stripped = stripped.replace(marker, "BOUND_VALUE", 1)
    stripped = re.sub(r"\$\$.*?\$\$|\$[^$]+\$", "MATH_DEFINITION", stripped, flags=re.S)
    stripped = re.sub(r"^#{1,6} .*?$", "HEADING", stripped, flags=re.M)
    stripped = re.sub(r"\[@[^\]]+\]", "CITATION", stripped)
    stripped = re.sub(r"<!--.*?-->", "", stripped, flags=re.S)
    stripped = re.sub(
        r"\bv0\.3\b|\bM[012]\b|\b(?:Task\s+\d+[A-Z]?|CG\d+|IFRS\s+9)\b", "IDENTIFIER", stripped
    )
    if re.search(r"\d", stripped):
        raise ValueError("Orphan numeric claim")


def validate_language(text):
    forbidden = [
        r"COVID caused",
        r"macro(?:economic)? (?:variables|features) always",
        r"first (?:study|paper) (?:ever|to)",
        r"structurally impossible",
        r"within-month.*structurally unavailable",
        r"10\.5%",
        r"M2 is closer than M1",
        r"identified numerical cause",
        r"Fannie (?:completed|confirms|validated)",
        r"fully verified provider release",
    ]
    if any(re.search(pattern, text, re.I) for pattern in forbidden):
        raise ValueError("Unsupported or corrected claim reintroduced")
    required = [
        "MIXED_TEMPORAL_TRANSPORT_ACROSS_POPULATIONS",
        "DISTINCTIVE COMBINATION PLAUSIBLE WITH MATERIAL LIMITATIONS",
        "ROBUST_FACILITY_ONLY",
        "DRAFT_NOT_YET_AUTHORIZED",
        "EXPLORATORY_ONLY",
        "diagnostic hypothesis",
        "sparse months",
        "one realized national macro",
        "not independent corroboration",
        "Current delinquency",
        "release lags were not certified",
        "CG03 implementation review remains open",
        "PUBLICATION TERMS REVIEW REQUIRED",
        "ETHICS AND AUTHOR REVIEW REQUIRED",
    ]
    if any(phrase.lower() not in text.lower() for phrase in required):
        raise ValueError("Required scope qualification missing")
    for section in ("## Abstract", "### 4.5", "## 5. Discussion"):
        start = text.index(section)
        end = text.find("\n## ", start + len(section))
        part = text[start : end if end != -1 else None]
        if "unseen" not in part.lower():
            raise ValueError("Unseen transport omitted from central section")


def verify(private=False):
    main = (ROOT / "paper/main_v0.3.md").read_text(encoding="utf-8")
    mapping = read("reports/paper/task19_claim_evidence_map.json")
    bindings = mapping["bindings"]
    validate_bindings(main, bindings)
    validate_language(main)
    builder = module("build_task19_manuscript")
    builder.definitions()
    calendar, cif = builder.table_tokens()
    template = (ROOT / "reports/paper/task19_manuscript_template.md").read_text(encoding="utf-8")
    template = template.replace("{{CALENDAR_TABLE}}", calendar).replace("{{CIF_TABLE}}", cif)
    template = template.replace("{{VERIFIED_REFERENCES}}", builder.references(template))
    expected, expected_bindings = builder.render(template)
    if main != expected or bindings != expected_bindings:
        raise ValueError("Draft/map differs from retained-source rendering")
    refs = {
        r["reference_id"]
        for r in read("docs/paper/literature/reference_registry.json")["references"]
    }
    citations = module("check_literature").citation_keys(main)
    if not citations <= refs:
        raise ValueError("Unverified manuscript citation")
    revision = read("reports/paper/task19_revision_register.json")
    original_changes = read("reports/paper/review/task17_manuscript_change_register.json")
    if {c["change_id"] for c in revision["changes"]} != {
        c["change_id"] for c in original_changes["changes"]
    } or revision["new_empirical_analysis"]:
        raise ValueError("Review concern omitted or scientific boundary changed")
    plan = read("reports/paper/task19_figure_table_plan.json")
    if len(plan["figures"]) != 6 or len(plan["tables"]) != 7:
        raise ValueError("Figure/table plan incomplete")
    for figure in plan["figures"]:
        if figure["requires_private_records"]:
            raise ValueError("Figure requires unauthorized private processing")
        for source in figure["sources"]:
            path, field = source.split("#", 1)
            builder.pointer(read(path), field)
    erratum = (ROOT / "reports/paper/TASK17_ERRATUM.md").read_text(encoding="utf-8")
    if "argument is withdrawn" not in erratum or "audit error" not in erratum:
        raise ValueError("Task 17 erratum missing")
    paper = module("check_paper_evidence")
    for path in [
        ROOT / "paper/main_v0.3.md",
        ROOT / "reports/paper/task19_claim_evidence_map.json",
    ]:
        paper.check_public_text(path.read_text(encoding="utf-8"))
    old = module("check_manuscript").verify()
    clean = re.sub(r"<!--.*?-->", "", main, flags=re.S)
    abstract = clean.split("## Abstract\n")[1].split("## 1.")[0]
    return dict(
        status="PASSED",
        quantitative_claims=len(bindings),
        mapped=len(bindings),
        orphan_numeric_claims=0,
        citations=len(citations),
        manuscript_words=len(clean.split("## References")[0].split()),
        abstract_words=len(abstract.split()),
        legacy_numeric_bindings=old["bindings"],
        preservation=preservation(private),
        local_evidence_admission=admission(),
        new_empirical_analysis=False,
        recommendation="READY_FOR_HOSTILE_MANUSCRIPT_REVIEW",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-hashes", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.private_hashes)))
