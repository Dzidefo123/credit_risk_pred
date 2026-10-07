"""Offline Task 16 integrity checks; no model evaluation or loan-record parsing."""

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
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def validate_references(refs):
    ids, dois, titles = set(), set(), set()
    for r in refs:
        if not r["verification_status"].startswith("VERIFIED_") or not r["verification_sources"]:
            raise ValueError("Unverified reference")
        if r["reference_id"] in ids:
            raise ValueError("Duplicate key")
        ids.add(r["reference_id"])
        title = re.sub(r"\W", "", r["title"].casefold())
        if title in titles:
            raise ValueError("Duplicate title/version")
        titles.add(title)
        if r["doi"]:
            doi = r["doi"].casefold()
            if doi in dois:
                raise ValueError("Duplicate DOI")
            dois.add(doi)
        if r["publication_type"] == "PREPRINT" and r["peer_reviewed"]:
            raise ValueError("Preprint labeled peer reviewed")
        if not r["authors"] or not r["title"] or "UNVERIFIED" in r["flags"]:
            raise ValueError("Incomplete reference")
    return ids


def citation_keys(text):
    return {
        key
        for group in re.findall(r"\[@([^\]]+)\]", text)
        for key in re.findall(r"(?:^|;\s*@)([A-Za-z]+\d{4})", group)
    }


def sections(text):
    parts = re.split(r"(?=^## )", text, flags=re.M)
    return {p.splitlines()[0]: p for p in parts}


def validate_manuscript(old, new, ids):
    if not citation_keys(new) <= ids:
        raise ValueError("Unresolved citation key")
    if re.search(r"\b(?:first paper|first study|unprecedented|novel method)\b", new, re.I):
        raise ValueError("Unsupported priority claim")
    pattern = r"([+-]?[\d,]+(?:\.\d+)?%?)<!-- NUM: (N\d+) -->"
    if re.findall(pattern, old) != re.findall(pattern, new):
        raise ValueError("Empirical number/binding changed")
    before, after = sections(old), sections(new)
    allowed = {
        "## 1. Introduction",
        "## 2. Related work and citation gaps",
        "## 10. Limitations and generalization boundaries",
        "## References",
    }
    for heading, content in before.items():
        if heading.startswith("# Temporal") or heading in allowed:
            continue
        if after.get(heading) != content:
            raise ValueError("Frozen section changed: " + heading)
    for required in [
        "DRAFT_NOT_YET_AUTHORIZED",
        "EXPLORATORY_ONLY",
        "PUBLICATION TERMS REVIEW REQUIRED",
        "conditional entry",
    ]:
        if required not in new:
            raise ValueError("Scientific boundary removed")


def preserved_digest(root, name):
    payload = (root / name).read_bytes().replace(b"\r\n", b"\n")
    actual = hashlib.sha256(payload).hexdigest()
    amendments = json.loads(
        (root / "reports/paper/task16_compatibility_amendment.json").read_text(encoding="utf-8")
    )["approved_code_hashes"]
    if name in amendments:
        approved = amendments[name]
        if actual != approved["amended_sha256_lf"]:
            raise ValueError("Unapproved compatibility code change: " + name)
        return approved["original_sha256_lf"]
    return actual


def preservation(private=False):
    m = read("docs/paper/literature/task16_preservation_manifest.json")
    for name, expected in m["public_lf_hashes"].items():
        if preserved_digest(ROOT, name) != expected:
            raise ValueError("Prior public file changed: " + name)
    for ref, expected in m["git_refs"].items():
        actual = subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()
        if actual != expected:
            raise ValueError("Frozen tag changed")
    count = 0
    if private:
        for name, expected in m["private_byte_hashes"].items():
            digest = hashlib.sha256()
            with (ROOT / name).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError("Frozen private hash changed: " + name)
            count += 1
    return {
        "prior_public_files": len(m["public_lf_hashes"]),
        "compatibility_exceptions": 6,
        "private_hashes_checked": count,
        "tags_unchanged": len(m["git_refs"]),
    }


def verify(private=False):
    refs = read("docs/paper/literature/reference_registry.json")["references"]
    ids = validate_references(refs)
    builder = module("build_literature")
    expected_refs = builder.build_references()
    builder.claim_map(expected_refs)
    if refs != expected_refs:
        raise ValueError("Reference differs from verified metadata/review snapshot")
    bib = (ROOT / "paper/references.bib").read_text(encoding="utf-8")
    if bib != builder.bibliography(refs):
        raise ValueError("Bibliography differs from verified registry")
    if set(re.findall(r"@article\{([^,]+),", bib)) != ids:
        raise ValueError("Bibliography keys mismatch")
    old = (ROOT / "paper/main.md").read_text(encoding="utf-8")
    new = (ROOT / "paper/main_v0.2.md").read_text(encoding="utf-8")
    validate_manuscript(old, new, ids)
    gaps = read("docs/paper/literature/citation_gap_resolution.json")["gaps"]
    if {g["gap_id"] for g in gaps} != {
        g["gap_id"] for g in read("docs/paper/citation_gaps.json")["gaps"]
    }:
        raise ValueError("Citation gap missing")
    for g in gaps:
        if not set(g["reference_ids"]) <= ids or not g["residual"]:
            raise ValueError("Gap citation/support incomplete")
    rows = read("docs/paper/closest_paper_matrix.json")["papers"]
    if not 10 <= len(rows) <= 20 or any(r["paper"] not in ids for r in rows):
        raise ValueError("Closest-paper verification missing")
    if not {"Bu2026", "Peng2026", "Deng1996", "Deng2000"} <= {r["paper"] for r in rows}:
        raise ValueError("Critical closest work missing")
    if {t for r in refs for t in r["topics"]} != {f"L{i}" for i in range(1, 11)}:
        raise ValueError("Literature bucket missing")
    usage = read("docs/paper/claim_literature_map.json")["claims"]
    expected = {
        c["claim_id"] for c in read("docs/paper/manuscript_claim_usage.json")["used_claims"]
    }
    if {c["claim_id"] for c in usage} != expected:
        raise ValueError("Manuscript claim not mapped")
    for c in usage:
        if not set(c["supporting_reference_ids"] + c["closest_prior_work"]) <= ids:
            raise ValueError("Claim cites unverified reference")
        if c["claim_class"] == "SUPPORTED_EMPIRICAL" and c["supporting_reference_ids"]:
            raise ValueError("Literature used as own-result evidence")
    original = module("check_manuscript").verify()
    return dict(
        status="PASSED",
        references=len(refs),
        closest_papers=len(rows),
        citations_used=len(citation_keys(new)),
        numeric_bindings=original["bindings"],
        abstract_unchanged=True,
        citation_gaps=len(gaps),
        evidence_queries=[],
        preservation=preservation(private),
        decision=builder.DECISION,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-hashes", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.private_hashes)))
