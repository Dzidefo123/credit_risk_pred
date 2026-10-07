"""Audit manuscript display bindings and language without new empirical analysis."""

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "manuscript_builder", ROOT / "scripts/build_manuscript.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def check_language(text):
    clean = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    if re.search(
        r"(?i)\b(?:sealed|pre-registered|registered)\s+Fannie|Fannie.{0,50}\b(?:sealed|pre-registered|registered)\b",
        clean,
    ):
        raise ValueError("Prohibited Fannie terminology")
    if re.search(
        r"(?i)\b(?:we|this study|the model)\s+(?:completed|demonstrates?|achieved)"
        r"\s+(?:an?\s+)?(?:external validation|independent replication|cross-provider validation)",
        clean,
    ):
        raise ValueError("Unsupported external-validation assertion")
    if re.search(
        r"(?i)\b(?:we|the study|this model)\s+(?:is|provides?|implements?)\s+(?:an?\s+)?"
        r"(?:IRB|IFRS\s?9|regulatory PD|production bank)\s+(?:production\s+)?model",
        clean,
    ):
        raise ValueError("Unsupported regulatory-model assertion")
    if re.search(
        r"(?i)\b(?:macroeconomic variables|macro variables|unemployment)"
        r"\s+(?:caused|causes?|drives?)\b",
        clean,
    ):
        raise ValueError("Unsupported causal macro assertion")
    if re.search(
        r"(?i)\b(?:we studied|we study|the cohort contains)\s+[\d,]+"
        r"\s+(?:unique|distinct)\s+borrowers",
        clean,
    ):
        raise ValueError("Borrower uniqueness assertion")
    flags = []
    patterns = {
        "causal": r"\b(?:cause|caused|effect|impact|drives|because of|mechanism|explains)\b",
        "external": (
            r"\b(?:external validation|independent replication|cross-provider validation|"
            r"industry-wide|Fannie replication)\b"
        ),
        "regulatory": r"\b(?:IFRS\s?9|IRB|regulatory|validated bank)\b",
    }
    for kind, pattern in patterns.items():
        for match in re.finditer(pattern, clean, flags=re.I):
            start = clean.rfind("\n", 0, match.start()) + 1
            end = clean.find("\n", match.end())
            context = clean[start : end if end >= 0 else len(clean)]
            flags.append(
                {
                    "kind": kind,
                    "term": match.group(),
                    "context": context,
                    "disposition": "RETAIN_NONCAUSAL_METHOD_OR_EXPLICIT_BOUNDARY",
                    "author_review_required": True,
                }
            )
    return flags


def check_unbound_numbers(text):
    # Result tokens are paired to numeric markers; remove them before auditing literals.
    text = re.sub(r"[+-]?[\d,]+(?:\.\d+)?%?<!-- NUM: N\d+ -->", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$|\$[^$]+\$", "", text, flags=re.S)
    text = text.replace("v0.1", "VERSION")
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.M)
    # Remaining decimal or comma-group values would be unsupported result literals.
    if re.search(r"(?<!\w)[+-]?\d+\.\d+|(?<!\w)\d{1,3}(?:,\d{3})+", text):
        raise ValueError("Unbound quantitative literal")
    structural_codes = {"01", "02", "03", "09", "99"}
    documented_years = {str(y) for y in range(2006, 2027)}
    for match in re.finditer(r"(?<![\w.])\d+(?![\w.])", text):
        value = match.group()
        prefix = text[max(0, match.start() - 6) : match.start()]
        if (
            value in structural_codes
            or value in documented_years
            or prefix.endswith("Task ")
            or prefix.endswith("IFRS ")
        ):
            continue
        raise ValueError("Unbound quantitative integer")


def verify():
    registry = read("docs/paper/track_b_claim_evidence_registry.json")
    claims = {c["claim_id"]: c for c in registry["claims"]}
    source = (ROOT / "paper/main.source.md").read_text(encoding="utf-8")
    main = (ROOT / "paper/main.md").read_text(encoding="utf-8")
    expected, bindings = builder.render(source, claims)
    if main != expected:
        raise ValueError("Manuscript differs from claim-bound source")
    stored = read("docs/paper/manuscript_numeric_bindings.json")
    if stored["bindings"] != bindings or stored["new_empirical_calculations"]:
        raise ValueError("Numeric traceability mismatch")
    usage = read("docs/paper/manuscript_claim_usage.json")["used_claims"]
    for item in usage:
        cid = item["claim_id"]
        if cid not in claims or claims[cid]["claim_class"] in {"UNSUPPORTED", "PROHIBITED"}:
            raise ValueError("Unsupported claim used")
    abstract = main.split("## Abstract\n")[1].split("## 1.")[0]
    abstract_ids = set(re.findall(r"NUM: (N\d+)", abstract))
    for binding in bindings:
        if binding["binding_id"] in abstract_ids:
            c = claims[binding["claim_id"]]
            if c["claim_class"] != "SUPPORTED_EMPIRICAL" or c["claim_strength"] != "PRIMARY":
                raise ValueError("Abstract numerical finding is not primary empirical evidence")
    annotations = re.findall(r"<!-- CLAIM: ([^>]+) -->", abstract)
    for annotation in annotations:
        for cid in (c.strip() for c in annotation.split(",")):
            if (
                claims[cid]["claim_class"] != "SUPPORTED_EMPIRICAL"
                or claims[cid]["claim_strength"] != "PRIMARY"
            ):
                raise ValueError("Abstract empirical annotation is not primary")
    gaps = read("docs/paper/citation_gaps.json")["gaps"]
    markers = re.findall(r"\[CITATION NEEDED: (CG\d+) \| ([^\]]+)\]", main)
    if {(g["gap_id"], g["claim_or_topic"]) for g in gaps} != set(markers):
        raise ValueError("Unstructured citation placeholder")
    if re.search(r"\b10\.\d{4,9}/\S+", main) or "\\cite{" in main:
        raise ValueError("Unverified bibliography/DOI")
    for required in [
        "conditional entry",
        "research default proxy",
        "EXPLORATORY_ONLY",
        "post-hoc diagnostic analysis",
        "DRAFT_NOT_YET_AUTHORIZED",
        "PUBLICATION TERMS REVIEW REQUIRED",
    ]:
        if required.lower() not in main.lower():
            raise ValueError("Required scientific qualification missing")
    for path in (ROOT / "paper").iterdir():
        # Task 16 validates this additive bibliography in check_literature.py.
        if path.name == "references.bib":
            continue
        if path.suffix != ".md":
            raise ValueError("Unreviewed paper artifact")
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bF\d{2}Q[1-4]\d{7}\b|C:\\\\Users\\\\|(?<![\d.])\b\d{12}\b(?![\d.])", text):
            raise ValueError("Possible private record or identifier")
    check_unbound_numbers(main)
    flags = check_language(main)
    return {
        "status": "PASSED",
        "bindings": len(bindings),
        "used_claims": len(usage),
        "body_word_count": builder.words(main.split("## Figure placeholders")[0]),
        "abstract_word_count": builder.words(abstract),
        "citation_gaps": len(gaps),
        "language_flags": flags,
        "evidence_queries": [],
        "decision": "ARXIV MANUSCRIPT V0.1 READY WITH MATERIAL GAPS",
    }


if __name__ == "__main__":
    result = verify()
    result["language_flags"] = len(result["language_flags"])
    print(json.dumps(result))
