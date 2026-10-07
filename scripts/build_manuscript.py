"""Render an internal Markdown draft using frozen claims; never fit or evaluate models."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "1a3206094147a2ba487bda5af044fc42e8d71649"
TOKEN = re.compile(r"\{\{([A-Z0-9_]+)\|([^|]*)\|([^{}|]+)\}\}")


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def write(name, value):
    path = ROOT / name
    if path.exists():
        raise ValueError("Existing draft artifact; use a versioned author-review amendment")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def pointer(value, address):
    for key in address.strip("/").split("/") if address else []:
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def claim_value(claim, address):
    if address.startswith("@uncertainty"):
        return pointer(claim["uncertainty"], address[len("@uncertainty") :])
    return pointer(claim["point_estimate"], address)


def render(source, claims):
    bindings = []

    def replace(match):
        cid, address, fmt = match.groups()
        claim = claims[cid]
        if claim["claim_class"] in {"UNSUPPORTED", "PROHIBITED"}:
            raise ValueError("Unsupported or prohibited numeric claim")
        if claim["verification_status"] not in {"VERIFIED_EXACT", "VERIFIED_WITH_QUALIFICATION"}:
            raise ValueError("Unverified numeric claim")
        if not re.fullmatch(r"[+,]?\.?\d*[df%]", fmt):
            raise ValueError("Unapproved display format")
        value = claim_value(claim, address)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Not a frozen numerical value")
        displayed = format(value, fmt)
        ident = f"N{len(bindings) + 1:04}"
        bindings.append(
            {
                "binding_id": ident,
                "claim_id": cid,
                "claim_pointer": address,
                "exact_frozen_value": value,
                "display_format": fmt,
                "displayed_value": displayed,
                "claim_class": claim["claim_class"],
                "source_artifacts": claim["source_artifacts"],
            }
        )
        return displayed + f"<!-- NUM: {ident} -->"

    return TOKEN.sub(replace, source), bindings


def words(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return len(re.findall(r"\b\w+(?:[-'][\w]+)*\b", text))


def build():
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != BASE:
        raise ValueError("Wrong authorized manuscript base")
    source_path = ROOT / "paper/main.source.md"
    text = source_path.read_text(encoding="utf-8")
    registry = read("docs/paper/track_b_claim_evidence_registry.json")
    claims = {c["claim_id"]: c for c in registry["claims"]}
    rendered, bindings = render(text, claims)
    output = ROOT / "paper/main.md"
    if output.exists():
        raise ValueError("Existing manuscript must not be overwritten")
    output.write_text(rendered, encoding="utf-8")
    write(
        "docs/paper/manuscript_numeric_bindings.json",
        {
            "version": "task15-v0.1",
            "new_empirical_calculations": False,
            "formatting_policy": "Display formatting only; percentages use the format specifier",
            "bindings": bindings,
        },
    )
    gaps = []
    section = None
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:]
        for gid, topic in re.findall(r"\[CITATION NEEDED: (CG\d+) \| ([^\]]+)\]", line):
            gaps.append(
                {
                    "gap_id": gid,
                    "section": section,
                    "claim_or_topic": topic,
                    "required_literature_type": "Original scholarly methodology/empirical work",
                    "suggested_search_terms": topic,
                    "priority": "HIGH",
                    "status": "UNRESOLVED; no search or citation fabrication in Task 15",
                }
            )
    write("docs/paper/citation_gaps.json", {"gaps": gaps, "novelty_established": False})
    notation = [
        ("i", "Mortgage facility, not uniquely linked borrower"),
        ("e_i", "Eligible conditional calendar entry"),
        ("k,K", "Monthly relative interval index and horizon after entry"),
        ("t", "Calendar reporting interval; not the model duration proxy"),
        ("Y_it", "No endpoint / research default proxy / payoff-maturity"),
        ("x_it", "Frozen static/duration inputs and permitted PIT information"),
        ("h_D,h_P,h_0", "Joint monthly conditional cause/no-event probabilities"),
        ("S_i", "Conditional event-free survival, initialized at entry"),
        ("F_D,F_P", "Cause cumulative incidence from joint probability recursion"),
        ("eta_c", "Multinomial class logit; probabilities use joint softmax"),
        ("G,G+,G-", "Original-coupon proxy minus PIT market rate and asymmetric terms"),
    ]
    write(
        "docs/paper/notation_registry.json",
        {
            "symbols": [{"symbol": s, "meaning": m} for s, m in notation],
            "conditional_entry": True,
            "prospective_macro_forecast_claim": False,
        },
    )
    terms = {
        "facility": "Mortgage record key; not a borrower identity",
        "mortgage": "Selected provider mortgage facility",
        "research default proxy": "Frozen composite monthly adverse endpoint",
        "payoff": "Verified payoff/maturity, not observed voluntary refinancing",
        "competing risk": "Joint probability system with default and payoff exits",
        "temporal evaluation": "Frozen later calendar comparison with documented prior exposure",
        "PIT macro": "Vintage-aware macro information, not verified mortgage knowledge time",
        "development": "Earlier fit population; not independent validation",
        "calibration": "Probability agreement diagnostics; no evaluation correction applied",
        "proper scoring": "Joint log loss or cause-specific Brier on their defined targets",
        "conditional entry": "Risk after eligible observed entry, not origination lifetime",
        "external replication": "Proposed Fannie extension; no outcome results",
    }
    write(
        "docs/paper/manuscript_terminology.json",
        {
            "approved_terms": terms,
            "prohibited_substitutions": [
                "borrower uniqueness",
                "regulatory default equivalence",
                "virgin holdout",
                "causal macro explanation",
                "completed external confirmation",
                "sealed Fannie protocol",
            ],
            "fannie_protocol": "DRAFT_NOT_YET_AUTHORIZED",
            "task13a_authorized": False,
        },
    )
    used = {x["claim_id"] for x in bindings}
    for annotation in re.findall(r"<!-- CLAIM: ([^>]+) -->", text):
        used.update(c.strip() for c in annotation.split(","))
    unknown = used - claims.keys()
    if unknown:
        raise ValueError("Unknown claim annotations: " + str(sorted(unknown)))
    write(
        "docs/paper/manuscript_claim_usage.json",
        {
            "used_claims": [
                {
                    "claim_id": cid,
                    "claim_class": claims[cid]["claim_class"],
                    "claim_strength": claims[cid]["claim_strength"],
                    "verification_status": claims[cid]["verification_status"],
                }
                for cid in sorted(used)
            ],
            "unsupported_or_prohibited_used": [],
            "unclassified_governance_statements": (
                "Author/publication/ethics review flags are not findings"
            ),
        },
    )
    baseline = read("docs/track_b/fannie_session_preservation_manifest.json")
    files = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .strip("\0")
        .split("\0")
    )
    write(
        "docs/paper/task15_preservation_manifest.json",
        {
            "base_commit": BASE,
            "public_lf_hashes": {
                n: hashlib.sha256((ROOT / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                for n in files
            },
            "private_byte_hashes": baseline["private_byte_hashes"],
            "git_refs": baseline["git_refs"],
        },
    )
    print(
        json.dumps(
            {
                "body_word_count": words(rendered.split("## Figure placeholders")[0]),
                "abstract_word_count": words(rendered.split("## Abstract\n")[1].split("## 1.")[0]),
                "numeric_bindings": len(bindings),
                "used_claims": len(used),
                "citation_gaps": len(gaps),
            }
        )
    )


if __name__ == "__main__":
    build()
