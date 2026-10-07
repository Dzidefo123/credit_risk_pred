"""Task13R schema reconciliation gates; no downloader, research parser or outcome reader."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "task13_schema", ROOT / "scripts/fannie_feasibility.py"
)
task13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task13)


def order_hash(fields):
    pairs = [[f["position"], f["name"]] for f in fields]
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=True).encode()).hexdigest()


def validate_fields(fields, width):
    positions = [f["position"] for f in fields]
    if positions != list(range(1, width + 1)):
        raise ValueError("Duplicate, missing or reordered documented position")
    for field in fields:
        if field["sf_applicability"] not in {"APPLICABLE", "NOT_APPLICABLE"}:
            raise ValueError("Unknown SF applicability")
        if not field["name"] or not field["source"]:
            raise ValueError("Uncited field")
    return fields


def sf_fields(fields):
    """Semantic inventory only; filtering never changes the physical layout."""
    return [
        f
        for f in fields
        if f["sf_applicability"] == "APPLICABLE"
        and f["sf_value_expectation"] != "ALWAYS_NOT_APPLICABLE"
    ]


def compare_layouts(old, new):
    """Documented positions, not loan records; no inference from row contents."""
    a = {f["position"]: f for f in old}
    b = {f["position"]: f for f in new}
    changes = []
    for position in sorted(a.keys() | b.keys()):
        previous, current = a.get(position), b.get(position)
        if previous is None:
            kind = "FIELD_ADDED_LATER"
        elif current is None:
            kind = "FIELD_REMOVED_LATER"
        elif previous["name"] != current["name"]:
            kind = "FIELD_RENAMED_ONLY"
        else:
            continue
        changes.append(
            dict(
                position=position,
                classification=kind,
                previous_name=previous["name"] if previous else None,
                current_name=current["name"] if current else None,
            )
        )
    return changes


def registry_contract(registry):
    if registry["research_authorized"] or registry["protocol_status"] != "DRAFT_NOT_YET_AUTHORIZED":
        raise ValueError("Research activation prohibited")
    ids = [layout["layout_id"] for layout in registry["layouts"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate layout ID")
    for layout in registry["layouts"]:
        if layout["fields"]:
            validate_fields(layout["fields"], layout["expected_physical_width"])
            if order_hash(layout["fields"]) != layout["ordering_sha256"]:
                raise ValueError("Ordering evidence changed")
        elif layout["status"] != "DOCUMENTED_WIDTH_ONLY_ADAPTER_DISABLED":
            raise ValueError("Missing field ordering evidence")
    return dict(status="PASSED", layouts=len(ids))


def validate_structure(line, layout_id, registry):
    registry_contract(registry)
    matches = [x for x in registry["layouts"] if x["layout_id"] == layout_id]
    if not matches or not matches[0]["fields"]:
        raise ValueError("Unknown/unordered layout; STOP")
    layout = matches[0]
    info = task13.structural_line(line)
    if info["field_count"] != layout["expected_physical_width"]:
        raise ValueError("Exact release width mismatch; STOP; no padding")
    return info


def research_gate(binding, registry):
    """A matching width alone can never bind a private archive to a release."""
    registry_contract(registry)
    if not (
        binding.get("provider_release_verified")
        and binding.get("ordering_verified")
        and binding.get("archive_sha256_verified")
    ):
        raise ValueError("Archive release/ordering binding unverified; STOP")
    raise ValueError("Task13R cannot authorize research or Task13A")


def canonical_metadata(layout):
    """Separate provider slots from research concepts; absence is a typed state, not padding."""
    positions = {f["position"] for f in layout["fields"]}
    return dict(
        provider_width=layout["expected_physical_width"],
        vantage_score_status="DOCUMENTED_POSITION" if 114 in positions else "STRUCTURAL_ABSENCE",
        no_provider_record_mutation=True,
    )


def preservation(root):
    frozen = json.loads(
        (root / "docs/track_b/fannie_release_preservation_manifest.json").read_text()
    )
    for name, sha in frozen["public_lf_hashes"].items():
        value = hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if value != sha:
            raise ValueError("Frozen public evidence changed: " + name)
    for name, sha in frozen["private_byte_hashes"].items():
        if task13.sha256(root / name) != sha:
            raise ValueError("Frozen private evidence changed: " + name)
    for name, sha in frozen["external_archives"].items():
        if task13.sha256(name) != sha:
            raise ValueError("Frozen external archive changed")
    return dict(
        status="PASSED",
        public=len(frozen["public_lf_hashes"]),
        private=len(frozen["private_byte_hashes"]),
        external_archives=1,
    )


def date_mentions(note):
    months = {
        m: i
        for i, m in enumerate(
            [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
            1,
        )
    }
    return sorted(
        {
            f"{y}-{months[m]:02}"
            for m, y in re.findall(r"(" + "|".join(months) + r")\s+(\d{4})", note)
        }
    )


if __name__ == "__main__":
    registry = json.loads((ROOT / "docs/track_b/fannie_release_layout_registry.json").read_text())
    print(json.dumps(dict(contracts=registry_contract(registry), preservation=preservation(ROOT))))
