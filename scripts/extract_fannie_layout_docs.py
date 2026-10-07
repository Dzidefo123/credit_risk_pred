"""Extract pinned official documentation only, using the optional bundled document runtime.

No network, licensed records, samples, model dependencies or outcome calculations.
PDF/XLSX bytes stay private; export only field metadata for the reconciliation builder.
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data/track_b/fannie/release_reconciliation"


def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def metadata_row(row, source, page=None):
    if len(row) != 11 or not str(row[0]).strip().isdigit():
        return None
    return dict(
        position=int(row[0]),
        name=clean(row[1]),
        sf_applicability=clean(row[8]),
        date_bound_note=clean(row[4]),
        disclosure_note=clean(row[5]),
        source=source,
        page=page,
    )


def extract_pdf(path, source, expected_sha):
    import pdfplumber
    from pypdf import PdfReader

    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("Pinned documentation hash mismatch")
    fields = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            for table in page.extract_tables():
                for cells in table:
                    row = metadata_row(cells, source, number)
                    if row:
                        fields.append(row)
    # Both PDFs overflow the name cell at105. Whole-page text retains the actual label.
    whole_text = " ".join(clean(p.extract_text()) for p in PdfReader(path).pages)
    if not re.search(r"105\s+Repurchase Make Whole Proceeds Flag\s*Indicates", whole_text):
        raise ValueError("Field105 overflow recovery not supported by document text")
    for row in fields:
        if row["position"] == 105:
            row["name"] = "Repurchase Make Whole Proceeds Flag"
            row["extraction_note"] = (
                "Recovered overflow label from full-page text; visually checked"
            )
    return fields


def extract_workbook(path, source, expected_sha):
    from openpyxl import load_workbook

    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("Pinned documentation hash mismatch")
    book = load_workbook(path, data_only=True, read_only=True)
    output = {}
    for sheet in book:
        rows = []
        for cells in sheet.values:
            row = metadata_row(cells, source + ":" + sheet.title)
            if row:
                rows.append(row)
        output[sheet.title] = rows
    book.close()
    return output


def extract_legacy(path, expected_sha):
    from pypdf import PdfReader

    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("Pinned legacy documentation hash mismatch")
    text = "\n".join(p.extract_text() for p in PdfReader(path).pages)
    acquisition = text.split("Acquisition File Layout", 1)[1].split("Performance File Layout", 1)[0]
    performance = text.split("Performance File Layout", 1)[1]
    groups = {}
    for name, content in [("acquisition", acquisition), ("performance", performance)]:
        rows = []
        for position, label in re.findall(
            r"^\s*(\d+)\s+(.+?)\s+(?:ALPHA-NUMERIC|NUMERIC|DATE)\s+", content, re.M
        ):
            rows.append(dict(position=int(position), name=clean(label), source="legacy_2017"))
        expected = 25 if name == "acquisition" else 31
        if [r["position"] for r in rows] != list(range(1, expected + 1)):
            raise ValueError("Incomplete legacy documentation extraction")
        groups[name] = rows
    return groups


def main():
    manifest = json.loads((PRIVATE / "documentation_retrieval_manifest.json").read_text())
    sources = {s["id"]: s for s in manifest}
    old = json.loads((ROOT / "docs/track_b/fannie_source_registry.json").read_text())
    current = next(s for s in old["sources"] if s["id"] == "layout")
    for tag, path, sha in [
        ("layout_current_pdf", ROOT / current["private_path"], current["sha256"]),
        ("layout_2024_pdf", PRIVATE / "layout_2024.pdf", sources["layout_2024"]["sha256"]),
    ]:
        (PRIVATE / (tag + "_rows.json")).write_text(
            json.dumps(extract_pdf(path, tag, sha), indent=2) + "\n", encoding="utf-8"
        )
    for tag in ["layout_1225", "layout_current", "vs4_changes"]:
        sheets = extract_workbook(PRIVATE / (tag + ".xlsx"), tag, sources[tag]["sha256"])
        for sheet, fields in sheets.items():
            suffix = "_definitions" if sheet == "Definitions" else ""
            (PRIVATE / (tag + suffix + "_rows.json")).write_text(
                json.dumps(fields, indent=2) + "\n", encoding="utf-8"
            )
    legacy = extract_legacy(PRIVATE / "legacy_2017.pdf", sources["legacy_2017"]["sha256"])
    (PRIVATE / "legacy_2017_rows.json").write_text(
        json.dumps(legacy, indent=2) + "\n", encoding="utf-8"
    )
    print("Official documentation tables extracted; no loan records accessed")


if __name__ == "__main__":
    main()
