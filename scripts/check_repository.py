"""Audit tracked-file hygiene, preserved evidence and current documentation links."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

FORBIDDEN_PARTS = {".venv", "venv", "__pycache__", "artifacts", "dist", "mlruns", "mlartifacts"}
FORBIDDEN_SUFFIXES = {".csv", ".pkl", ".joblib", ".pyc", ".db", ".h5", ".keras"}
DOCUMENTS = ("README.md", "docs/history/README.md", "reports/final_repository_audit.md")


def audit(root):
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], text=True).split(
        "\0"
    )
    tracked = [name for name in tracked if name]
    for name in tracked:
        path = PurePosixPath(name)
        if (
            FORBIDDEN_PARTS.intersection(path.parts)
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
            or path.name.startswith(".env")
            or (path.suffix == ".ipynb" and name != "credit_risk(1).ipynb")
        ):
            raise ValueError(f"Generated/private file tracked: {name}")
        if (root / name).stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f"Tracked file exceeds 10 MiB audit limit: {name}")
    inventory = json.loads((root / "reports/phase1_inventory.json").read_text(encoding="utf-8"))
    for name, expected in inventory["preserved_sha256"].items():
        relative = "docs/history/README_v1.md" if name == "README.md" else name
        if relative not in tracked:
            raise ValueError(f"Historical evidence untracked: {relative}")
        payload = (root / relative).read_bytes()
        # V1 files may be checked out as LF on Linux; the README archive is binary.
        candidates = [payload]
        if name != "README.md":
            candidates.append(payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        if expected not in [hashlib.sha256(value).hexdigest() for value in candidates]:
            raise ValueError(f"Historical evidence changed: {relative}")
    evidence = json.loads((root / "reports/phase4_experiment.json").read_text(encoding="utf-8"))
    locations = {
        "pd.py": "models/pd.py",
        "origination.py": "features/origination.py",
        "config.py": "utils/config.py",
        "metrics.py": "validation/metrics.py",
    }
    for name, expected in evidence["experiment"]["source_code_sha256"].items():
        payload = (root / "src/credit_risk" / locations[name]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Frozen model source bytes changed: {name}")
    links = 0
    for name in DOCUMENTS:
        document = root / name
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            url = urlsplit(target)
            if url.scheme or not url.path:
                continue
            destination = (document.parent / unquote(url.path)).resolve()
            if not destination.is_relative_to(root) or not destination.is_file():
                raise ValueError(f"Broken/outside-project link: {name}: {target}")
            links += 1
    return {
        "status": "passed",
        "tracked_files": len(tracked),
        "preserved_v1_files": len(inventory["preserved_sha256"]),
        "frozen_training_sources": len(locations),
        "current_document_links": links,
        "history_rewritten": False,
        "original_data_or_model_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(audit(args.root.resolve())))


if __name__ == "__main__":
    main()
