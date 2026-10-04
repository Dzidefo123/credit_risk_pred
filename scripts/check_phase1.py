"""Phase 1 preservation and cleanup checks; uses only the standard library."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    inventory = json.loads((root / "reports/phase1_inventory.json").read_text(encoding="utf-8"))
    assert git("branch", "--show-current") == "feature/risk-modeling-lab"
    tracked = set(git("ls-files").splitlines())
    for group in ("preserved_sha256", "local_artifact_sha256"):
        for name, expected in inventory[group].items():
            relative = "docs/history/README_v1.md" if group == "preserved_sha256" and name == "README.md" else name
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, name
            if group == "preserved_sha256":
                assert relative in tracked, relative
            else:
                assert name not in tracked, name
                assert git("check-ignore", "--", name) == name, name
    assert not any(p.startswith(("venv/", "__pycache__/")) for p in tracked)
    assert (root / "venv/pyvenv.cfg").is_file(), "Local environment was removed"
    assert (root / "__pycache__/app.cpython-38.pyc").is_file(), "Local cache was removed"
    for name in ("app.py", "main.py", "scripts/check_phase1.py"):
        ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
    notebook = json.loads((root / "credit_risk(1).ipynb").read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 73
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"notebook-cell-{index}")
    print("PASS: preserved V1 hashes, retained local artifacts, index cleanup, ignore rules, and syntax")


if __name__ == "__main__":
    main()
