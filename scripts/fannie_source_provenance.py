"""Task 13S read-only provenance verification; never open loan members."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path, normalize=False):
    if normalize:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_archive(path, expected):
    if path.name != expected["archive_name"]:
        raise ValueError("Archive filename changed")
    if path.stat().st_size != expected["archive_bytes"]:
        raise ValueError("Archive size changed")
    if digest(path) != expected["archive_sha256"]:
        raise ValueError("Archive hash changed")
    with zipfile.ZipFile(path) as archive:
        actual = [
            (m.filename, m.compress_size, m.file_size, m.compress_type, list(m.date_time))
            for m in archive.infolist()
        ]
    wanted = [
        (
            m["name"],
            m["compressed_bytes"],
            m["uncompressed_bytes"],
            m["compression_method"],
            m["zip_timestamp"],
        )
        for m in expected["members"]
    ]
    if actual != wanted:
        raise ValueError("Archive directory changed")
    return {"status": "PASSED", "member_bodies_opened": False}


def verify_preservation(root, manifest):
    for name, expected in manifest["public_lf_hashes"].items():
        if digest(root / name, normalize=True) != expected:
            raise ValueError("Prior public file changed: " + name)
    for name, expected in manifest["private_byte_hashes"].items():
        if digest(root / name) != expected:
            raise ValueError("Prior private evidence changed: " + name)
    for ref, expected in manifest["git_refs"].items():
        actual = subprocess.check_output(["git", "rev-parse", ref], cwd=root, text=True).strip()
        if actual != expected:
            raise ValueError("Frozen Git ref changed")
    return {
        "status": "PASSED",
        "public": len(manifest["public_lf_hashes"]),
        "private": len(manifest["private_byte_hashes"]),
        "git_refs": len(manifest["git_refs"]),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(
        (ROOT / "docs/track_b/fannie_source_preservation_manifest.json").read_text()
    )
    expected = json.loads(
        (ROOT / "reports/track_b/fannie_release_archive_structure.json").read_text()
    )
    print(
        json.dumps(
            {
                "preservation": verify_preservation(ROOT, manifest),
                "archive": verify_archive(args.archive, expected),
            }
        )
    )
