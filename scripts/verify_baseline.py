"""Verify the local baseline snapshot against committed SHA256/size records. No ML dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(snapshot, manifest):
    issues = []
    for artifact in manifest["artifacts"]:
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact paths must stay within the snapshot.")
        path = snapshot / relative
        if path.is_symlink() or not path.is_file():
            issues.append(f"Missing or non-regular file: {relative}")
        elif path.stat().st_size != artifact["bytes"]:
            issues.append(f"Size mismatch: {relative}")
        elif file_sha256(path) != artifact["sha256"]:
            issues.append(f"SHA256 mismatch: {relative}")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/baselines/v1.json")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    snapshot = args.snapshot or ROOT / manifest["local_snapshot"]
    issues = verify(snapshot, manifest)
    print(json.dumps({"ok": not issues, "snapshot": str(snapshot), "checked_files": len(manifest["artifacts"]),
                      "issues": issues}, indent=2))
    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
