"""Explicit verification of archived, hash-locked scientific protocol records."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT/"docs/provenance/frozen-protocols.json"


def frozen_protocol_sha256(identifier, archive=ARCHIVE):
    """Hash the original protocol bytes, never the revised reader-facing report.

    Identifier is a historical archive key, not a promise that the old Markdown
    path still exists. The general file sha256 helper retains its normal behavior.
    """
    records = json.loads(Path(archive).read_text(encoding="utf-8"))["protocols"]
    key = Path(identifier).as_posix()
    if key not in records:
        raise ValueError(f"No frozen protocol record: {key}")
    record = records[key]
    actual = hashlib.sha256(record["content_utf8"].encode("utf-8")).hexdigest()
    if actual != record["sha256"]:
        raise ValueError(f"Frozen protocol archive was modified: {key}")
    return actual
