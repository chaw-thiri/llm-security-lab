"""Validate human review attestations before exporting any trainable task/document data."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from promptshield.common import read_json, read_jsonl, write_json, write_jsonl, sha256


def validate_reviews(reviews, ids):
    by_id = {}
    for row in reviews:
        key = row.get("id")
        if key not in ids or key in by_id:
            raise ValueError("Unknown or duplicate review ID")
        label = row.get("human_label")
        if not ((type(label) is int and label in (0, 1)) or label == "exclude"):
            raise ValueError(f"Human review pending or invalid label: {key}")
        if row.get("human_review_confirmed") is not True:
            raise ValueError(f"Missing human review attestation: {key}")
        if not isinstance(row.get("reviewer_id"), str) or not row["reviewer_id"].strip() or not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
            raise ValueError(f"Missing reviewer identity/rationale: {key}")
        try:
            datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
        except (ValueError, KeyError, AttributeError, TypeError):
            raise ValueError(f"Missing valid ISO review timestamp: {key}") from None
        by_id[key] = row
    if set(by_id) != ids:
        raise ValueError(f"Human review missing for {len(ids - set(by_id))} candidates")
    return by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    args = parser.parse_args()
    root = Path("data/splits/external-tasks-v2-candidates")
    destination = Path("data/splits/external-tasks-v2-reviewed")
    if destination.exists():
        raise FileExistsError(destination)
    manifest = read_json(root / "manifest.json")
    rows = []
    for split, record in manifest["partitions"].items():
        path = root / f"{split}.candidates.jsonl"
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Candidate partition changed: {split}")
        rows.extend(read_jsonl(path))
    reviews = validate_reviews(read_jsonl(args.reviews), {r["id"] for r in rows})
    groups = {"train": [], "validation": [], "test": []}
    counts = Counter()
    for row in rows:
        review = reviews[row["id"]]
        if review["human_label"] == "exclude":
            counts["human_excluded"] += 1
            continue
        if not row["within_512_tokens"]:
            counts["long_context_retained_outside_512_token_export"] += 1
            continue
        split = "test" if row["split"] == "challenge" else row["split"]
        groups[split].append({"id": row["id"], "prompt": f"TRUSTED TASK:\n{row['task']}\n\nUNTRUSTED DOCUMENT:\n{row['document']}",
                              "label": review["human_label"], "source": "BIPIA-external-tasks-v2",
                              "context_group": row["context_group"], "task_type": row["task_type"]})
        counts["corrected_labels"] += review["human_label"] != row["proposed_label"]
    for split, items in groups.items():
        if {r["label"] for r in items} != {0, 1}:
            raise ValueError(f"Reviewed 512-token {split} lacks both labels; do not export a misleading dataset")
    output = {"status": "human_review_attested", "candidate_manifest_sha256": sha256(root / "manifest.json"),
              "reviews_sha256": sha256(args.reviews), "reviewed_rows": len(rows), "counts": dict(counts),
              "note": "Review attestations are recorded, not independently verified by software. Long rows are not truncated.", "splits": {}}
    for split, items in groups.items():
        path = destination / f"{split}.jsonl"
        write_jsonl(path, items)
        output["splits"][split] = {"rows": len(items), "sha256": sha256(path),
                                    "labels": dict(Counter(str(r["label"]) for r in items))}
    write_json(destination / "manifest.json", output)
    print(output)


if __name__ == "__main__":
    main()
