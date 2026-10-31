"""Evaluate false alarms on pinned NotInject prompts after exact-overlap exclusion."""
from pathlib import Path
from collections import defaultdict

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from torch.utils.data import DataLoader

from promptshield.classifier import InjectionClassifier
from promptshield.common import choose_device, read_json, read_jsonl, sha256, write_json
from promptshield.data import Collator, TextDataset, normalized
from promptshield.train import predict_loader

REPO = "leolee99/NotInject"
REVISION = "9fdfe4368fc43c592e601f8fc6217de7c777f9e7"
FILES = {
    "one": "data/NotInject_one-00000-of-00001.parquet",
    "two": "data/NotInject_two-00000-of-00001.parquet",
    "three": "data/NotInject_three-00000-of-00001.parquet",
}
CHECKPOINTS = {
    "baseline": "runs/baseline-v1/models/full/best",
    **{f"ce-{seed}": f"runs/context-v2-ce-{seed}/best" for seed in (42, 43, 44)},
    **{f"ranking-{seed}": f"runs/context-v2-ranking-{seed}/best" for seed in (42, 43, 44)},
}
BASELINE_THRESHOLD = 0.002023694571107626


def load_benchmark(destination):
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError(f"Hub resolved an unexpected benchmark revision: {info.sha}")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    rows, artifacts = [], {}
    for subset, filename in FILES.items():
        local = Path(hf_hub_download(REPO, filename, repo_type="dataset", revision=REVISION,
                                     local_dir=root))
        subset_rows = pq.read_table(local).to_pylist()
        for index, row in enumerate(subset_rows):
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"Invalid prompt in {filename} row {index}")
            rows.append({"id": f"{subset}-{index}", "prompt": row["prompt"], "label": 0,
                         "category": row.get("category", "unknown"),
                         "trigger_count": len(row.get("word_list") or []), "subset": subset})
        artifacts[filename] = {"sha256": sha256(local), "bytes": local.stat().st_size,
                               "rows": len(subset_rows)}
    if len(rows) != 339:
        raise ValueError(f"Expected 339 published NotInject prompts, found {len(rows)}")
    return rows, artifacts


def main():
    rows, artifacts = load_benchmark("data/raw/external/notinject")
    split_root = Path("data/splits/piguard-v1")
    split_manifest = read_json(split_root / "manifest.json")
    seen, overlap_by_split = set(), {}
    for split in ("train", "validation", "test"):
        path = split_root / f"{split}.jsonl"
        if sha256(path) != split_manifest["splits"][split]["sha256"]:
            raise ValueError(f"Original {split} split changed; refusing an overlap claim.")
        keys = {normalized(row["prompt"]) for row in read_jsonl(path)}
        overlap_by_split[split] = [row["id"] for row in rows if normalized(row["prompt"]) in keys]
        seen.update(keys)
    # This upstream holdout contributed 48 NotInject rows to preprocessing. Although
    # the trained checkpoints never consumed it, exclude those examples for a strict blind test.
    reserved_path = Path("data/raw/piguard/valid.json")
    reserved_rows = read_json(reserved_path)
    reserved_keys = {normalized(row["prompt"]) for row in reserved_rows}
    overlap_by_split["reserved_upstream_validation"] = [
        row["id"] for row in rows if normalized(row["prompt"]) in reserved_keys
    ]
    seen.update(reserved_keys)
    eligible = [row for row in rows if normalized(row["prompt"]) not in seen]
    report = {
        "benchmark": {"repo": REPO, "revision": REVISION, "license": "MIT",
                      "published_rows": len(rows), "files": artifacts},
        "overlap_audit": {
            "normalization": "lowercase and collapse whitespace, matching PromptShield split preparation",
            "overlap_ids_by_split": overlap_by_split,
            "unique_overlapping_ids": sorted({i for ids in overlap_by_split.values() for i in ids}),
            "excluded_rows": len(rows) - len(eligible), "eligible_rows": len(eligible),
            "original_manifest_sha256": sha256(split_root / "manifest.json"),
            "reserved_upstream_validation_sha256": sha256(reserved_path),
            "limitation": "Exact normalized overlap only; paraphrase/template overlap is not detected."
        },
        "scope": "All remaining examples are benign trigger-word probes. This measures over-defense, not attack recall or end-to-end document safety.",
        "models": {}
    }
    device = choose_device("mps")
    for name, checkpoint in CHECKPOINTS.items():
        path = Path(checkpoint)
        model, tokenizer, metadata = InjectionClassifier.load(path, device)
        predictions = []
        dataset = TextDataset(eligible, tokenizer, metadata["max_length"])
        loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=Collator(tokenizer))
        labels, scores, _ = predict_loader(model, loader, device)
        for row, score in zip(eligible, scores):
            predictions.append({"id": row["id"], "subset": row["subset"],
                                "category": row["category"], "trigger_count": row["trigger_count"],
                                "score": score,
                                "flagged_at_checkpoint_threshold": score >= metadata["threshold"],
                                "flagged_at_baseline_threshold": score >= BASELINE_THRESHOLD})
        result = {
            "checkpoint": checkpoint,
            "weights_sha256": sha256(path / "model.safetensors"),
            "threshold_selected_on_original_validation": metadata["threshold"],
            "eligible_rows": len(eligible),
            "flagged_at_own_threshold": sum(r["flagged_at_checkpoint_threshold"] for r in predictions),
            "false_positive_rate_at_own_threshold": sum(r["flagged_at_checkpoint_threshold"] for r in predictions) / len(eligible),
            "flagged_at_fixed_baseline_threshold": sum(r["flagged_at_baseline_threshold"] for r in predictions),
            "false_positive_rate_at_fixed_baseline_threshold": sum(r["flagged_at_baseline_threshold"] for r in predictions) / len(eligible),
            "by_subset": {}, "by_category": {}, "by_trigger_count": {}, "predictions": predictions
        }
        for field, target in (("subset", result["by_subset"]), ("category", result["by_category"]),
                              ("trigger_count", result["by_trigger_count"])):
            groups = defaultdict(list)
            for prediction in predictions:
                groups[str(prediction[field])].append(prediction)
            for key, group in sorted(groups.items()):
                target[key] = {"rows": len(group),
                               "flagged_at_own_threshold": sum(r["flagged_at_checkpoint_threshold"] for r in group),
                               "fpr_at_own_threshold": sum(r["flagged_at_checkpoint_threshold"] for r in group) / len(group)}
        report["models"][name] = result
        write_json("docs/notinject-evaluation.json", report)
        print(name, result["flagged_at_own_threshold"], "/", len(eligible),
              f"({result['false_positive_rate_at_own_threshold']:.1%})", flush=True)
        del model
    print("Excluded exact overlaps:", report["overlap_audit"]["excluded_rows"], flush=True)


if __name__ == "__main__":
    main()
