"""Minimal deterministic preparation; source is metadata, never a tokenized feature."""
import argparse
from collections import Counter, defaultdict
import hashlib
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from promptshield.common import read_json, read_jsonl, sha256, write_json, write_jsonl


def normalized(text):
    return " ".join(text.lower().split())


def clean_rows(rows, reserved=()):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row.get("prompt"), str) or type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError(f"Invalid prompt/label at row {index}; do not coerce unknown labels.")
        if not isinstance(row.get("source"), str):
            raise ValueError(f"Missing source metadata at row {index}.")
        groups[normalized(row["prompt"])].append(row)
    excluded = {normalized(row["prompt"]) for row in reserved}
    stats = Counter(raw_rows=len(rows), normalized_duplicate_excess=len(rows) - len(groups))
    clean = []
    for key, group in sorted(groups.items()):
        if not key:
            stats["blank_rows"] += len(group)
            continue
        if len({row["label"] for row in group}) != 1:
            stats["conflicting_groups"] += 1
            stats["conflicting_rows"] += len(group)
            continue
        if key in excluded:
            stats["reserved_overlap_groups"] += 1
            continue
        # Preserve the original text, but choose a representative independently of file order.
        row = min(group, key=lambda r: (r["source"], r["prompt"]))
        clean.append({"id": hashlib.sha256(key.encode()).hexdigest(), "prompt": row["prompt"],
                      "label": row["label"], "source": row["source"],
                      "sources": sorted({r["source"] for r in group})})
    stats["usable_rows"] = len(clean)
    return clean, dict(stats)


def split_rows(rows, seed=42):
    strata = defaultdict(list)
    for row in rows:
        strata[(row["label"], row["source"])].append(row)
    result = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda r: r["id"])
        rng.shuffle(group)
        # Small strata remain in training; report this rather than duplicate observations.
        n_eval = max(1, round(len(group) * 0.1)) if len(group) >= 10 else 0
        result["validation"].extend(group[:n_eval])
        result["test"].extend(group[n_eval:2 * n_eval])
        result["train"].extend(group[2 * n_eval:])
    for group in result.values():
        rng.shuffle(group)
    sets = [set(r["id"] for r in result[name]) for name in result]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(map(len, result.values())) == len(rows)
    return result


def prepare(raw, output, seed=42):
    raw, output = Path(raw), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; choose a new directory to preserve locked splits.")
    rows, stats = clean_rows(read_json(raw / "train.json"), read_json(raw / "valid.json"))
    splits = split_rows(rows, seed)
    manifest = {"seed": seed, "cleaning": stats, "splits": {},
                "raw_provenance": read_json(raw / "provenance.json"),
                "limitations": ["Case/whitespace duplicate exclusion only; semantic/template overlap may remain.",
                                "Some sources are single-class; random stratification is not a source-held-out test.",
                                "Official validation is reserved, not mixed with held-out test."]}
    for name, group in splits.items():
        path = output / f"{name}.jsonl"
        write_jsonl(path, group)
        manifest["splits"][name] = {"rows": len(group), "sha256": sha256(path),
            "labels": dict(Counter(str(r["label"]) for r in group)),
            "sources": dict(Counter(r["source"] for r in group))}
    write_json(output / "manifest.json", manifest)
    print(manifest["cleaning"])
    print({name: len(rows) for name, rows in splits.items()})
    return manifest


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        encoded = tokenizer([r["prompt"] for r in rows], truncation=True, max_length=max_length, padding=False)
        self.encodings = [{"input_ids": ids, "attention_mask": mask} for ids, mask in
                          zip(encoded["input_ids"], encoded["attention_mask"])]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return dict(self.encodings[index], labels=self.rows[index]["label"])


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = torch.tensor([r["labels"] for r in features], dtype=torch.long)
        batch = self.tokenizer.pad([{k: v for k, v in r.items() if k != "labels"} for r in features],
                                   padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/piguard")
    parser.add_argument("--output", default="data/splits/piguard-v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.seed)


if __name__ == "__main__":
    main()
