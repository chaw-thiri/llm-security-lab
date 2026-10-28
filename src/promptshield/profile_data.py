"""Measure training-input token lengths, without inspecting or tuning on test outcomes."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

from promptshield.common import read_jsonl, sha256, write_json


def profile(split, backbone, output, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(backbone, local_files_only=True, trust_remote_code=False)
    rows = read_jsonl(split)
    lengths = []
    for start in range(0, len(rows), 256):
        encoded = tokenizer([r["prompt"] for r in rows[start:start+256]], truncation=False,
                            padding=False, return_length=True, verbose=False)
        lengths.extend(encoded["length"])
    def summarize(values):
        a = np.asarray(values)
        return {"rows": len(a), "median_tokens": float(np.median(a)), "p95_tokens": float(np.percentile(a, 95)),
                "max_tokens": int(a.max()), "above_cap": int((a > max_length).sum()),
                "fraction_above_cap": float((a > max_length).mean())}
    groups = defaultdict(list)
    for row, length in zip(rows, lengths):
        groups[str(row["label"])].append(length)
    report = {"split_sha256": sha256(split), "split": str(split), "max_length_including_special_tokens": max_length,
              "all": summarize(lengths), "by_label": {key: summarize(value) for key, value in groups.items()},
              "scope": "Training input lengths only. Exceeding cap means content would be truncated; it does not identify the attack's location."}
    write_json(output, report)
    print(report, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="data/splits/piguard-v1/train.jsonl")
    parser.add_argument("--backbone", default="data/raw/piguard/backbone")
    parser.add_argument("--output", default="docs/token-length-profile.json")
    args = parser.parse_args()
    profile(args.split, args.backbone, args.output)


if __name__ == "__main__":
    main()
