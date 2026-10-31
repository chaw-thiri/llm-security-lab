"""Offline checkpoint inference; optional overlapping chunks for long documents."""
import argparse
import json
from pathlib import Path
import time

import torch

from promptshield.classifier import InjectionClassifier
from promptshield.common import choose_device, synchronize


class Detector:
    def __init__(self, checkpoint, device="auto"):
        self.device = choose_device(device)
        self.model, self.tokenizer, self.metadata = InjectionClassifier.load(checkpoint, self.device)
        self.threshold = self.metadata["threshold"]
        self.max_length = self.metadata["max_length"]

    def score(self, text, chunk=False, task=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Provide nonempty text.")
        input_format = self.metadata.get("config", {}).get("input_format", "document")
        if input_format == "task_document":
            if not isinstance(task, str) or not task.strip():
                raise ValueError("This checkpoint requires the trusted task as well as the document.")
            text = f"TRUSTED TASK:\n{task}\n\nUNTRUSTED DOCUMENT:\n{text}"
        ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        capacity = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        if capacity < 1:
            raise ValueError("Checkpoint max_length leaves no room for text.")
        if chunk:
            encoded = self.tokenizer(text, max_length=self.max_length, truncation=True,
                                     stride=capacity // 2, return_overflowing_tokens=True, padding=False)
            windows, masks = encoded["input_ids"], encoded["attention_mask"]
        else:
            encoded = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False)
            windows, masks = [encoded["input_ids"]], [encoded["attention_mask"]]
        probabilities = []
        synchronize(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            for sequence, mask in zip(windows, masks):
                batch = {"input_ids": torch.tensor([sequence], device=self.device),
                         "attention_mask": torch.tensor([mask], dtype=torch.long, device=self.device)}
                probabilities.append(float(self.model(**batch).softmax(-1)[0, 1].item()))
        synchronize(self.device)
        score = max(probabilities)
        return {"injection_score": score, "threshold": self.threshold, "blocked": score >= self.threshold,
                "label": "injection" if score >= self.threshold else "benign", "tokens": len(ids),
                "chunks": len(windows), "truncated": len(ids) > capacity and not chunk,
                "latency_seconds": time.perf_counter() - start,
                "policy": "max_over_chunks" if chunk else "truncate", "diagnostic_checkpoint": self.metadata["diagnostic_run"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--file", type=Path)
    parser.add_argument("--task", help="Required by task-conditioned checkpoints.")
    parser.add_argument("--chunk", action="store_true", help="Max score across overlapping windows; threshold needs separate validation for this policy.")
    args = parser.parse_args()
    text = args.file.read_text(encoding="utf-8") if args.file else args.text
    print(json.dumps(Detector(args.checkpoint, args.device).score(text, args.chunk, args.task), indent=2))


if __name__ == "__main__":
    main()
