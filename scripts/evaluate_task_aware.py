"""Score the locked task-aware challenge and fixed-task NotInject probe set."""
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from torch.utils.data import DataLoader

from promptshield.classifier import InjectionClassifier
from promptshield.common import choose_device, read_json, read_jsonl, sha256, write_json
from promptshield.data import Collator, TextDataset, normalized
from promptshield.scoring import metrics
from promptshield.train import predict_loader
from prepare_task_aware import PAYLOADS


ROOT = Path("results/task-aware-v1")
NOTINJECT_FILES = {
    "one": "data/NotInject_one-00000-of-00001.parquet",
    "two": "data/NotInject_two-00000-of-00001.parquet",
    "three": "data/NotInject_three-00000-of-00001.parquet",
}
FIXED_TASK = "Summarize the following document in one factual sentence."


def make_notinject_rows():
    base = Path("data/raw/external/notinject")
    rows = []
    for subset, filename in NOTINJECT_FILES.items():
        table = pq.read_table(base / filename).to_pylist()
        for i, row in enumerate(table):
            rows.append({"id": f"{subset}-{i}", "prompt": row["prompt"], "label": 0,
                         "subset": subset, "trigger_count": len(row.get("word_list") or [])})
    if len(rows) != 339:
        raise ValueError(f"Expected 339 pinned NotInject rows, found {len(rows)}")
    excluded = set()
    split_root = Path("data/splits/piguard-v1")
    split_manifest = read_json(split_root / "manifest.json")
    for split in ("train", "validation", "test"):
        path = split_root / f"{split}.jsonl"
        if sha256(path) != split_manifest["splits"][split]["sha256"]:
            raise ValueError(f"Original {split} split changed.")
        excluded.update(normalized(r["prompt"]) for r in read_jsonl(path))
    reserved = Path("data/raw/piguard/valid.json")
    reserved_rows = read_json(reserved)
    excluded.update(normalized(r["prompt"]) for r in reserved_rows)
    eligible = [r for r in rows if normalized(r["prompt"]) not in excluded]
    if len(eligible) != 291:
        raise ValueError(f"Expected the locked 291-row NotInject subset, found {len(eligible)}")
    return eligible, {"published_rows": len(rows), "excluded_exact_overlaps": 48,
                      "eligible_rows": len(eligible), "fixed_task": FIXED_TASK,
                      "normalization": "lowercase and collapsed whitespace"}


def evaluate(checkpoint, rows):
    device = choose_device("mps")
    model, tokenizer, metadata = InjectionClassifier.load(checkpoint, device)
    dataset = TextDataset(rows, tokenizer, metadata["max_length"])
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=Collator(tokenizer))
    labels, scores, loss = predict_loader(model, loader, device)
    result = {"checkpoint": checkpoint, "weights_sha256": sha256(Path(checkpoint) / "model.safetensors"),
              "threshold": metadata["threshold"], "loss": loss,
              "metrics": metrics(labels, scores, metadata["threshold"]),
              "predictions": [{"id": r["id"], "label": y, "score": s}
                              for r, y, s in zip(rows, labels, scores)]}
    del model
    return result


def main():
    study = read_json("docs/task-aware-study.json")
    model_paths = {"baseline": "runs/baseline-v1/models/full/best",
                   "context-v2-ce-42": "runs/context-v2-ce-42/best"}
    for seed in (42, 43, 44):
        model_paths[f"document-{seed}"] = f"runs/task-aware-document-{seed}/best"
        model_paths[f"task-document-{seed}"] = f"runs/task-aware-task-document-{seed}/best"

    modes = {"baseline": "document", "context-v2-ce-42": "document"}
    modes.update({f"document-{seed}": "document" for seed in (42, 43, 44)})
    modes.update({f"task-document-{seed}": "task_document" for seed in (42, 43, 44)})
    challenge_rows = {}
    for arm in ("document", "task-document"):
        path = Path(f"data/splits/task-aware-v1/{arm}/test.jsonl")
        manifest = read_json(path.parent / "manifest.json")
        if sha256(path) != manifest["splits"]["test"]["sha256"]:
            raise ValueError(f"Locked {arm} challenge changed.")
        challenge_rows[arm] = read_jsonl(path)

    notinject, overlap = make_notinject_rows()
    report = {"scope": "Synthetic family-held-out challenge plus a fixed-task external benign trigger-word probe; no production claim.",
              "challenge_sha256": {arm: sha256(Path(f"data/splits/task-aware-v1/{arm}/test.jsonl")) for arm in challenge_rows},
              "notinject": overlap, "models": {}}
    for name, checkpoint in model_paths.items():
        arm = "task-document" if modes[name] == "task_document" else "document"
        challenge = challenge_rows[arm]
        score_rows = [{"id": row["id"], "label": row["label"], "prompt": row["prompt"]} for row in challenge]
        result = evaluate(checkpoint, score_rows)
        result["challenge_by_family"] = {}
        for family in sorted({r["family"] for r in challenge}):
            indexes = [i for i, row in enumerate(challenge) if row["family"] == family]
            result["challenge_by_family"][family] = metrics(
                [score_rows[i]["label"] for i in indexes],
                [result["predictions"][i]["score"] for i in indexes], result["threshold"])
        # Post-hoc paired stress check. Quotes are added around each payload after
        # the locked challenge was scored; scores do not affect thresholds or selection.
        quote_rows = []
        for row in challenge:
            parts = row["pair_id"].split("-")
            if len(parts) != 4 or parts[0].lower() != "challenge":
                raise ValueError(f"Unexpected challenge pair ID: {row['pair_id']}")
            family_i, payload_i, task_i = map(int, parts[1:])
            marker = f"REDIRECT-CHALLENGE-{family_i}-{payload_i}-{task_i}"
            payload = PAYLOADS["challenge"][payload_i].format(marker=marker)
            document = row["document"]
            if payload not in document:
                raise ValueError(f"Could not locate challenge payload for {row['id']}")
            document = document.replace(payload, f'"{payload}"', 1)
            prompt = document if modes[name] == "document" else f"TRUSTED TASK:\n{row['task']}\n\nUNTRUSTED DOCUMENT:\n{document}"
            quote_rows.append({"id": row["id"], "label": row["label"], "prompt": prompt})
        quote_result = evaluate(checkpoint, quote_rows)
        result["challenge_quoted_payload_stress"] = {"metrics": quote_result["metrics"],
                                                       "predictions": quote_result["predictions"],
                                                       "method": "Added double quotation marks around the redirect payload in each locked challenge document; same checkpoint and development-selected threshold; post-hoc diagnostic only."}
        if modes[name] == "task_document":
            ni_rows = [dict(r, prompt=f"TRUSTED TASK:\n{FIXED_TASK}\n\nUNTRUSTED DOCUMENT:\n{r['prompt']}") for r in notinject]
        else:
            ni_rows = [dict(r, prompt=r["prompt"]) for r in notinject]
        ni = evaluate(checkpoint, ni_rows)
        result["notinject"] = {"metrics": ni["metrics"], "predictions": ni["predictions"]}
        report["models"][name] = result
        write_json(ROOT / "classification-comparison.json", report)
        print(name, "challenge", result["metrics"], "NotInject FPR", ni["metrics"]["benign_fpr"], flush=True)
    write_json("docs/task-aware-results.json", report)
    print("TASK-AWARE CLASSIFICATION EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
