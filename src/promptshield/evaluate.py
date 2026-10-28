"""Evaluate a fixed checkpoint/threshold without fitting on the held-out test set."""
import argparse
from collections import defaultdict
from pathlib import Path
import time

from torch.utils.data import DataLoader

from promptshield.classifier import InjectionClassifier
from promptshield.common import choose_device, read_json, read_jsonl, sha256, synchronize, write_json, write_jsonl
from promptshield.data import Collator, TextDataset
from promptshield.scoring import metrics
from promptshield.train import predict_loader


def evaluate(checkpoint, split, output, device_name="auto", batch_size=8):
    checkpoint, split, output = Path(checkpoint), Path(split), Path(output)
    device = choose_device(device_name)
    model, tokenizer, metadata = InjectionClassifier.load(checkpoint, device)
    manifest_path = Path(metadata["config"]["data_dir"]) / "manifest.json"
    manifest = read_json(manifest_path)
    if sha256(manifest_path) != metadata["data_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from checkpoint training provenance.")
    if sha256(split) != manifest["splits"][split.stem]["sha256"]:
        raise ValueError("Evaluation split failed manifest verification.")
    rows = read_jsonl(split)
    dataset = TextDataset(rows, tokenizer, metadata["max_length"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=Collator(tokenizer))
    synchronize(device)
    start = time.perf_counter()
    labels, scores, loss = predict_loader(model, loader, device)
    synchronize(device)
    duration = time.perf_counter() - start
    report = {"checkpoint": str(checkpoint), "checkpoint_weights_sha256": sha256(checkpoint / "model.safetensors"),
              "split": str(split), "split_sha256": sha256(split), "diagnostic_checkpoint": metadata["diagnostic_run"],
              "metrics": metrics(labels, scores, metadata["threshold"]), "loss": loss,
              "inference_seconds": duration, "batched_seconds_per_example": duration / len(rows),
              "threshold_fitted_on": "validation only", "by_source": {}, "by_character_length": {}}
    slices = defaultdict(list)
    lengths = defaultdict(list)
    predictions = []
    for row, label, score in zip(rows, labels, scores):
        slices[row["source"]].append((label, score))
        band = "short_lt500" if len(row["prompt"]) < 500 else "medium_500_2000" if len(row["prompt"]) < 2000 else "long_ge2000"
        lengths[band].append((label, score))
        predictions.append({"id": row["id"], "source": row["source"], "label": label,
                            "injection_score": score, "predicted": int(score >= metadata["threshold"])})
    for target, collection in (("by_source", slices), ("by_character_length", lengths)):
        for key, pairs in collection.items():
            ys, ps = zip(*pairs)
            report[target][key] = metrics(ys, ps, metadata["threshold"])
    write_json(output / "metrics.json", report)
    write_jsonl(output / "predictions.jsonl", predictions)
    # Plots are generated from actual measurements only.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import PrecisionRecallDisplay
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cm = report["metrics"]["confusion_matrix"]
    array = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    axes[0].imshow(array, cmap="Blues")
    for (i, j), value in np.ndenumerate(array):
        axes[0].text(j, i, str(value), ha="center", va="center")
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benign", "Injection"], yticklabels=["Benign", "Injection"],
                xlabel="Predicted", ylabel="Actual", title="Held-out confusion matrix")
    PrecisionRecallDisplay.from_predictions(labels, scores, ax=axes[1])
    axes[1].set_title("Precision–recall curve")
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=160)
    plt.close(fig)
    history_path = checkpoint.parent / "history.jsonl"
    if history_path.exists():
        history = read_jsonl(history_path)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["update"] for r in history], [r["train_loss"] for r in history], label="Train loss (epoch-to-date)")
        ax.plot([r["update"] for r in history], [r["validation_loss"] for r in history], label="Validation loss")
        ax.set(xlabel="Optimizer update", ylabel="Cross-entropy", title="Measured learning curves")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "learning-curves.png", dpi=160)
        plt.close(fig)
    print(report["metrics"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="data/splits/piguard-v1/test.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split, args.output, args.device, args.batch_size)


if __name__ == "__main__":
    main()
