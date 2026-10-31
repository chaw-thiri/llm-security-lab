"""Inspectable fine-tuning: explicit gradients, optimizer, scheduler and resumable state."""
import argparse
import importlib.metadata
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from promptshield.classifier import InjectionClassifier
from promptshield.common import choose_device, read_json, read_jsonl, seed_everything, sha256, synchronize, write_json
from promptshield.data import Collator, TextDataset
from promptshield.scoring import metrics, select_threshold
from promptshield.pairwise import PairedDataset, PairBatchSampler, ranking_sum


def load_config(path):
    config = read_json(path)
    required = {"backbone", "data_dir", "run_dir", "seed", "epochs", "batch_size", "accumulation_steps",
                "encoder_lr", "head_lr", "weight_decay", "warmup_ratio", "max_length", "pooling",
                "dropout", "frozen", "weighted_loss", "max_fpr", "device"}
    optional = {"initial_checkpoint", "initial_weights_sha256", "paired_batches", "ranking_weight",
                "ranking_margin", "input_format"}
    if set(config) - optional != required:
        missing = required - set(config)
        extra = set(config) - required - optional
        raise ValueError(f"Config keys differ: missing={missing}, extra={extra}")
    if config.get("input_format", "document") not in ("document", "task_document"):
        raise ValueError("input_format must be document or task_document")
    if bool(config.get("initial_checkpoint")) != bool(config.get("initial_weights_sha256")):
        raise ValueError("Continuation requires both initial_checkpoint and initial_weights_sha256.")
    for key in ("epochs", "batch_size", "accumulation_steps"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 2 <= config["max_length"] <= 512 or not 0 <= config["warmup_ratio"] < 1:
        raise ValueError("Invalid length/warm-up configuration")
    if min(config["encoder_lr"], config["head_lr"]) <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid optimizer configuration")
    if config.get("ranking_weight", 0) < 0 or config.get("ranking_margin", 1) <= 0:
        raise ValueError("Ranking weight must be nonnegative and margin positive.")
    if config.get("ranking_weight", 0) and not config.get("paired_batches"):
        raise ValueError("Ranking requires paired_batches.")
    return config


def optimizer_for(model, config):
    groups = []
    for name, module, lr in (("encoder", model.encoder, config["encoder_lr"]), ("head", model.head, config["head_lr"])):
        decay, no_decay = [], []
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                (no_decay if param.ndim < 2 or param_name.endswith("bias") else decay).append(param)
        for params, wd in ((decay, config["weight_decay"]), (no_decay, 0.0)):
            if params:
                groups.append({"params": params, "lr": lr, "weight_decay": wd, "name": name})
    return torch.optim.AdamW(groups)


def schedule(optimizer, total_steps, warmup_ratio):
    warmup = int(total_steps * warmup_ratio)
    def multiplier(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def rng_state(device):
    npstate = np.random.get_state()
    state = {"python": random.getstate(), "numpy": (npstate[0], npstate[1].tolist(), *npstate[2:]),
             "torch": torch.get_rng_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"])
    if "mps" in state and device.type == "mps":
        torch.mps.set_rng_state(state["mps"])
    if "cuda" in state and device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def predict_loader(model, loader, device, weights=None):
    was_training = model.training
    model.eval()
    labels, scores, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            count = len(batch["labels"])
            total_loss += nn.functional.cross_entropy(logits, batch["labels"], weight=weights, reduction="none").sum().item()
            labels.extend(batch["labels"].cpu().tolist())
            scores.extend(logits.softmax(-1)[:, 1].cpu().tolist())
    model.train(was_training)
    return labels, scores, total_loss / len(labels)


def benchmark(model, dataset, tokenizer, device, config, path, steps=30):
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]), collate_fn=Collator(tokenizer))
    optimizer = optimizer_for(model, config)
    model.train()
    durations, counts = [], []
    for index, batch in enumerate(loader):
        if index >= steps + 5:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = nn.functional.cross_entropy(logits, batch["labels"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        synchronize(device)
        if index >= 5:
            durations.append(time.perf_counter() - start)
            counts.append(len(batch["labels"]))
    if not durations:
        raise ValueError("Benchmark needs more than five warm-up batches.")
    throughput = sum(counts) / sum(durations)
    result = {"device": str(device), "measured_batches": len(durations), "warmup_batches": 5,
              "batch_size": config["batch_size"], "max_length": config["max_length"],
              "seconds_per_batch": float(np.mean(durations)), "examples_per_second": throughput,
              "estimated_training_hours": len(dataset) * config["epochs"] / throughput / 3600,
              "scope": "Real encoder/head forward, backward, clipping and AdamW; updates every batch. Excludes tokenization, validation, saves; accumulation cadence differs from main run."}
    if device.type == "mps":
        result["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    write_json(path, result)
    print(result, flush=True)
    return result


def train(config_path, resume=False, limit_train=None, limit_validation=None, max_updates=None, benchmark_steps=None):
    config = load_config(config_path)
    run = Path(config["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists() and not resume:
        raise FileExistsError(f"{run} already contains a run; use --resume or a new run_dir.")
    if resume and (limit_train or limit_validation or max_updates or benchmark_steps):
        raise ValueError("Resume uses the saved data selection and schedule; diagnostic overrides are not allowed.")
    device = choose_device(config["device"])
    seed_everything(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["backbone"], local_files_only=True, trust_remote_code=False)
    data_dir = Path(config["data_dir"])
    manifest = read_json(data_dir / "manifest.json")
    paths = {name: data_dir / f"{name}.jsonl" for name in ("train", "validation")}
    for name, path in paths.items():
        if sha256(path) != manifest["splits"][name]["sha256"]:
            raise ValueError(f"Changed {name} split; reproduce the manifest before training.")
    rows, valid_rows = read_jsonl(paths["train"]), read_jsonl(paths["validation"])
    full_train_count = len(rows)
    if limit_train:
        random.Random(config["seed"]).shuffle(rows)
        rows = rows[:limit_train]
    if limit_validation:
        random.Random(config["seed"] + 1).shuffle(valid_rows)
        valid_rows = valid_rows[:limit_validation]
    if {r["label"] for r in rows} != {0, 1} or {r["label"] for r in valid_rows} != {0, 1}:
        raise ValueError("Training and validation must each contain both labels.")
    dataset_type = PairedDataset if config.get("paired_batches") else TextDataset
    dataset = dataset_type(rows, tokenizer, config["max_length"])
    validation = TextDataset(valid_rows, tokenizer, config["max_length"])
    model = InjectionClassifier.from_backbone(config["backbone"], pooling=config["pooling"],
                  dropout=config["dropout"], frozen=config["frozen"]).to(device)
    if config.get("initial_checkpoint") and not resume:
        initial = Path(config["initial_checkpoint"])
        if sha256(initial / "model.safetensors") != config.get("initial_weights_sha256"):
            raise ValueError("Initial checkpoint hash differs from declared experiment provenance.")
        loaded, initial_tokenizer, initial_metadata = InjectionClassifier.load(initial, device)
        if (initial_metadata["pooling"] != config["pooling"] or
                initial_metadata["frozen"] != config["frozen"] or
                initial_metadata["dropout"] != config["dropout"] or
                initial_metadata["max_length"] != config["max_length"] or
                initial_tokenizer.get_vocab() != tokenizer.get_vocab()):
            raise ValueError("Continuation architecture/tokenizer differs from the initial checkpoint.")
        model.load_state_dict(loaded.state_dict())
        del loaded, initial_tokenizer
    if benchmark_steps:
        return benchmark(model, dataset, tokenizer, device, config, run / "benchmark.json", benchmark_steps)
    optimizer = optimizer_for(model, config)
    batches_per_epoch = math.ceil(len(dataset) / config["batch_size"])
    updates_per_epoch = math.ceil(batches_per_epoch / config["accumulation_steps"])
    scheduler = schedule(optimizer, updates_per_epoch * config["epochs"], config["warmup_ratio"])
    weights = None
    if config["weighted_loss"]:
        counts = torch.bincount(torch.tensor([r["label"] for r in rows]), minlength=2)
        weights = (len(rows) / (2 * counts.float())).to(device)
    metadata = {"config": config, "data_manifest_sha256": sha256(data_dir / "manifest.json"),
                "training_split_sha256": sha256(paths["train"]), "validation_split_sha256": sha256(paths["validation"]),
                "device": str(device), "diagnostic_run": bool(limit_train or limit_validation or max_updates),
                "training_rows": len(rows), "full_training_rows": full_train_count, "validation_rows": len(valid_rows),
                "max_length": config["max_length"], "threshold": 0.5,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "scikit-learn")}}
    epoch0 = batch0 = updates = 0
    best_ap = -1.0
    epoch_loss0 = epoch_examples0 = 0
    if resume:
        if read_json(run / "config.json") != config:
            raise ValueError("Resume configuration differs from the saved run.")
        old = read_json(run / "last/detector.json")
        if old["diagnostic_run"] or old["data_manifest_sha256"] != metadata["data_manifest_sha256"]:
            raise ValueError("Cannot resume a diagnostic run or changed dataset as a full run.")
        loaded, _, _ = InjectionClassifier.load(run / "last", device)
        model.load_state_dict(loaded.state_dict())
        del loaded
        state = torch.load(run / "last/training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch0, batch0, updates, best_ap = state["epoch"], state["next_batch"], state["updates"], state["best_ap"]
        epoch_loss0, epoch_examples0 = state.get("epoch_loss", 0), state.get("epoch_examples", 0)
        restore_rng(state["rng"], device)
    else:
        write_json(run / "config.json", config)
    write_json(run / "run_metadata.json", metadata)
    lock = run / ".training.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    valid_loader = DataLoader(validation, batch_size=config["batch_size"], shuffle=False,
                              generator=torch.Generator().manual_seed(config["seed"] + 100000),
                              collate_fn=Collator(tokenizer))
    start = time.perf_counter()
    stopped = False
    paused = False
    try:
        for epoch in range(epoch0, config["epochs"]):
            if config.get("paired_batches"):
                loader = DataLoader(dataset, batch_sampler=PairBatchSampler(rows, config["batch_size"], config["seed"] + epoch),
                                    generator=torch.Generator().manual_seed(config["seed"] + epoch), collate_fn=Collator(tokenizer))
            else:
                loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0,
                                    generator=torch.Generator().manual_seed(config["seed"] + epoch),
                                    collate_fn=Collator(tokenizer))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss, epoch_examples = (epoch_loss0, epoch_examples0) if epoch == epoch0 else (0, 0)
            grad_norm = 0.0
            for index, batch in enumerate(loader):
                if epoch == epoch0 and index < batch0:
                    continue
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = model(batch["input_ids"], batch["attention_mask"])
                losses = nn.functional.cross_entropy(logits, batch["labels"], weight=weights, reduction="none")
                if not torch.isfinite(losses).all():
                    raise FloatingPointError("Non-finite training loss; stop and inspect.")
                # Correct sample weighting even for a final partial accumulation window.
                window_start = index // config["accumulation_steps"] * config["accumulation_steps"] * config["batch_size"]
                window_examples = min(config["accumulation_steps"] * config["batch_size"], len(dataset) - window_start)
                objective = losses.sum()
                if config.get("ranking_weight", 0):
                    objective = objective + config["ranking_weight"] * ranking_sum(
                        logits, batch["labels"], batch["pair_ids"], config.get("ranking_margin", 1.0))
                if not torch.isfinite(objective):
                    raise FloatingPointError("Non-finite combined objective.")
                (objective / window_examples).backward()
                epoch_loss += objective.detach().item()
                epoch_examples += len(batch["labels"])
                boundary = (index + 1) % config["accumulation_steps"] == 0 or index + 1 == len(loader)
                if boundary:
                    grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True).item())
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    if updates % 25 == 0 or updates == 1:
                        print(f"epoch={epoch+1} update={updates} loss={epoch_loss/epoch_examples:.4f} grad_norm={grad_norm:.3f}", flush=True)
                    if updates % 25 == 0 or updates == 1:
                        write_json(run / "progress.json", {"status": "training", "epoch": epoch + 1,
                            "epochs": config["epochs"], "update": updates,
                            "total_updates": updates_per_epoch * config["epochs"],
                            "batch": index + 1, "batches_per_epoch": batches_per_epoch,
                            "train_loss": epoch_loss / epoch_examples,
                            "elapsed_seconds_this_invocation": time.perf_counter() - start})
                    if max_updates and updates >= max_updates:
                        stopped = True
                    if (run / "PAUSE").exists() or (run.parent / "PAUSE_STUDY").exists():
                        stopped = paused = True
                # Validate/checkpoint only after complete updates. No partial gradients are serialized.
                if boundary and (updates % 500 == 0 or index + 1 == len(loader) or stopped):
                    labels, scores, val_loss = predict_loader(model, valid_loader, device, weights)
                    threshold = select_threshold(labels, scores, config["max_fpr"])
                    validation_metrics = metrics(labels, scores, threshold)
                    ap = validation_metrics["average_precision"]
                    record = {"epoch": epoch + 1, "update": updates, "batch": index + 1,
                              "train_loss": epoch_loss/epoch_examples, "validation_loss": val_loss,
                              "gradient_norm": grad_norm, "head_lr": optimizer.param_groups[-1]["lr"],
                              "elapsed_seconds": time.perf_counter() - start, "validation": validation_metrics}
                    with (run / "history.jsonl").open("a", encoding="utf-8") as handle:
                        import json
                        handle.write(json.dumps(record, allow_nan=False) + "\n")
                    current = dict(metadata, epoch=epoch+1, updates=updates, threshold=threshold, validation=validation_metrics)
                    if ap > best_ap:
                        best_ap = ap
                        model.save(run / "best", tokenizer, current)
                    model.save(run / "last", tokenizer, current)
                    next_epoch, next_batch = (epoch + 1, 0) if index + 1 == len(loader) else (epoch, index + 1)
                    state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                             "epoch": next_epoch, "next_batch": next_batch, "updates": updates,
                             "best_ap": best_ap, "rng": rng_state(device),
                             "epoch_loss": epoch_loss if next_epoch == epoch else 0,
                             "epoch_examples": epoch_examples if next_epoch == epoch else 0}
                    torch.save(state, run / "last/training_state.tmp")
                    os.replace(run / "last/training_state.tmp", run / "last/training_state.pt")
                    print(f"validation: AP={ap:.4f} F1={validation_metrics['f1']:.4f} FPR={validation_metrics['benign_fpr']:.4f}; saved", flush=True)
                if stopped:
                    break
            batch0 = 0
            if stopped:
                break
        if paused:
            (run / "PAUSE").unlink(missing_ok=True)
        write_json(run / "status.json", {"status": "paused" if paused else "diagnostic_complete" if stopped else "complete",
                   "updates": updates, "best_validation_average_precision": best_ap,
                   "seconds_this_invocation": time.perf_counter() - start})
    finally:
        lock.unlink(missing_ok=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-validation", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--benchmark-steps", type=int)
    args = parser.parse_args()
    train(args.config, args.resume, args.limit_train, args.limit_validation, args.max_updates, args.benchmark_steps)


if __name__ == "__main__":
    main()
