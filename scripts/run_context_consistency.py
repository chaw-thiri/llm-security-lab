"""Run the fixed CE/consistency ablation and record a development-only replication decision."""
from promptshield.provenance import frozen_protocol_sha256
from promptshield.reporting import write_report as write_compact_report
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from torch.utils.data import DataLoader
import torch
import numpy as np

from promptshield.classifier import InjectionClassifier
from promptshield.common import read_json, read_jsonl, write_json, sha256, choose_device
from promptshield.data import TextDataset, Collator
from promptshield.scoring import metrics, select_grouped_threshold

DATA = Path("data/splits/context-consistency-v1")
OUT = Path("results/context-consistency-v1")
DOMAINS = ("technical", "security", "general")


def score(checkpoint, cases, split_hash, tag):
    checkpoint = Path(checkpoint)
    weight_hash = sha256(checkpoint/"model.safetensors")
    path = OUT/f"{tag}-{weight_hash[:16]}.json"
    provenance = dict(weights_sha256=weight_hash, split_sha256=split_hash)
    if path.exists():
        saved = read_json(path)
        if saved["provenance"] != provenance or len(saved["predictions"]) != len(cases):
            raise ValueError("Stale or incomplete consistency score cache")
        if any((p["id"], p["label"], p["domain"]) != (r["id"], r["label"], r.get("domain", "original"))
               for p, r in zip(saved["predictions"], cases)):
            raise ValueError("Score/case alignment changed")
        return saved["predictions"]
    device = choose_device("mps")
    model, tok, meta = InjectionClassifier.load(checkpoint, device)
    if meta["pooling"] != "cls" or meta["max_length"] != 512 or meta.get("config", {}).get("input_format", "document") != "document":
        raise ValueError("The fixed study requires first-token document-only checkpoints")
    loader = DataLoader(TextDataset(cases, tok, 512), batch_size=8, shuffle=False, collate_fn=Collator(tok))
    predictions = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probabilities = logits.softmax(-1)[:, 1].cpu().tolist()
            margins = (logits[:, 1]-logits[:, 0]).cpu().tolist()
            start = len(predictions)
            predictions.extend(dict(id=r["id"], label=r["label"], domain=r.get("domain", "original"),
                                    score=p, logit_margin=m) for r,p,m in zip(cases[start:start+len(probabilities)], probabilities, margins))
    del model
    if len(predictions) != len(cases) or any(not np.isfinite(r["score"]) or not np.isfinite(r["logit_margin"]) for r in predictions):
        raise ValueError("Incomplete or nonfinite inference")
    write_json(path, dict(provenance=provenance, predictions=predictions))
    return predictions


def describe(predictions, threshold):
    return dict(metrics=metrics([p["label"] for p in predictions], [p["score"] for p in predictions], threshold),
                by_domain={d:metrics([p["label"] for p in predictions if p["domain"] == d],
                                    [p["score"] for p in predictions if p["domain"] == d], threshold)
                           for d in sorted({p["domain"] for p in predictions})})


def margin_gaps(predictions, cases):
    grouped = {}
    for p,r in zip(predictions,cases):
        if r.get("context_group"):
            grouped.setdefault((r["context_group"],r["label"]),[]).append(p["logit_margin"])
    values = [v[0]-v[1] for v in grouped.values() if len(v) == 2]
    if len(values) != 2*len({r["context_group"] for r in cases if r.get("context_group")}):
        raise ValueError("Incomplete held-out context groups")
    return dict(role_pairs=len(values), mean_squared_gap=float(np.mean(np.square(values))),
                median_absolute_gap=float(np.median(np.abs(values))),
                q95_absolute_gap=float(np.quantile(np.abs(values),.95)))


def calibrate(name, checkpoint, development):
    predictions = score(checkpoint, development, sha256(DATA/"validation.jsonl"), "development")
    threshold = select_grouped_threshold([p["label"] for p in predictions], [p["score"] for p in predictions],
                                         [p["domain"] for p in predictions], .05)
    destination = Path("runs/context-consistency-calibrated")/name
    provenance = dict(weights_sha256=sha256(Path(checkpoint)/"model.safetensors"),
                      development_sha256=sha256(DATA/"validation.jsonl"), method="maximize development recall subject to <=5% benign FPR in each domain")
    if destination.exists():
        meta = read_json(destination/"detector.json")
        if meta.get("consistency_calibration") != provenance or meta["threshold"] != threshold or sha256(destination/"model.safetensors") != provenance["weights_sha256"]:
            raise ValueError("Calibrated copy changed")
    else:
        shutil.copytree(checkpoint,destination)
        meta = read_json(destination/"detector.json")
        meta["threshold_before_consistency_calibration"] = meta["threshold"]
        meta["validation_before_consistency_calibration"] = meta["validation"]
        meta.update(threshold=threshold, consistency_calibration=provenance,
                    validation=metrics([p["label"] for p in predictions],[p["score"] for p in predictions],threshold))
        write_json(destination/"detector.json",meta)
    result = dict(checkpoint=str(destination), threshold=threshold, calibration=provenance,
                  development=describe(predictions,threshold), held_out_context_gaps=margin_gaps(predictions,development))
    if any(m["benign_fpr"] > .05 for m in result["development"]["by_domain"].values()):
        raise AssertionError("A development domain exceeds the fixed FPR cap")
    result["worst_real_recall"] = min(result["development"]["by_domain"][d]["recall"] for d in DOMAINS)
    result["mean_real_recall"] = float(np.mean([result["development"]["by_domain"][d]["recall"] for d in DOMAINS]))
    write_json(OUT/f"{name}-development.json",result)
    print(name,"development worst/mean real recall",result["worst_real_recall"],result["mean_real_recall"],
          "context gap MSE",result["held_out_context_gaps"]["mean_squared_gap"],flush=True)
    return result


def train_arm(arm,seed,development):
    run = Path(f"runs/context-consistency-{arm}-{seed}")
    status = run/"status.json"
    if not status.exists() or read_json(status)["status"] != "complete":
        command = [sys.executable,"-m","promptshield.train","--config",f"configs/context-consistency-{arm}-{seed}.json"]
        if (run/"config.json").exists():
            command.append("--resume")
        subprocess.run(command,check=True)
    if read_json(status)["status"] != "complete":
        raise RuntimeError("Run is paused/incomplete; keep final challenge unscored")
    result = calibrate(f"{arm}-{seed}",run/"best",development)
    result["training_history"] = read_jsonl(run/"history.jsonl")
    return result


def evaluate(manifest,references,runs):
    fresh_path = DATA/"test.jsonl"
    if sha256(fresh_path) != manifest["splits"]["test"]["sha256"]:
        raise ValueError("Final challenge changed after the development decision")
    fresh = read_jsonl(fresh_path)
    original_path = Path("data/splits/piguard-v1/test.jsonl")
    if sha256(original_path) != read_json("data/splits/piguard-v1/manifest.json")["splits"]["test"]["sha256"]:
        raise ValueError("Original regression split changed")
    original = read_jsonl(original_path)
    old_path = Path("data/splits/grounded-upgrade-v1/fresh-challenge.jsonl")
    if sha256(old_path) != read_json("data/splits/grounded-upgrade-v1/manifest.json")["hashes"]["fresh-challenge.jsonl"]:
        raise ValueError("Inspected challenge regression changed")
    old = read_jsonl(old_path)
    all_models = {"baseline-original":dict(checkpoint="runs/baseline-v1/models/full/best",
                    threshold=read_json("runs/baseline-v1/models/full/best/detector.json")["threshold"]),
                  "previous-grounded-original":dict(checkpoint="runs/grounded-upgrade-calibrated/grounded-42",
                    threshold=read_json("runs/grounded-upgrade-calibrated/grounded-42/detector.json")["threshold"]),
                  **references,**runs}
    report = dict(scope="Construction-labeled CE/consistency loss ablation; same-publisher source and wording holdouts, not independent human-reviewed production validation",
                  protocol_sha256=manifest["protocol_sha256"], data_manifest_sha256=sha256(DATA/"manifest.json"),
                  execution_sha256=sha256(OUT/"execution.json"), replication_decision=read_json(OUT/"replication-decision.json"),
                  fresh_rows=len(fresh), fresh_source_pages=18, models={}, serving_default_changed=False)
    for name,entry in all_models.items():
        checkpoint,threshold = entry["checkpoint"],entry["threshold"]
        predictions = score(checkpoint,fresh,sha256(fresh_path),"final")
        result = dict(checkpoint=checkpoint, weights_sha256=sha256(Path(checkpoint)/"model.safetensors"), threshold=threshold,
                      final=describe(predictions,threshold), original_regression=describe(score(checkpoint,original,sha256(original_path),"original-regression"),threshold),
                      exposed_challenge_regression=describe(score(checkpoint,old,sha256(old_path),"exposed-regression"),threshold),
                      final_breakdowns={})
        for field in ("role_family","quoted","position","payload_index"):
            result["final_breakdowns"][field] = {}
            for value in sorted({str(r.get(field,"none")) for r in fresh}):
                selected = [p for p,r in zip(predictions,fresh) if str(r.get(field,"none")) == value]
                result["final_breakdowns"][field][value] = describe(selected,threshold)
        if "development" in entry:
            result["development"] = entry["development"]
            result["held_out_context_gaps"] = entry["held_out_context_gaps"]
        if "training_history" in entry:
            result["training_history"] = entry["training_history"]
        report["models"][name] = result
        write_json(OUT/"evaluation-progress.json",report)
        print(name,"FINAL",result["final"]["metrics"],flush=True)
    write_json("docs/context-consistency-results.json",report)
    write_compact_report("context-consistency")
    write_json(OUT/"status.json",dict(status="complete",models=list(report["models"]),serving_default_changed=False))




def main():
    if (OUT/"status.json").exists() and read_json(OUT/"status.json")["status"] == "complete":
        write_compact_report("context-consistency")
        print("Completed study: regenerated compact report only; no training or diary overwrite",flush=True)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    os.environ["TMPDIR"] = str(Path.cwd()/".tmp")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    manifest = read_json(DATA/"manifest.json")
    if frozen_protocol_sha256("docs/context-consistency-protocol.md") != manifest["protocol_sha256"]:
        raise ValueError("Study protocol changed after data freeze")
    for split in ("train","validation","test"):
        if sha256(DATA/f"{split}.jsonl") != manifest["splits"][split]["sha256"]:
            raise ValueError("A frozen split changed")
    audit = read_json("docs/context-consistency-semantic-audit.json")
    if audit["training_sha256"] != manifest["splits"]["train"]["sha256"] or not audit["assistant_semantic_audit_completed"] or audit["independent_human_review_completed"]:
        raise ValueError("Require the honest, completed assistant audit of these construction-labeled cases")
    execution = dict(data_manifest_sha256=sha256(DATA/"manifest.json"),protocol_sha256=manifest["protocol_sha256"],
                     code={p:sha256(p) for p in ("src/promptshield/train.py","src/promptshield/data.py","src/promptshield/consistency.py","scripts/prepare_context_consistency.py","scripts/run_context_consistency.py")},
                     configs={f"{arm}-{seed}":sha256(f"configs/context-consistency-{arm}-{seed}.json") for arm in ("ce","consistent") for seed in (42,43,44)})
    execution_path = OUT/"execution.json"
    if execution_path.exists() and read_json(execution_path) != execution:
        raise ValueError("Execution code/config changed; use a new study version")
    write_json(execution_path,execution)
    development = read_jsonl(DATA/"validation.jsonl")
    references = {"baseline-calibrated":calibrate("reference-baseline","runs/baseline-v1/models/full/best",development),
                  "previous-grounded-calibrated":calibrate("reference-grounded","runs/grounded-upgrade-grounded-42/best",development)}
    runs = {}
    for arm in ("ce","consistent"):
        write_json(OUT/"status.json",dict(status="training",arm=arm,seed=42))
        runs[f"{arm}-42"] = train_arm(arm,42,development)
    control,candidate = runs["ce-42"],runs["consistent-42"]
    replicate = (candidate["worst_real_recall"] >= control["worst_real_recall"]+.10
                 and candidate["mean_real_recall"] >= control["mean_real_recall"]
                 and candidate["development"]["by_domain"]["original"]["recall"] >= control["development"]["by_domain"]["original"]["recall"]-.02)
    write_json(OUT/"replication-decision.json",dict(replicate=replicate,decision_source="development only, before new challenge scoring",control=control,candidate=candidate))
    if replicate:
        for seed in (43,44):
            for arm in ("ce","consistent"):
                write_json(OUT/"status.json",dict(status="training",arm=arm,seed=seed))
                runs[f"{arm}-{seed}"] = train_arm(arm,seed,development)
    write_json(OUT/"development-comparison.json",dict(references=references,runs=runs))
    write_json(OUT/"status.json",dict(status="evaluating_fixed_final_challenge",replicated=replicate))
    evaluate(manifest,references,runs)


if __name__ == "__main__":
    main()
