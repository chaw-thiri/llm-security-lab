"""Run a matched pooling comparison from the repository root, using local artifacts only."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys

from torch.utils.data import DataLoader

from promptshield.classifier import InjectionClassifier
from promptshield.common import choose_device, read_json, read_jsonl, sha256, write_json
from promptshield.data import Collator, TextDataset
from promptshield.predict import Detector
from promptshield.scoring import metrics, select_grouped_threshold
from promptshield.train import load_config, predict_loader, train
from replay_task_verifier import decision, load_cohort, outcome_row, summarize


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("configs/pooling/study.json")
OUT = Path("results/pooling-v1")
RUNS = Path("runs/pooling-v1")
INPUTS = ("train", "validation", "calibration")  # Never enumerate the test/review directories.


def same_or_write(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing record differs; refusing to replace {path}")
    else:
        write_json(path, value)


def run_configs(study, base):
    return {f"{arm}-{seed}": dict(base, run_dir=str(RUNS / f"{arm}-{seed}"), pooling=pooling, seed=seed)
            for seed in study["seeds"] for arm, pooling in study["arms"].items()}


def preflight():
    study = read_json(PROTOCOL)
    if sha256(study["base_config"]) != study["base_config_sha256"]:
        raise ValueError("Base training configuration changed.")
    base = load_config(study["base_config"])
    data = Path(base["data_dir"])
    if sha256(data / "manifest.json") != study["data_manifest_sha256"]:
        raise ValueError("Training manifest changed.")
    manifest = read_json(data / "manifest.json")["splits"]
    for split in INPUTS:
        if (sha256(data / f"{split}.jsonl") != manifest[split]["sha256"]
                or manifest[split]["rows"] != study["training_splits"][split]):
            raise ValueError(f"Changed {split} split.")
    if sha256(Path(base["initial_checkpoint"]) / "model.safetensors") != base["initial_weights_sha256"]:
        raise ValueError("Shared starting weights changed.")
    code = ("scripts/run_pooling_study.py", "scripts/replay_task_verifier.py", "scripts/validate_real_documents.py",
            "src/promptshield/classifier.py", "src/promptshield/train.py", "src/promptshield/data.py",
            "src/promptshield/wording.py", "src/promptshield/scoring.py", "src/promptshield/predict.py",
            "src/promptshield/consistency.py", "src/promptshield/pairwise.py",
            "src/promptshield/security_eval.py", "src/promptshield/common.py")
    provenance = {"protocol": sha256(PROTOCOL), "base_config": sha256(study["base_config"]),
                  "initial_weights": base["initial_weights_sha256"], "manifest": study["data_manifest_sha256"],
                  "tokenizer": sha256(Path(base["backbone"]) / "tokenizer.json"),
                  "splits": {split: manifest[split]["sha256"] for split in INPUTS},
                  "code": {path: sha256(path) for path in code}}
    return study, base, provenance


def calibrate(config, study, rows):
    best = Path(config["run_dir"]) / "best"
    device = choose_device(config["device"])
    model, tokenizer, meta = InjectionClassifier.load(best, device)
    if meta["config"] != config or meta["diagnostic_run"] or meta["pooling"] != config["pooling"]:
        raise ValueError(f"Unexpected checkpoint at {best}")
    domains = [r["domain"] for r in rows]
    if set(domains) != set(study["calibration_domains"]):
        raise ValueError("Calibration domains changed.")
    loader = DataLoader(TextDataset(rows, tokenizer, config["max_length"]), batch_size=config["batch_size"],
                        shuffle=False, collate_fn=Collator(tokenizer))
    labels, scores, _ = predict_loader(model, loader, device)
    if labels != [r["label"] for r in rows]:
        raise ValueError("Calibration rows and predictions are misaligned.")
    cap = study["calibration_max_domain_fpr"]
    threshold = select_grouped_threshold(labels, scores, domains, cap)
    by_domain = {}
    for d in sorted(set(domains)):
        indices = [i for i, value in enumerate(domains) if value == d]
        by_domain[d] = metrics([labels[i] for i in indices], [scores[i] for i in indices], threshold)
        if by_domain[d]["benign_fpr"] > cap:
            raise ValueError(f"Calibration cap exceeded in {d}")
    benign_roles = {}
    for role in sorted({r["role_family"] for r in rows if "role_family" in r}):
        members = [i for i, r in enumerate(rows) if r.get("role_family") == role and r["label"] == 0]
        benign_roles[role] = {"benign": len(members), "false_alarms": sum(scores[i] >= threshold for i in members)}
    return {"checkpoint": str(best), "weights_sha256": sha256(best / "model.safetensors"),
            "detector_sha256": sha256(best / "detector.json"), "tokenizer_sha256": sha256(best / "tokenizer.json"),
            "threshold": threshold, "parameters": meta["parameters"], "selection_update": meta["updates"],
            "calibration": metrics(labels, scores, threshold), "by_domain": by_domain,
            "calibration_benign_descriptions": benign_roles}


def exposed_replay(policy, cohorts, device):
    best = Path(policy["checkpoint"])
    for filename, key in (("model.safetensors", "weights_sha256"), ("detector.json", "detector_sha256"),
                          ("tokenizer.json", "tokenizer_sha256")):
        if sha256(best / filename) != policy[key]:
            raise ValueError(f"Frozen policy artifact changed: {best / filename}")
    detector = Detector(best, device)
    detector.threshold = policy["threshold"]  # Applies this run's calibration receipt in memory only.
    result = {}
    for name in cohorts:
        cases, responses, controls, source = load_cohort(name)
        rows = []
        for case in cases:
            row = outcome_row(name, case, responses[case["id"]])
            row["detector"], _ = decision(detector, case)
            rows.append(row)
        ungated = summarize(rows, "ungated")
        if (ungated["correct_benign_answers"], ungated["exact_hijacks"]) != controls:
            raise ValueError(f"Ungated replay did not reproduce archived outcomes: {name}")
        result[name] = {"source": source, "ungated": ungated, "detector": summarize(rows, "detector"),
                        "by_group": {g: summarize([r for r in rows if r["group"] == g], "detector")
                                     for g in sorted({r["group"] for r in rows})}}
        if name == "real-documents-v1":
            attacks = [r for r in rows if r["label"] == 1]
            result[name]["by_attack_position"] = {
                p: summarize([r for r in attacks if r["position"] == p], "detector")
                for p in sorted({r["position"] for r in attacks})}
    return result


def paired_differences(replays, study):
    result = {}
    for cohort in study["exposed_replay"]:
        slices = {"all": lambda data: data["detector"]}
        if cohort == "real-documents-v1":
            for position in ("start", "middle", "end"):
                slices[position] = lambda data, p=position: data["by_attack_position"][p]
        result[cohort] = {}
        for slice_name, get_counts in slices.items():
            changes = {}
            for metric in ("attacks_detected", "false_alarms", "correct_benign_answers", "exact_hijacks"):
                if slice_name != "all" and metric in ("false_alarms", "correct_benign_answers"):
                    continue
                values = {str(s): get_counts(replays[f"attention-{s}"][cohort])[metric]
                                  - get_counts(replays[f"cls-{s}"][cohort])[metric] for s in study["seeds"]}
                changes[metric] = {"attention_minus_cls_count": values, "mean_count_change": mean(values.values())}
            result[cohort][slice_name] = changes
    return result


def status(phase, **kwargs):
    write_json(OUT / "status.json", {"phase": phase, "pid": os.getpid(),
                                     "as_of_utc": datetime.now(timezone.utc).isoformat(), **kwargs})


def run():
    study, base, provenance = preflight()
    same_or_write(OUT / "inputs.json", provenance)
    configs = run_configs(study, base)
    rows = read_jsonl(Path(base["data_dir"]) / "calibration.jsonl")
    policies = {}
    for name, config in configs.items():
        if (RUNS / "PAUSE_STUDY").exists():
            status("paused", next_run=name, calibrated=len(policies))
            return
        path = OUT / "configs" / f"{name}.json"
        same_or_write(path, config)
        run_dir = Path(config["run_dir"])
        saved = run_dir / "config.json"
        if saved.exists() and read_json(saved) != config:
            raise ValueError(f"Training config changed for {name}")
        state = run_dir / "status.json"
        if not state.exists() or read_json(state)["status"] != "complete":
            status("training", current_run=name, run_number=len(policies)+1, total=len(configs))
            print(f"Starting {name}: {config['pooling']}, seed {config['seed']}, {config['device']}", flush=True)
            train(path, resume=saved.exists())
            if read_json(state)["status"] == "paused":
                status("paused", current_run=name, calibrated=len(policies))
                return
        if read_json(state)["status"] != "complete" or read_json(state)["updates"] != study["updates_per_run"]:
            raise ValueError(f"Incomplete training for {name}")
        receipt = OUT / "calibration" / f"{name}.json"
        if not receipt.exists():
            status("calibrating", current_run=name, run_number=len(policies)+1, total=len(configs))
            write_json(receipt, calibrate(config, study, rows))
        policies[name] = read_json(receipt)
        if (policies[name]["checkpoint"] != str(run_dir / "best") or
                sha256(run_dir / "best/model.safetensors") != policies[name]["weights_sha256"]):
            raise ValueError(f"Calibration receipt does not match {name}")
    # No case or cached Llama response is opened before every checkpoint and threshold is locked.
    same_or_write(OUT / "frozen-policies.json", {"inputs": provenance, "policies": policies})
    replays = {}
    for name, policy in policies.items():
        status("exposed_replay", current_run=name, calibrated=len(policies))
        replay = OUT / "replay" / f"{name}.json"
        if not replay.exists():
            write_json(replay, exposed_replay(policy, study["exposed_replay"], base["device"]))
        replays[name] = read_json(replay)
    write_json(OUT / "paired-replay.json", {"scope": "Previously examined cases; count differences, not independent validation.",
                                          "policies_sha256": sha256(OUT / "frozen-policies.json"),
                                          "comparisons": paired_differences(replays, study)})
    status("complete", trained=len(configs), calibrated=len(policies), serving_changed=False)
    print("Finished all runs and exposed replays. Independent validation remains pending.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detach", action="store_true", help="Start a separate local process and append output to train.log.")
    parser.add_argument("--check", action="store_true", help="Verify workflow and input hashes only, without writing files.")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.check:
        study, base, _ = preflight()
        print("Inputs verified:", ", ".join(run_configs(study, base)))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, "-u", str(Path(__file__).resolve())]
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            command[:0] = ["caffeinate", "-i"]
        with (OUT / "train.log").open("ab") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
        print(f"Launcher PID {child.pid}; follow {OUT / 'train.log'}")
        return
    with (OUT / "study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("Pooling study is already running.")
        try:
            run()
        except Exception as exc:
            status("failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
