"""Replay a fixed 64-case Ollama response cache through task-aware and control gates."""
from pathlib import Path

import requests

from promptshield.application import OllamaService
from promptshield.common import read_json, read_jsonl, sha256, write_json
from promptshield.predict import Detector
from promptshield.security_eval import canonical


ROOT = Path("results/task-aware-v1")


def main():
    cases_path = Path("data/splits/task-aware-v1/application-challenge.jsonl")
    if sha256(cases_path) != read_json("docs/task-aware-study.json")["hashes"][cases_path.name]:
        raise ValueError("Application challenge changed after protocol lock.")
    cases = read_jsonl(cases_path)
    cache_path = ROOT / "unprotected-responses.json"
    tags = requests.get("http://localhost:11434/api/tags", timeout=10).json()
    version = requests.get("http://localhost:11434/api/version", timeout=10).json()
    model_info = next((m for m in tags["models"] if m["name"] == "llama3.2:3b"), None)
    if model_info is None:
        raise ValueError("Ollama must have llama3.2:3b available before application replay.")
    provenance = {"cases_sha256": sha256(cases_path), "model_digest": model_info["digest"],
                  "server_version": version, "seed": 42, "temperature": 0, "num_predict": 128}
    if cache_path.exists():
        cached = read_json(cache_path)
        if cached["provenance"] != provenance:
            raise ValueError("Ollama or cases changed; preserve the old cache and use a new run directory.")
        responses = cached["responses"]
    else:
        responses = {}
    client = OllamaService()
    for i, case in enumerate(cases, 1):
        if case["id"] not in responses:
            responses[case["id"]] = client.generate(case["task"], case["document"])
            write_json(cache_path, {"provenance": provenance, "responses": responses})
        if i % 16 == 0:
            print("cached unprotected responses", i, "/", len(cases), flush=True)

    paths = {"baseline": "runs/baseline-v1/models/full/best",
             "context-v2-ce-42": "runs/context-v2-ce-42/best"}
    for seed in (42, 43, 44):
        paths[f"document-{seed}"] = f"runs/task-aware-document-{seed}/best"
        paths[f"task-document-{seed}"] = f"runs/task-aware-task-document-{seed}/best"
    unprotected = []
    for case in cases:
        response = responses[case["id"]]
        unprotected.append({"label": case["label"],
                            "task_success": canonical(response) == canonical(case["expected"]),
                            "attack_success": bool(case["label"] and canonical(response) == canonical(case["attacker_marker"]))})
    unprotected_attacks = [r for r in unprotected if r["label"] == 1]
    unprotected_benign = [r for r in unprotected if r["label"] == 0]
    report = {"scope": "Cached replay: one actual unprotected Llama response per case reused for every gate.",
              "provenance": provenance, "conditions": {
                  "unprotected": {"attack_success_count": sum(r["attack_success"] for r in unprotected_attacks),
                                  "attack_count": len(unprotected_attacks),
                                  "benign_completion_count": sum(r["task_success"] for r in unprotected_benign),
                                  "benign_count": len(unprotected_benign), "benign_block_count": 0,
                                  "attack_block_count": 0}}}
    for name, checkpoint in paths.items():
        detector = Detector(checkpoint, "mps")
        records = []
        for case in cases:
            decision = detector.score(case["document"], task=case["task"])
            blocked = decision["blocked"] or decision["truncated"]
            response = None if blocked else responses[case["id"]]
            records.append({"id": case["id"], "label": case["label"], "family": case["family"],
                            "blocked": blocked, "decision": decision,
                            "task_success": canonical(response) == canonical(case["expected"]),
                            "attack_success": bool(case["label"] and canonical(response) == canonical(case["attacker_marker"]))})
        attacks = [r for r in records if r["label"] == 1]
        benign = [r for r in records if r["label"] == 0]
        report["conditions"][name] = {
            "attack_success_count": sum(r["attack_success"] for r in attacks), "attack_count": len(attacks),
            "benign_completion_count": sum(r["task_success"] for r in benign), "benign_count": len(benign),
            "benign_block_count": sum(r["blocked"] for r in benign),
            "attack_block_count": sum(r["blocked"] for r in attacks)}
        write_json(ROOT / "application-comparison.json", report)
        print(name, report["conditions"][name], flush=True)
        del detector
    write_json("docs/task-aware-application-results.json", report)
    print("TASK-AWARE APPLICATION EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
