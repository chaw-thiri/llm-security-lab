"""Join existing saved scores with inspected synthetic cases; no inference or tuning."""
from promptshield.reporting import write_report as write_compact_report
from collections import Counter
from pathlib import Path
from promptshield.common import read_json, read_jsonl, write_json, write_jsonl, sha256


def main():
    rows_path = Path("data/splits/task-aware-v1/task-document/test.jsonl")
    scores_path = Path("docs/task-aware-results.json")
    report = read_json(scores_path)
    if sha256(rows_path) != report["challenge_sha256"]["task-document"]:
        raise ValueError("Challenge hash changed")
    cases = {r["id"]: r for r in read_jsonl(rows_path)}
    output = {"scope": "Post-hoc audit of previously inspected synthetic challenge; no new evaluation or threshold fitting.",
              "scores_sha256": sha256(scores_path), "challenge_sha256": sha256(rows_path), "models": {}}
    errors = []
    for name, model in report["models"].items():
        false_negatives, false_positives = [], []
        for prediction in model["predictions"]:
            case = cases[prediction["id"]]
            predicted = int(prediction["score"] >= model["threshold"])
            if predicted != prediction["label"]:
                error = {"model": name, **case, "score": prediction["score"], "threshold": model["threshold"],
                         "error": "false_negative" if case["label"] else "false_positive"}
                errors.append(error)
                (false_negatives if case["label"] else false_positives).append(error)
        output["models"][name] = {
            "false_negatives": len(false_negatives), "false_positives": len(false_positives),
            "misses_by_family": dict(Counter(r["family"] for r in false_negatives)),
            "misses_by_payload_index": dict(Counter(r["pair_id"].split("-")[2] for r in false_negatives)),
            "misses_by_task": dict(Counter(r["task"] for r in false_negatives)),
            "missed_ids": [r["id"] for r in false_negatives],
            "threshold": model["threshold"], "challenge_average_precision": model["metrics"]["average_precision"]}
    task_errors = [r for r in errors if r["model"].startswith("task-document-") and r["label"]]
    votes = Counter(r["id"] for r in task_errors)
    output["missed_by_all_task_seeds"] = sorted(k for k, n in votes.items() if n == 3)
    train = read_jsonl("data/splits/task-aware-v1/task-document/train.jsonl")
    output["input_audit"] = {"training_rows": len(train), "training_rows_with_task": sum("task" in r for r in train),
                             "unique_task_phrasings": len({r["task"] for r in cases.values()}),
                             "semantic_task_types": ["appointment reference extraction"]}
    write_jsonl("results/task-aware-v1/error-cases.jsonl", errors)
    write_json("docs/task-aware-error-audit.json", output)
    write_compact_report("task-aware")
    print(json_summary(output))


def json_summary(report):
    return {"models": len(report["models"]), "consensus_misses": len(report["missed_by_all_task_seeds"]), "input_audit": report["input_audit"]}


if __name__ == "__main__":
    main()
