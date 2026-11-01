"""Summarize every attempted seed and export inspected errors; no fitting or inference."""
from promptshield.reporting import write_report as write_compact_report
from pathlib import Path
import statistics

from promptshield.common import read_json, read_jsonl, write_json, write_jsonl, sha256
from investigate_domain_scoring import checked_scores

OUT = Path("results/context-consistency-v1")


def average(values):
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                minimum=min(values), maximum=max(values), seeds=len(values))


def main():
    if read_json(OUT/"status.json")["status"] != "complete":
        raise RuntimeError("Wait for all fixed evaluations and the diary update to finish")
    study = read_json("docs/context-consistency-results.json")
    path = Path("data/splits/context-consistency-v1/test.jsonl")
    manifest = read_json(path.parent/"manifest.json")
    if sha256(path) != manifest["splits"]["test"]["sha256"] or sha256(path.parent/"manifest.json") != study["data_manifest_sha256"]:
        raise ValueError("Frozen final cases changed")
    cases = read_jsonl(path)
    arms = {arm:[name for name in study["models"] if name.startswith(arm+"-")] for arm in ("ce","consistent")}
    predictions = {}
    for name in sum(arms.values(),[]):
        model = study["models"][name]
        saved = read_json(OUT/f"final-{model['weights_sha256'][:16]}.json")
        predictions[name] = checked_scores(saved,cases,model["weights_sha256"],sha256(path))
    report = dict(scope="Post-hoc descriptive summary of the fixed loss ablation; no thresholds or models selected here",
                  source_results_sha256=sha256("docs/context-consistency-results.json"), final_cases_sha256=sha256(path),
                  arms={}, paired_seed_changes={}, common_errors={})
    for arm,names in arms.items():
        report["arms"][arm] = dict(seeds=names,final={},by_domain={},original_regression={},development={})
        for key in ("recall","benign_fpr","roc_auc","f1"):
            report["arms"][arm]["final"][key] = average([study["models"][n]["final"]["metrics"][key] for n in names])
        for domain in ("technical","security","general"):
            report["arms"][arm]["by_domain"][domain] = {key:average([study["models"][n]["final"]["by_domain"][domain][key] for n in names])
                                                      for key in ("recall","benign_fpr","roc_auc")}
        for key in ("recall","benign_fpr","f1"):
            report["arms"][arm]["original_regression"][key] = average([study["models"][n]["original_regression"]["metrics"][key] for n in names])
        report["arms"][arm]["development"]["margin_gap_mse"] = average([study["models"][n]["held_out_context_gaps"]["mean_squared_gap"] for n in names])
    errors = []
    for i,case in enumerate(cases):
        models = {}
        for name,rows in predictions.items():
            p = rows[i]; blocked = p["score"] >= study["models"][name]["threshold"]
            models[name] = dict(score=p["score"], logit_margin=p["logit_margin"], threshold=study["models"][name]["threshold"],
                                blocked=blocked, correct=int(blocked) == case["label"])
        if any(not p["correct"] for p in models.values()):
            errors.append(dict(case,models=models,independent_human_review_completed=False))
    write_jsonl(OUT/"per-example-errors.jsonl",errors)
    for name in predictions:
        expected = study["models"][name]["final"]["metrics"]["confusion_matrix"]
        fn = sum(r["label"] == 1 and not r["models"][name]["blocked"] for r in errors)
        fp = sum(r["label"] == 0 and r["models"][name]["blocked"] for r in errors)
        if (fn,fp) != (expected["fn"],expected["fp"]):
            raise ValueError("Joined error records disagree with the independent evaluation counts")
    report["per_example_errors_sha256"] = sha256(OUT/"per-example-errors.jsonl")
    for seed in (42,43,44):
        ce,consistent = f"ce-{seed}",f"consistent-{seed}"
        if ce not in predictions or consistent not in predictions:
            continue
        counts = dict(attacks_recovered=0,new_attack_misses=0,benign_false_alarms_removed=0,new_benign_false_alarms=0)
        for case,a,b in zip(cases,predictions[ce],predictions[consistent]):
            ac = a["score"] >= study["models"][ce]["threshold"]
            bc = b["score"] >= study["models"][consistent]["threshold"]
            if case["label"]:
                counts["attacks_recovered"] += int(not ac and bc)
                counts["new_attack_misses"] += int(ac and not bc)
            else:
                counts["benign_false_alarms_removed"] += int(ac and not bc)
                counts["new_benign_false_alarms"] += int(not ac and bc)
        report["paired_seed_changes"][str(seed)] = counts
    for arm,names in arms.items():
        common = {"missed_attacks":[],"false_alarms":[]}
        for i,case in enumerate(cases):
            decisions = [predictions[n][i]["score"] >= study["models"][n]["threshold"] for n in names]
            if case["label"] and not any(decisions):
                common["missed_attacks"].append(case["id"])
            if not case["label"] and all(decisions):
                common["false_alarms"].append(case["id"])
        report["common_errors"][arm] = common
    write_json("docs/context-consistency-summary.json",report)
    write_compact_report("context-consistency")
    print("Summary and local per-example errors written",flush=True)


if __name__ == "__main__":
    main()
