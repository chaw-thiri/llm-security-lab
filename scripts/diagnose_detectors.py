"""Exploratory score-frontier diagnostic using only already inspected datasets."""
from promptshield.reporting import write_report as write_compact_report
from collections import Counter, defaultdict
import math
from pathlib import Path

from promptshield.common import read_json, read_jsonl, write_json, write_jsonl, sha256
from promptshield.scoring import metrics

OUT = Path("results/detector-diagnostic-v1")
BUDGETS = (.01, .02, .05)
RECALLS = (.90, .95, .99)


def frontier(rows):
    """All attainable >= operating points; equal scores move as one group."""
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Need nonempty unique case IDs")
    if any(type(r["label"]) is not int or r["label"] not in (0, 1) or not math.isfinite(r["score"]) or not 0 <= r["score"] <= 1 for r in rows):
        raise ValueError("Invalid saved prediction")
    positives = sum(r["label"] for r in rows)
    negatives = Counter(r["source"] for r in rows if not r["label"])
    if not positives or not negatives:
        raise ValueError("Both labels required")
    ordered = sorted(rows, key=lambda r: -r["score"])
    tp, fp = 0, Counter()
    def point(threshold):
        rates = {source: fp[source] / count for source, count in negatives.items()}
        return {"threshold": threshold, "recall": tp / positives, "tp": tp, "fn": positives-tp,
                "fp": sum(fp.values()), "fpr": sum(fp.values()) / sum(negatives.values()),
                "source_fpr": rates, "worst_source_fpr": max(rates.values())}
    result = [point(math.nextafter(1., 2.))]
    for i, row in enumerate(ordered):
        if row["label"]:
            tp += 1
        else:
            fp[row["source"]] += 1
        if i+1 == len(ordered) or ordered[i+1]["score"] != row["score"]:
            result.append(point(row["score"]))
    return result


def best_recall(points, budget, constraint="worst_source_fpr"):
    return max((p for p in points if p[constraint] <= budget),
               key=lambda p: (p["recall"], -p[constraint], p["threshold"]))


def least_fpr(points, recall, constraint="worst_source_fpr"):
    return min((p for p in points if p["recall"] >= recall),
               key=lambda p: (p[constraint], p["fpr"], -p["recall"], -p["threshold"]))


def add_missing_ce(study):
    """Score other saved CE seeds on the same old original/quoted cases only."""
    from torch.utils.data import DataLoader
    from promptshield.classifier import InjectionClassifier
    from promptshield.common import choose_device
    from promptshield.data import TextDataset, Collator
    from promptshield.train import predict_loader
    from prepare_task_aware import PAYLOADS
    device = choose_device("mps")
    path = Path("data/splits/task-aware-v1/document/test.jsonl")
    if sha256(path) != study["challenge_sha256"]["document"]:
        raise ValueError("Previously inspected challenge changed")
    old_rows = read_jsonl(path)
    ni = read_json("docs/notinject-evaluation.json")
    for seed in (43, 44):
        checkpoint = Path(f"runs/context-v2-ce-{seed}/best")
        provenance = {"weights": sha256(checkpoint/"model.safetensors"),
                      "metadata": sha256(checkpoint/"detector.json"), "challenge": sha256(path)}
        cache = OUT / f"ce-{seed}-scores.json"
        if cache.exists():
            saved = read_json(cache)
            if saved["provenance"] != provenance:
                raise ValueError("Stale diagnostic cache")
            entry = saved["entry"]
        else:
            model, tok, meta = InjectionClassifier.load(checkpoint, device)
            entry = {"checkpoint": str(checkpoint), "weights_sha256": provenance["weights"],
                     "threshold": meta["threshold"]}
            for condition in ("original", "quoted"):
                rows = []
                for row in old_rows:
                    document = row["document"]
                    if condition == "quoted":
                        _, fi, pi, ti = row["pair_id"].split("-")
                        payload = PAYLOADS["challenge"][int(pi)].format(marker=f"REDIRECT-CHALLENGE-{fi}-{pi}-{ti}")
                        if payload not in document:
                            raise ValueError("Expected old payload missing")
                        document = document.replace(payload, f'"{payload}"', 1)
                    rows.append(dict(row, prompt=document))
                loader = DataLoader(TextDataset(rows, tok, meta["max_length"]), batch_size=8, shuffle=False, collate_fn=Collator(tok))
                y, p, _ = predict_loader(model, loader, device)
                predictions = [{"id": r["id"], "label": label, "score": score} for r, label, score in zip(rows, y, p)]
                if condition == "original":
                    entry["predictions"] = predictions
                else:
                    entry["challenge_quoted_payload_stress"] = {"predictions": predictions}
            del model
            external = ni["models"][f"ce-{seed}"]
            if external["weights_sha256"] != provenance["weights"]:
                raise ValueError("NotInject checkpoint mismatch")
            entry["notinject"] = {"predictions": external["predictions"]}
            write_json(cache, {"provenance": provenance, "entry": entry})
        study["models"][f"context-v2-ce-{seed}"] = entry
        print("Prepared same-case CE seed", seed, flush=True)


def analyze(entry, condition):
    original = entry["predictions"] if condition == "original" else entry["challenge_quoted_payload_stress"]["predictions"]
    synthetic = [{"id": "synthetic:"+r["id"], "label": r["label"], "score": r["score"], "source": "synthetic"} for r in original]
    if any(r.get("label", 0) != 0 for r in entry["notinject"]["predictions"]):
        raise ValueError("NotInject probe must be benign")
    external = [{"id": "notinject:"+r["id"], "label": 0, "score": r["score"], "source": "notinject"} for r in entry["notinject"]["predictions"]]
    rows = synthetic + external
    points = frontier(rows)
    synthetic_points = frontier(synthetic)
    at_saved = {"all": metrics([r["label"] for r in rows], [r["score"] for r in rows], entry["threshold"]),
                "synthetic": metrics([r["label"] for r in synthetic], [r["score"] for r in synthetic], entry["threshold"]),
                "notinject_fpr": sum(r["score"] >= entry["threshold"] for r in external)/len(external)}
    return {"saved": at_saved,
            "recall_at_fpr_budgets_per_source": {str(b): best_recall(points, b) for b in BUDGETS},
            "min_fpr_at_recall_per_source": {str(r): least_fpr(points, r) for r in RECALLS},
            "recall_at_pooled_fpr_budgets": {str(b): best_recall(points, b, "fpr") for b in BUDGETS},
            "min_pooled_fpr_at_recall": {str(r): least_fpr(points, r, "fpr") for r in RECALLS},
            "synthetic_only_recall_at_fpr_budgets": {str(b): best_recall(synthetic_points, b) for b in BUDGETS},
            "curve": points, "synthetic_curve": synthetic_points}, rows


def main():
    study = read_json("docs/task-aware-results.json")
    add_missing_ce(study)
    names = ["baseline"] + [f"{method}-{seed}" for method in ("context-v2-ce", "document", "task-document") for seed in (42, 43, 44)]
    results, rows_by_model = {}, {}
    for name in names:
        entry = study["models"][name]
        if sha256(Path(entry["checkpoint"])/"model.safetensors") != entry["weights_sha256"]:
            raise ValueError(f"Checkpoint changed: {name}")
        results[name] = {"checkpoint": entry["checkpoint"], "weights_sha256": entry["weights_sha256"],
                         "saved_threshold": entry["threshold"], "conditions": {}}
        for condition in ("original", "quoted"):
            data, rows = analyze(entry, condition)
            results[name]["conditions"][condition] = data
            rows_by_model[(name, condition)] = rows
        print(name, "best recall at <=5% FPR on EACH benign source:", results[name]["conditions"]["original"]["recall_at_fpr_budgets_per_source"]["0.05"]["recall"], flush=True)
    # Require identical semantic case IDs/labels for every compared model.
    expected = {(r["id"], r["label"]) for r in rows_by_model[("baseline", "original")]}
    if any({(r["id"], r["label"]) for r in rows} != expected for rows in rows_by_model.values()):
        raise ValueError("Models were evaluated on different semantic cases")
    groups = {"all_models": names, **{method: [f"{method}-{seed}" for seed in (42,43,44)] for method in ("context-v2-ce", "document", "task-document")}}
    common = {}
    for condition in ("original", "quoted"):
        common[condition] = {}
        for group, models in groups.items():
            member_sets = defaultdict(list)
            for name in models:
                data = results[name]["conditions"][condition]
                rows = rows_by_model[(name, condition)]
                for policy, threshold in (("saved", results[name]["saved_threshold"]),
                                          ("oracle_5pct_per_source", data["recall_at_fpr_budgets_per_source"]["0.05"]["threshold"]),
                                          ("oracle_95pct_recall", data["min_fpr_at_recall_per_source"]["0.95"]["threshold"])):
                    member_sets[policy+"_false_negatives"].append({r["id"] for r in rows if r["label"] and r["score"] < threshold})
                    member_sets[policy+"_false_positives"].append({r["id"] for r in rows if not r["label"] and r["score"] >= threshold})
            common[condition][group] = {key: sorted(set.intersection(*sets)) for key, sets in member_sets.items()}
    report = {"scope": "Exploratory oracle frontiers on previously inspected examples. Thresholds chosen from these sets are optimistic descriptive limits, NOT independently validated settings or deployment recommendations.",
              "inputs": {"scores_sha256": sha256("docs/task-aware-results.json"), "notinject_sha256": sha256("docs/notinject-evaluation.json"), "challenge_sha256": study["challenge_sha256"]},
              "population": {"synthetic_attacks":256, "synthetic_benign":256, "notinject_benign":291,
                             "note":"Original and quoted conditions reuse the same cases; they are not independent replications. Pooled mixture is artificial; main comparisons bound FPR separately on each benign source."},
              "models": results, "common_errors": common}
    report["shared_error_patterns"] = export_common_errors(common)
    write_json(OUT/"full-frontiers.json", report)
    compact = dict(report)
    compact["models"] = {name: {**model, "conditions": {
        condition: {key: value for key, value in data.items() if key not in ("curve", "synthetic_curve")}
        for condition, data in model["conditions"].items()}} for name, model in results.items()}
    write_json("docs/detector-diagnostic.json", compact)
    write_report(report)
    plot(report)


def pct(value):
    return f"{value*100:.1f}%"


def export_common_errors(common):
    import pyarrow.parquet as pq
    cases = {"synthetic:"+r["id"]: {"task": r["task"], "document": r["document"],
              "family": r["family"], "source": "synthetic", "label": r["label"]}
             for r in read_jsonl("data/splits/task-aware-v1/task-document/test.jsonl")}
    source = read_json("docs/notinject-evaluation.json")
    for subset in ("one", "two", "three"):
        filename = f"data/NotInject_{subset}-00000-of-00001.parquet"
        path = Path("data/raw/external/notinject")/filename
        if sha256(path) != source["benchmark"]["files"][filename]["sha256"]:
            raise ValueError("NotInject cached source changed")
        for i, row in enumerate(pq.read_table(path).to_pylist()):
            cases[f"notinject:{subset}-{i}"] = {"task": "Summarize the following document in one factual sentence.",
                "document": row["prompt"], "family": row.get("category", "unknown"), "source": "notinject", "label": 0}
    records, counts = [], {}
    for group, errors in common["original"].items():
        counts[group] = {}
        for policy, ids in errors.items():
            counts[group][policy] = dict(Counter(cases[key]["source"]+":"+cases[key]["family"] for key in ids))
            records.extend({"id": key, "model_group": group, "condition": "original", "policy_and_error": policy,
                            **cases[key]} for key in ids)
    write_jsonl(OUT/"shared-errors-with-text.jsonl", records)
    return counts


def write_report(report):
    write_compact_report("detector-diagnostic")


def plot(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    budgets=[i/1000 for i in range(201)]
    methods={"Baseline":["baseline"], "Contextual CE":[f"context-v2-ce-{s}" for s in (42,43,44)],
             "Document-only":[f"document-{s}" for s in (42,43,44)], "Task + document":[f"task-document-{s}" for s in (42,43,44)]}
    fig, axes=plt.subplots(1,3,figsize=(14,4.3),sharey=True,layout="constrained")
    panels=(("original", "synthetic_curve", "Synthetic benign only"),
            ("original", "curve", "Original + NotInject benign"),
            ("quoted", "curve", "Quoted + NotInject benign"))
    for ax, (condition, curve_key, title) in zip(axes,panels):
        for label, names in methods.items():
            curves=[[100*best_recall(report["models"][name]["conditions"][condition][curve_key],b)["recall"] for b in budgets] for name in names]
            mean=[sum(v)/len(v) for v in zip(*curves)]
            line,=ax.step([b*100 for b in budgets],mean,label=label,where="post")
            if len(curves)>1:
                ax.fill_between([b*100 for b in budgets],[min(v) for v in zip(*curves)],[max(v) for v in zip(*curves)],alpha=.13,color=line.get_color(),step="post")
        ax.set(title=title,xlabel="Maximum false alarms per benign source (%)",xlim=(0,20),ylim=(0,102))
        ax.axvline(5,color="gray",linestyle=":",linewidth=1)
        ax.axhline(95,color="gray",linestyle=":",linewidth=1)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Attacks detected (%)")
    axes[2].legend(fontsize=8,loc="lower right")
    fig.suptitle("Exploratory limits on inspected cases — means and seed ranges")
    fig.savefig("docs/detector-diagnostic.png",dpi=180)
    fig.savefig("docs/detector-diagnostic.svg")
    svg = Path("docs/detector-diagnostic.svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
