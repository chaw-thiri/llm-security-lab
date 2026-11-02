"""Post-hoc domain diagnostic; never fits a serving policy or reads reserved data."""
from promptshield.provenance import frozen_protocol_sha256
from promptshield.reporting import write_report as write_compact_report
import argparse
import hashlib
import math
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from promptshield.common import read_json, read_jsonl, write_json, write_jsonl, sha256
from promptshield.data import normalized
from promptshield.scoring import metrics, select_threshold
from diagnose_detectors import frontier, best_recall, least_fpr

DATA = Path("data/splits/grounded-upgrade-v1")
CACHE = Path("results/grounded-upgrade-v1")
OUT = Path("results/domain-scoring-v1")
DOMAINS = ("technical", "security", "general")
BUDGET = .05


def distribution(values):
    if not values:
        return None
    p = np.asarray(values, dtype=float)
    return dict(n=len(p), minimum=float(p.min()), maximum=float(p.max()),
                **{name: float(np.quantile(p, q)) for name, q in
                   (("q05", .05), ("q25", .25), ("median", .5), ("q75", .75), ("q95", .95))},
                exact_zero=int((p == 0).sum()), exact_one=int((p == 1).sum()), unique=len(set(values)))


def attainable(rows):
    return frontier([dict(r, source=r["domain"]) for r in rows])


def summarize(rows, threshold):
    result = {"fixed": metrics([r["label"] for r in rows], [r["score"] for r in rows], threshold),
              "oracle_global_5pct_each_domain": best_recall(attainable(rows), BUDGET), "domains": {}}
    for domain in sorted({r["domain"] for r in rows}):
        selected = [r for r in rows if r["domain"] == domain]
        benign = sorted((r["score"] for r in selected if not r["label"]), reverse=True)
        result["domains"][domain] = {
            "fixed": metrics([r["label"] for r in selected], [r["score"] for r in selected], threshold),
            "scores": {str(label): distribution([r["score"] for r in selected if r["label"] == label]) for label in (0, 1)},
            "oracle_local_5pct": best_recall(attainable(selected), BUDGET),
            "oracle_min_fpr_at_95pct_recall": least_fpr(attainable(selected), .95),
            "minimum_threshold_for_5pct_benign_fpr": math.nextafter(benign[math.floor(BUDGET * len(benign))], math.inf)}
    positives = sum(r["label"] for r in rows)
    result["oracle_domain_routed_recall"] = sum(d["oracle_local_5pct"]["tp"] for d in result["domains"].values()) / positives
    result["cross_domain_auc"] = {}
    for attack_domain in result["domains"]:
        attacks = [r["score"] for r in rows if r["label"] and r["domain"] == attack_domain]
        result["cross_domain_auc"][attack_domain] = {}
        for benign_domain in result["domains"]:
            negatives = [r["score"] for r in rows if not r["label"] and r["domain"] == benign_domain]
            result["cross_domain_auc"][attack_domain][benign_domain] = float(roc_auc_score([1]*len(attacks)+[0]*len(negatives), attacks+negatives))
    return result


def checked_scores(saved, cases, weights_hash, split_hash):
    if saved["provenance"] != {"weights_sha256": weights_hash, "split_sha256": split_hash}:
        raise ValueError("Saved score provenance does not match frozen inputs")
    rows = saved["predictions"]
    if len(rows) != len(cases) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Score count or IDs do not match cases")
    for row, case in zip(rows, cases):
        if (row["id"], row["label"], row["domain"]) != (case["id"], case["label"], case["domain"]):
            raise ValueError("Saved scores are reordered, relabeled or assigned to the wrong domain")
        if not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
            raise ValueError("Invalid saved probability")
    return rows


def transfer_local_thresholds(development, fresh):
    thresholds = {}
    for domain in sorted({r["domain"] for r in fresh}):
        rows = [r for r in development if r["domain"] == domain]
        thresholds[domain] = select_threshold([r["label"] for r in rows], [r["score"] for r in rows], BUDGET)
    decisions = [float(r["score"] >= thresholds[r["domain"]]) for r in fresh]
    return {"scope": "Post-hoc diagnostic with known source-domain metadata; not a validated router or serving change",
            "thresholds_from_development_only": thresholds,
            "fresh": metrics([r["label"] for r in fresh], decisions, .5)}


def paired_fresh(rows, cases, threshold):
    by_id = {r["id"]: r for r in rows}
    result = {}
    for domain in DOMAINS:
        differences, quote_differences, misses_above_both = [], [], 0
        attacks = [c for c in cases if c["domain"] == domain and c["label"]]
        for case in attacks:
            prefix = case["source_id"] + "-" + case["id"][len(case["source_id"])+1:].split("-")[0]
            attack, clean, quoted = (by_id[key]["score"] for key in (case["id"], prefix+"-clean", prefix+"-quoted"))
            differences.append(attack-clean)
            quote_differences.append(attack-quoted)
            misses_above_both += int(attack < threshold and attack > max(clean, quoted))
        result[domain] = {"attacks": len(attacks), "above_own_clean": sum(x > 0 for x in differences),
                          "above_own_quotation": sum(x > 0 for x in quote_differences),
                          "missed_despite_outranking_both_controls": misses_above_both,
                          "score_delta_from_clean": distribution(differences),
                          "score_delta_from_quotation": distribution(quote_differences)}
    return result


def annotate_development(cases):
    raw = {}
    result = []
    for case in cases:
        kind = "original"
        if case.get("source_id"):
            source = case["source_id"]
            if source not in raw:
                raw[source] = normalized((Path("data/raw/real-documents-v1")/(source+".txt")).read_text())
            kind = "attack" if case["label"] else "clean" if normalized(case["prompt"]) in raw[source] else "descriptive_example"
        result.append(dict(case, benign_kind=kind))
    return result


def source_summary(rows, cases, threshold):
    result = {}
    for source in sorted({c.get("source_id") for c in cases if c.get("source_id")}):
        subset = [(r, c) for r, c in zip(rows, cases) if c.get("source_id") == source]
        selected = [r for r, _ in subset]
        result[source] = {"domain": selected[0]["domain"], "rows": len(selected),
                          "fixed": metrics([r["label"] for r in selected], [r["score"] for r in selected], threshold),
                          "benign_by_kind": {}}
        for kind in sorted({c["benign_kind"] for _, c in subset if not c["label"]}):
            p = [r["score"] for r, c in subset if not c["label"] and c["benign_kind"] == kind]
            result[source]["benign_by_kind"][kind] = distribution(p)
    return result


def build_probes(contexts, payloads, descriptions):
    """Identical instruction and marker across contexts; every insertion is at the end."""
    rows = []
    for context in contexts:
        base = {key: context[key] for key in ("context_id", "context_split", "source_id", "domain")}
        rows.append(dict(base, id=context["context_id"]+"-clean", prompt=context["document"], label=0,
                         condition="clean", payload_split="none", payload_index=-1))
        for payload_split in ("validation", "fresh"):
            for index, template in enumerate(payloads[payload_split]):
                marker = "TOKEN_"+hashlib.sha256(f"domain-probe-v1-{payload_split}-{index}".encode()).hexdigest()[:12].upper()
                payload = template.format(marker=marker)
                variants = {"active": payload, **{"descriptive_"+split: descriptions[split][0].format(p=payload)
                                                         for split in ("train", "validation", "fresh")}}
                for condition, insertion in variants.items():
                    rows.append(dict(base, id=f"{context['context_id']}-{payload_split}-{index}-{condition}",
                                     prompt=context["document"]+"\n"+insertion, label=int(condition == "active"),
                                     condition=condition, payload_split=payload_split, payload_index=index))
    return rows


def probe_cases(development, fresh):
    from transformers import AutoTokenizer
    from prepare_grounded_upgrade import windows, PAYLOADS, DESCRIPTIONS
    tok = AutoTokenizer.from_pretrained("data/raw/piguard/backbone", local_files_only=True)
    contexts = []
    for source in sorted({c["source_id"] for c in development if c.get("source_id")}):
        case = next(c for c in development if c.get("source_id") == source)
        candidates = windows((Path("data/raw/real-documents-v1")/(source+".txt")).read_text().strip(), tok)
        for i, passage in enumerate((candidates[0], candidates[-1])):
            contexts.append(dict(context_id=f"development-{source}-{i}", context_split="development", source_id=source,
                                 domain=case["domain"], document=passage))
    for case in fresh:
        if case["family"] == "clean":
            contexts.append(dict(context_id="fresh-"+case["id"], context_split="fresh", source_id=case["source_id"],
                                 domain=case["domain"], document=case["prompt"]))
    rows = build_probes(contexts, PAYLOADS, DESCRIPTIONS)
    lengths = [len(ids) for ids in tok([r["prompt"] for r in rows], truncation=False)["input_ids"]]
    if max(lengths) > 512 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Probe IDs overlap or a complete probe would be truncated")
    rows = [dict(r, tokens=n, construction_label=True, human_reviewed=False) for r, n in zip(rows, lengths)]
    path = OUT/"controlled-probes.jsonl"
    if path.exists() and read_jsonl(path) != rows:
        raise ValueError("Existing controlled probes changed")
    write_jsonl(path, rows)
    return rows


def score_probes(entry, rows):
    from torch.utils.data import DataLoader
    import torch
    from promptshield.classifier import InjectionClassifier
    from promptshield.common import choose_device
    from promptshield.data import TextDataset, Collator
    weight_hash = entry["weights_sha256"]
    provenance = {"weights_sha256": weight_hash, "probe_sha256": sha256(OUT/"controlled-probes.jsonl"),
                  "protocol_sha256": frozen_protocol_sha256("docs/domain-scoring-protocol.md"), "batch_size": 8,
                  "input": "document only, no metadata; full probe <=512 tokens"}
    path = OUT/f"probes-{weight_hash[:16]}.json"
    if path.exists():
        saved = read_json(path)
        if saved["provenance"] != provenance:
            raise ValueError("Stale probe cache")
        if len(saved["predictions"]) != len(rows) or any(
                (p["id"], p["label"], p["domain"]) != (r["id"], r["label"], r["domain"])
                or not math.isfinite(p["logit_margin"]) or not 0 <= p["score"] <= 1
                for p, r in zip(saved["predictions"], rows)):
            raise ValueError("Probe cache is misaligned or invalid")
        return saved["predictions"]
    device = choose_device("mps")
    model, tok, meta = InjectionClassifier.load(entry["checkpoint"], device)
    if meta["max_length"] != 512 or meta.get("config", {}).get("input_format", "document") != "document":
        raise ValueError("Probe experiment requires the existing document-only checkpoints")
    loader = DataLoader(TextDataset(rows, tok, meta["max_length"]), batch_size=8, shuffle=False, collate_fn=Collator(tok))
    predictions = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            p = logits.softmax(-1)[:, 1].cpu().tolist()
            margins = (logits[:, 1]-logits[:, 0]).cpu().tolist()
            predictions.extend(dict(id=r["id"], label=r["label"], domain=r["domain"], score=score, logit_margin=margin)
                               for r, score, margin in zip(rows[len(predictions):len(predictions)+len(p)], p, margins))
    del model
    if len(predictions) != len(rows):
        raise ValueError("Incomplete probe inference")
    write_json(path, {"provenance": provenance, "predictions": predictions})
    return predictions


def summarize_probes(scores, cases, threshold):
    result = []
    for split in ("development", "fresh"):
        for domain in DOMAINS:
            for payload_split in ("validation", "fresh"):
                for condition in ("active", "descriptive_train", "descriptive_validation", "descriptive_fresh"):
                    subset = [r for r, c in zip(scores, cases) if
                              (c["context_split"], c["domain"], c["payload_split"], c["condition"]) == (split, domain, payload_split, condition)]
                    result.append(dict(context_split=split, domain=domain, payload_split=payload_split, condition=condition,
                                       n=len(subset), flagged=sum(r["score"] >= threshold for r in subset),
                                       scores=distribution([r["score"] for r in subset]),
                                       logit_margin=distribution([r["logit_margin"] for r in subset])))
    return result


def write_report(report):
    write_compact_report("domain-scoring")


def plot(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def logit(p):
        p = min(max(p, 1e-7), 1-1e-7)
        return math.log(p/(1-p))
    entry = report["models"]["grounded-42"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True, sharey=True)
    for ax, split, title in zip(axes, ("development", "fresh"), ("Development (6 source pages)", "Measured challenge (18 source pages)")):
        for label, offset, color, name in ((0, -.13, "#2674a6", "Benign"), (1, .13, "#bd4d32", "Attack")):
            stats = [entry[split]["domains"][domain]["scores"][str(label)] for domain in DOMAINS]
            medians = np.array([logit(s["median"]) for s in stats])
            lows = np.array([logit(s["q05"]) for s in stats])
            highs = np.array([logit(s["q95"]) for s in stats])
            ax.errorbar(np.arange(3)+offset, medians, yerr=[medians-lows, highs-medians],
                        fmt="o", color=color, capsize=5, linewidth=2, label=name)
        ax.axhline(logit(entry["saved_threshold"]), linestyle="--", color="#444444", label="Saved threshold")
        ax.set_xticks(range(3), [s.title() for s in DOMAINS])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        ax.set_ylabel("Log-odds transform of saved probability")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Grounded seed 42: score separation and threshold transfer", fontsize=13)
    fig.savefig("docs/figures/domain-scoring-distributions.png", dpi=170)
    svg = Path("docs/figures/domain-scoring-distributions.svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-probes", action="store_true", required=True,
                        help="Required for the complete report: run/cache the fixed controlled inference; no training")
    args = parser.parse_args()
    if (OUT/"status.json").exists() and read_json(OUT/"status.json")["status"] == "complete":
        write_compact_report("domain-scoring")
        print("Completed diagnostic: regenerated compact report only; no inference",flush=True)
        return
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd()/".tmp/matplotlib"))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_json(DATA/"manifest.json")
    study = read_json("docs/records/grounded-upgrade-results.json")
    cases, hashes = {}, {}
    for split, filename in (("development", "development.jsonl"), ("fresh", "fresh-challenge.jsonl")):
        path = DATA/filename
        hashes[split] = sha256(path)
        if hashes[split] != manifest["hashes"][filename]:
            raise ValueError("Frozen study cases changed")
        cases[split] = read_jsonl(path)
    source_manifest = read_json("docs/records/real-document-source-manifest.json")
    sources = {r["id"]: r for r in source_manifest["sources"]}
    for source in {r["source_id"] for r in cases["development"] if r.get("source_id")}:
        if sha256(Path("data/raw/real-documents-v1")/(source+".txt")) != sources[source]["text_sha256"]:
            raise ValueError("Public source snapshot changed")
    cases["development"] = annotate_development(cases["development"])
    report = {"scope": "Exploratory investigation of already completed development/challenge results and controlled post-hoc public-text probes. No new generalization claim or tuned serving policy.",
              "provenance": {"study_sha256": sha256("docs/records/grounded-upgrade-results.json"), "case_hashes": hashes,
                             "protocol_sha256": frozen_protocol_sha256("docs/domain-scoring-protocol.md"),
                             "analysis_script_sha256": sha256(__file__)},
              "models": {}, "probes": {}, "probe_clean": {}}
    for name in ("baseline-calibrated", "control-42", "control-43", "control-44", "grounded-42", "grounded-43", "grounded-44"):
        entry = study["models"][name]
        if sha256(Path(entry["checkpoint"])/"model.safetensors") != entry["weights_sha256"]:
            raise ValueError("Checkpoint changed")
        scores = {split: checked_scores(read_json(CACHE/f"{split}-{entry['weights_sha256'][:16]}.json"), rows,
                                       entry["weights_sha256"], hashes[split]) for split, rows in cases.items()}
        report["models"][name] = {"checkpoint": entry["checkpoint"], "weights_sha256": entry["weights_sha256"],
            "saved_threshold": entry["threshold"], **{split: summarize(rows, entry["threshold"]) for split, rows in scores.items()},
            "development_domain_threshold_transfer": transfer_local_thresholds(scores["development"], scores["fresh"]),
            "matched_fresh_controls": paired_fresh(scores["fresh"], cases["fresh"], entry["threshold"]),
            "development_sources": source_summary(scores["development"], cases["development"], entry["threshold"])}
        print(name, "saved/oracle global/oracle domain recall", entry["fresh"]["metrics"]["recall"],
              report["models"][name]["fresh"]["oracle_global_5pct_each_domain"]["recall"],
              report["models"][name]["fresh"]["oracle_domain_routed_recall"], flush=True)
    if args.with_probes:
        probes = probe_cases(cases["development"], cases["fresh"])
        report["provenance"]["probe_sha256"] = sha256(OUT/"controlled-probes.jsonl")
        for name in ("baseline-calibrated", "control-42", "grounded-42", "grounded-43", "grounded-44"):
            entry = study["models"][name]
            predictions = score_probes(entry, probes)
            report["probes"][name] = summarize_probes(predictions, probes, entry["threshold"])
            report["probe_clean"][name] = [dict(context_split=split, domain=domain,
                scores=distribution([r["score"] for r, c in zip(predictions, probes)
                    if (c["context_split"], c["domain"], c["condition"]) == (split, domain, "clean")]))
                for split in ("development", "fresh") for domain in DOMAINS]
            print("Controlled inference complete:", name, flush=True)
    write_json("docs/records/domain-scoring-investigation.json", report)
    plot(report)
    write_report(report)
    write_json(OUT/"status.json", {"status": "complete", "models": list(report["models"]),
                                 "probe_models": list(report["probes"]), "serving_policy_changed": False})


if __name__ == "__main__":
    main()
