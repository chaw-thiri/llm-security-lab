"""Execute all six preregistered runs; choose on calibration before opening test."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
from statistics import mean
import numpy as np
from promptshield.common import read_json,read_jsonl,write_json,write_jsonl,sha256
from promptshield.scoring import select_grouped_threshold
import run_context_consistency as inference
from prepare_wording_final import DATA,OUT,DOMAINS,INITIAL,INITIAL_SHA

# Reuse validated inference/alignment checks, with a separate study cache directory.
ARMS = ("context","wording")
SEEDS = (42,43,44)
PROTOCOL = Path("docs/records/wording-final-protocol.json")


def gaps(predictions,cases):
    grouped={}
    for p,r in zip(predictions,cases):
        if r.get("context_group"):
            grouped.setdefault(r["context_group"],{})[r["context_index"],r["wording_index"],r["label"]]=p["logit_margin"]
    result={"context":[],"wording":[]}
    for cells in grouped.values():
        if len(cells)!=8: raise ValueError("Incomplete diagnostic grid")
        for y in (0,1):
            for w in (0,1): result["context"].append(cells[0,w,y]-cells[1,w,y])
            for c in (0,1): result["wording"].append(cells[c,0,y]-cells[c,1,y])
    return {k:dict(edges=len(v),mean_squared_gap=float(np.mean(np.square(v))),median_absolute_gap=float(np.median(np.abs(v)))) for k,v in result.items()}


def calibrate(name,checkpoint,rows):
    predictions=inference.score(checkpoint,rows,sha256(DATA/"calibration.jsonl"),"calibration")
    threshold=select_grouped_threshold([p["label"] for p in predictions],[p["score"] for p in predictions],
        [p["domain"] for p in predictions],.1)
    provenance=dict(weights_sha256=sha256(Path(checkpoint)/"model.safetensors"),calibration_sha256=sha256(DATA/"calibration.jsonl"),
                    protocol_sha256=sha256(PROTOCOL),method="shared threshold maximizes calibration recall with <=10% benign FPR per domain")
    destination=Path("runs/wording-final-calibrated")/name
    if destination.exists():
        meta=read_json(destination/"detector.json")
        if meta.get("wording_calibration")!=provenance or meta["threshold"]!=threshold or sha256(destination/"model.safetensors")!=provenance["weights_sha256"]:
            raise ValueError("Calibrated checkpoint changed")
    else:
        shutil.copytree(checkpoint,destination)
        meta=read_json(destination/"detector.json")
        meta["threshold_before_wording_calibration"]=meta["threshold"]
        meta["validation_before_wording_calibration"]=meta["validation"]
        meta.update(threshold=threshold,wording_calibration=provenance)
        # 'validation' stays checkpoint-selection metrics; calibration has its own field.
        meta["calibration_metrics"]=inference.describe(predictions,threshold)
        write_json(destination/"detector.json",meta)
    description=inference.describe(predictions,threshold)
    if any(v["benign_fpr"]>.1 for v in description["by_domain"].values()):raise AssertionError("Calibration cap breached")
    result=dict(checkpoint=str(destination),weights_sha256=provenance["weights_sha256"],threshold=threshold,
                calibration=description,calibration_provenance=provenance,margin_gaps=gaps(predictions,rows))
    result["mean_real_recall"]=mean(description["by_domain"][d]["recall"] for d in DOMAINS)
    result["worst_real_recall"]=min(description["by_domain"][d]["recall"] for d in DOMAINS)
    return result


def select_family(runs):
    summaries={a:dict(mean_real_recall=mean(runs[f"{a}-{s}"]["mean_real_recall"] for s in SEEDS),
                      worst_seed_domain_recall=min(runs[f"{a}-{s}"]["worst_real_recall"] for s in SEEDS)) for a in ARMS}
    chosen=max(ARMS,key=lambda a:(summaries[a]["mean_real_recall"],summaries[a]["worst_seed_domain_recall"],a=="context"))
    return dict(selected_family=chosen,summaries=summaries,decision_source="calibration only; before any final scoring",thresholds={n:r["threshold"] for n,r in runs.items()})


def render(report):
    # Presentation reads the immutable measured record; execution bytes are archived.
    from promptshield.reporting import write_report
    write_report("wording-final")


def evaluate(manifest,runs,decision):
    if sha256(PROTOCOL)!=manifest["protocol_sha256"] or sha256(DATA/"test.jsonl")!=manifest["splits"]["test"]["sha256"]:
        raise ValueError("Final challenge/protocol changed after the calibration decision")
    test=read_jsonl(DATA/"test.jsonl")
    original_path=Path("data/splits/piguard-v1/test.jsonl")
    if sha256(original_path)!=read_json("data/splits/piguard-v1/manifest.json")["splits"]["test"]["sha256"]:
        raise ValueError("PIGuard regression changed")
    original=read_jsonl(original_path)
    old_path=Path("data/splits/context-consistency-v1/test.jsonl")
    if sha256(old_path)!=read_json("data/splits/context-consistency-v1/manifest.json")["splits"]["test"]["sha256"]:
        raise ValueError("Exposed challenge changed")
    old=read_jsonl(old_path)
    models={"previous-consistency-original":dict(checkpoint=INITIAL,threshold=read_json(Path(INITIAL)/"detector.json")["threshold"]),**runs}
    report=dict(study="wording-final-v2",protocol_sha256=manifest["protocol_sha256"],data_manifest_sha256=sha256(DATA/"manifest.json"),
        execution_sha256=sha256(OUT/"execution.json"),family_decision=decision,models={},fresh_rows=len(test),fresh_source_pages=18,
        serving_default_changed=False,independent_human_review_completed=False,reserved_external_challenge_used=False)
    all_predictions={}
    for name,entry in models.items():
        checkpoint,t=entry["checkpoint"],entry["threshold"]
        predictions=inference.score(checkpoint,test,sha256(DATA/"test.jsonl"),"final")
        all_predictions[name]=predictions
        result=dict(entry,weights_sha256=sha256(Path(checkpoint)/"model.safetensors"),final=inference.describe(predictions,t),
            original_regression=inference.describe(inference.score(checkpoint,original,sha256(original_path),"original-regression"),t),
            exposed_challenge_regression=inference.describe(inference.score(checkpoint,old,sha256(old_path),"exposed-regression"),t),final_breakdowns={})
        for field in ("role_family","quoted","position","payload_index"):
            result["final_breakdowns"][field]={}
            for value in sorted({str(r.get(field,"none")) for r in test}):
                selected=[p for p,r in zip(predictions,test) if str(r.get(field,"none"))==value]
                result["final_breakdowns"][field][value]=inference.describe(selected,t)
        result["meets_target"]=result["final"]["metrics"]["recall"]>.8 and result["final"]["metrics"]["benign_fpr"]<=.1
        report["models"][name]=result
        write_json(OUT/"evaluation-progress.json",report)
        print(name,"FINAL",result["final"]["metrics"],flush=True)
    chosen=decision["selected_family"]
    report["milestone_passed"]=all(report["models"][f"{chosen}-{s}"]["meets_target"] for s in SEEDS)
    report["arm_summary"]={a:{k:mean(report["models"][f"{a}-{s}"]["final"]["metrics"][k] for s in SEEDS) for k in ("recall","benign_fpr","roc_auc","f1")} for a in ARMS}
    errors=[]
    for i,r in enumerate(test):
        values={n:dict(score=ps[i]["score"],logit_margin=ps[i]["logit_margin"],prediction=int(ps[i]["score"]>=models[n]["threshold"])) for n,ps in all_predictions.items()}
        if any(v["prediction"]!=r["label"] for v in values.values()):errors.append(dict(r,predictions=values))
    write_jsonl(OUT/"per-example-errors.jsonl",errors)
    for n,result in report["models"].items():
        c=result["final"]["metrics"]["confusion_matrix"]
        if sum(r["label"]==0 and r["predictions"][n]["prediction"]==1 for r in errors)!=c["fp"] or sum(r["label"]==1 and r["predictions"][n]["prediction"]==0 for r in errors)!=c["fn"]:
            raise AssertionError("Joined errors do not reproduce measured confusion matrix")
    write_json("docs/records/wording-final-results.json",report)
    render(report)
    write_json(OUT/"status.json",dict(status="complete",milestone_passed=report["milestone_passed"],serving_default_changed=False))


def main():
    if (OUT/"status.json").exists() and read_json(OUT/"status.json")["status"]=="complete":
        render(read_json("docs/records/wording-final-results.json"));return
    inference.OUT = OUT
    OUT.mkdir(parents=True,exist_ok=True)
    os.environ["TMPDIR"]=str(Path.cwd()/".tmp");os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    manifest=read_json(DATA/"manifest.json")
    if sha256(PROTOCOL)!=manifest["protocol_sha256"]:raise ValueError("Protocol changed")
    for split in ("train","validation","calibration","test"):
        if sha256(DATA/f"{split}.jsonl")!=manifest["splits"][split]["sha256"]:raise ValueError("Frozen split changed")
    audit=read_json("docs/records/wording-final-semantic-audit.json")
    if audit["data_manifest_sha256"]!=sha256(DATA/"manifest.json") or not audit["assistant_semantic_audit_completed"] or audit["independent_human_review_completed"]:
        raise ValueError("Honest completed assistant audit required")
    execution=dict(data_manifest_sha256=sha256(DATA/"manifest.json"),protocol_sha256=sha256(PROTOCOL),
        code={p:sha256(p) for p in ("src/promptshield/train.py","src/promptshield/data.py","src/promptshield/wording.py","scripts/prepare_wording_final.py","scripts/run_wording_final.py","scripts/run_context_consistency.py")},
        configs={f"{a}-{s}":sha256(f"configs/wording-final/{a}-{s}.json") for a in ARMS for s in SEEDS})
    path=OUT/"execution.json"
    if path.exists() and read_json(path)!=execution:raise ValueError("Execution changed; choose a new study version")
    write_json(path,execution)
    calibration=read_jsonl(DATA/"calibration.jsonl");runs={}
    for seed in SEEDS:
        for arm in ARMS:
            name=f"{arm}-{seed}";run=Path(f"runs/wording-final-{name}")
            write_json(OUT/"status.json",dict(status="training",arm=arm,seed=seed))
            if not (run/"status.json").exists() or read_json(run/"status.json")["status"]!="complete":
                command=[sys.executable,"-m","promptshield.train","--config",f"configs/wording-final/{name}.json"]
                if (run/"config.json").exists():command.append("--resume")
                subprocess.run(command,check=True)
            if read_json(run/"status.json")["status"]!="complete":raise RuntimeError("Paused/incomplete; final remains unscored")
            runs[name]=calibrate(name,run/"best",calibration)
            runs[name]["training_history"]=read_jsonl(run/"history.jsonl")
            write_json(OUT/"calibration-progress.json",runs)
            print(name,"calibration real mean recall",runs[name]["mean_real_recall"],flush=True)
    decision=select_family(runs);decision_path=OUT/"family-decision.json"
    if decision_path.exists() and read_json(decision_path)!=decision:raise ValueError("Frozen family decision changed")
    write_json(decision_path,decision)
    write_json(OUT/"status.json",dict(status="evaluating_frozen_final",selected_family=decision["selected_family"]))
    evaluate(manifest,runs,decision)


if __name__=="__main__":main()
