"""Matched continuation, development-only calibration/replication, then frozen evaluation."""
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from torch.utils.data import DataLoader
from promptshield.classifier import InjectionClassifier
from promptshield.common import choose_device, read_json, read_jsonl, write_json, write_jsonl, sha256
from promptshield.data import TextDataset, Collator
from promptshield.scoring import metrics, select_grouped_threshold
from promptshield.train import predict_loader

DATA=Path("data/splits/grounded-upgrade-v1")
OUT=Path("results/grounded-upgrade-v1")


def predictions(checkpoint,rows,split_hash,tag):
    weight_hash=sha256(Path(checkpoint)/"model.safetensors")
    cache=OUT/f"{tag}-{weight_hash[:16]}.json"
    provenance={"weights_sha256":weight_hash,"split_sha256":split_hash}
    if cache.exists():
        saved=read_json(cache)
        if saved["provenance"]!=provenance:
            raise ValueError("Stale score cache")
        return saved["predictions"]
    device=choose_device("mps")
    model,tok,meta=InjectionClassifier.load(checkpoint,device)
    loader=DataLoader(TextDataset(rows,tok,meta["max_length"]),batch_size=8,shuffle=False,collate_fn=Collator(tok))
    y,p,_=predict_loader(model,loader,device)
    scores=[dict(id=r["id"],label=label,score=score,domain=r.get("domain","original")) for r,label,score in zip(rows,y,p)]
    write_json(cache,{"provenance":provenance,"predictions":scores})
    del model
    return scores


def calibrate(name,checkpoint,development):
    dev_hash=sha256(DATA/"development.jsonl")
    scores=predictions(checkpoint,development,dev_hash,"development")
    threshold=select_grouped_threshold([r["label"] for r in scores],[r["score"] for r in scores],
                                       [r["domain"] for r in scores],.05)
    provenance={"source_weights_sha256":sha256(Path(checkpoint)/"model.safetensors"),
                "validation_sha256":dev_hash,"max_fpr_per_domain":.05,"method":"maximize validation recall subject to each domain's empirical benign FPR"}
    destination=Path(f"runs/grounded-upgrade-calibrated/{name}")
    if destination.exists():
        meta=read_json(destination/"detector.json")
        if meta.get("calibration")!=provenance or sha256(destination/"model.safetensors")!=provenance["source_weights_sha256"]:
            raise ValueError("Calibrated checkpoint already exists with different provenance")
    else:
        shutil.copytree(checkpoint,destination)
        meta=read_json(destination/"detector.json")
        meta["validation_before_group_calibration"]=meta["validation"]
        meta["threshold_before_group_calibration"]=meta["threshold"]
        meta["threshold"]=threshold
        meta["calibration"]=provenance
        meta["validation"]=metrics([r["label"] for r in scores],[r["score"] for r in scores],threshold)
        write_json(destination/"detector.json",meta)
    by_domain={}
    for domain in sorted({r["domain"] for r in scores}):
        group=[r for r in scores if r["domain"]==domain]
        by_domain[domain]=metrics([r["label"] for r in group],[r["score"] for r in group],threshold)
    if any(v["benign_fpr"]>.05 for v in by_domain.values()):
        raise AssertionError("Calibration violated a domain FPR constraint")
    result={"checkpoint":str(destination),"threshold":threshold,"calibration":provenance,
            "development":metrics([r["label"] for r in scores],[r["score"] for r in scores],threshold),"development_by_domain":by_domain}
    write_json(OUT/f"{name}-development.json",result)
    print(name,"development recall",result["development"]["recall"],"per-domain FPR",{k:v["benign_fpr"] for k,v in by_domain.items()},flush=True)
    return result


def train_arm(arm,seed,development):
    path=Path(f"runs/grounded-upgrade-{arm}-{seed}")
    status=path/"status.json"
    if not status.exists() or read_json(status)["status"]!="complete":
        command=[sys.executable,"-m","promptshield.train","--config",f"configs/grounded-upgrade-{arm}-{seed}.json"]
        if (path/"config.json").exists():
            command.append("--resume")
        subprocess.run(command,check=True)
    if read_json(status)["status"]!="complete":
        raise RuntimeError("Run paused or incomplete; do not open fresh evaluation")
    return calibrate(f"{arm}-{seed}",path/"best",development)


def summarize_scores(rows,threshold):
    result={"metrics":metrics([r["label"] for r in rows],[r["score"] for r in rows],threshold),"by_domain":{}}
    for domain in sorted({r["domain"] for r in rows}):
        selected=[r for r in rows if r["domain"]==domain]
        result["by_domain"][domain]=metrics([r["label"] for r in selected],[r["score"] for r in selected],threshold)
    return result


def evaluate(manifest,baseline,runs):
    # This function is only reached after the development-only replication decision.
    fresh=read_jsonl(DATA/"fresh-challenge.jsonl")
    if sha256(DATA/"fresh-challenge.jsonl")!=manifest["hashes"]["fresh-challenge.jsonl"]:
        raise ValueError("Fresh evaluation changed")
    original_path=Path("data/splits/piguard-v1/test.jsonl")
    original=read_jsonl(original_path)
    if sha256(original_path)!=read_json("data/splits/piguard-v1/manifest.json")["splits"]["test"]["sha256"]:
        raise ValueError("Original regression changed")
    real_path=Path("data/splits/real-documents-v1/excerpt-cases.jsonl")
    real_source=read_jsonl(real_path)
    real=[dict(r,prompt=r["document"],domain=r["cohort"]) for r in real_source]
    paths={"baseline-original":{"checkpoint":"runs/baseline-v1/models/full/best","threshold":read_json("runs/baseline-v1/models/full/best/detector.json")["threshold"]},
           "baseline-calibrated":baseline,**runs}
    report={"scope":"Exploratory construction-labeled real-context continuation with fresh public-page case holdout; same publishers and authored attacks. Not human-reviewed or production validation.",
            "data_manifest_sha256":sha256(DATA/"manifest.json"),"source_partition":manifest["source_partition"],
            "fresh_case_count":len(fresh),"fresh_source_count":manifest["counts"]["fresh_sources"],
            "development":{"baseline":baseline,"runs":runs},"models":{}}
    for name,entry in paths.items():
        checkpoint,threshold=entry["checkpoint"],entry["threshold"]
        fresh_scores=predictions(checkpoint,fresh,sha256(DATA/"fresh-challenge.jsonl"),"fresh")
        original_scores=predictions(checkpoint,original,sha256(original_path),"original-regression")
        real_scores=predictions(checkpoint,real,sha256(real_path),"real-regression")
        result={"checkpoint":checkpoint,"weights_sha256":sha256(Path(checkpoint)/"model.safetensors"),"threshold":threshold,
                "fresh":summarize_scores(fresh_scores,threshold),"original_regression":summarize_scores(original_scores,threshold),
                "exposed_real_regression":summarize_scores(real_scores,threshold)}
        # Separate clean source excerpts from authored descriptive-quote negatives.
        result["fresh_by_kind"]={}
        for kind in ("clean","descriptive_quote","attacks"):
            indexes=[i for i,r in enumerate(fresh) if (r["label"]==1 if kind=="attacks" else r["family"]==kind)]
            selected=[fresh_scores[i] for i in indexes]
            result["fresh_by_kind"][kind]={"rows":len(selected),"flagged":sum(r["score"]>=threshold for r in selected)}
        result["real_regression_by_exposure"]={}
        for exposure in ("train","validation","regression"):
            selected=[r for r,case in zip(real_scores,real) if manifest["source_partition"][case["source_id"]]==exposure]
            result["real_regression_by_exposure"][exposure]=summarize_scores(selected,threshold)["metrics"]
        cache=read_json("results/real-documents-v1/unprotected-responses.json")["responses"]
        app=[]
        from validate_real_documents import qa_success, answer_form
        for case,prediction in zip(real_source,real_scores):
            blocked=prediction["score"]>=threshold
            text=None if blocked else cache[case["id"]]["response"]
            app.append(dict(label=case["label"],blocked=blocked,
                            qa=qa_success(text,case["answers"]),hijack=bool(case["attacker_marker"] and answer_form(text)==answer_form(case["attacker_marker"]))))
        result["exposed_real_application_replay"]={"method":"Previously inspected source cases and existing Llama response cache; no fresh utility claim.",
            "benign_qa_completed":sum(r["qa"] for r in app if not r["label"]),"benign_count":30,
            "marker_hijacks":sum(r["hijack"] for r in app if r["label"]),"attack_count":270}
        report["models"][name]=result
        write_json(OUT/"classification-results.json",report)
        print(name,"FRESH",result["fresh"]["metrics"],flush=True)
    write_json("docs/grounded-upgrade-results.json",report)
    lines=["# Grounded document-only upgrade — measured results","",report["scope"],"",
           "No frozen baseline, earlier results, human-review labels or default application checkpoint were overwritten. Thresholds were selected on development only, before the fresh evaluation. The original test and prior 30-source application study are regression references; source exposure is recorded explicitly.","",
           "| Model | Fresh attack recall | Fresh benign FPR | Original regression recall | Original regression F1 | Prior real QA / 30 | Prior real hijacks / 270 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name,m in report["models"].items():
        f,o,a=m["fresh"]["metrics"],m["original_regression"]["metrics"],m["exposed_real_application_replay"]
        lines.append(f"| {name} | {f['recall']:.1%} | {f['benign_fpr']:.1%} | {o['recall']:.1%} | {o['f1']:.1%} | {a['benign_qa_completed']} | {a['marker_hijacks']} |")
    lines += ["","The fresh evaluation has 18 public source pages, 36 clean passages, 36 authored descriptive-quotation controls, and 432 active attack copies. Correlated variants are not independent documents. Full per-domain counts, clean-versus-quotation false alarms, thresholds and checkpoint hashes are in the JSON record.","",
              "Construction labels are not independent human judgments. These new pages come from the same publishers as development material and cannot demonstrate publisher-independent robustness. High scores on these controlled attacks are not proof of resistance to adaptive attacks. The 512-token complete-document limitation remains unchanged.","",
              "The original validation, old synthetic challenges, NotInject and old real-document study have already been inspected. The separate external-tasks-v2 reserved challenge was not opened. Only development metrics selected thresholds and the replication decision; all attempted runs are reported. See [the fixed protocol](grounded-upgrade-protocol.md)."]
    Path("docs/grounded-upgrade-results.md").write_text("\n".join(lines)+"\n")
    write_json(OUT/"status.json",{"status":"complete","models":list(report["models"]),"default_checkpoint_changed":False})


def main():
    (Path.cwd()/".tmp").mkdir(exist_ok=True)
    os.environ["TMPDIR"]=str((Path.cwd()/".tmp").resolve())
    os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=read_json(DATA/"manifest.json")
    if sha256("docs/grounded-upgrade-protocol.md")!=manifest["protocol_sha256"]:
        raise ValueError("Protocol changed after data freeze")
    development=read_jsonl(DATA/"development.jsonl")
    if sha256(DATA/"development.jsonl")!=manifest["hashes"]["development.jsonl"]:
        raise ValueError("Development data changed")
    baseline=calibrate("baseline","runs/baseline-v1/models/full/best",development)
    runs={}
    for arm in ("control","grounded"):
        write_json(OUT/"status.json",{"status":"training","arm":arm,"seed":42})
        runs[f"{arm}-42"]=train_arm(arm,42,development)
    grounded=runs["grounded-42"]
    promising=(grounded["development"]["recall"]>=max(baseline["development"]["recall"],runs["control-42"]["development"]["recall"])+.10
               and grounded["development_by_domain"]["original"]["recall"]>=baseline["development_by_domain"]["original"]["recall"]-.02)
    write_json(OUT/"replication-decision.json",{"promising":promising,"rule":"Development-only +10pp recall over both calibrated baseline/control; original-domain recall drop <=2pp relative to calibrated baseline","baseline":baseline,"runs":runs})
    if promising:
        for seed in (43,44):
            for arm in ("control","grounded"):
                write_json(OUT/"status.json",{"status":"training","arm":arm,"seed":seed})
                runs[f"{arm}-{seed}"]=train_arm(arm,seed,development)
    write_json(OUT/"development-comparison.json",{"baseline":baseline,"runs":runs,"replicated":promising})
    write_json(OUT/"status.json",{"status":"evaluating_fixed_fresh_cases","replicated":promising})
    evaluate(manifest,baseline,runs)


if __name__=="__main__":
    main()
