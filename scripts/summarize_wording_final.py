"""Audit joined final errors without refitting any model or threshold."""
from collections import Counter
from pathlib import Path
from statistics import mean
from promptshield.common import read_json,read_jsonl,write_json,write_jsonl,sha256
from prepare_wording_final import DATA,OUT


def main():
    report=read_json("docs/wording-final-results.json")
    if read_json(OUT/"status.json")["status"]!="complete":raise RuntimeError("Study must be complete")
    if sha256(DATA/"test.jsonl")!=read_json(DATA/"manifest.json")["splits"]["test"]["sha256"]:
        raise ValueError("Final examples changed")
    cases=read_jsonl(DATA/"test.jsonl");by_model={}
    for name,r in report["models"].items():
        path=OUT/f"final-{r['weights_sha256'][:16]}.json"
        saved=read_json(path)
        if saved["provenance"]!={"weights_sha256":r["weights_sha256"],"split_sha256":sha256(DATA/"test.jsonl")}:
            raise ValueError("Score provenance changed")
        predictions=saved["predictions"]
        if len(predictions)!=len(cases) or any((p["id"],p["label"],p["domain"])!=(c["id"],c["label"],c["domain"]) for p,c in zip(predictions,cases)):
            raise ValueError("Score/example alignment failed")
        by_model[name]=[int(p["score"]>=r["threshold"]) for p in predictions]
        counts=Counter((c["label"],p) for c,p in zip(cases,by_model[name]))
        cm=r["final"]["metrics"]["confusion_matrix"]
        if [counts[0,0],counts[0,1],counts[1,0],counts[1,1]]!=[cm[k] for k in ("tn","fp","fn","tp")]:
            raise ValueError("Confusion totals disagree")
    summary=dict(scope="Post-evaluation error audit only; no training/calibration decisions",milestone_passed=report["milestone_passed"],
        selected_family=report["family_decision"]["selected_family"],paired_changes={},arms={},source_page_count=18,
        effective_independence="Cases share pages, payloads and outer styles; three seeds share initial weights")
    for s in (42,43,44):
        a,b=by_model[f"context-{s}"],by_model[f"wording-{s}"]
        summary["paired_changes"][str(s)]={
            "attack_recoveries":sum(c["label"]==1 and x==0 and y==1 for c,x,y in zip(cases,a,b)),
            "new_attack_misses":sum(c["label"]==1 and x==1 and y==0 for c,x,y in zip(cases,a,b)),
            "benign_recoveries":sum(c["label"]==0 and x==1 and y==0 for c,x,y in zip(cases,a,b)),
            "new_benign_false_alarms":sum(c["label"]==0 and x==0 and y==1 for c,x,y in zip(cases,a,b))}
    for arm in ("context","wording"):
        values=[report["models"][f"{arm}-{s}"] for s in (42,43,44)]
        shared=[c for i,c in enumerate(cases) if all(by_model[f"{arm}-{s}"][i]!=c["label"] for s in (42,43,44))]
        summary["arms"][arm]=dict(
            final_mean=report["arm_summary"][arm],
            original_f1_mean=mean(v["original_regression"]["metrics"]["f1"] for v in values),
            exposed_challenge_mean={k:mean(v["exposed_challenge_regression"]["metrics"][k] for v in values) for k in ("recall","benign_fpr","f1")},
            final_rate_ranges={k:[min(v["final"]["metrics"][k] for v in values),max(v["final"]["metrics"][k] for v in values)] for k in ("recall","benign_fpr")},
            calibration_context_mse=mean(v["margin_gaps"]["context"]["mean_squared_gap"] for v in values),
            calibration_wording_mse=mean(v["margin_gaps"]["wording"]["mean_squared_gap"] for v in values),
            shared_attack_misses=sum(c["label"]==1 for c in shared),
            shared_benign_false_alarms=sum(c["label"]==0 for c in shared),
            shared_errors_by_role_and_label={f"{k[0]}:{k[1]}":v for k,v in Counter((c["role_family"],c["label"]) for c in shared).items()},
            mean_domain_rates={d:{k:mean(v["final"]["by_domain"][d][k] for v in values) for k in ("recall","benign_fpr")} for d in ("technical","security","general")},
            mean_clean_false_alarms=mean(v["final_breakdowns"]["role_family"]["clean"]["metrics"]["confusion_matrix"]["fp"] for v in values))
    old_path=Path("data/splits/context-consistency-v1/test.jsonl")
    old_hash=read_json(old_path.parent/"manifest.json")["splits"]["test"]["sha256"]
    if sha256(old_path)!=old_hash:raise ValueError("Exposed regression changed")
    old_cases=read_jsonl(old_path); selected=report["family_decision"]["selected_family"];old_predictions={}
    for seed in (42,43,44):
        name=f"{selected}-{seed}";entry=report["models"][name]
        saved=read_json(OUT/f"exposed-regression-{entry['weights_sha256'][:16]}.json")
        if saved["provenance"]!={"weights_sha256":entry["weights_sha256"],"split_sha256":old_hash}:
            raise ValueError("Exposed score provenance differs")
        ps=saved["predictions"]
        if len(ps)!=len(old_cases) or any((p["id"],p["label"],p["domain"])!=(c["id"],c["label"],c["domain"]) for p,c in zip(ps,old_cases)):
            raise ValueError("Exposed score/example alignment failed")
        old_predictions[name]=[dict(p,prediction=int(p["score"]>=entry["threshold"])) for p in ps]
        cm=entry["exposed_challenge_regression"]["metrics"]["confusion_matrix"]
        if sum(p["label"]==0 and p["prediction"]==1 for p in old_predictions[name])!=cm["fp"] or sum(p["label"]==1 and p["prediction"]==0 for p in old_predictions[name])!=cm["fn"]:
            raise ValueError("Exposed error totals disagree")
    old_errors=[dict(c,predictions={n:ps[i] for n,ps in old_predictions.items()}) for i,c in enumerate(old_cases)
                if any(ps[i]["prediction"]!=c["label"] for ps in old_predictions.values())]
    common=[r for r in old_errors if all(p["prediction"]!=r["label"] for p in r["predictions"].values())]
    summary["exposed_regression_audit"]=dict(shared_attack_misses=sum(r["label"]==1 for r in common),
        shared_benign_false_alarms=sum(r["label"]==0 for r in common),
        shared_errors_by_role_and_label={f"{k[0]}:{k[1]}":v for k,v in Counter((r["role_family"],r["label"]) for r in common).items()},
        local_errors=str(OUT/"exposed-regression-errors.jsonl"),review_status="Posthoc assistant inspection; no new independent review or tuning")
    write_jsonl(OUT/"exposed-regression-errors.jsonl",old_errors)
    write_json("docs/wording-final-summary.json",summary)
    print(summary,flush=True)


if __name__=="__main__":main()
