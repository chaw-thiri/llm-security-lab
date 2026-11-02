"""Validation-only threshold sensitivity; fixed, previously inspected tests are descriptive."""
from promptshield.reporting import write_report as write_compact_report
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
from promptshield.classifier import InjectionClassifier
from promptshield.common import read_json,read_jsonl,write_json,sha256,choose_device
from promptshield.data import TextDataset,Collator
from promptshield.train import predict_loader
from promptshield.scoring import metrics,select_threshold

BUDGETS=(0,.005,.01,.02,.05)
ARMS={'baseline':'runs/baseline-v1/models/full/best','control':'runs/context-pilot-control/best','targeted':'runs/context-pilot-targeted/best'}
SPLITS={'validation':'data/splits/context-pilot-v1/control/validation.jsonl','challenge':'data/splits/context-pilot-v1/control/test.jsonl','regression':'data/splits/piguard-v1/test.jsonl'}

def analyze(validation,evaluations,budgets=BUDGETS):
    policies=[]
    for population in ('original_only','with_synthetic'):
        selected=[r for r in validation if population=='with_synthetic' or not r['source'].startswith('context-')]
        for budget in budgets:
            threshold=select_threshold([r['label'] for r in selected],[r['score'] for r in selected],budget)
            policies.append({'population':population,'fpr_budget':budget,'threshold':threshold,'validation':metrics([r['label'] for r in selected],[r['score'] for r in selected],threshold),'evaluations':{name:metrics([r['label'] for r in rows],[r['score'] for r in rows],threshold) for name,rows in evaluations.items()}})
    return policies

def run(device_name='mps'):
    output=Path('results/threshold-analysis-v1');device=choose_device(device_name)
    report={'scope':'Post-pilot diagnostic; no test-driven threshold selection or deployment changes. Original-only validation here is a balanced 512-example subset, not the full original baseline validation.','arms':{}}
    for arm,checkpoint in ARMS.items():
        provenance={'weights_sha256':sha256(Path(checkpoint)/'model.safetensors'),'splits':{name:sha256(file) for name,file in SPLITS.items()}}
        cache=output/f'{arm}-scores.json'
        if cache.exists():
            stored=read_json(cache)
            if stored['provenance']!=provenance:raise ValueError('Stale score cache')
            scores=stored['scores']
        else:
            model,tok,meta=InjectionClassifier.load(checkpoint,device);scores={}
            for name,file in SPLITS.items():
                rows=read_jsonl(file)
                y,p,_=predict_loader(model,DataLoader(TextDataset(rows,tok,meta['max_length']),batch_size=8,shuffle=False,collate_fn=Collator(tok)),device)
                scores[name]=[{'id':r['id'],'source':r['source'],'label':label,'score':score} for r,label,score in zip(rows,y,p)]
                print(arm,name,'scored',len(rows),flush=True)
            write_json(cache,{'provenance':provenance,'scores':scores});del model
        policies=analyze(scores['validation'],{k:v for k,v in scores.items() if k!='validation'})
        # Same numerical threshold across checkpoints is a diagnostic, not matched operating conditions.
        fixed=read_json('runs/baseline-v1/models/full/best/detector.json')['threshold']
        report['arms'][arm]={'provenance':provenance,'policies':policies,'fixed_baseline_threshold':{k:metrics([r['label'] for r in rows],[r['score'] for r in rows],fixed) for k,rows in scores.items() if k!='validation'}}
        write_json(output/'comparison.json',report)
    write_json("docs/records/threshold-analysis.json",report)
    write_compact_report("threshold-analysis")
    print('Saved compact threshold report and measured JSON',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='mps');run(parser.parse_args().device)
