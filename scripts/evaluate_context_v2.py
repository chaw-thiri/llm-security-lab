"""Locked challenge/regression evaluation after development-only replication decision."""
from pathlib import Path
from collections import defaultdict
import requests
from torch.utils.data import DataLoader
from promptshield.classifier import InjectionClassifier
from promptshield.common import read_json,read_jsonl,write_json,sha256,choose_device
from promptshield.data import TextDataset,Collator
from promptshield.train import predict_loader
from promptshield.scoring import metrics
from promptshield.application import OllamaService
from promptshield.security_eval import canonical

ROOT=Path('results/context-v2')

def score(checkpoint,rows):
    model,tok,meta=InjectionClassifier.load(checkpoint,choose_device('mps'))
    labels,scores,loss=predict_loader(model,DataLoader(TextDataset(rows,tok,meta['max_length']),batch_size=8,collate_fn=Collator(tok)),choose_device('mps'))
    result={'metrics':metrics(labels,scores,meta['threshold']),'loss':loss,'predictions':[{'id':r['id'],'source':r['source'],'label':label,'score':s} for r,label,s in zip(rows,labels,scores)]}
    result['by_family']={}
    for family in sorted({r.get('family','original') for r in rows}):
        indexes=[i for i,r in enumerate(rows) if r.get('family','original')==family]
        result['by_family'][family]=metrics([labels[i] for i in indexes],[scores[i] for i in indexes],meta['threshold'])
    del model
    return result

def main():
    study=read_json(ROOT/'development-comparison.json')
    paths={'baseline':'runs/baseline-v1/models/full/best',**{key:value['checkpoint'] for key,value in study['runs'].items()}}
    challenge_path=Path('data/splits/context-v2/challenge.jsonl');test_path=Path('data/splits/piguard-v1/test.jsonl')
    review=read_json('docs/context-v2-data-review.json')
    if sha256(challenge_path)!=review['hashes']['challenge.jsonl']:raise ValueError('Challenge changed')
    if sha256(test_path)!=read_json('data/splits/piguard-v1/manifest.json')['splits']['test']['sha256']:raise ValueError('Regression split changed')
    challenge=read_jsonl(challenge_path);regression=read_jsonl(test_path)
    report={'scope':'One authored synthetic study. Challenge families held out, but same author/templates and correlated examples; original test is a disclosed regression reference. No deployment change.','runs':{},'challenge_sha256':sha256(challenge_path),'regression_sha256':sha256(test_path)}
    for key,path in paths.items():
        cache=ROOT/f'{key}-evaluation.json';weight_hash=sha256(Path(path)/'model.safetensors');threshold=read_json(Path(path)/'detector.json')['threshold']
        provenance={'weights_sha256':weight_hash,'challenge_sha256':report['challenge_sha256'],'regression_sha256':report['regression_sha256'],'threshold':threshold,'selected_update':read_json(Path(path)/'detector.json').get('updates'),'metadata_sha256':sha256(Path(path)/'detector.json')}
        if cache.exists():
            result=read_json(cache)
            if result['provenance']!=provenance:raise ValueError('Stale evaluation cache')
        else:
            result={'checkpoint':path,'provenance':provenance,'challenge':score(path,challenge)}
            print(key,'challenge',result['challenge']['metrics'],flush=True)
            baseline_cache=Path('results/threshold-analysis-v1/baseline-scores.json')
            if key=='baseline' and baseline_cache.exists():
                old=read_json(baseline_cache)
                if old['provenance']['weights_sha256']!=weight_hash or old['provenance']['splits']['regression']!=report['regression_sha256']:raise ValueError('Baseline cache provenance mismatch')
                rows=old['scores']['regression'];result['regression']={'metrics':metrics([r['label'] for r in rows],[r['score'] for r in rows],threshold),'predictions':rows}
            else:result['regression']=score(path,regression)
            write_json(cache,result)
        report['runs'][key]=result
        write_json(ROOT/'classification-comparison.json',report)
    cases_file=Path('data/splits/context-v2/application-challenge.jsonl')
    if sha256(cases_file)!=review['hashes']['application-challenge.jsonl']:raise ValueError('Application cases changed')
    cases=read_jsonl(cases_file)
    challenge_by_id={row['id']:row for row in challenge}
    if len(cases)!=len(challenge) or any(case['id'] not in challenge_by_id or case['document']!=challenge_by_id[case['id']]['prompt'] or case['label']!=challenge_by_id[case['id']]['label'] for case in cases):raise ValueError('Application and classification cases differ')
    tags=requests.get('http://localhost:11434/api/tags',timeout=10).json()
    version=requests.get('http://localhost:11434/api/version',timeout=10).json()
    app_cache=ROOT/'unprotected-responses.json'
    provenance={'cases_sha256':sha256(cases_file),'model':next(m for m in tags['models'] if m['name']=='llama3.2:3b')['digest'],'server_version':version,'seed':42,'temperature':0,'num_predict':128}
    if app_cache.exists():
        saved=read_json(app_cache)
        if saved['provenance']!=provenance:raise ValueError('Application environment changed; retain old run and use new study output')
        responses=saved['responses']
    else:responses={}
    client=OllamaService()
    for i,case in enumerate(cases):
        if case['id'] not in responses:
            responses[case['id']]=client.generate(case['task'],case['document'])
            write_json(app_cache,{'provenance':provenance,'responses':responses})
        if (i+1)%16==0:print('application unprotected',i+1,'/',len(cases),flush=True)
    application={'provenance':provenance,'method':'One actual unprotected Llama response per case, reused for each gate when forwarded. Identical trusted task/document/generation settings; gates only block. This isolates gating from generation variation; it is cached response replay, not separate Llama runs per checkpoint. All documents checked to fit context before training.','conditions':{}}
    for key in ['unprotected']+list(paths):
        predictions={r['id']:r for r in report['runs'][key]['challenge']['predictions']} if key!='unprotected' else {}
        threshold=report['runs'][key]['provenance']['threshold'] if predictions else 0
        records=[]
        for case in cases:
            blocked=bool(predictions and predictions[case['id']]['score']>=threshold)
            response=None if blocked else responses[case['id']]
            records.append({'id':case['id'],'label':case['label'],'blocked':blocked,'task_success':canonical(response)==canonical(case['expected']),'attack_success':bool(case['label'] and canonical(response)==canonical(case['attacker_marker']))})
        attacks=[r for r in records if r['label']];benign=[r for r in records if not r['label']]
        application['conditions'][key]={'attack_success_count':sum(r['attack_success'] for r in attacks),'attack_count':len(attacks),'benign_completion_count':sum(r['task_success'] for r in benign),'benign_count':len(benign),'benign_block_count':sum(r['blocked'] for r in benign),'attack_block_count':sum(r['blocked'] for r in attacks)}
    write_json(ROOT/'application-comparison.json',application)
    public={'scope':report['scope'],'development':study,'classification':{k:{'checkpoint':v['checkpoint'],'provenance':v['provenance'],'challenge':v['challenge']['metrics'],'challenge_by_family':v['challenge']['by_family'],'regression':v['regression']['metrics']} for k,v in report['runs'].items()},'application':application}
    write_json('docs/context-v2-results.json',public)
    write_json(ROOT/'study-status.json',{'status':'complete','runs':list(paths)})
    print('STUDY COMPLETE',flush=True)
if __name__=='__main__':main()
