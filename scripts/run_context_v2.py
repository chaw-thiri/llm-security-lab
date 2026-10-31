"""Run preregistered arms; replicate promising development results before opening challenge scores."""
from pathlib import Path
import subprocess
import sys
from torch.utils.data import DataLoader
from promptshield.classifier import InjectionClassifier
from promptshield.common import read_json,read_jsonl,write_json,choose_device,sha256
from promptshield.data import TextDataset,Collator
from promptshield.train import predict_loader
from promptshield.scoring import metrics

ROOT=Path('results/context-v2')

def development(checkpoint):
    model,tok,meta=InjectionClassifier.load(checkpoint,choose_device('mps'))
    rows=read_jsonl('data/splits/context-v2/development.jsonl')
    loader=DataLoader(TextDataset(rows,tok,512),batch_size=8,collate_fn=Collator(tok))
    labels,scores,_=predict_loader(model,loader,choose_device('mps'))
    result={'checkpoint':str(checkpoint),'weights_sha256':sha256(Path(checkpoint)/'model.safetensors'),'development':metrics(labels,scores,meta['threshold']),'original_validation':meta['validation']}
    del model
    return result

def train_arm(arm,seed):
    run=Path(f'runs/context-v2-{arm}-{seed}');status=run/'status.json'
    if not status.exists() or read_json(status)['status']!='complete':
        cmd=[sys.executable,'-m','promptshield.train','--config',f'configs/context-v2-{arm}-{seed}.json']
        if (run/'config.json').exists():cmd+=['--resume']
        subprocess.run(cmd,check=True)
    if read_json(status)['status']!='complete':raise RuntimeError('Run paused or incomplete; stop study before evaluation.')
    result=development(run/'best');write_json(ROOT/f'{arm}-{seed}-development.json',result)
    return result

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    baseline=development('runs/baseline-v1/models/full/best');write_json(ROOT/'baseline-development.json',baseline)
    outcomes={}
    for arm in ('control','ce','ranking'):
        write_json(ROOT/'study-status.json',{'status':'training','arm':arm,'seed':42})
        outcomes[f'{arm}-42']=train_arm(arm,42)
    reference=outcomes['control-42']['development'];promising=[]
    for arm in ('ce','ranking'):
        result=outcomes[f'{arm}-42'];d=result['development']
        if (d['benign_fpr']<=min(reference['benign_fpr'],baseline['development']['benign_fpr'])-.10 and
            d['recall']>=baseline['development']['recall']-.02 and
            result['original_validation']['recall']>=baseline['original_validation']['recall']-.01):promising.append(arm)
    write_json(ROOT/'replication-decision.json',{'promising':promising,'rule':'Development benign FPR improves >=10pp over baseline and control; development attack recall drops <=2pp; full original validation recall drops <=1pp. No challenge/test selection.'})
    if promising:
        for seed in (43,44):
            for arm in ('control','ce','ranking'):
                write_json(ROOT/'study-status.json',{'status':'training','arm':arm,'seed':seed})
                outcomes[f'{arm}-{seed}']=train_arm(arm,seed)
    write_json(ROOT/'development-comparison.json',{'baseline':baseline,'runs':outcomes,'promising':promising})
    write_json(ROOT/'study-status.json',{'status':'training_complete','runs':list(outcomes),'promising':promising})
    print('DEVELOPMENT COMPLETE',promising,flush=True)
if __name__=='__main__':main()
