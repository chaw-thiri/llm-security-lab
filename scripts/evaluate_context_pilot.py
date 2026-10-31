"""Compare fixed checkpoints on the locked pilot challenge and disclosed baseline regression set."""
from pathlib import Path
from torch.utils.data import DataLoader
from promptshield.classifier import InjectionClassifier
from promptshield.common import read_jsonl,write_json,sha256,choose_device
from promptshield.data import TextDataset,Collator
from promptshield.train import predict_loader
from promptshield.scoring import metrics,select_threshold

def run():
 out=Path('results/context-pilot-v1');out.mkdir(parents=True,exist_ok=True)
 report={'scope':'Single-seed synthetic pilot, not independent external generalization evidence. Baseline test is a previously inspected regression set.', 'arms':{}, 'training_counts':{'rows_per_arm':1024,'synthetic_rows_targeted':32,'updates_per_arm':64}}
 device=choose_device('mps')
 for arm,path in [('baseline','runs/baseline-v1/models/full/best'),('control','runs/context-pilot-control/best'),('targeted','runs/context-pilot-targeted/best')]:
  model,tok,meta=InjectionClassifier.load(path,device)
  def predict(rows):
   return predict_loader(model,DataLoader(TextDataset(rows,tok,512),batch_size=8,shuffle=False,collate_fn=Collator(tok)),device)
  entry={'checkpoint':path,'weights_sha256':sha256(Path(path)/'model.safetensors'),'threshold':meta['threshold'],'evaluations':{}}
  thresholds={'saved':meta['threshold']}
  if arm=='baseline':
   y,p,_=predict(read_jsonl('data/splits/context-pilot-v1/control/validation.jsonl'))
   thresholds['pilot_validation_recalibrated']=select_threshold(y,p,.02)
  entry['thresholds']=thresholds
  for name,file in [('challenge','data/splits/context-pilot-v1/control/test.jsonl'),('baseline_regression','data/splits/piguard-v1/test.jsonl')]:
   rows=read_jsonl(file);y,p,loss=predict(rows)
   data={'split_sha256':sha256(file),'rows':len(rows),'loss':loss,'metrics_by_threshold':{k:metrics(y,p,t) for k,t in thresholds.items()}}
   if name=='challenge':
    data['predictions']=[{'id':r['id'],'label':label,'quoted':r['quoted'],'score':score} for r,label,score in zip(rows,y,p)]
    data['by_quotation']={str(q):metrics([label for r,label in zip(rows,y) if r['quoted']==q],[score for r,score in zip(rows,p) if r['quoted']==q],meta['threshold']) for q in (False,True)}
   entry['evaluations'][name]=data
   print(arm,name,data['metrics_by_threshold']['saved'],flush=True)
  report['arms'][arm]=entry;write_json(out/'comparison.json',report)
  del model
 return report
if __name__=='__main__':run()
