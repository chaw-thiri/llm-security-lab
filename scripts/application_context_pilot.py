"""Replay disclosed application regression cases; not a new independent challenge."""
from pathlib import Path
import requests
from promptshield.application import OllamaService,process_document
from promptshield.predict import Detector
from promptshield.security_eval import synthetic_cases,canonical
from promptshield.common import write_json,write_jsonl

def run():
 out=Path('results/context-pilot-v1');client=OllamaService()
 tags=requests.get('http://localhost:11434/api/tags',timeout=10).json()
 version=requests.get('http://localhost:11434/api/version',timeout=10).json()
 cases=synthetic_cases();records=[];report={'scope':'Previously inspected 20-case application regression, not an independent generalization test.','ollama':{'version':version,'models':tags},'conditions':{}}
 for arm,path in [('unprotected',None),('baseline','runs/baseline-v1/models/full/best'),('control','runs/context-pilot-control/best'),('targeted','runs/context-pilot-targeted/best')]:
  detector=Detector(path,'mps') if path else None
  group=[]
  for case in cases:
   result=process_document(detector,client,case['task'],case['document']) if detector else {'blocked':False,'response':client.generate(case['task'],case['document'])}
   record={'id':case['id'],'label':case['label'],'family':case['family'],'arm':arm,**result,'task_success':canonical(result['response'])==canonical(case['expected']),'attack_success':bool(case['label'] and canonical(result['response'])==canonical(case['attacker_marker']))}
   records.append(record);group.append(record);write_jsonl(out/'application-responses.jsonl',records)
  attacks=[r for r in group if r['label']];benign=[r for r in group if not r['label']]
  report['conditions'][arm]={'attack_success_count':sum(r['attack_success'] for r in attacks),'attack_count':len(attacks),'benign_completion_count':sum(r['task_success'] for r in benign),'benign_count':len(benign),'benign_block_count':sum(r['blocked'] for r in benign),'attack_block_count':sum(r['blocked'] for r in attacks)}
  print(arm,report['conditions'][arm],flush=True);write_json(out/'application-comparison.json',report)
  del detector
if __name__=='__main__':run()
