"""Dependency-free release verification; no model invocation."""
import copy,hashlib,json
from pathlib import Path
from foodmap_parser.meal_context_contract import load_dataset,parse_response
from foodmap_parser.meal_context_training import approved_partitions,load_config,training_schedule,split_keys
from foodmap_parser.meal_context_experiment import evaluate
ROOT=Path(__file__).resolve().parent
def main():
 m=json.loads((ROOT/'reproduction_manifest.json').read_text())
 for name,wanted in m['files'].items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted,name
 rows=load_dataset(ROOT/'data/data_n224.jsonl');parts=approved_partitions(rows)
 confirm=load_dataset(ROOT/'data/reserved_development63.jsonl')
 assert len(parts['train'])==224 and len(parts['development'])==37 and len(confirm)==63
 assert not set(split_keys(parts['train']))&set(split_keys(confirm))
 assert not set(split_keys(parts['development']))&set(split_keys(confirm))
 for p in (ROOT/'configs').glob('*.json'):
  cfg=load_config(p);schedule=list(training_schedule(224,cfg))
  assert len(schedule)==224 and sum(len(x['indices']) for x in schedule)==896
 try:evaluate(parts['train'],[])
 except ValueError:pass
 else:raise AssertionError('train evaluation should be rejected')
 bad=copy.deepcopy(rows);bad[0]['review_status']='pending'
 try:approved_partitions(bad)
 except ValueError:pass
 else:raise AssertionError('pending training label should be rejected')
 print(json.dumps({'status':'passed','source_and_data_hashes_verified':len(m['files']),
                  'train':224,'development':100,'model_execution':False}))
if __name__=='__main__':main()
