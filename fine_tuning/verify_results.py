"""Replay all published raw generations without loading a language model."""
import hashlib,json
from pathlib import Path
from foodmap_parser.meal_context_contract import load_dataset
from foodmap_parser.meal_context_experiment import evaluate
from foodmap_parser.meal_context_inference import verify_saved_predictions,strict_annotation_audit
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def main():
 for name,wanted in read(ROOT/'release_manifest.json')['files'].items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted,name
 protocol=read(ROOT/'protocol.json');n=0
 for arm in protocol['arms']:
  for split,file in [('known','evaluation_original37.jsonl'),('confirm','reserved_development63.jsonl')]:
   rows=load_dataset(ROOT/'data'/file);prefix=ROOT/'results'/f'eval_{split}_{arm["name"]}'
   predictions=[json.loads(l) for l in Path(str(prefix)+'_predictions.jsonl').read_text().splitlines()]
   assert len(rows)==len(predictions)
   verify_saved_predictions(rows,predictions)
   result=evaluate(rows,predictions);result['strict_annotation_audit']=strict_annotation_audit(rows,predictions)
   assert result==read(Path(str(prefix)+'_evaluation.json'))
   n+=len(rows)
 assert n==600
 print(json.dumps({'status':'verified','raw_predictions_replayed':n,'model_execution':False}))
if __name__=='__main__':main()
