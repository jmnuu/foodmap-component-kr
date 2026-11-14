"""Portable rerun of fixed conditions; no network, no original adapter required."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ORDER=['qv_r8_s42','qkvo_r4_s42','qkvo_r4_s43','qv_r8_s43','qv_r8_s44','qkvo_r4_s44']

def build_plan(model,out):
    stages=[]
    def add(name,module,args):
        stages.append({'stage':name,'command':[sys.executable,'-u','-m','foodmap_parser.'+module,*map(str,args)]})
    def evaluation(split,name,file):
        add('eval_'+split+'_'+name,'meal_context_inference',[
            '--input',ROOT/'data'/file,'--split','development','--model-dir',model,
            '--adapter-run-dir',out/('train_'+name),'--precision','nf4','--max-new-tokens','1024',
            '--output-dir',out/('eval_'+split+'_'+name)])
    for name in ORDER:
        add('train_'+name,'meal_context_training',['--input',ROOT/'data/data_n224.jsonl',
            '--config',ROOT/'configs'/f'{name}.json','--variant','uniform','--model-dir',model,
            '--output-dir',out/('train_'+name)])
        evaluation('known',name,'evaluation_original37.jsonl')
    for name in ORDER:evaluation('confirm',name,'reserved_development63.jsonl')
    return stages

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();model=Path(a.model_dir).resolve();out=Path(a.output_dir).resolve()
    plan=build_plan(model,out)
    if a.dry_run:
        print(json.dumps({'model_execution':False,'stages':plan},indent=2));return
    if not model.is_dir():raise ValueError('local checkpoint required')
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    subprocess.run([sys.executable,str(ROOT/'checks.py')],cwd=ROOT,env=env,check=True)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for stage in plan:
        name=stage['stage']
        if (out/'STOP_AFTER_STAGE').exists():
            (out/'stopped.json').write_text(json.dumps({'before_stage':name}));return
        (out/'progress.json').write_text(json.dumps(stage))
        with (out/f'{name}.log').open('x') as log:
            subprocess.run(stage['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'completed.json').write_text(json.dumps({'status':'completed','new_trainings':6,'evaluations':12}))
if __name__=='__main__':main()
