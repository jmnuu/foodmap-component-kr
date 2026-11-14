"""Reviewed v3 QLoRA experiments. No downloads, automatic approvals or test tuning."""
import argparse
import copy
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .consumption_experiment import measure_peak_rss
from .meal_context_contract import PROMPT, VERSION, digest, input_digest, load_dataset, messages_for, validate_row
from .meal_model_contract import serialize_target, project_token_weights
from .meal_model_experiment import write_json, write_jsonl
from .meal_model_training import checkpoint_fingerprint, file_digest, check_gpu_budget, load_local_model

FOCUS = {'date','time','subject','consumption_status','status_evidence'}


def loss_normalization(config):
    """Legacy example mean or token-weight mean over one optimizer update."""
    value=config['training'].get('loss_normalization','example_mean')
    if value not in ('example_mean','batch_token_mean'):
        raise ValueError('invalid loss_normalization')
    return value


def lora_learning_rates(config):
    """The configured rate is A's rate; B uses an explicit positive ratio.

    Missing ratio preserves the legacy single-group AdamW path. This is the
    LoRA+ A/B learning-rate intervention, not a new optimizer or rank change.
    """
    params=config['training']
    base=params['learning_rate']
    ratio=params.get('lora_lr_ratio',1.0)
    for name,value in [('learning_rate',base),('lora_lr_ratio',ratio)]:
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
            raise ValueError('positive finite '+name+' required')
    rate_b=base*ratio
    if not math.isfinite(rate_b):raise ValueError('nonfinite LoRA B learning rate')
    return {'A':base,'B':rate_b,'B_over_A':ratio}


def lora_optimizer_groups(named_parameters,config):
    """Cover every trainable A/B tensor once; reject unsupported adapters.

    Retain original parameter order and one group when the ratio is one, so
    previously saved baseline runs remain a meaningful numerical control.
    """
    named=list(named_parameters);rates=lora_learning_rates(config)
    buckets={'A':[],'B':[]};seen=set()
    if not named:raise ValueError('nonempty LoRA parameters required')
    for name,param in named:
        if id(param) in seen:raise ValueError('duplicate optimizer parameter')
        seen.add(id(param))
        if not param.requires_grad:raise ValueError('frozen optimizer parameter')
        matches=[key for key in buckets if '.lora_'+key+'.' in name]
        if len(matches)!=1:raise ValueError('only explicit LoRA A/B tensors supported: '+name)
        buckets[matches[0]].append((name,param))
    if not all(buckets.values()):raise ValueError('both LoRA A and B required')
    if rates['B_over_A']==1:
        groups=[{'params':[p for _,p in named],'lr':rates['A']}]
    else:
        groups=[{'params':[p for _,p in buckets[key]],'lr':rates[key]} for key in ('A','B')]
    audit={'method':'standard_lora' if rates['B_over_A']==1 else 'loraplus_ab_rates',
           'learning_rates':rates,'optimizer_group_count':len(groups),
           'factors':{key:{'names':[n for n,_ in values],
                           'parameters':sum(p.numel() for _,p in values)}
                      for key,values in buckets.items()}}
    return groups,audit


def supervised_weight_sum(feature):
    """Denominator after causal shift; masked prompt tokens never contribute."""
    labels,weights=feature['labels'],feature['loss_weights']
    if len(labels)!=len(weights) or len(labels)<2:
        raise ValueError('aligned causal labels and loss weights required')
    for label,weight in zip(labels,weights):
        if not math.isfinite(weight) or weight<0 or (label==-100 and weight!=0):
            raise ValueError('invalid supervised loss weight')
    mass=sum(weights[1:])
    if mass<=0:raise ValueError('positive shifted supervised weight required')
    return mass


def normalize_loss_sum(loss_sum, example_weight_sum, group_weight_sum, group_size, normalization):
    """Accept scalar tensors or floats; do not divide token mean by group size again."""
    if normalization=='example_mean':
        return loss_sum/example_weight_sum/group_size
    if normalization=='batch_token_mean':
        return loss_sum/group_weight_sum
    raise ValueError('invalid loss_normalization')


def load_config(path):
    cfg=json.loads(Path(path).read_text())
    if cfg['version']!='meal-context-qlora-v1' or cfg['contract_version']!=VERSION:
        raise ValueError('v3 QLoRA config required')
    t,r=cfg['training'],cfg['runtime']
    loss_normalization(cfg)
    lora_learning_rates(cfg)
    for key in ('epochs','gradient_accumulation_steps','lora_rank','lora_alpha'):
        if type(t[key]) is not int or t[key]<1: raise ValueError('invalid '+key)
    if 'max_optimizer_steps' in t and (type(t['max_optimizer_steps']) is not int or t['max_optimizer_steps']<1):
        raise ValueError('positive integer max_optimizer_steps required')
    for key in ('learning_rate','focus_weight','max_grad_norm'):
        if type(t[key]) not in (int,float) or not math.isfinite(t[key]) or t[key]<=0: raise ValueError('invalid '+key)
    if t['focus_weight']<1 or t['gradient_checkpointing'] is not True or type(t['seed']) is not int:
        raise ValueError('invalid weighting/checkpointing/seed')
    if not isinstance(t['lora_dropout'],(int,float)) or not 0<=t['lora_dropout']<1:
        raise ValueError('invalid dropout')
    modules=t['target_modules']
    if not isinstance(modules,list) or not modules or len(set(modules))!=len(modules) or not set(modules)<={'q_proj','k_proj','v_proj','o_proj'}:
        raise ValueError('explicit nonduplicated attention projections required')
    for key in ('max_input_tokens','max_new_tokens','max_sequence_tokens'):
        if type(r[key]) is not int or r[key]<1: raise ValueError('invalid '+key)
    if r['quantization']!='nf4_double' or r['micro_batch_size']!=1 or any(r[k] is not False for k in ('enable_thinking','cpu_offload','network_downloads')):
        raise ValueError('requires NF4, batch 1, non-thinking, offline, no offload')
    return cfg


def planned_optimizer_steps(example_count, config):
    """Bounded update count; epoch-based v1 behavior remains the default."""
    if type(example_count) is not int or example_count<1:
        raise ValueError('positive example count required')
    t=config['training']
    available=math.ceil(example_count/t['gradient_accumulation_steps'])*t['epochs']
    requested=t.get('max_optimizer_steps',available)
    if type(requested) is not int or requested<1 or requested>available:
        raise ValueError('optimizer step budget exceeds epoch capacity or is invalid')
    return requested


def training_schedule(example_count, config):
    """Repeatable shuffled groups; the cap never drops a partial accumulation group."""
    budget=planned_optimizer_steps(example_count,config)
    t=config['training'];rng=random.Random(t['seed']);steps=0
    for epoch in range(1,t['epochs']+1):
        order=list(range(example_count));rng.shuffle(order)
        for start in range(0,example_count,t['gradient_accumulation_steps']):
            group=order[start:start+t['gradient_accumulation_steps']]
            steps+=1
            yield {'epoch':epoch,'indices':group,'epoch_completed':start+len(group)==example_count}
            if steps==budget:return


def split_keys(rows):
    keys=set()
    for row in rows:
        keys.add(digest(['group',row['source_group_id']]))
        if row['speaker_id'] is not None: keys.add(digest(['speaker',row['speaker_id']]))
        for text in [row['input'],*[x['text'] for x in row['context']]]:
            keys.add(digest(['text',' '.join(unicodedata.normalize('NFKC',text).casefold().split())]))
    return sorted(keys)


def approved_partitions(rows):
    if not rows or len({r['id'] for r in rows})!=len(rows): raise ValueError('nonempty unique IDs required')
    for row in rows: validate_row(row)
    parts={name:[r for r in rows if r['split']==name] for name in ('train','development','test')}
    if not parts['train'] or not parts['development']: raise ValueError('separate train and development required')
    for row in parts['train']+parts['development']:
        if row['review_status']!='approved' or row['target'] is None:
            raise ValueError('human-reviewed train/development required; no synthetic bypass')
    for a,b in (('train','development'),('train','test'),('development','test')):
        if set(split_keys(parts[a])) & set(split_keys(parts[b])):
            raise ValueError('cross-split source/speaker/text leakage')
    return parts


def training_features(row, tokenizer, config, variant):
    if variant not in {'uniform','event_weighted'}: raise ValueError('invalid training variant')
    if row['target'] is None or not getattr(tokenizer,'is_fast',False) or tokenizer.eos_token_id is None:
        raise ValueError('labeled row and fast tokenizer with EOS required')
    prompt=tokenizer.apply_chat_template(messages_for(row),tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix=tokenizer.encode(prompt,add_special_tokens=False)
    focus=config['training']['focus_weight'] if variant=='event_weighted' else 1.0
    target,ranges=serialize_target(row['target'],focus,focus_fields=FOCUS)
    encoded=tokenizer(target,add_special_tokens=False,return_offsets_mapping=True)
    labels=encoded['input_ids']+[tokenizer.eos_token_id]
    runtime=config['runtime']
    if len(prefix)>runtime['max_input_tokens'] or len(labels)>runtime['max_new_tokens'] or len(prefix)+len(labels)>runtime['max_sequence_tokens']:
        raise ValueError(f'{row["id"]}: token budget exceeded; never truncate')
    return {'input_ids':prefix+labels,'labels':[-100]*len(prefix)+labels,
            'loss_weights':[0.0]*len(prefix)+project_token_weights(encoded['offset_mapping'],ranges)+[1.0]}


def adapter_fingerprint(run_dir, base_checkpoint, evaluation_rows=None):
    root=Path(run_dir).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    done=json.loads((root/'completed.json').read_text())
    if manifest['contract_version']!=VERSION or manifest['prompt_sha256']!=digest(PROMPT) or manifest['checkpoint']!=base_checkpoint:
        raise ValueError('adapter contract/prompt/base mismatch')
    if done['status']!='trained' or done['manifest_sha256']!=file_digest(root/'manifest.json'):
        raise ValueError('adapter run not completed or manifest changed')
    files={name:file_digest(root/'adapter'/name) for name in ('adapter_config.json','adapter_model.safetensors')}
    if files!=done['adapter_files']: raise ValueError('adapter files changed')
    if evaluation_rows and set(split_keys(evaluation_rows)) & set(manifest['training_split_keys']):
        raise ValueError('adapter evaluation overlaps training data')
    return {'files':files,'sha256':digest(files),'training_manifest_sha256':done['manifest_sha256'],
            'training_dataset_sha256':manifest['dataset_sha256'],'variant':manifest['variant']}


def train(rows,config,variant,model_dir,output_dir,gpu_budget_mib=7000):
    parts=approved_partitions(rows)  # fail before imports or GPU allocation
    normalization=loss_normalization(config)
    lora_learning_rates(config)
    step_budget=planned_optimizer_steps(len(parts['train']),config)
    if variant not in {'uniform','event_weighted'}: raise ValueError('unsupported variant')
    if not math.isfinite(gpu_budget_mib) or gpu_budget_mib<=0: raise ValueError('positive GPU budget required')
    import torch
    from torch.nn import functional as F
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise ValueError('CUDA BF16 required')
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    params=config['training']
    random.seed(params['seed']); torch.manual_seed(params['seed']); torch.cuda.manual_seed_all(params['seed'])
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    manifest={'contract_version':VERSION,'variant':variant,'config':config,'config_sha256':digest(config),
              'created_at':datetime.now(timezone.utc).isoformat(),'prompt_sha256':digest(PROMPT),
              'dataset_sha256':digest(rows),'training_ids':[r['id'] for r in parts['train']],
              'training_split_keys':split_keys(parts['train']), 'checkpoint':checkpoint_fingerprint(model_dir),
              'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','accelerate')},
              'code_sha256':{p.name:file_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
              'hardware':{'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'platform':platform.platform(),
                          'driver':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],text=True).strip()},
              'data_scope':'human-reviewed synthetic pilot; not real ASR or an external benchmark',
              'planned_optimizer_steps':step_budget,
              'loss_normalization':normalization,
              'process_hash_seed':os.environ.get('PYTHONHASHSEED'),
              'selection':f'fixed {step_budget} optimizer steps; no early stopping or checkpoint choice using evaluation outputs'}
    out=Path(output_dir); out.mkdir(parents=True,mode=0o700,exist_ok=False)
    write_json(out/'manifest.json',manifest); write_jsonl(out/'dataset_snapshot.jsonl',rows)
    started=time.perf_counter()
    try:
        model,tokenizer,dtype=load_local_model(model_dir,config,train=True)
        features=[training_features(r,tokenizer,config,variant) for r in parts['train']]
        weight_sums=[supervised_weight_sum(f) for f in features]
        lengths=[len(f['input_ids']) for f in features]
        if max(lengths)>model.config.max_position_embeddings: raise ValueError('model context exceeded')
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not named or any('lora_' not in n for n,p in named): raise ValueError('only LoRA may be trainable')
        trainable=[p for n,p in named]
        initial={n:p.detach().cpu().clone() for n,p in named}
        groups,optimizer_audit=lora_optimizer_groups(named,config)
        optimizer=torch.optim.AdamW(groups,lr=params['learning_rate'])
        optimizer_audit.update(optimizer='torch.optim.AdamW',defaults=optimizer.defaults)
        write_json(out/'optimizer.json',optimizer_audit)
        model.train(); steps=0;examples_processed=0;target_tokens_processed=0
        epochs_completed=0;epochs_started=0;examples_in_epoch=0
        write_json(out/'tokenization.json',{'train_examples':len(features),'min_tokens':min(lengths),'max_tokens':max(lengths),
                  'target_tokens_per_epoch':sum(sum(x!=-100 for x in f['labels']) for f in features),
                  'trainable_names':[n for n,p in named],'compute_dtype':str(dtype)})
        check_gpu_budget(gpu_budget_mib)
        with (out/'training_steps.jsonl').open('x') as log:
            for batch in training_schedule(len(features),config):
                group=batch['indices']
                group_weight_sum=sum(weight_sums[i] for i in group)
                if batch['epoch']!=epochs_started:examples_in_epoch=0
                epochs_started=batch['epoch']
                optimizer.zero_grad(set_to_none=True);group_loss=0.0
                for i in group:
                    f=features[i]
                    ids=torch.tensor([f['input_ids']],device='cuda')
                    labels=torch.tensor([f['labels'][1:]],device='cuda')
                    weights=torch.tensor([f['loss_weights'][1:]],device='cuda',dtype=torch.float32)
                    with torch.autocast('cuda',dtype=dtype):
                        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[:,:-1,:]
                        losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),labels.reshape(-1),ignore_index=-100,reduction='none')
                        loss=normalize_loss_sum((losses*weights.reshape(-1)).sum(),weights.sum(),
                                                group_weight_sum,len(group),normalization)
                    if not torch.isfinite(loss).item(): raise ValueError('nonfinite training loss')
                    loss.backward();group_loss+=loss.detach().item()
                    examples_processed+=1;examples_in_epoch+=1
                    target_tokens_processed+=sum(x!=-100 for x in f['labels'])
                    del logits,losses,loss,ids,labels,weights
                    check_gpu_budget(gpu_budget_mib)
                norm=torch.nn.utils.clip_grad_norm_(trainable,params['max_grad_norm'])
                if not torch.isfinite(norm).item(): raise ValueError('nonfinite gradient norm')
                optimizer.step();steps+=1
                if batch['epoch_completed']:epochs_completed+=1
                record={'epoch':epochs_started,'step':steps,'loss':group_loss,'gradient_norm':norm.item(),
                        'loss_normalization':normalization,'group_weight_sum':group_weight_sum,
                        'group_size':len(group),'training_ids':[parts['train'][i]['id'] for i in group],
                        'examples_processed':examples_processed,'target_tokens_processed':target_tokens_processed,**check_gpu_budget(gpu_budget_mib)}
                log.write(json.dumps(record)+'\n');log.flush();os.fsync(log.fileno())
                print(json.dumps(record),flush=True)
        if steps!=step_budget:raise ValueError('training ended before fixed step budget')
        torch.cuda.synchronize()
        changed=[n for n,p in named if not torch.equal(initial[n],p.detach().cpu())]
        if not changed: raise ValueError('LoRA weights did not update')
        model.save_pretrained(out/'adapter',safe_serialization=True)
        files={n:file_digest(out/'adapter'/n) for n in ('adapter_config.json','adapter_model.safetensors')}
        result={'status':'trained','model_execution_performed':True,'optimizer_steps':steps,'train_examples':len(features),
                'loss_normalization':normalization,
                'optimizer_audit_sha256':file_digest(out/'optimizer.json'),
                'epochs':epochs_completed,'epoch_limit':params['epochs'],'epochs_started':epochs_started,
                'partial_epoch_examples':examples_in_epoch if epochs_completed<epochs_started else 0,
                'examples_processed':examples_processed,'trainable_parameters':sum(p.numel() for p in trainable),'updated_tensors':len(changed),
                'training_target_tokens':target_tokens_processed,
                'training_seconds':time.perf_counter()-started,'peak_cpu_rss_mib':measure_peak_rss(),**check_gpu_budget(gpu_budget_mib),
                'manifest_sha256':file_digest(out/'manifest.json'),'adapter_files':files,'flops_measured':False,
                'evaluation_performed':False,'training_resume_supported':False}
        write_json(out/'completed.json',result)
        return result
    except Exception as exc:
        write_json(out/'failed.json',{'error_type':type(exc).__name__,'error':str(exc),'completed':False,'seconds':time.perf_counter()-started})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--config',required=True)
    p.add_argument('--variant',choices=['uniform','event_weighted'],required=True)
    p.add_argument('--model-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--gpu-budget-mib',type=float,default=7000)
    a=p.parse_args()
    result=train(load_dataset(a.input),load_config(a.config),a.variant,a.model_dir,a.output_dir,a.gpu_budget_mib)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
