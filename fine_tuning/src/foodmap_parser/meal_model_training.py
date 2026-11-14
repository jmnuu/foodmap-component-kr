"""Optional local CUDA Qwen3/QLoRA backend; never downloads or silently offloads.

Data preparation is dependency-free except tokenization. GPU execution requires
separately installed optional packages and remains unverified on this Mac.
"""

import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from pathlib import Path

from .meal_model_contract import (PROMPT, VERSION, digest, input_digest, messages_for,
                                  parse_response, project_token_weights, serialize_target)


def load_config(path):
    config = parse_response(Path(path).read_text(encoding="utf-8"))
    if config["version"] != "meal-study-v1" or config["contract_version"] != VERSION:
        raise ValueError("unsupported study/contract version")
    runtime, train = config["runtime"], config["training"]
    if runtime["enable_thinking"] is not False or runtime["cpu_offload"] is not False or runtime["network_downloads"] is not False:
        raise ValueError("v1 requires non-thinking, no offload, no download")
    if runtime["quantization"] != "nf4_double" or runtime["micro_batch_size"] != 1:
        raise ValueError("only NF4 double quantization and micro-batch 1 are supported")
    for name in ("max_input_tokens", "max_new_tokens", "max_sequence_tokens"):
        if type(runtime[name]) is not int or runtime[name] < 1:
            raise ValueError(f"invalid {name}")
    for name in ("epochs", "gradient_accumulation_steps", "lora_rank", "lora_alpha"):
        if type(train[name]) is not int or train[name] < 1:
            raise ValueError(f"invalid {name}")
    for name in ("learning_rate", "focus_weight", "max_grad_norm"):
        if not isinstance(train[name], (int, float)) or not math.isfinite(train[name]) or train[name] <= 0:
            raise ValueError(f"invalid {name}")
    if type(train["seed"]) is not int or not isinstance(train["lora_dropout"], (int, float)) or not 0 <= train["lora_dropout"] < 1:
        raise ValueError("invalid seed or LoRA dropout")
    if not train["target_modules"] or any(name not in {"q_proj", "k_proj", "v_proj", "o_proj"} for name in train["target_modules"]):
        raise ValueError("v1 only supports explicitly selected attention projections")
    if train["focus_weight"] < 1 or train["gradient_checkpointing"] is not True:
        raise ValueError("focus weight >=1 and checkpointing required")
    expected = {"T0": ("teacher", "none", None), "T1": ("teacher", "uniform", None),
                "T2": ("teacher", "event_weighted", None), "S0": ("student", "uniform", None),
                "S1": ("student", "uniform", "T0"), "S2": ("student", "uniform", "T2")}
    if set(config["experiments"]) != set(expected):
        raise ValueError("unsupported experiment matrix")
    for variant, values in expected.items():
        entry = config["experiments"][variant]
        if (entry["role"], entry["objective"], entry["teacher_variant"]) != values:
            raise ValueError("experiment definitions must remain fixed")
    return config


def approved_training_rows(rows, config, variant):
    if variant not in config["experiments"] or variant == "T0":
        raise ValueError("T0 is inference-only; unsupported training variant")
    training = [row for row in rows if row["split"] == "train"]
    development = [row for row in rows if row["split"] == "development"]
    if not training or not development:
        raise ValueError("need both training and separate reviewed development records")
    for row in training + development:
        if row["target"] is None or row["review_status"] != "approved":
            raise ValueError("training/development labels require declared human approval")
    expected_teacher = config["experiments"][variant]["teacher_variant"]
    pseudo = [row for row in training if row["label_origin"] == "teacher_pseudo"]
    if expected_teacher is None and pseudo:
        raise ValueError("non-distillation baseline cannot use teacher labels")
    if expected_teacher is not None:
        if not pseudo or any(row["teacher"]["variant"] != expected_teacher for row in pseudo):
            raise ValueError("distillation arm needs the designated teacher's reviewed pseudo-labels")
    return training


def training_features(row, tokenizer, config, variant):
    if row["target"] is None:
        raise ValueError("target required")
    if not getattr(tokenizer, "is_fast", False) or tokenizer.eos_token_id is None:
        raise ValueError("a fast tokenizer with EOS is required for weight alignment")
    prompt = tokenizer.apply_chat_template(messages_for(row), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    focus = config["training"]["focus_weight"] if config["experiments"][variant]["objective"] == "event_weighted" else 1.0
    target, ranges = serialize_target(row["target"], focus)
    encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    target_ids = encoded["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt_ids) > config["runtime"]["max_input_tokens"]:
        raise ValueError(f"{row['id']}: prompt over limit; truncation is forbidden")
    if len(target_ids) > config["runtime"]["max_new_tokens"]:
        raise ValueError(f"{row['id']}: target exceeds inference output budget")
    if len(prompt_ids) + len(target_ids) > config["runtime"]["max_sequence_tokens"]:
        raise ValueError(f"{row['id']}: sequence over limit; truncation is forbidden")
    return {"input_ids": prompt_ids + target_ids,
            "labels": [-100] * len(prompt_ids) + target_ids,
            "loss_weights": [0.0] * len(prompt_ids) + project_token_weights(encoded["offset_mapping"], ranges) + [1.0]}


def weighted_mean(losses, weights):
    """CPU reference for the normalized token-loss objective."""
    if len(losses) != len(weights) or not weights or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("invalid loss weights")
    return sum(loss * weight for loss, weight in zip(losses, weights)) / sum(weights)


def checkpoint_files(model_dir):
    root = Path(model_dir).expanduser().resolve()
    config_path = root / "config.json"
    config = parse_response(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError("dense text Qwen3 checkpoint required, not Qwen3-Omni/MoE source")
    index = root / "model.safetensors.index.json"
    names = set(parse_response(index.read_text(encoding="utf-8"))["weight_map"].values()) if index.exists() else {"model.safetensors"}
    files = list(root.glob("*.json")) + list(root.glob("*.jinja"))
    if not names:
        raise ValueError("empty weight map")
    for name in names:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("invalid weight shard path")
        files.append(root / name)
    if not (root / "tokenizer.json").is_file() or not (root / "tokenizer_config.json").is_file():
        raise ValueError("local fast tokenizer files are required")
    for path in files:
        if not path.is_file() or path.stat().st_size == 0 or path.resolve().parent != root:
            raise ValueError(f"missing/empty/escaping checkpoint file: {path.name}")
    return root, sorted(set(files))


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def checkpoint_fingerprint(model_dir):
    root, files = checkpoint_files(model_dir)
    hashes = {path.name: file_digest(path) for path in files}
    return {"sha256": digest(hashes), "files": hashes}


def preflight(model_dir=None):
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"packages": packages, "cuda_available": False, "checkpoint_present": False,
              "checkpoint_integrity_verified": False, "gpu_runtime_validated": False,
              "downloads_performed": False, "issues": []}
    if packages["torch"]:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_total_mib"] = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    if model_dir:
        try:
            root, files = checkpoint_files(model_dir)
            result.update(checkpoint_present=True, checkpoint_path=str(root), checkpoint_bytes=sum(p.stat().st_size for p in files))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["issues"].append(str(exc))
    return result


def load_local_model(model_dir, config, *, train=False, adapter_dir=None):
    report = preflight(model_dir)
    if not report["checkpoint_present"] or not report["cuda_available"] or any(v is None for v in report["packages"].values()):
        raise ValueError("local text checkpoint, CUDA and all optional packages required; no fallback/download")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    root = str(Path(model_dir).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    torch.cuda.reset_peak_memory_stats(0)
    model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True, trust_remote_code=False,
        quantization_config=quantization, device_map={"": 0}, dtype=dtype, attn_implementation="sdpa")
    if train:
        params = config["training"]
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=params["lora_rank"], lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"], target_modules=params["target_modules"],
            bias="none", task_type="CAUSAL_LM"))
        model.config.use_cache = False
    elif adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(Path(adapter_dir).resolve()), is_trainable=False,
                                         local_files_only=True)
    return model, tokenizer, dtype


def check_gpu_budget(budget_mib):
    import torch
    peak = torch.cuda.max_memory_allocated(0) / 1024 ** 2
    if not math.isfinite(budget_mib) or budget_mib <= 0 or peak > budget_mib:
        raise ValueError(f"GPU allocated memory budget exceeded/invalid: {peak:.1f} MiB")
    return {"peak_gpu_allocated_mib": peak,
            "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024 ** 2}


def train_adapter(rows, config, variant, model_dir, output_dir, gpu_budget_mib):
    # Review/split gates run before importing torch or loading a large model.
    training = approved_training_rows(rows, config, variant)
    import torch
    from torch.nn import functional as F
    params = config["training"]
    random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    torch.cuda.manual_seed_all(params["seed"])
    model, tokenizer, dtype = load_local_model(model_dir, config, train=True)
    features = [training_features(row, tokenizer, config, variant) for row in training]
    # Check reviewed development lengths too, but never backpropagate its labels.
    for row in rows:
        if row["split"] == "development":
            training_features(row, tokenizer, config, variant)
    check_gpu_budget(gpu_budget_mib)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=params["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    model.train()
    rng = random.Random(params["seed"])
    steps = 0
    started = time.perf_counter()
    log_path = Path(output_dir) / "training_steps.jsonl"
    with log_path.open("x", encoding="utf-8") as log:
        for epoch in range(params["epochs"]):
            order = list(range(len(features)))
            rng.shuffle(order)
            size = params["gradient_accumulation_steps"]
            for start in range(0, len(order), size):
                group = order[start:start + size]
                optimizer.zero_grad(set_to_none=True)
                group_loss = 0.0
                for index in group:
                    feature = features[index]
                    ids = torch.tensor([feature["input_ids"]], device="cuda")
                    labels = torch.tensor([feature["labels"][1:]], device="cuda")
                    weights = torch.tensor([feature["loss_weights"][1:]], device="cuda", dtype=torch.float32)
                    with torch.autocast("cuda", dtype=dtype):
                        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, :-1, :]
                        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1),
                                                 ignore_index=-100, reduction="none")
                        loss = (losses * weights.reshape(-1)).sum() / weights.sum()
                    if not torch.isfinite(loss).item():
                        raise ValueError("non-finite training loss; checkpoint not accepted")
                    scaler.scale(loss / len(group)).backward()
                    group_loss += loss.detach().item() / len(group)
                    del logits, losses, loss, ids, labels, weights
                    check_gpu_budget(gpu_budget_mib)
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(trainable, params["max_grad_norm"])
                if not torch.isfinite(norm).item():
                    raise ValueError("non-finite gradient norm; checkpoint not accepted")
                scaler.step(optimizer)
                scaler.update()
                steps += 1
                memory = check_gpu_budget(gpu_budget_mib)
                log.write(json.dumps({"epoch": epoch + 1, "step": steps, "loss": group_loss, **memory}) + "\n")
                log.flush()
    torch.cuda.synchronize()
    model.save_pretrained(Path(output_dir) / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(Path(output_dir) / "adapter")
    return {"optimizer_steps": steps, "train_examples": len(training), "training_seconds": time.perf_counter() - started,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "training_target_tokens": sum(sum(label != -100 for label in f["labels"]) for f in features) * params["epochs"],
            "compute_dtype": str(dtype), **check_gpu_budget(gpu_budget_mib),
            "evaluation_performed": False, "flops_measured": False}
