"""Local CUDA Qwen3 inference for meal-context-v3; no training or downloads.

Keep raw generations, strict validation failures and incremental progress. Pending
synthetic labels can only be scored explicitly as developmental diagnostics.
"""

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .consumption_experiment import measure_peak_rss
from .meal_context_contract import (PROMPT, VERSION, digest, input_digest, load_dataset,
                                    messages_for, normalize_temporal, validate_target)
from .meal_context_experiment import evaluate
from .meal_model_experiment import read_jsonl, write_json, write_jsonl
from .meal_model_training import check_gpu_budget, checkpoint_fingerprint, file_digest
from .qwen_meal_contract import parse_response
from .meal_context_training import adapter_fingerprint


def prediction_from_raw(row, raw):
    result = {"id": row["id"], "contract_version": VERSION,
              "input_sha256": input_digest(row), "status": "ok",
              "raw_output": raw, "output": None}
    try:
        result["output"] = validate_target(parse_response(raw), row)
        result["temporal_preview"] = [normalize_temporal(e, row) for e in result["output"]["events"]]
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        result.update(status="invalid_output", output=None, error=str(exc))
    return result


def requests_for(rows, prompt):
    requests = []
    for row in rows:
        messages = messages_for(row)
        messages[0]["content"] = prompt
        requests.append({"id": row["id"], "input_sha256": input_digest(row), "messages": messages})
    return requests


def strict_annotation_audit(rows, predictions):
    """Supplement existing joint metrics without changing their definition.

    Exact annotation equality includes evidence, subject basis and components.
    Alternative equivalent evidence/list ordering may fail this conservative audit.
    """
    matches = [p["status"] == "ok" and p["output"] == r["target"]
               for r, p in zip(rows, predictions)]
    return {"version": "strict-annotation-v1", "matched_rows": sum(matches),
            "rows": len(rows), "error_ids": [r["id"] for r, ok in zip(rows, matches) if not ok],
            "definition": "Full target equality including evidence, subject basis, components and list order; not semantic equivalence."}


def error_analysis_rows(rows, predictions, evaluation):
    joint_errors = set(evaluation["error_ids"])
    strict_errors = set(evaluation["strict_annotation_audit"]["error_ids"])
    return [{"id": r["id"], "input": r["input"], "candidate_target": r["target"],
             "joint_mismatch": r["id"] in joint_errors,
             "strict_annotation_mismatch": r["id"] in strict_errors,
             "prediction": p} for r, p in zip(rows, predictions)
            if r["id"] in joint_errors | strict_errors]


def verify_saved_predictions(rows, predictions):
    """Resume only the exact ordered prefix; never silently repair a torn file."""
    if len(predictions) > len(rows):
        raise ValueError("saved predictions exceed dataset")
    for row, saved in zip(rows, predictions):
        if (saved.get("id") != row["id"] or saved.get("contract_version") != VERSION
                or saved.get("input_sha256") != input_digest(row)):
            raise ValueError("saved prediction order/contract/input mismatch")
        rebuilt = prediction_from_raw(row, saved["raw_output"])
        for field in ("status", "output", "temporal_preview", "error"):
            if saved.get(field) != rebuilt.get(field):
                raise ValueError("saved output differs from its raw generation")
    return predictions


def open_run(output_dir, signature, requests, *, resume=False):
    out = Path(output_dir)
    if resume:
        previous = parse_response((out / "manifest.json").read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError("resume requires identical data, code, prompt, model, config and environment")
        if file_digest(out / "requests.jsonl") != previous["requests_file_sha256"]:
            raise ValueError("saved requests changed")
        if digest(read_jsonl(out / "requests.jsonl")) != digest(requests):
            raise ValueError("new requests differ from saved exact prompt text")
    else:
        out.mkdir(parents=True, mode=0o700, exist_ok=False)
        write_jsonl(out / "requests.jsonl", requests)
        write_json(out / "manifest.json", {
            "created_at": datetime.now(timezone.utc).isoformat(), "signature": signature,
            "requests_file_sha256": file_digest(out / "requests.jsonl"),
            "privacy": "local raw text; not anonymized or encrypted",
            "purpose": "adapter inference" if signature.get("adapter") else "unadapted base-model inference",
        })
    return out


def execute(rows, model_dir, output_dir, *, precision="bf16", prompt=PROMPT,
            max_input_tokens=2048, max_new_tokens=1024, gpu_budget_mib=7000,
            diagnostic_only=False, resume=False, adapter_run_dir=None):
    # Apply evaluation gates before loading weights or creating output files.
    if not rows or any(r["split"] == "train" or r["target"] is None for r in rows):
        raise ValueError("this baseline needs labeled development/test rows")
    if not diagnostic_only and any(r["review_status"] != "approved" for r in rows):
        raise ValueError("pending labels require diagnostic-only")
    if precision not in {"bf16", "nf4"}:
        raise ValueError("unsupported precision")
    if any(type(n) is not int or n < 1 for n in (max_input_tokens, max_new_tokens)):
        raise ValueError("positive integer token limits required")
    if not math.isfinite(gpu_budget_mib) or gpu_budget_mib <= 0 or not prompt.strip():
        raise ValueError("positive GPU budget and nonempty prompt required")
    packages = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "accelerate", "bitsandbytes")}
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA with BF16 required; no silent CPU/dtype fallback")
    properties = torch.cuda.get_device_properties(0)
    hardware = {"gpu_name": properties.name, "gpu_total_bytes": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(0)),
                "cuda_build": torch.version.cuda, "platform": platform.platform(),
                "driver_and_device": subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"], text=True).strip()}
    config = {"precision": precision, "max_input_tokens": max_input_tokens,
              "max_new_tokens": max_new_tokens, "gpu_budget_mib": gpu_budget_mib,
              "diagnostic_only": diagnostic_only, "enable_thinking": False,
              "do_sample": False, "batch_size": 1, "seed": 42,
              "cpu_offload": False, "runtime_downloads": False, "attention": "sdpa"}
    base_checkpoint = checkpoint_fingerprint(model_dir)
    adapter = None
    if adapter_run_dir is not None:
        if precision != "nf4" or prompt != PROMPT:
            raise ValueError("v3 adapter requires matching NF4 precision and training prompt")
        adapter = adapter_fingerprint(adapter_run_dir, base_checkpoint, rows)
        packages["peft"] = importlib.metadata.version("peft")
    signature = {"contract_version": VERSION, "dataset_sha256": digest(rows),
                 "prompt_sha256": digest(prompt), "config": config,
                 "checkpoint": base_checkpoint, "packages": packages, "adapter": adapter,
                 "hardware": hardware,
                 "python": platform.python_version(),
                 "code_sha256": {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    out = open_run(output_dir, signature, requests_for(rows, prompt), resume=resume)
    saved_requests = read_jsonl(out / "requests.jsonl")
    prediction_path = out / "predictions.jsonl"
    predictions = read_jsonl(prediction_path) if prediction_path.exists() and prediction_path.stat().st_size else []
    verify_saved_predictions(rows, predictions)
    if (out / "completed.json").exists():
        completed = parse_response((out / "completed.json").read_text(encoding="utf-8"))
        if len(predictions) != len(rows) or completed["predictions_file_sha256"] != file_digest(prediction_path):
            raise ValueError("completed run has missing or changed predictions")
        return completed
    attempt_start = time.perf_counter()
    attempted_records = 0
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA with BF16 required; no silent CPU/dtype fallback")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(0)
        root = str(Path(model_dir).resolve())
        model_context_limit = parse_response((Path(root) / "config.json").read_text(encoding="utf-8"))["max_position_embeddings"]
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False, use_fast=True)
        prompts = []
        for request in saved_requests:
            rendered = tokenizer.apply_chat_template(request["messages"], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            if encoded.input_ids.shape[1] > max_input_tokens:
                raise ValueError(f"{request['id']}: prompt exceeds token budget; no silent truncation")
            if encoded.input_ids.shape[1] + max_new_tokens > model_context_limit:
                raise ValueError(f"{request['id']}: prompt plus output budget exceeds model context")
            prompts.append(encoded)
        kwargs = {"local_files_only": True, "trust_remote_code": False,
                  "device_map": {"": 0}, "dtype": torch.bfloat16, "attn_implementation": "sdpa"}
        if precision == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(root, **kwargs)
        if adapter_run_dir is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(Path(adapter_run_dir).resolve() / "adapter"),
                is_trainable=False, local_files_only=True)
        model.eval()
        check_gpu_budget(gpu_budget_mib)
        generation = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens,
            use_cache=True, eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.eos_token_id)
        with prediction_path.open("a", encoding="utf-8") as handle:
            for row, encoded in zip(rows[len(predictions):], prompts[len(predictions):]):
                inputs = encoded.to("cuda:0")
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    tokens = model.generate(**inputs, generation_config=generation)
                torch.cuda.synchronize()
                latency = time.perf_counter() - start
                generated = tokens[0, inputs.input_ids.shape[1]:]
                raw = tokenizer.decode(generated, skip_special_tokens=True)
                prediction = prediction_from_raw(row, raw)
                memory = check_gpu_budget(gpu_budget_mib)
                eos = generation.eos_token_id
                eos_ids = eos if isinstance(eos, list) else [eos]
                prediction.update(input_tokens=inputs.input_ids.shape[1], generated_tokens=len(generated),
                    ended_with_eos=bool(len(generated) and generated[-1].item() in eos_ids),
                    reached_token_cap=len(generated) >= max_new_tokens,
                    latency_seconds=latency, **memory)
                handle.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                predictions.append(prediction)
                attempted_records += 1
                print(json.dumps({"id": row["id"], "status": prediction["status"],
                                  "generated_tokens": len(generated)}, ensure_ascii=False), flush=True)
                del tokens, generated, inputs
        evaluation = evaluate(rows, predictions, diagnostic_only=diagnostic_only)
        evaluation["strict_annotation_audit"] = strict_annotation_audit(rows, predictions)
        if not (out / "evaluation.json").exists():
            write_json(out / "evaluation.json", evaluation)
        else:
            if parse_response((out / "evaluation.json").read_text(encoding="utf-8")) != evaluation:
                raise ValueError("existing evaluation differs; preserve original run")
        error_rows = error_analysis_rows(rows, predictions, evaluation)
        if not (out / "errors.jsonl").exists():
            write_jsonl(out / "errors.jsonl", error_rows)
        memory = check_gpu_budget(gpu_budget_mib)
        result = {"status": "inferred_and_evaluated", "model_execution_performed": True,
                  "records": len(predictions), "attempt_records": attempted_records,
                  "resumed": resume, "diagnostic_only": diagnostic_only,
                  "invalid_outputs": evaluation["invalid_outputs"],
                  "exact_event_match": evaluation["exact_event_match"],
                  "peak_gpu_allocated_mib": max([memory["peak_gpu_allocated_mib"]] +
                      [p["peak_gpu_allocated_mib"] for p in predictions]),
                  "peak_gpu_reserved_mib": max([memory["peak_gpu_reserved_mib"]] +
                      [p["peak_gpu_reserved_mib"] for p in predictions]),
                  "attempt_peak_cpu_rss_mib": measure_peak_rss(),
                  "attempt_seconds": time.perf_counter() - attempt_start,
                  "generation_seconds_total": sum(p["latency_seconds"] for p in predictions),
                  "gpu_name": torch.cuda.get_device_name(0), "cuda_build": torch.version.cuda,
                  "predictions_file_sha256": file_digest(prediction_path),
                  "flops_measured": False,
                  "memory_scope": "PyTorch allocated/reserved; not total board VRAM. CPU RSS is this attempt only.",
                  "limitations": ["pending synthetic labels are not a validated research benchmark",
                      "joint exact match excludes subject basis, evidence spans and explicit components",
                      "inference only; any training is recorded in the referenced adapter run; no personalization, distillation or wearable validation",
                      "resume skips durable predictions; interrupted generation is rerun"]}
        write_json(out / "completed.json", result)
        return result
    except Exception as exc:
        stamp = time.time_ns()
        write_json(out / f"failed-{stamp}.json", {"error_type": type(exc).__name__, "error": str(exc),
            "completed": False, "durable_predictions": len(predictions),
            "attempt_seconds": time.perf_counter() - attempt_start})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--precision", choices=("bf16", "nf4"), default="bf16")
    parser.add_argument("--prompt-file")
    parser.add_argument("--adapter-run-dir", help="Completed, fingerprinted v3 QLoRA training run")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-budget-mib", type=float, default=7000)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    rows = [r for r in load_dataset(args.input) if r["split"] == args.split]
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else PROMPT
    result = execute(rows, args.model_dir, args.output_dir, precision=args.precision, prompt=prompt,
        max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
        gpu_budget_mib=args.gpu_budget_mib, diagnostic_only=args.diagnostic_only, resume=args.resume,
        adapter_run_dir=args.adapter_run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
