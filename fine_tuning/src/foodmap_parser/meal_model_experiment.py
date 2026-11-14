"""Offline study CLI: validate, prepare, preflight, train, infer, evaluate, distill.

No implicit model download, automatic label approval, or parser fallback.
"""

import argparse
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .consumption_experiment import measure_peak_rss
from .meal_model_contract import (PROMPT, VERSION, digest, event_counters, input_digest,
                                  load_dataset, messages_for, parse_response, serialize_target,
                                  validate_row, validate_target)
from .meal_model_training import (approved_training_rows, check_gpu_budget, checkpoint_fingerprint,
                                  file_digest, load_config, load_local_model, preflight, train_adapter)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "meal_teacher_student_v1.json"


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    rows = [parse_response(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("predictions must be nonempty with unique IDs")
    return rows


def dataset_summary(rows):
    return {"rows": len(rows), "splits": dict(Counter(r["split"] for r in rows)),
            "label_origins": dict(Counter(r["label_origin"] for r in rows)),
            "review_statuses": dict(Counter(r["review_status"] for r in rows)),
            "dataset_sha256": digest(rows), "model_execution_performed": False}


def manifest(rows, config, variant):
    return {"created_at": datetime.now(timezone.utc).isoformat(), "contract_version": VERSION,
            "variant": variant, "config": config, "config_sha256": digest(config),
            "prompt_sha256": digest(PROMPT), "dataset": dataset_summary(rows),
            "code_sha256": {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "python": sys.version, "platform": platform.platform(),
            "privacy": "local only; raw inputs/outputs are NOT anonymized or encrypted",
            "limitations": ["structural evidence checks do not establish truth", "not validated on GPU yet by the implementation author"]}


def prepare(rows, config, variant, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    write_json(out / "manifest.json", manifest(rows, config, variant))
    write_jsonl(out / "requests.jsonl", [{"id": r["id"], "input_sha256": input_digest(r), "messages": messages_for(r)} for r in rows])
    focus = config["training"]["focus_weight"] if config["experiments"][variant]["objective"] == "event_weighted" else 1.0
    targets = []
    for row in rows:
        if row["target"] is None:
            continue
        target, ranges = serialize_target(row["target"], focus)
        targets.append({"id": row["id"], "split": row["split"], "review_status": row["review_status"],
                        "target_json": target, "focus_character_ranges": ranges})
    write_jsonl(out / "target_design.jsonl", targets)
    result = {"status": "prepared_not_tokenized_or_trained", "records": len(rows),
              "model_execution_performed": False, "training_approval_granted": False}
    write_json(out / "completed.json", result)
    return result


def require_runtime(model_dir, budget_mib):
    if budget_mib is None or not math.isfinite(budget_mib) or budget_mib <= 0:
        raise ValueError("a positive measured GPU allocation budget is required")
    report = preflight(model_dir)
    if not report["checkpoint_present"] or not report["cuda_available"] or any(v is None for v in report["packages"].values()):
        raise ValueError("runtime not ready: " + json.dumps(report, ensure_ascii=False))
    return report


def execute(rows, config, variant, model_dir, output_dir, *, training=False, adapter_dir=None, gpu_budget_mib=None):
    if training:
        approved_training_rows(rows, config, variant)
        if adapter_dir is not None:
            raise ValueError("training starts from the fixed base checkpoint, not an adapter")
    elif (variant == "T0") != (adapter_dir is None):
        raise ValueError("T0 requires the unadapted base; other inference arms require their trained adapter")
    environment = require_runtime(model_dir, gpu_budget_mib)
    checkpoint = checkpoint_fingerprint(model_dir)
    adapter_hashes = None
    if adapter_dir:
        adapter = Path(adapter_dir).resolve()
        previous = parse_response((adapter.parent / "manifest.json").read_text(encoding="utf-8"))
        completion = parse_response((adapter.parent / "completed.json").read_text(encoding="utf-8"))
        if (previous["variant"] != variant or previous["checkpoint"]["sha256"] != checkpoint["sha256"]
                or previous["contract_version"] != VERSION or previous["config_sha256"] != digest(config)
                or completion["status"] != "trained_not_evaluated"):
            raise ValueError("adapter/base/contract/config mismatch or incomplete training")
        adapter_hashes = {p.name: file_digest(p) for p in sorted(adapter.iterdir()) if p.is_file()}
        if adapter_hashes != completion["adapter_files"]:
            raise ValueError("adapter files changed after training")
    out = Path(output_dir)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    metadata = {**manifest(rows, config, variant), "checkpoint": checkpoint,
                "adapter_files": adapter_hashes, "environment": environment,
                "gpu_allocated_budget_mib": gpu_budget_mib,
                "operation": "train" if training else "infer"}
    write_json(out / "manifest.json", metadata)
    try:
        if training:
            result = train_adapter(rows, config, variant, model_dir, out, gpu_budget_mib)
            result.update(status="trained_not_evaluated", adapter_files={p.name: file_digest(p) for p in sorted((out / "adapter").iterdir()) if p.is_file()})
        else:
            import torch
            model, tokenizer, dtype = load_local_model(model_dir, config, adapter_dir=adapter_dir)
            model.eval()
            check_gpu_budget(gpu_budget_mib)
            invalid = 0
            with (out / "predictions.jsonl").open("x", encoding="utf-8") as handle:
                for row in rows:
                    prompt = tokenizer.apply_chat_template(messages_for(row), tokenize=False,
                                                           add_generation_prompt=True, enable_thinking=False)
                    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                    if encoded.input_ids.shape[1] > config["runtime"]["max_input_tokens"]:
                        raise ValueError(f"{row['id']}: input too long; not silently truncated")
                    inputs = encoded.to("cuda:0")
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.inference_mode():
                        tokens = model.generate(**inputs, do_sample=False, max_new_tokens=config["runtime"]["max_new_tokens"],
                                                pad_token_id=tokenizer.eos_token_id, use_cache=True)
                    torch.cuda.synchronize()
                    raw = tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    prediction = {"id": row["id"], "input_sha256": input_digest(row), "contract_version": VERSION,
                                  "variant": variant, "status": "ok", "raw_output": raw,
                                  "output": None, "latency_seconds": time.perf_counter() - start,
                                  "input_tokens": inputs.input_ids.shape[1], "generated_tokens": tokens.shape[1] - inputs.input_ids.shape[1]}
                    try:
                        prediction["output"] = validate_target(parse_response(raw), row)
                    except (ValueError, KeyError, TypeError) as exc:
                        prediction.update(status="invalid_output", error=str(exc))
                        invalid += 1
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    handle.flush()
                    del tokens, inputs, encoded
                    check_gpu_budget(gpu_budget_mib)
            result = {"status": "inferred_not_evaluated", "records": len(rows), "invalid_outputs": invalid,
                      "predictions_sha256": file_digest(out / "predictions.jsonl"), "compute_dtype": str(dtype),
                      "model_execution_performed": True, "flops_measured": False, **check_gpu_budget(gpu_budget_mib)}
        result["peak_cpu_process_rss_mib"] = measure_peak_rss()
        result["memory_note"] = "GPU allocated/reserved and whole-process CPU RSS are separate; RSS unavailable on some platforms"
        write_json(out / "completed.json", result)
        return result
    except Exception as exc:
        write_json(out / "failed.json", {"error_type": type(exc).__name__, "error": str(exc), "completed": False})
        raise


def evaluate(rows, predictions, *, diagnostic_only=False):
    if any(row["split"] == "train" or row["target"] is None for row in rows):
        raise ValueError("evaluation requires labeled development/test rows, never training")
    if not diagnostic_only and any(row["review_status"] != "approved" for row in rows):
        raise ValueError("unreviewed gold: use diagnostic-only, not a validated research score")
    by_id = {row["id"]: row for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError("prediction IDs must exactly cover the selected gold split")
    variants = {row.get("variant") for row in predictions}
    if len(variants) != 1 or not variants <= {"T0", "T1", "T2", "S0", "S1", "S2"}:
        raise ValueError("mixed/unsupported variants")
    names = ("food", "food_time", "food_subject_status", "joint", "self_consumed", "evidence_links")
    totals = {name: Counter(tp=0, fp=0, fn=0) for name in names}
    errors, invalid, exact = [], 0, 0
    for row in rows:
        prediction = by_id[row["id"]]
        if prediction.get("input_sha256") != input_digest(row) or prediction.get("contract_version") != VERSION:
            raise ValueError("prediction input/contract mismatch")
        gold = event_counters(row["target"], row)
        if prediction["status"] == "invalid_output":
            invalid += 1
            actual = {name: Counter({("INVALID_OUTPUT", row["id"]): 1}) for name in names}
        elif prediction["status"] == "ok":
            actual = event_counters(validate_target(prediction["output"], row), row)
        else:
            raise ValueError("unsupported prediction status")
        equal = prediction["status"] == "ok" and actual["joint"] == gold["joint"]
        exact += equal
        if not equal:
            errors.append(row["id"])
        for name in names:
            totals[name]["tp"] += sum((gold[name] & actual[name]).values())
            totals[name]["fp"] += sum((actual[name] - gold[name]).values())
            totals[name]["fn"] += sum((gold[name] - actual[name]).values())
    metrics = {}
    for name, values in totals.items():
        tp, fp, fn = (values[key] for key in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        metrics[name] = {**values, "precision": precision, "recall": recall,
                         "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    return {"variant": next(iter(variants)), "examples": len(rows), "exact_event_match": exact,
            "invalid_outputs": invalid, "metrics": metrics, "error_ids": errors,
            "diagnostic_only": diagnostic_only, "label_origins": dict(Counter(r["label_origin"] for r in rows)),
            "limitations": ["no ingredient accuracy or absolute timestamp evaluation", "quote validation is not semantic proof",
                            "invalid output gets an FP penalty, including on empty gold", "human approval metadata is not independently verified"],
            "gold_sha256": digest(rows), "predictions_sha256": digest(predictions)}


def distillation_candidates(rows, run_dir, output_dir):
    if any(row["split"] != "train" for row in rows):
        raise ValueError("distillation must preserve training-only source splits")
    run_dir = Path(run_dir)
    metadata = parse_response((run_dir / "manifest.json").read_text(encoding="utf-8"))
    completed = parse_response((run_dir / "completed.json").read_text(encoding="utf-8"))
    if metadata["variant"] not in {"T0", "T2"} or metadata["operation"] != "infer" or completed["status"] != "inferred_not_evaluated":
        raise ValueError("a completed T0/T2 inference run is required")
    if completed["predictions_sha256"] != file_digest(run_dir / "predictions.jsonl"):
        raise ValueError("teacher prediction artifact changed")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    by_id = {p["id"]: p for p in predictions}
    if set(by_id) != {r["id"] for r in rows}:
        raise ValueError("teacher run must exactly cover training inputs")
    candidates, rejected = [], []
    for row in rows:
        prediction = by_id[row["id"]]
        if (prediction["input_sha256"] != input_digest(row) or prediction["variant"] != metadata["variant"]
                or prediction["contract_version"] != VERSION):
            raise ValueError("teacher case/variant/contract mismatch")
        if prediction["status"] != "ok":
            rejected.append({"id": row["id"], "reason": prediction["status"]})
            continue
        target = validate_target(prediction["output"], row)
        candidate = {**row, "target": target, "label_origin": "teacher_pseudo", "review_status": "pending", "reviewer": None,
                     "teacher": {"variant": metadata["variant"], "run_sha256": digest(metadata), "input_sha256": input_digest(row)}}
        candidates.append(validate_row(candidate))
    out = Path(output_dir)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    write_jsonl(out / "candidates.pending.jsonl", candidates)
    write_jsonl(out / "rejected.jsonl", rejected)
    result = {"status": "review_queue_not_training_gold", "candidates": len(candidates), "rejected": len(rejected),
              "teacher_manifest_sha256": digest(metadata), "source_dataset_sha256": digest(rows)}
    write_json(out / "completed.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--model-dir")
    check.add_argument("--output-dir", type=Path)
    for name in ("validate", "prepare", "train", "infer", "evaluate", "distill"):
        command = commands.add_parser(name)
        command.add_argument("--input", required=True, type=Path)
        if name != "validate":
            command.add_argument("--output-dir", required=True, type=Path)
        if name in {"prepare", "train", "infer"}:
            command.add_argument("--variant", choices=("T0", "T1", "T2", "S0", "S1", "S2"), required=True)
        if name in {"train", "infer"}:
            command.add_argument("--model-dir", required=True, type=Path)
            command.add_argument("--gpu-budget-mib", required=True, type=float)
        if name == "infer":
            command.add_argument("--adapter-dir", type=Path)
        if name in {"infer", "evaluate"}:
            command.add_argument("--split", choices=("train", "development", "test"), default="development")
        if name == "evaluate":
            command.add_argument("--predictions", required=True, type=Path)
            command.add_argument("--diagnostic-only", action="store_true")
        if name == "distill":
            command.add_argument("--teacher-run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "preflight":
        result = preflight(args.model_dir)
        if args.output_dir:
            args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
            write_json(args.output_dir / "preflight.json", result)
    else:
        rows = load_dataset(args.input)
        if hasattr(args, "split"):
            rows = [r for r in rows if r["split"] == args.split]
            if not rows:
                raise ValueError("selected split is empty")
        if args.command == "validate":
            result = dataset_summary(rows)
        elif args.command == "prepare":
            result = prepare(rows, config, args.variant, args.output_dir)
        elif args.command in {"train", "infer"}:
            result = execute(rows, config, args.variant, args.model_dir, args.output_dir,
                             training=args.command == "train", adapter_dir=getattr(args, "adapter_dir", None),
                             gpu_budget_mib=args.gpu_budget_mib)
        elif args.command == "evaluate":
            result = evaluate(rows, read_jsonl(args.predictions), diagnostic_only=args.diagnostic_only)
            args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
            write_json(args.output_dir / "evaluation.json", result)
        else:
            training_rows = [row for row in rows if row["split"] == "train"]
            if not training_rows:
                raise ValueError("no training inputs for distillation")
            result = distillation_candidates(training_rows, args.teacher_run_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
