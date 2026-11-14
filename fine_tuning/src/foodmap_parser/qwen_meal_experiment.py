"""Offline-first Qwen trial harness, separate from the production parser.

No automatic downloads, paid API calls, training, or fallback to baseline.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .qwen_meal_contract import (PROMPT, VERSION, case_digest, digest, messages_for,
                                 normalize_case, parse_response, validate_response)


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(parse_response(line))
    if not rows:
        raise ValueError("empty JSONL input")
    ids = [r.get("id") if isinstance(r, dict) else None for r in rows]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("each row needs a unique nonempty string id")
    return rows


def load_cases(path):
    return [normalize_case(row, Path(path).resolve().parent) for row in read_jsonl(path)]


def checkpoint_info(directory):
    root = Path(directory).expanduser().resolve()
    info = {"path": str(root), "complete_weight_files": False, "issues": []}
    try:
        config = parse_response((root / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen3_omni_moe":
            raise ValueError("expected a Qwen3-Omni checkpoint config")
        info["config_sha256"] = digest(config)
        index = root / "model.safetensors.index.json"
        if index.exists():
            weight_map = parse_response(index.read_text(encoding="utf-8"))["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("empty weight index")
            names = sorted(set(weight_map.values()))
        else:
            names = ["model.safetensors"]
        for name in names:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("invalid shard path in weight index")
            path = root / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty weight shard: {name}")
        info["weights"] = [{"name": name, "bytes": (root / name).stat().st_size} for name in names]
        info["complete_weight_files"] = True
        info["verification"] = "file_presence_and_size_only_not_weight_integrity"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        info["issues"].append(str(exc))
    return info


def preflight(model_dir=None):
    packages = {}
    for name in ("torch", "transformers", "accelerate", "qwen-omni-utils"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "packages": packages,
            "checkpoint": checkpoint_info(model_dir) if model_dir else None,
            "runtime_validated": False, "downloads_performed": False,
            "note": "This runner requires CUDA. Source checkout is not model weights."}


def generated_sequences(result):
    # Qwen examples return (text, audio); newer Transformers returns text directly.
    text_result = result[0] if isinstance(result, tuple) else result
    sequences = getattr(text_result, "sequences", None)
    if sequences is None:
        raise ValueError("unsupported generate result; expected return_dict_in_generate sequences")
    return sequences


def code_fingerprint():
    import foodmap_parser
    sources = {}
    for directory in reversed(list(foodmap_parser.__path__)):
        for path in sorted(Path(directory).glob("*.py")):
            sources[path.name] = digest(path.read_text(encoding="utf-8"))
    return digest(sources)


def load_qwen(model_dir):
    if not model_dir or not checkpoint_info(model_dir)["complete_weight_files"]:
        raise ValueError("complete local Qwen3-Omni weights required; no download attempted")
    report = preflight(model_dir)
    if any(value is None for value in report["packages"].values()):
        raise ValueError("missing optional Qwen packages; use a separate GPU environment")
    version = re.match(r"(\d+)\.(\d+)\.(\d+)", report["packages"]["transformers"])
    if not version or tuple(map(int, version.groups())) < (5, 2, 0):
        raise ValueError("this experimental runner requires transformers >= 5.2.0")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU required; CPU/MPS inference is not validated by this runner")
    root = str(Path(model_dir).expanduser().resolve())
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        root, dtype="auto", device_map="auto", local_files_only=True, trust_remote_code=False)
    model.disable_talker()
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(root, local_files_only=True,
                                                    trust_remote_code=False)

    def infer(case):
        messages = messages_for(case)
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs = processor(text=prompt, audio=audios, images=images, videos=videos,
                           return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(model.device).to(model.dtype)
        with torch.inference_mode():
            result = model.generate(**inputs, return_audio=False, thinker_do_sample=False,
                                    thinker_max_new_tokens=1536,
                                    thinker_return_dict_in_generate=True)
        tokens = generated_sequences(result)
        return processor.batch_decode(tokens[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]

    return infer


def predict(case, backend, lookup, infer=None):
    start = time.perf_counter()
    output = {"id": case["id"], "case_sha256": case_digest(case), "backend": backend,
              "contract_version": VERSION, "events": [], "status": "ok"}
    if backend == "baseline":
        if case["media"] or not case["input"].strip():
            raise ValueError("baseline accepts text only; use a separately labeled modality comparison")
        from .meal_events import parse_meal_events
        output["events"] = parse_meal_events(case["input"])["events"]
    elif backend == "qwen":
        if infer is None:
            raise ValueError("Qwen inference backend is required")
        raw = infer(case)  # Runtime errors stop the run; never substitute baseline.
        output["raw_output"] = raw
        try:
            output["events"] = validate_response(parse_response(raw), case, lookup)
        except (ValueError, TypeError, KeyError) as exc:
            output.update(status="invalid_output", error=str(exc))
    else:
        raise ValueError("unsupported backend")
    output["latency_seconds"] = time.perf_counter() - start
    return output


def evaluate(gold_path, predictions_path):
    from .evaluate_meal_events import precision_recall_f1
    gold = read_jsonl(gold_path)
    predictions = read_jsonl(predictions_path)
    by_id = {p["id"]: p for p in predictions}
    if set(by_id) != {g["id"] for g in gold}:
        raise ValueError("prediction IDs must exactly match gold; missing rows cannot be dropped")
    if len({p.get("backend") for p in predictions}) != 1:
        raise ValueError("mixed prediction backends")
    if predictions[0].get("backend") not in {"baseline", "qwen"}:
        raise ValueError("unsupported prediction backend")
    scores = []
    errors = []
    invalid = 0
    unlinked = 0
    totals = Counter()

    def mentions(events):
        return Counter((food["food_id"] if food.get("food_id") else
                        "UNLINKED:" + food["normalized_name"],
                        event["meal_time"], event["consumption_status"])
                       for event in events for food in event["foods"])

    for row in gold:
        if row.get("split", "development") not in {"development", "dev"}:
            raise ValueError("this command only evaluates development data, not a frozen test set")
        case = normalize_case(row, Path(gold_path).resolve().parent)
        prediction = by_id[row["id"]]
        if prediction.get("case_sha256") != case_digest(case):
            raise ValueError(f"input mismatch for {row['id']}")
        if prediction.get("contract_version") != VERSION:
            raise ValueError("prediction contract version mismatch")
        if prediction.get("status") not in {"ok", "invalid_output"}:
            raise ValueError("unsupported prediction status")
        expected = mentions(row["events"])
        failed = prediction["status"] != "ok"
        actual = Counter() if failed else mentions(prediction["events"])
        score = (0.0, 0.0, 0.0) if failed else precision_recall_f1(actual, expected)
        scores.append(score)
        invalid += failed
        unlinked += sum(n for item, n in actual.items() if item[0].startswith("UNLINKED:"))
        totals["tp"] += sum((actual & expected).values())
        totals["fp"] += sum((actual - expected).values())
        totals["fn"] += sum((expected - actual).values())
        if score[2] < 1:
            errors.append(row["id"])
    return {"dataset_role": "development_only", "backend": predictions[0]["backend"],
            "examples": len(gold), "invalid_outputs": invalid, "unlinked_mentions": unlinked,
            "coarse_food_time_status_macro": dict(zip(("precision", "recall", "f1"),
                [sum(score[i] for score in scores) / len(scores) for i in range(3)])),
            "mention_counts": dict(totals), "error_ids": errors,
            "limitations": ["Not independent generalization evidence", "Not exact timestamp accuracy",
                            "Not component inference accuracy", "Invalid outputs score zero"],
            "gold_sha256": digest(gold), "predictions_sha256": digest(predictions)}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--model-dir")
    for name in ("prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--input", required=True)
        command.add_argument("--output-dir", required=True)
        if name == "run":
            command.add_argument("--backend", required=True, choices=("baseline", "qwen"))
            command.add_argument("--model-dir")
    scoring = commands.add_parser("evaluate")
    scoring.add_argument("--gold", required=True)
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(preflight(args.model_dir), ensure_ascii=False, indent=2))
        return
    if args.command == "evaluate":
        report = evaluate(args.gold, args.predictions)
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    cases = load_cases(args.input)
    out = Path(args.output_dir)
    if out.exists():
        raise ValueError("output directory already exists; use a new run directory")
    from .parser import build_food_lookup
    lookup = build_food_lookup()
    backend = args.backend if args.command == "run" else "not_run"
    if backend == "baseline" and any(c["media"] for c in cases):
        raise ValueError("baseline rejects multimodal input rather than silently ignoring it")
    infer = load_qwen(args.model_dir) if backend == "qwen" else None
    out.mkdir(parents=True, mode=0o700)
    manifest = {"status": "started", "backend": backend, "contract_version": VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(), "cases": len(cases),
                "input_sha256": digest([case_digest(c) for c in cases]),
                "prompt_sha256": digest(PROMPT), "food_lookup_sha256": digest(lookup),
                "environment": preflight(getattr(args, "model_dir", None)),
                "code_sha256": code_fingerprint(),
                "decoding": {"thinker_do_sample": False, "thinker_max_new_tokens": 1536},
                "privacy": "local only; input/output and paths may contain personal data"}
    write_json(out / "manifest.json", manifest)
    filename = "requests.jsonl" if args.command == "prepare" else "predictions.jsonl"
    with (out / filename).open("x", encoding="utf-8") as handle:
        for case in cases:
            result = ({"id": case["id"], "case_sha256": case_digest(case),
                       "messages": messages_for(case)} if args.command == "prepare" else
                      predict(case, backend, lookup, infer))
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
    write_json(out / "completed.json", {"status": "prepared_not_inferred" if args.command == "prepare"
                                       else "completed", "records": len(cases)})
    print(f"{args.command}: {len(cases)} records -> {out}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
