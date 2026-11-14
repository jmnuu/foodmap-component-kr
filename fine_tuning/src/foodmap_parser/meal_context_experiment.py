"""CPU-only v3 validate/prepare/evaluate CLI. Does not train or invoke any model."""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .meal_context_contract import (PROMPT, VERSION, digest, event_counters, input_digest,
                                    load_dataset, messages_for, normalize_temporal)
from .meal_model_experiment import file_digest, read_jsonl, write_json, write_jsonl


def summary(rows):
    return {"contract_version": VERSION, "rows": len(rows),
            "source_groups": len({r["source_group_id"] for r in rows}),
            "splits": dict(Counter(r["split"] for r in rows)),
            "review_statuses": dict(Counter(r["review_status"] for r in rows)),
            "label_origins": dict(Counter(r["label_origin"] for r in rows)),
            "dataset_sha256": digest(rows), "model_execution_performed": False}


def prepare(rows, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    sources = Path(__file__).parent
    write_json(out / "manifest.json", {**summary(rows), "created_at": datetime.now(timezone.utc).isoformat(),
               "prompt_sha256": digest(PROMPT),
               "code_sha256": {p.name: file_digest(p) for p in sorted(sources.glob("*.py"))},
               "privacy": "local files contain raw text; not anonymized or encrypted"})
    write_jsonl(out / "requests.jsonl", [{"id": r["id"], "input_sha256": input_digest(r), "messages": messages_for(r)} for r in rows])
    write_jsonl(out / "candidate_label_preview.jsonl", [
        {"id": r["id"], "review_status": r["review_status"], "target": r["target"],
         "temporal_preview": [normalize_temporal(e, r) for e in r["target"]["events"]],
         "source": "candidate_annotations_not_predictions"}
        for r in rows if r["target"] is not None])
    result = {"status": "prepared_not_tokenized_or_trained", "model_execution_performed": False,
              "training_approval_granted": False, "records": len(rows)}
    write_json(out / "completed.json", result)
    return result


def evaluate(rows, predictions, *, diagnostic_only=False):
    if not rows or any(r["split"] == "train" or r["target"] is None for r in rows):
        raise ValueError("evaluation needs labeled development/test data")
    if not diagnostic_only and any(r["review_status"] != "approved" for r in rows):
        raise ValueError("unreviewed labels require diagnostic-only mode")
    by_id = {p["id"]: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r["id"] for r in rows}:
        raise ValueError("prediction IDs must exactly cover the evaluation rows")
    names = tuple(event_counters({"events": []}, rows[0]))
    totals = {name: Counter(tp=0, fp=0, fn=0) for name in names}
    invalid, exact, errors = 0, 0, []
    slices = {"explicit": Counter(), "discourse_context": Counter(), "diary_default": Counter(), "unresolved": Counter()}
    for row in rows:
        p = by_id[row["id"]]
        if p.get("contract_version") != VERSION or p.get("input_sha256") != input_digest(row):
            raise ValueError("prediction contract/input mismatch")
        gold = event_counters(row["target"], row)
        output_valid = False
        if p["status"] == "ok":
            try:
                actual = event_counters(p["output"], row)
                output_valid = True
            except (ValueError, KeyError, TypeError, OverflowError):
                pass
        elif p["status"] != "invalid_output":
            raise ValueError("invalid prediction status")
        if not output_valid:
            invalid += 1
            actual = {name: Counter({("INVALID_OUTPUT", row["id"]): 1}) for name in names}
        matched = output_valid and actual["joint"] == gold["joint"]
        exact += matched
        if not matched:
            errors.append(row["id"])
        for basis in {e["subject"]["basis"] for e in row["target"]["events"]}:
            slices[basis]["rows"] += 1
            slices[basis]["exact_event_match"] += matched
        for name in names:
            totals[name]["tp"] += sum((gold[name] & actual[name]).values())
            totals[name]["fp"] += sum((actual[name] - gold[name]).values())
            totals[name]["fn"] += sum((gold[name] - actual[name]).values())
    metrics = {}
    for name, count in totals.items():
        tp, fp, fn = (count[key] for key in ("tp", "fp", "fn"))
        metrics[name] = {**count, "precision": tp / (tp + fp) if tp + fp else None,
                         "recall": tp / (tp + fn) if tp + fn else None,
                         "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    return {"contract_version": VERSION, "examples": len(rows), "exact_event_match": exact,
            "invalid_outputs": invalid, "error_ids": errors, "metrics": metrics,
            "subject_basis_slices": {k: dict(v) for k, v in slices.items()},
            "diagnostic_only": diagnostic_only, "gold_sha256": digest(rows), "predictions_sha256": digest(predictions),
            "prediction_provenance_verified": False,
            "limitations": ["evaluation does not invoke a model or establish prediction provenance",
                            "quote existence does not prove semantic truth; review declarations are not authenticated",
                            "normalized-date metric includes unresolved values; inspect date_reference and joint too",
                            "subject basis slices overlap on multi-event rows",
                            "no ingredient, wearable response, ASR intent, GPU memory or UTC-window accuracy evaluation"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "prepare", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--input", required=True)
        if name != "validate":
            command.add_argument("--output-dir", required=True)
        if name == "evaluate":
            command.add_argument("--split", choices=("development", "test"), required=True)
            command.add_argument("--predictions", required=True)
            command.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args(argv)
    rows = load_dataset(args.input)
    if args.command == "validate":
        result = summary(rows)
    elif args.command == "prepare":
        result = prepare(rows, args.output_dir)
    else:
        selected = [r for r in rows if r["split"] == args.split]
        result = evaluate(selected, read_jsonl(args.predictions), diagnostic_only=args.diagnostic_only)
        out = Path(args.output_dir)
        out.mkdir(parents=True, mode=0o700, exist_ok=False)
        write_json(out / "evaluation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
