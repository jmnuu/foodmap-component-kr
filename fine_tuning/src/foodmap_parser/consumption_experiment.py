"""Reproducible, CPU-only paired diagnostic for consumption-status extraction.

The authored challenge is NOT an independently annotated real-world test set.
Each invocation runs one strategy in a fresh process; no downloads or training.
"""

import argparse
import hashlib
import inspect
import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .evaluate_meal_events import food_event_mentions, precision_recall_f1
from .meal_events import parse_meal_events

ROOT = Path(__file__).resolve().parents[2]
CHALLENGE = ROOT / "samples" / "consumption_scope_challenge.jsonl"
LEGACY = ROOT / "data" / "v0.2" / "meal_event_examples.jsonl"
STATUSES = {"consumed", "not_consumed", "planned", "unknown"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(include_audit=False):
    rows = []
    for line in LEGACY.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.append({"id": row["id"], "split": "legacy_development",
                     "category": "legacy", "input": row["input"],
                     "expected": list(food_event_mentions(row["events"]).elements())})
    for line in CHALLENGE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] not in {"development", "audit"}:
            raise ValueError("unsupported challenge split")
        if include_audit or row["split"] != "audit":
            rows.append(row)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate case IDs")
    for row in rows:
        if not row["input"].strip():
            raise ValueError("empty input")
        for mention in row["expected"]:
            if len(mention) != 3 or mention[2] not in STATUSES:
                raise ValueError("invalid expected mention")
    return rows


def summarize(rows):
    tp = fp = fn = false_consumed = 0
    macro = []
    exact = []
    consumed_gold = consumed_predicted = consumed_tp = 0
    for row in rows:
        gold = Counter(map(tuple, row["expected"]))
        actual = Counter(map(tuple, row["predicted"]))
        tp += sum((gold & actual).values())
        fp += sum((actual - gold).values())
        fn += sum((gold - actual).values())
        macro.append(precision_recall_f1(actual, gold)[2])
        exact.append(actual == gold)
        gold_consumed = Counter({m: n for m, n in gold.items() if m[2] == "consumed"})
        actual_consumed = Counter({m: n for m, n in actual.items() if m[2] == "consumed"})
        false_consumed += sum((actual_consumed - gold_consumed).values())
        consumed_gold += sum(gold_consumed.values())
        consumed_predicted += sum(actual_consumed.values())
        consumed_tp += sum((gold_consumed & actual_consumed).values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"examples": len(rows), "exact_match_count": sum(exact),
            "macro_example_f1": statistics.mean(macro) if macro else None,
            "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            "false_consumed_mentions": false_consumed,
            "consumed_precision": consumed_tp / consumed_predicted if consumed_predicted else None,
            "consumed_recall": consumed_tp / consumed_gold if consumed_gold else None,
            "error_ids": [r["id"] for r, equal in zip(rows, exact) if not equal]}


def measure_peak_rss():
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes; Linux returns KiB. Other platforms stay unreported.
    if sys.platform == "darwin":
        return value / (1024 ** 2)
    if sys.platform.startswith("linux"):
        return value / 1024
    return None


def run(strategy, output_dir, include_audit=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    cases = load_cases(include_audit)
    signature = inspect.signature(parse_meal_events)
    if strategy != "legacy" and "status_strategy" not in signature.parameters:
        raise ValueError("this source revision supports only legacy")
    kwargs = {"status_strategy": strategy} if "status_strategy" in signature.parameters else {}
    sources = {str(p.relative_to(ROOT)): sha256(p)
               for p in sorted((ROOT / "src" / "foodmap_parser").glob("*.py"))}
    references = {str(p.relative_to(ROOT)): sha256(p)
                  for p in sorted((ROOT / "data").rglob("*"))
                  if p.is_file() and p.suffix in {".csv", ".json", ".jsonl"}}
    manifest = {"strategy": strategy, "created_at": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "platform": platform.platform(),
                "challenge_sha256": sha256(CHALLENGE), "legacy_sha256": sha256(LEGACY),
                "sources": sources, "reference_files": references,
                "include_audit": include_audit, "training_performed": False,
                "label_origin": "assistant_authored_synthetic_needs_human_review",
                "audit_note": "frozen diagnostic audit, NOT independent real-world validation",
                "metric_scope": "food_id + coarse meal_time + consumption_status; not components or exact timestamps",
                "memory_scope": "whole-process peak RSS including imports, dataset and parser; NOT GPU or training memory"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions = []
    started = time.perf_counter()
    for row in cases:
        result = parse_meal_events(row["input"], **kwargs)
        predictions.append({**row, "predicted": list(food_event_mentions(result["events"]).elements()),
                            "evidence": [{"food_id": food["food_id"],
                                          "status_evidence": food.get("consumption_evidence")}
                                         for event in result["events"] for food in event["foods"]]})
    elapsed = time.perf_counter() - started
    by_split = defaultdict(list)
    by_category = defaultdict(list)
    for row in predictions:
        by_split[row["split"]].append(row)
        by_category[row["split"] + ":" + row["category"]].append(row)
    metrics = {"strategy": strategy, "by_split": {k: summarize(v) for k, v in by_split.items()},
               "by_category": {k: summarize(v) for k, v in by_category.items()},
               "elapsed_seconds": elapsed, "peak_rss_mib": measure_peak_rss()}
    (output_dir / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "completed.json").write_text(json.dumps({"completed": True, "examples": len(cases)}) + "\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("legacy", "scoped", "unscoped"), default="legacy")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-audit", action="store_true")
    args = parser.parse_args()
    metrics = run(args.strategy, args.output_dir, args.include_audit)
    print(json.dumps({k: v for k, v in metrics.items() if k != "by_category"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
