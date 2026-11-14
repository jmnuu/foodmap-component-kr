"""Score saved component predictions against separately reviewed partial gold.

This evaluates components for annotated food mentions, not ASR or food linking.
Unreviewed labels are masked, never filled from the FoodMap dictionary.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .label_validation import validate_labels, validate_review
except ImportError:
    from label_validation import validate_labels, validate_review


def _metrics(counts):
    tp, fp, fn = counts
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def score_partial_labels(gold, predictions):
    gold_ids = [r["id"] for r in gold]
    pred_ids = [r["id"] for r in predictions]
    if len(set(gold_ids)) != len(gold_ids) or len(set(pred_ids)) != len(pred_ids):
        raise ValueError("duplicate evaluation record ID")
    if set(gold_ids) != set(pred_ids):
        raise ValueError("prediction IDs must exactly match gold IDs")
    pred_by_id = {r["id"]: r for r in predictions}
    by_label = defaultdict(lambda: [0, 0, 0])
    known, unknown = 0, 0
    for record in gold:
        if record.get("split") != "test" or record.get("review_status") != "approved":
            raise ValueError("gold requires approved test records")
        validate_review(record)
        validate_labels(record["food_id"], record["component_labels"])
        prediction = pred_by_id[record["id"]]
        if prediction["food_id"] != record["food_id"]:
            raise ValueError("food linking must be evaluated separately")
        predicted = prediction["present_component_ids"]
        if predicted:
            validate_labels(record["food_id"], [{"component_id": cid, "role": "variant", "state": "unknown"} for cid in predicted])
        predicted = set(predicted)
        for label in record["component_labels"]:
            if label["state"] == "unknown":
                unknown += 1
                continue
            known += 1
            cid = label["component_id"]
            actual, guess = label["state"] == "present", cid in predicted
            counts = by_label[cid]
            counts[0] += int(actual and guess)
            counts[1] += int(not actual and guess)
            counts[2] += int(actual and not guess)
    per_label = {cid: _metrics(counts) for cid, counts in sorted(by_label.items())}
    f1s = [r["f1"] for r in per_label.values() if r["f1"] is not None]
    return {
        "records": len(gold), "known_labels": known, "unknown_labels_masked": unknown,
        "micro": _metrics([sum(v[i] for v in by_label.values()) for i in range(3)]),
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "per_component": per_label,
        "scope": "reviewed component labels only; unannotated predictions are not scored",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    def read(path):
        with open(path, encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    print(json.dumps(score_partial_labels(read(args.gold), read(args.predictions)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
