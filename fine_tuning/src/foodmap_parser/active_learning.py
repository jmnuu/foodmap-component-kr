"""Human-in-the-loop selection and export for FoodMap text labels.

The queue selects examples where a reviewer adds the most value: unknown food
phrases and recognized composite foods whose component candidates are uncertain.
Raw input, model output, review decisions, and approved training labels remain
separate so an updated model can always be trained from an auditable snapshot.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

try:
    from .label_validation import validate_training_record, validate_review
except ImportError:
    from label_validation import validate_training_record, validate_review


ROOT = Path(__file__).resolve().parents[2]
HUMAN_REVIEW_SOURCE_PREFIX = "SRC_ACTIVE_LEARNING_REVIEW"
VALID_DECISIONS = {"approved", "rejected", "needs_more_context"}


def _binary_entropy(probability):
    """Return normalized binary entropy, where 0.5 is maximally uncertain."""
    probability = min(max(float(probability), 0.0001), 0.9999)
    entropy = -probability * math.log2(probability) - (1 - probability) * math.log2(1 - probability)
    return round(entropy, 4)


def _candidate_probability(component):
    # Uncalibrated text-model scores are not component occurrence probabilities.
    return float(component.get("posterior_probability", component.get("prior_probability", component["confidence"])))


def _component_review_task(food_row, event_row):
    """Create a task only when uncertain inferred components need a decision."""
    inferred = food_row.get("representative_components", [])
    reviewable = [
        component for component in inferred
        if component["role"] in {"typical", "optional", "variant"}
        and component["inference_method"] != "text_override"
    ]
    if not reviewable:
        return None

    entropy_scores = [_binary_entropy(_candidate_probability(component)) for component in reviewable]
    ambiguous = [score for score in entropy_scores if score >= 0.75]
    # A food with multiple unconfirmed alternatives remains useful even when a
    # hand-curated prior is sharp; recipe variation must be checked separately.
    variation_score = min(1.0, len(reviewable) / 4)
    uncertainty_score = round(0.7 * (sum(entropy_scores) / len(entropy_scores)) + 0.3 * variation_score, 4)

    if uncertainty_score < 0.45 and not ambiguous:
        return None

    reasons = []
    if ambiguous:
        reasons.append("component_probability_near_decision_boundary")
    if len(reviewable) >= 2:
        reasons.append("multiple_recipe_dependent_components")
    if any(component.get("model_probability") is None for component in reviewable):
        reasons.append("no_direct_human_language_evidence")

    return {
        "review_task_id": f"AL:{food_row['food_mention_id']}:components",
        "task_type": "component_confirmation",
        "input_record_id": event_row["source_record_id"],
        "event_id": food_row["event_id"],
        "food_mention_id": food_row["food_mention_id"],
        "food_id": food_row["food_id"],
        "raw_text": food_row["raw_text"],
        "source_text": event_row["source_text"],
        "mention_context": food_row.get("mention_context", food_row["raw_text"]),
        "split": event_row.get("split", "unassigned"),
        "meal_time": event_row["meal_time"],
        "consumption_status": event_row["consumption_status"],
        "candidate_components": [
            {
                "component_id": component["component_id"],
                "component_name_ko": component["component_name_ko"],
                "role": component["role"],
                "probability": _candidate_probability(component),
                "inference_method": component["inference_method"],
                "source_ids": component.get("source_ids", []),
            }
            for component in reviewable
        ],
        "selection_reasons": reasons,
        "uncertainty_score": uncertainty_score,
        "priority_score": uncertainty_score,
        "review_status": "pending_review",
        "source_ids": food_row.get("source_ids", []),
    }


def build_active_learning_queue(pipeline_result, max_tasks_per_food=2):
    """Select diverse, high-uncertainty review tasks from pipeline output.

    The per-food cap prevents a high-volume common food from consuming the whole
    review budget. Unknown phrases get maximum priority because they are absent
    from the current ontology; all other tasks preserve model candidates rather
    than automatically accepting them as labels.
    """
    events_by_id = {event["event_id"]: event for event in pipeline_result["meal_events"]}
    tasks = []

    for candidate in pipeline_result["unresolved_food_candidates"]:
        if candidate.get("split") in {"validation", "test"}:
            continue
        tasks.append({
            "review_task_id": f"AL:{candidate['candidate_id']}:food",
            "task_type": "unknown_food_normalization",
            "input_record_id": candidate["source_record_id"],
            "event_id": None,
            "food_mention_id": None,
            "food_id": None,
            "raw_text": candidate["candidate_text"],
            "source_text": candidate["source_text"],
            "meal_time": None,
            "consumption_status": None,
            "candidate_components": [],
            "selection_reasons": ["food_phrase_not_in_reference_ontology"],
            "uncertainty_score": 1.0,
            "priority_score": 1.0,
            "review_status": "pending_review",
            "source_ids": candidate["source_ids"],
        })

    for food_row in pipeline_result["meal_event_foods"]:
        event_row = events_by_id[food_row["event_id"]]
        if event_row.get("split") in {"validation", "test"}:
            continue
        if event_row["consumption_status"] != "consumed":
            continue
        task = _component_review_task(food_row, event_row)
        if task:
            tasks.append(task)

    tasks.sort(key=lambda task: (-task["priority_score"], task["review_task_id"]))
    per_food_count = Counter()
    selected = []
    for task in tasks:
        food_key = task["food_id"] or task["raw_text"]
        if per_food_count[food_key] >= max_tasks_per_food:
            continue
        selected.append(task)
        per_food_count[food_key] += 1

    return selected


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def export_approved_training_records(tasks, decisions):
    """Convert approved component reviews into trainable, provenance-rich rows.

    Decisions are supplied separately from the queue. This prevents a model
    prediction or an unreviewed queue row from silently becoming training data.
    """
    task_ids = [task["review_task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate review task ID")
    decisions_by_task = {}
    for decision in decisions:
        task_id = decision.get("review_task_id")
        if task_id not in task_ids or task_id in decisions_by_task:
            raise ValueError("unknown or duplicate decision task ID")
        if decision.get("decision") not in VALID_DECISIONS:
            raise ValueError("invalid review decision")
        validate_review(decision)
        decisions_by_task[task_id] = decision
    records = []

    for task in tasks:
        decision = decisions_by_task.get(task["review_task_id"])
        if not decision or decision.get("decision") != "approved":
            continue
        if task["task_type"] != "component_confirmation":
            raise ValueError("food normalization requires a separate ontology review")
        if task.get("split") in {"validation", "test"}:
            raise ValueError("evaluation records cannot enter training")
        record = {
            "schema_version": "v0.4-partial-labels",
            "id": f"ALTRAIN:{task['review_task_id']}",
            "text": task.get("mention_context", task["raw_text"]),
            "food_id": task["food_id"],
            "component_labels": decision.get("component_labels"),
            "label_origin": "active_learning_human_review",
            "data_origin": task.get("data_origin", "unknown"),
            "training_source_ids": [*task.get("source_ids", []), f"{HUMAN_REVIEW_SOURCE_PREFIX}:{task['review_task_id']}"],
            "review_task_id": task["review_task_id"],
            "input_record_id": task.get("input_record_id"),
            "review_status": "approved",
            "split": "train",
            "reviewer": decision["reviewer"],
            "reviewed_at": decision["reviewed_at"],
            "rationale": decision.get("rationale", ""),
        }
        validate_training_record(record)
        records.append(record)

    return records


def main():
    argument_parser = argparse.ArgumentParser(description="Export approved FoodMap active-learning labels.")
    argument_parser.add_argument("--queue", type=Path, required=True)
    argument_parser.add_argument("--decisions", type=Path, required=True)
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "active_learning_approved.jsonl",
    )
    args = argument_parser.parse_args()

    records = export_approved_training_records(load_jsonl(args.queue), load_jsonl(args.decisions))
    if args.output.exists():
        raise ValueError("output already exists; choose a new training snapshot path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, records)
    print(f"approved_training_records: {len(records)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
