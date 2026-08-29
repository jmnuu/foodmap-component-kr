"""Evaluate food, time, and consumption-status extraction on MealEvent JSONL."""

import json
from collections import Counter
from pathlib import Path

try:
    from .meal_events import parse_meal_events
except ImportError:
    from meal_events import parse_meal_events


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_PATH = ROOT / "data" / "v0.2" / "meal_event_examples.jsonl"


def load_examples():
    with open(EXAMPLES_PATH, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def food_mentions(events):
    return Counter(
        food["food_id"]
        for event in events
        for food in event["foods"]
    )


def food_event_mentions(events):
    return Counter(
        (food["food_id"], event["meal_time"], event["consumption_status"])
        for event in events
        for food in event["foods"]
    )


def precision_recall_f1(predicted, gold):
    true_positive = sum((predicted & gold).values())
    false_positive = sum((predicted - gold).values())
    false_negative = sum((gold - predicted).values())

    if not predicted and not gold:
        return 1.0, 1.0, 1.0

    precision = true_positive / (true_positive + false_positive) if predicted else 0.0
    recall = true_positive / (true_positive + false_negative) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    examples = load_examples()
    food_scores = []
    event_scores = []
    errors = []

    for example in examples:
        predicted_events = parse_meal_events(example["input"])["events"]
        gold_events = example["events"]

        food_score = precision_recall_f1(
            food_mentions(predicted_events),
            food_mentions(gold_events),
        )
        event_score = precision_recall_f1(
            food_event_mentions(predicted_events),
            food_event_mentions(gold_events),
        )
        food_scores.append(food_score)
        event_scores.append(event_score)

        if event_score[2] < 1.0:
            errors.append(example["id"])

    def mean(scores, index):
        return sum(score[index] for score in scores) / len(scores)

    print("FoodMap MealEvent Evaluation")
    print("============================")
    print(f"examples: {len(examples)}")
    print("\nFood ID matching")
    print(f"precision: {mean(food_scores, 0):.3f}")
    print(f"recall:    {mean(food_scores, 1):.3f}")
    print(f"f1:        {mean(food_scores, 2):.3f}")
    print("\nFood-time-status matching")
    print(f"precision: {mean(event_scores, 0):.3f}")
    print(f"recall:    {mean(event_scores, 1):.3f}")
    print(f"f1:        {mean(event_scores, 2):.3f}")

    if errors:
        print("\nExamples needing review:")
        print(", ".join(errors))


if __name__ == "__main__":
    main()
