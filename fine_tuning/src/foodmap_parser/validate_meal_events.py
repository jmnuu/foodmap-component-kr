import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FOODS_PATH = ROOT / "data" / "v0.1" / "food_items.csv"
EVENTS_PATH = ROOT / "data" / "v0.2" / "meal_event_examples.jsonl"

MEAL_TIMES = {
    "breakfast",
    "lunch",
    "dinner",
    "snack",
    "night",
    "late_night",
    "unknown",
}
CONSUMPTION_STATUSES = {"consumed", "not_consumed", "planned", "unknown"}


def load_food_ids():
    with open(FOODS_PATH, "r", encoding="utf-8") as file:
        return {row["food_id"] for row in csv.DictReader(file)}


def load_examples():
    examples = []

    with open(EVENTS_PATH, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if line:
                examples.append((line_number, json.loads(line)))

    return examples


def validate_example(line_number, example, food_ids):
    errors = []
    example_id = example.get("id", f"line {line_number}")

    if not isinstance(example.get("input"), str) or not example["input"].strip():
        errors.append(f"{example_id}: input must be a non-empty string")

    events = example.get("events")
    if not isinstance(events, list):
        return errors + [f"{example_id}: events must be a list"]

    for expected_index, event in enumerate(events):
        prefix = f"{example_id} event {expected_index}"

        if event.get("event_index") != expected_index:
            errors.append(f"{prefix}: event_index must match its list position")
        if event.get("meal_time") not in MEAL_TIMES:
            errors.append(f"{prefix}: invalid meal_time")
        if event.get("consumption_status") not in CONSUMPTION_STATUSES:
            errors.append(f"{prefix}: invalid consumption_status")

        foods = event.get("foods")
        if not isinstance(foods, list):
            errors.append(f"{prefix}: foods must be a list")
            continue
        if event.get("consumption_status") in {"consumed", "planned"} and not foods:
            errors.append(f"{prefix}: consumed or planned events require a food")

        for food_index, food in enumerate(foods):
            food_prefix = f"{prefix} food {food_index}"
            if not isinstance(food.get("raw_text"), str) or not food["raw_text"].strip():
                errors.append(f"{food_prefix}: raw_text must be a non-empty string")
            if food.get("food_id") not in food_ids:
                errors.append(f"{food_prefix}: unknown food_id {food.get('food_id')}")
            if food.get("quantity") is not None and not isinstance(food["quantity"], (int, float)):
                errors.append(f"{food_prefix}: quantity must be numeric or null")
            if food.get("unit") is not None and not isinstance(food["unit"], str):
                errors.append(f"{food_prefix}: unit must be a string or null")
            if not isinstance(food.get("modifiers"), list):
                errors.append(f"{food_prefix}: modifiers must be a list")

    return errors


def main():
    food_ids = load_food_ids()
    examples = load_examples()
    errors = []

    for line_number, example in examples:
        errors.extend(validate_example(line_number, example, food_ids))

    print("FoodMap MealEvent Validation")
    print("============================")
    print(f"examples: {len(examples)}")
    print(f"known food IDs: {len(food_ids)}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nNo validation errors found.")


if __name__ == "__main__":
    main()
