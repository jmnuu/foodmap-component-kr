import json
from pathlib import Path
from parser import parse_food_log, build_component_lookup


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "v0.1"
EXAMPLES_PATH = DATA_DIR / "sentence_examples.jsonl"


def load_examples():
    examples = []

    with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    return examples


def precision_recall_f1(pred_set, gold_set):
    pred_set = set(pred_set)
    gold_set = set(gold_set)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1, tp, fp, fn


def get_components_for_food_ids(food_ids):
    component_lookup = build_component_lookup()

    components = []

    for food_id in food_ids:
        for item in component_lookup.get(food_id, []):
            components.append(item["component"])

    return sorted(set(components))


def main():
    examples = load_examples()

    food_precisions = []
    food_recalls = []
    food_f1s = []

    component_precisions = []
    component_recalls = []
    component_f1s = []

    error_cases = []

    for example in examples:
        text = example["input"]

        gold_food_ids = [
            item["food_id"]
            for item in example["food_mentions"]
        ]

        result = parse_food_log(text)

        pred_food_ids = [
            item["food_id"]
            for item in result["foods"]
        ]

        fp, fr, ff1, _, _, _ = precision_recall_f1(
            pred_food_ids,
            gold_food_ids
        )

        food_precisions.append(fp)
        food_recalls.append(fr)
        food_f1s.append(ff1)

        gold_components = get_components_for_food_ids(gold_food_ids)
        pred_components = get_components_for_food_ids(pred_food_ids)

        cp, cr, cf1, _, _, _ = precision_recall_f1(
            pred_components,
            gold_components
        )

        component_precisions.append(cp)
        component_recalls.append(cr)
        component_f1s.append(cf1)

        if ff1 < 1.0:
            error_cases.append({
                "id": example.get("id"),
                "input": text,
                "gold_food_ids": gold_food_ids,
                "pred_food_ids": pred_food_ids,
            })

    n = len(examples)

    print("FoodMap Parser Evaluation")
    print("=========================")
    print(f"examples: {n}")

    print("\nFood ID Matching")
    print("----------------")
    print(f"precision: {sum(food_precisions) / n:.3f}")
    print(f"recall:    {sum(food_recalls) / n:.3f}")
    print(f"f1:        {sum(food_f1s) / n:.3f}")

    print("\nComponent Matching")
    print("------------------")
    print(f"precision: {sum(component_precisions) / n:.3f}")
    print(f"recall:    {sum(component_recalls) / n:.3f}")
    print(f"f1:        {sum(component_f1s) / n:.3f}")

    if error_cases:
        print("\nError Cases")
        print("-----------")
        for case in error_cases:
            print(f"[{case['id']}] {case['input']}")
            print(f"  gold: {case['gold_food_ids']}")
            print(f"  pred: {case['pred_food_ids']}")
    else:
        print("\nNo food extraction errors found.")


if __name__ == "__main__":
    main()