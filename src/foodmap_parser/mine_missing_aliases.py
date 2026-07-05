import csv
import json
from pathlib import Path
from parser import parse_food_log

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "v0.1"
REPORT_DIR = ROOT / "reports"
EXAMPLES_PATH = DATA_DIR / "sentence_examples.jsonl"
OUTPUT_PATH = REPORT_DIR / "missing_aliases.csv"


def load_examples():
    examples = []
    with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main():
    REPORT_DIR.mkdir(exist_ok=True)

    examples = load_examples()
    missing_rows = []

    for ex in examples:
        text = ex["input"]
        gold_mentions = ex.get("food_mentions", [])

        result = parse_food_log(text)
        pred_food_ids = {item["food_id"] for item in result["foods"]}

        for gold in gold_mentions:
            gold_food_id = gold["food_id"]
            raw_text = gold["raw_text"]
            normalized_food = gold["normalized_food"]

            if gold_food_id not in pred_food_ids:
                missing_rows.append({
                    "sentence_id": ex.get("id", ""),
                    "input": text,
                    "missing_raw_text": raw_text,
                    "suggested_food_id": gold_food_id,
                    "suggested_normalized_food": normalized_food,
                    "action": "add_to_food_aliases",
                    "review_status": "pending"
                })

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sentence_id",
            "input",
            "missing_raw_text",
            "suggested_food_id",
            "suggested_normalized_food",
            "action",
            "review_status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_rows)

    print(f"Missing alias candidates saved to: {OUTPUT_PATH}")
    print(f"missing count: {len(missing_rows)}")


if __name__ == "__main__":
    main()