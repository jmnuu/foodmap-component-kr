import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "v0.1"


def load_csv(filename):
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, skipinitialspace=True))


def build_food_lookup():
    foods = load_csv("food_items.csv")
    aliases = load_csv("food_aliases.csv")

    lookup = {}

    for row in foods:
        name = row["normalized_food_ko"]
        lookup[name] = {
            "food_id": row["food_id"],
            "normalized_food_ko": row["normalized_food_ko"],
        }

    for row in aliases:
        lookup[row["alias"]] = {
            "food_id": row["food_id"],
            "normalized_food_ko": row["normalized_food_ko"],
        }

    return lookup


def build_component_lookup():
    components = load_csv("component_vocab.csv")
    annotations = load_csv("food_component_annotations.csv")

    component_name_by_id = {
        row["component_id"]: row["component_name_en"]
        for row in components
    }

    component_lookup = {}

    for row in annotations:
        if row["status"] not in {"confirmed", "possible"}:
            continue

        food_id = row["food_id"]
        component_id = row["component_id"]
        component_name = component_name_by_id.get(component_id)

        if component_name is None:
            continue

        component_lookup.setdefault(food_id, []).append({
            "component": component_name,
            "status": row["status"],
            "confidence": float(row["confidence"]),
        })

    return component_lookup


def parse_food_log(text):
    food_lookup = build_food_lookup()
    component_lookup = build_component_lookup()

    found_foods = []

    # 긴 표현을 먼저 찾기 위해 길이 기준 정렬
    for alias, info in sorted(food_lookup.items(), key=lambda x: len(x[0]), reverse=True):
        if alias in text:
            food_id = info["food_id"]

            found_foods.append({
                "raw_text": alias,
                "food_id": food_id,
                "normalized_food": info["normalized_food_ko"],
                "components": component_lookup.get(food_id, []),
            })

    return {
        "input": text,
        "foods": found_foods,
    }


if __name__ == "__main__":
    text = "어제 밤에 라면이랑 김밥 먹고 아아 마셨어."
    result = parse_food_log(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))