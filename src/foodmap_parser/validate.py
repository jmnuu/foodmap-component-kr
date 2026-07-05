import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "v0.1"


def load_csv(filename):
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    components = load_csv("component_vocab.csv")
    foods = load_csv("food_items.csv")
    annotations = load_csv("food_component_annotations.csv")
    aliases = load_csv("food_aliases.csv")

    component_ids = {row["component_id"] for row in components}
    food_ids = {row["food_id"] for row in foods}

    errors = []

    for row in annotations:
        if row["food_id"] not in food_ids:
            errors.append(f"Unknown food_id in annotations: {row['food_id']}")
        if row["component_id"] not in component_ids:
            errors.append(f"Unknown component_id in annotations: {row['component_id']}")

    for row in aliases:
        if row["food_id"] not in food_ids:
            errors.append(f"Unknown food_id in aliases: {row['food_id']}")

    print("FoodMap Dataset Validation")
    print("==========================")
    print(f"components: {len(components)}")
    print(f"foods: {len(foods)}")
    print(f"annotations: {len(annotations)}")
    print(f"aliases: {len(aliases)}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
    else:
        print("\nNo validation errors found.")


if __name__ == "__main__":
    main()