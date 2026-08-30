try:
    from .reference_data import load_reference_csv
except ImportError:  # Allows direct module execution during early local use.
    from reference_data import load_reference_csv


def load_csv(filename):
    return load_reference_csv(filename)


def main():
    components = load_csv("component_vocab.csv")
    foods = load_csv("food_items.csv")
    annotations = load_csv("food_component_annotations.csv")
    aliases = load_csv("food_aliases.csv")
    component_aliases = load_csv("component_aliases.csv")
    mfds_links = load_csv("mfds_food_links.csv")

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

    for row in component_aliases:
        if row["component_id"] not in component_ids:
            errors.append(f"Unknown component_id in component_aliases: {row['component_id']}")

    for row in mfds_links:
        if row["food_id"] not in food_ids:
            errors.append(f"Unknown food_id in mfds_food_links: {row['food_id']}")

    print("FoodMap Dataset Validation")
    print("==========================")
    print(f"components: {len(components)}")
    print(f"foods: {len(foods)}")
    print(f"annotations: {len(annotations)}")
    print(f"aliases: {len(aliases)}")
    print(f"component aliases: {len(component_aliases)}")
    print(f"mfds links: {len(mfds_links)}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
    else:
        print("\nNo validation errors found.")


if __name__ == "__main__":
    main()
