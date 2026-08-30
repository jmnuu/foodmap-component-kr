"""Build a reviewable collection queue for high-uncertainty food families."""

import argparse
import csv
from pathlib import Path

try:
    from .food_uncertainty import build_food_uncertainty_inventory
    from .general_component_priors import load_general_component_priors
except ImportError:  # Allows direct module execution during early local use.
    from food_uncertainty import build_food_uncertainty_inventory
    from general_component_priors import load_general_component_priors


QUEUE_FIELDS = [
    "priority_rank",
    "food_id",
    "food_name_ko",
    "uncertainty_score",
    "target_observations",
    "minimum_independent_sources",
    "priority_component_ids",
    "collection_strata",
    "review_status",
]


def _sampling_target(priority_rank):
    """Allocate the 70% research effort to uncertainty, not new food names."""
    if priority_rank <= 5:
        return 60, 5
    if priority_rank <= 15:
        return 45, 4
    return 30, 3


def build_recipe_collection_queue():
    """Return high-uncertainty foods ordered for independent recipe collection."""
    priors = load_general_component_priors()
    high_uncertainty = [
        item for item in build_food_uncertainty_inventory()
        if item["uncertainty_level"] == "high"
    ]
    queue = []

    for rank, item in enumerate(high_uncertainty, start=1):
        target_observations, minimum_sources = _sampling_target(rank)
        variable_component_ids = [
            component["component_id"]
            for component in priors[item["food_id"]]["components"]
            if component["role"] in {"optional", "variant"}
        ]
        queue.append({
            "priority_rank": rank,
            "food_id": item["food_id"],
            "food_name_ko": item["food_name_ko"],
            "uncertainty_score": item["uncertainty_score"],
            "target_observations": target_observations,
            "minimum_independent_sources": minimum_sources,
            "priority_component_ids": "|".join(variable_component_ids),
            "collection_strata": "household_recipe|restaurant_or_delivery|commercial_or_meal_kit",
            "review_status": "pending",
        })
    return queue


def main():
    argument_parser = argparse.ArgumentParser(
        description="Create the high-uncertainty FoodMap recipe-collection queue."
    )
    argument_parser.add_argument("--output", type=Path, required=True)
    args = argument_parser.parse_args()

    queue = build_recipe_collection_queue()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(queue)

    print(f"high_uncertainty_foods={len(queue)}")
    print(f"target_observations={sum(row['target_observations'] for row in queue)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
