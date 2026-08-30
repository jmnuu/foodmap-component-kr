"""Validate recipe observations and update FoodMap component priors.

This module accepts only externally observed recipe or ingredient-list records.
It keeps those records separate from the curated FoodMap priors and never adds
an unreviewed component to a food family automatically.
"""

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

try:
    from .general_component_priors import load_general_component_priors
    from .reference_data import load_reference_csv
except ImportError:  # Allows direct module execution during early local use.
    from general_component_priors import load_general_component_priors
    from reference_data import load_reference_csv


OBSERVATION_TYPES = {"recipe", "product_ingredient_list", "meal_kit"}
REVIEW_STATUSES = {"approved", "pending_review", "rejected"}
REQUIRED_FIELDS = {
    "recipe_observation_id",
    "food_id",
    "recipe_variant",
    "component_ids",
    "source_id",
    "retrieved_at",
    "quality_weight",
    "observation_type",
    "independence_group_id",
    "review_status",
}
POSTERIOR_VERSION = "v0.3-beta-recipe-observation-update"


def validate_recipe_observation(record):
    """Return validation errors for one external recipe-observation record."""
    errors = []
    missing = sorted(field for field in REQUIRED_FIELDS if field not in record)
    if missing:
        return [f"missing fields: {', '.join(missing)}"]

    if not isinstance(record["component_ids"], list) or not record["component_ids"]:
        errors.append("component_ids must be a non-empty list")
    elif len(record["component_ids"]) != len(set(record["component_ids"])):
        errors.append("component_ids must not contain duplicates")

    if not record.get("source_url") and not record.get("source_locator"):
        errors.append("one of source_url or source_locator is required")
    if record["observation_type"] not in OBSERVATION_TYPES:
        errors.append(f"observation_type must be one of {sorted(OBSERVATION_TYPES)}")
    if not record["independence_group_id"]:
        errors.append("independence_group_id is required")
    if record["review_status"] not in REVIEW_STATUSES:
        errors.append(f"review_status must be one of {sorted(REVIEW_STATUSES)}")

    try:
        weight = float(record["quality_weight"])
        if not 0 < weight <= 1:
            errors.append("quality_weight must be greater than 0 and at most 1")
    except (TypeError, ValueError):
        errors.append("quality_weight must be numeric")

    try:
        date.fromisoformat(record["retrieved_at"])
    except (TypeError, ValueError):
        errors.append("retrieved_at must be an ISO date (YYYY-MM-DD)")

    return errors


def load_recipe_observations(path):
    """Load and validate a JSONL observation file without silently dropping rows."""
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            errors = validate_recipe_observation(record)
            if errors:
                raise ValueError(f"{path}:{line_number}: {'; '.join(errors)}")
            records.append(record)
    return records


def build_component_posteriors(observations):
    """Update existing candidate components with weighted independent records.

    An observed component outside a food family's current reviewed candidate
    set is reported as `unmodeled_component_ids`. It is intentionally excluded
    from the posterior until a human adds a reviewed food-component annotation.
    """
    known_food_ids = {row["food_id"] for row in load_reference_csv("food_items.csv")}
    known_component_ids = {
        row["component_id"] for row in load_reference_csv("component_vocab.csv")
    }
    observations_by_food = defaultdict(list)

    for observation in observations:
        errors = validate_recipe_observation(observation)
        if errors:
            raise ValueError(
                f"{observation.get('recipe_observation_id', '<unknown>')}: "
                f"{'; '.join(errors)}"
            )
        if observation["review_status"] != "approved":
            raise ValueError(
                f"{observation['recipe_observation_id']}: "
                "only approved observations may update a posterior"
            )
        if observation["food_id"] not in known_food_ids:
            raise ValueError(f"unknown food_id: {observation['food_id']}")
        unknown_components = set(observation["component_ids"]) - known_component_ids
        if unknown_components:
            raise ValueError(
                "unknown component_ids: " + ", ".join(sorted(unknown_components))
            )
        observations_by_food[observation["food_id"]].append(observation)

    posterior_by_food = {}
    for food_id, prior in load_general_component_priors().items():
        food_observations = observations_by_food[food_id]
        modeled_component_ids = {
            component["component_id"] for component in prior["components"]
        }
        unmodeled_component_ids = sorted({
            component_id
            for observation in food_observations
            for component_id in observation["component_ids"]
            if component_id not in modeled_component_ids
        })
        components = []

        for component in prior["components"]:
            component_id = component["component_id"]
            present_weight = sum(
                float(observation["quality_weight"])
                for observation in food_observations
                if component_id in observation["component_ids"]
            )
            absent_weight = sum(
                float(observation["quality_weight"])
                for observation in food_observations
                if component_id not in observation["component_ids"]
            )
            alpha = round(float(component["prior_alpha"]) + present_weight, 4)
            beta = round(float(component["prior_beta"]) + absent_weight, 4)
            components.append({
                "component_id": component_id,
                "role": component["role"],
                "prior_alpha": component["prior_alpha"],
                "prior_beta": component["prior_beta"],
                "posterior_alpha": alpha,
                "posterior_beta": beta,
                "posterior_probability": round(alpha / (alpha + beta), 4),
                "observed_present_weight": round(present_weight, 4),
                "observed_absent_weight": round(absent_weight, 4),
                "prior_source_ids": component.get("source_ids", []),
            })

        posterior_by_food[food_id] = {
            "observation_count": len(food_observations),
            "observation_source_ids": sorted({
                observation["source_id"] for observation in food_observations
            }),
            "unmodeled_component_ids": unmodeled_component_ids,
            "components": components,
        }

    return {
        "posterior_version": POSTERIOR_VERSION,
        "food_posteriors": posterior_by_food,
    }


def main():
    argument_parser = argparse.ArgumentParser(
        description="Validate recipe observations and create FoodMap component posteriors."
    )
    argument_parser.add_argument("--input", type=Path, required=True)
    argument_parser.add_argument("--output", type=Path, required=True)
    args = argument_parser.parse_args()

    observations = load_recipe_observations(args.input)
    result = build_component_posteriors(observations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"validated_observations={len(observations)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
