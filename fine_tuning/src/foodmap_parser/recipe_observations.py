"""Validate recipe observations and update FoodMap component priors.

This module accepts only externally observed recipe or ingredient-list records.
It keeps those records separate from the curated FoodMap priors and never adds
an unreviewed component to a food family automatically.
"""

import argparse
import json
import hashlib
from collections import defaultdict
from datetime import date
from pathlib import Path

try:
    from .general_component_priors import load_general_component_priors
    from .reference_data import load_reference_csv
    from .label_validation import validate_review
except ImportError:  # Allows direct module execution during early local use.
    from general_component_priors import load_general_component_priors
    from reference_data import load_reference_csv
    from label_validation import validate_review


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
POSTERIOR_VERSION = "v0.4-partial-observations"


def validate_recipe_observation(record):
    """Return validation errors for one external recipe-observation record."""
    errors = []
    missing = sorted(field for field in REQUIRED_FIELDS if field not in record)
    if missing:
        return [f"missing fields: {', '.join(missing)}"]

    present = record["component_ids"]
    absent = record.get("absent_component_ids", [])
    for name, values in (("component_ids", present), ("absent_component_ids", absent)):
        if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
            errors.append(f"{name} must be a list of IDs")
        elif len(values) != len(set(values)):
            errors.append(f"{name} must not contain duplicates")
    if errors:
        return errors
    if not present and not absent:
        errors.append("at least one observed present or absent component is required")
    if set(present) & set(absent):
        errors.append("a component cannot be both present and absent")
    for field in ("recipe_observation_id", "food_id", "source_id", "independence_group_id"):
        if not isinstance(record[field], str) or not record[field].strip():
            errors.append(f"{field} must be non-empty")

    if not record.get("source_url") and not record.get("source_locator"):
        errors.append("one of source_url or source_locator is required")
    if record["observation_type"] not in OBSERVATION_TYPES:
        errors.append(f"observation_type must be one of {sorted(OBSERVATION_TYPES)}")
    if not record["independence_group_id"]:
        errors.append("independence_group_id is required")
    if record["review_status"] not in REVIEW_STATUSES:
        errors.append(f"review_status must be one of {sorted(REVIEW_STATUSES)}")
    if record["review_status"] == "approved":
        try:
            validate_review(record.get("review_metadata", {}), allow_date=True)
        except ValueError as error:
            errors.append(str(error))

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
    seen_ids = set()
    seen_locators = set()

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
        oid = observation["recipe_observation_id"]
        locator = observation.get("source_url") or observation.get("source_locator")
        if oid in seen_ids or locator in seen_locators:
            raise ValueError("duplicate recipe observation ID or source locator")
        seen_ids.add(oid)
        seen_locators.add(locator)
        unknown_components = (set(observation["component_ids"]) | set(observation.get("absent_component_ids", []))) - known_component_ids
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
            groups = defaultdict(lambda: [0.0, 0.0])
            supporting = []
            for observation in food_observations:
                state = (0 if component_id in observation["component_ids"] else
                         1 if component_id in observation.get("absent_component_ids", []) else None)
                if state is None:
                    continue
                groups[observation["independence_group_id"]][state] += float(observation["quality_weight"])
                supporting.append(observation["recipe_observation_id"])
            # Each author/source group contributes at most one observation per
            # component. Conflicting observations retain their weighted ratio.
            present_weight = sum(p / max(1.0, p + a) for p, a in groups.values())
            absent_weight = sum(a / max(1.0, p + a) for p, a in groups.values())
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
                "observation_ids": supporting,
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
        "reference_fingerprint": hashlib.sha256(json.dumps(load_general_component_priors(), sort_keys=True).encode()).hexdigest(),
        "observations": observations,
        "food_posteriors": posterior_by_food,
    }


def load_posterior_snapshot(path):
    """Revalidate a snapshot and its reference version before inference uses it."""
    with open(path, encoding="utf-8") as file:
        snapshot = json.load(file)
    if snapshot.get("posterior_version") != POSTERIOR_VERSION:
        raise ValueError("unsupported posterior snapshot version")
    rebuilt = build_component_posteriors(snapshot.get("observations", []))
    if rebuilt != snapshot:
        raise ValueError("posterior snapshot is stale or inconsistent; rebuild it")
    priors = json.loads(json.dumps(load_general_component_priors()))
    for food_id, food in priors.items():
        by_id = {c["component_id"]: c for c in snapshot["food_posteriors"][food_id]["components"]}
        for component in food["components"]:
            posterior = by_id[component["component_id"]]
            if not posterior["observation_ids"]:
                continue
            component.update(posterior)
            relevant = [o for o in snapshot["observations"] if o["recipe_observation_id"] in posterior["observation_ids"]]
            component["source_ids"] = sorted(set(component["source_ids"]) | {o["source_id"] for o in relevant})
    return priors


def main():
    argument_parser = argparse.ArgumentParser(
        description="Validate recipe observations and create FoodMap component posteriors."
    )
    argument_parser.add_argument("--input", type=Path, required=True)
    argument_parser.add_argument("--output", type=Path, required=True)
    argument_parser.add_argument("--approved-only", action="store_true", help="Explicitly select approved rows from a mixed reviewed collection")
    args = argument_parser.parse_args()

    observations = load_recipe_observations(args.input)
    skipped = 0
    if args.approved_only:
        approved = [o for o in observations if o["review_status"] == "approved"]
        skipped = len(observations) - len(approved)
        observations = approved
        if not observations:
            raise ValueError("no approved observations; review source data first")
    result = build_component_posteriors(observations)
    if args.output.exists():
        raise ValueError("output already exists; choose a new posterior snapshot path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"validated_observations={len(observations)}")
    print(f"skipped_unapproved={skipped}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
