"""Rank foods whose representative components need more population evidence."""

import argparse
import json
import math
from pathlib import Path

try:
    from .general_component_priors import load_general_component_priors
    from .reference_data import load_reference_csv
except ImportError:  # Allows `python food_uncertainty.py` during early local use.
    from general_component_priors import load_general_component_priors
    from reference_data import load_reference_csv


ROOT = Path(__file__).resolve().parents[2]
def _binary_entropy(probability):
    probability = min(max(probability, 1e-6), 1 - 1e-6)
    return -(
        probability * math.log2(probability)
        + (1 - probability) * math.log2(1 - probability)
    )


def _load_foods():
    return load_reference_csv("food_items.csv")


def _uncertainty_level(score):
    if score >= 0.50:
        return "high"
    if score >= 0.30:
        return "medium"
    return "low"


def build_food_uncertainty_inventory():
    """Return a transparent ranking of foods needing recipe-observation data.

    The score prioritizes recipe variability, uncertain component probabilities,
    and weak evidence strength. It does not measure nutritional risk or a health
    effect. A high score means that generic food text has too many plausible
    component configurations to treat as a fixed recipe.
    """
    priors = load_general_component_priors()
    inventory = []

    for food in _load_foods():
        components = priors.get(food["food_id"], {}).get("components", [])
        if not components:
            continue

        probabilities = [float(component["confidence"]) for component in components]
        mean_entropy = sum(_binary_entropy(probability) for probability in probabilities) / len(probabilities)
        variable_components = [
            component for component in components
            if component["role"] in {"optional", "variant"}
        ]
        variation_ratio = len(variable_components) / len(components)
        mean_strength = sum(
            float(component.get("prior_alpha", 1)) + float(component.get("prior_beta", 1))
            for component in components
        ) / len(components)
        evidence_deficit = max(0.0, min(1.0, (8.0 - mean_strength) / 8.0))
        composite_bonus = 1.0 if food["food_category"] in {"mixed_meal", "stew", "fast_food"} else 0.0

        score = round(
            0.45 * mean_entropy
            + 0.30 * variation_ratio
            + 0.15 * evidence_deficit
            + 0.10 * composite_bonus,
            4,
        )
        reasons = []
        if variation_ratio >= 0.4:
            reasons.append("many_optional_or_variant_components")
        if mean_entropy >= 0.7:
            reasons.append("component_probabilities_are_ambiguous")
        if evidence_deficit >= 0.35:
            reasons.append("limited_prior_evidence_strength")
        if composite_bonus:
            reasons.append("composite_food_category")

        inventory.append({
            "food_id": food["food_id"],
            "food_name_ko": food["normalized_food_ko"],
            "food_category": food["food_category"],
            "uncertainty_score": score,
            "uncertainty_level": _uncertainty_level(score),
            "component_candidate_count": len(components),
            "variable_component_count": len(variable_components),
            "mean_component_entropy": round(mean_entropy, 4),
            "mean_prior_strength": round(mean_strength, 4),
            "reasons": reasons,
            "next_action": "collect_independent_recipe_observations"
            if _uncertainty_level(score) == "high"
            else "retain_prior_and_monitor",
        })

    return sorted(
        inventory,
        key=lambda item: (-item["uncertainty_score"], item["food_id"]),
    )


def main():
    argument_parser = argparse.ArgumentParser(
        description="Rank FoodMap foods by representative-component uncertainty."
    )
    argument_parser.add_argument("--output", type=Path, required=True)
    args = argument_parser.parse_args()

    inventory = build_food_uncertainty_inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(inventory, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"foods_ranked={len(inventory)}")
    print(f"high_uncertainty={sum(item['uncertainty_level'] == 'high' for item in inventory)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
