import unittest
from pathlib import Path

from foodmap_parser.recipe_collection_queue import build_recipe_collection_queue
from foodmap_parser.recipe_observations import (
    build_component_posteriors,
    load_recipe_observations,
    validate_recipe_observation,
)


class RecipeObservationTests(unittest.TestCase):
    def test_updates_existing_candidate_with_weighted_recipe_observations(self):
        observations = [
            {
                "recipe_observation_id": "TEST:001",
                "food_id": "F0018",
                "recipe_variant": "ham_cheese_sandwich",
                "component_ids": ["C033", "C060", "C067", "C059"],
                "source_id": "SRC_TEST_A",
                "source_url": "https://example.org/a",
                "retrieved_at": "2026-08-30",
                "quality_weight": 0.8,
                "observation_type": "recipe",
                "independence_group_id": "test:author-a",
                "review_status": "approved",
            },
            {
                "recipe_observation_id": "TEST:002",
                "food_id": "F0018",
                "recipe_variant": "vegetable_sandwich",
                "component_ids": ["C033", "C060", "C048"],
                "source_id": "SRC_TEST_B",
                "source_locator": "local:reviewed-recipe-b",
                "retrieved_at": "2026-08-30",
                "quality_weight": 1.0,
                "observation_type": "recipe",
                "independence_group_id": "test:author-b",
                "review_status": "approved",
            },
        ]

        result = build_component_posteriors(observations)
        ham = next(
            item for item in result["food_posteriors"]["F0018"]["components"]
            if item["component_id"] == "C067"
        )

        self.assertEqual(ham["observed_present_weight"], 0.8)
        self.assertEqual(ham["observed_absent_weight"], 1.0)
        self.assertEqual(ham["posterior_alpha"], 2.85)
        self.assertEqual(ham["posterior_beta"], 3.95)

    def test_requires_a_stable_external_locator(self):
        errors = validate_recipe_observation({
            "recipe_observation_id": "TEST:BAD",
            "food_id": "F0018",
            "recipe_variant": "generic_sandwich",
            "component_ids": ["C033"],
            "source_id": "SRC_TEST",
            "retrieved_at": "2026-08-30",
            "quality_weight": 1.0,
            "observation_type": "recipe",
            "independence_group_id": "test:author-bad",
            "review_status": "approved",
        })

        self.assertIn("one of source_url or source_locator is required", errors)

    def test_uses_only_high_uncertainty_foods_in_the_collection_queue(self):
        queue = build_recipe_collection_queue()

        self.assertEqual(len(queue), 25)
        self.assertEqual(queue[0]["food_id"], "F0016")
        self.assertEqual(queue[0]["target_observations"], 60)
        self.assertEqual(queue[-1]["target_observations"], 30)

    def test_pilot_observations_are_valid_but_cannot_update_until_reviewed(self):
        root = Path(__file__).resolve().parents[1]
        records = load_recipe_observations(
            root / "data" / "v0.3" / "recipe_observations_pilot.jsonl"
        )

        self.assertEqual(len(records), 10)
        self.assertTrue(all(record["review_status"] == "pending_review" for record in records))
        with self.assertRaisesRegex(ValueError, "only approved observations"):
            build_component_posteriors(records)
