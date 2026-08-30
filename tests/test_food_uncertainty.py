import unittest

from foodmap_parser.food_uncertainty import build_food_uncertainty_inventory


class FoodUncertaintyTests(unittest.TestCase):
    def test_ranks_every_food_and_prioritizes_composite_foods(self):
        inventory = build_food_uncertainty_inventory()
        by_food_id = {item["food_id"]: item for item in inventory}

        self.assertEqual(len(inventory), 80)
        self.assertGreater(
            by_food_id["F0018"]["uncertainty_score"],
            by_food_id["F0027"]["uncertainty_score"],
        )
        self.assertIn("composite_food_category", by_food_id["F0018"]["reasons"])
        self.assertEqual(
            by_food_id["F0018"]["next_action"],
            "collect_independent_recipe_observations",
        )
