import unittest

from foodmap_parser.component_language_model import load_training_records
from foodmap_parser.general_component_priors import load_general_component_priors
from foodmap_parser.meal_events import parse_meal_events


class GeneralComponentPriorsTests(unittest.TestCase):
    def test_builds_population_priors_for_every_current_food(self):
        priors = load_general_component_priors()

        self.assertEqual(len(priors), 80)
        self.assertTrue(priors["F0005"]["components"])
        self.assertTrue(all(
            component["prior_alpha"] > 0 and component["prior_beta"] > 0
            for component in priors["F0005"]["components"]
        ))
        self.assertEqual(
            priors["F0018"]["components"][0]["prior_kind"],
            "manual_composite_food_prior",
        )

    def test_combines_human_seed_and_weak_reference_training_records(self):
        records = load_training_records()
        human_records = [
            record for record in records
            if "SRC_COMPONENT_LANGUAGE_TRAINING_V03" in record["training_source_ids"]
        ]
        weak_records = [
            record for record in records
            if record.get("label_origin") == "weak_reference_annotation"
        ]

        self.assertEqual(len(human_records), 14)
        self.assertGreaterEqual(len(weak_records), 140)
        self.assertGreater(len({record["food_id"] for record in weak_records}), 70)
        self.assertTrue(any(record["food_id"] == "F0040" for record in weak_records))

    def test_weak_only_food_uses_population_prior_without_fake_model_certainty(self):
        food = parse_meal_events("저녁에 김치찌개 먹었어.")["events"][0]["foods"][0]

        self.assertTrue(food["representative_components"])
        self.assertTrue(all(
            "model_probability" not in component
            for component in food["representative_components"]
        ))
