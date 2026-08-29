import tempfile
import unittest
from pathlib import Path

from foodmap_parser.synthetic_foodmap import generate_synthetic_foodmap


class SyntheticFoodMapTests(unittest.TestCase):
    def test_generates_profiles_events_and_personalized_foodmaps(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "synthetic"
            result = generate_synthetic_foodmap(
                output_dir,
                profile_count=3,
                days=2,
                seed=7,
                write_parquet=False,
            )

            self.assertEqual(len(result["profiles"]), 3)
            self.assertGreater(len(result["meal_logs"]), 0)
            self.assertGreater(len(result["observations"]), 0)
            self.assertEqual(len(result["personal_foodmaps"]), 9)
            self.assertTrue((output_dir / "personal_foodmaps.jsonl").exists())
            self.assertTrue((output_dir / "association" / "event_relations.jsonl").exists())
            self.assertEqual(result["manifest"]["data_origin"], "synthetic")
            self.assertTrue(all(
                observation["source_ids"]
                for observation in result["observations"]
            ))
            self.assertTrue(all(
                foodmap["source_ids"]
                for foodmap in result["personal_foodmaps"]
            ))
