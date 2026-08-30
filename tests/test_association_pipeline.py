import json
import tempfile
import unittest
from pathlib import Path

from foodmap_parser.association_pipeline import run_pipeline


class AssociationPipelineTests(unittest.TestCase):
    def test_builds_relations_for_newly_expanded_food(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input.jsonl"
            output_dir = temporary_path / "output"
            input_path.write_text(
                "{\"id\":\"LOG001\",\"input\":\"저녁에 비빔밥이랑 김밥 먹었어.\"}\n",
                encoding="utf-8",
            )

            result = run_pipeline(input_path, output_dir, write_parquet=False)

            self.assertEqual(result["coverage_report"]["meal_events"], 1)
            self.assertEqual(result["coverage_report"]["food_mentions"], 2)
            self.assertTrue(any(
                row["relation_type"] == "HAS_COMPONENT"
                for row in result["event_relations"]
            ))
            self.assertFalse(result["unresolved_food_candidates"])
            self.assertTrue((output_dir / "event_relations.jsonl").exists())
            self.assertTrue((output_dir / "source_ledger.jsonl").exists())
            self.assertTrue((output_dir / "coverage_report.json").exists())

            report = json.loads((output_dir / "coverage_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["unresolved_food_candidates"], 0)
            self.assertTrue(all(
                row["source_ids"]
                for row in result["event_relations"]
            ))

    def test_writes_representative_component_relations_for_ambiguous_food(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input.jsonl"
            output_dir = temporary_path / "output"
            input_path.write_text(
                "{\"id\":\"LOG002\",\"input\":\"점심에 샌드위치 먹었어.\"}\n",
                encoding="utf-8",
            )

            result = run_pipeline(input_path, output_dir, write_parquet=False)
            relations = [
                row for row in result["event_relations"]
                if row["relation_type"] == "HAS_REPRESENTATIVE_COMPONENT"
            ]
            roles_by_component = {row["object_id"]: row["component_role"] for row in relations}

            self.assertEqual(roles_by_component["C060"], "core")
            self.assertEqual(roles_by_component["C065"], "typical")
            self.assertEqual(roles_by_component["C067"], "variant")
            self.assertIsNotNone(next(
                row["model_probability"]
                for row in relations
                if row["object_id"] == "C060"
            ))
            self.assertEqual(result["coverage_report"]["representative_component_relation_count"], len(relations))
