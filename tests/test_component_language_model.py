import unittest

from foodmap_parser.component_language_model import predict_seed_components


class ComponentLanguageModelTests(unittest.TestCase):
    def test_text_model_ranks_ham_and_cheese_above_generic_sandwich(self):
        generic = {
            item["component_id"]: item["probability"]
            for item in predict_seed_components("점심에 샌드위치 먹었어.", "F0018")
        }
        ham_cheese = {
            item["component_id"]: item["probability"]
            for item in predict_seed_components("점심에 햄치즈 샌드위치 먹었어.", "F0018")
        }

        self.assertGreater(ham_cheese["C067"], generic["C067"])
        self.assertGreater(ham_cheese["C059"], generic["C059"])
        self.assertGreater(ham_cheese["C060"], 0.5)

    def test_text_model_marks_its_training_provenance(self):
        result = predict_seed_components("점심에 에그 샌드위치 먹었어.", "F0018")
        egg = next(item for item in result if item["component_id"] == "C068")

        self.assertEqual(egg["model_name"], "korean_char_ngram_multilabel_naive_bayes")
        self.assertEqual(egg["training_source_id"], "SRC_COMPONENT_LANGUAGE_TRAINING_V03")
