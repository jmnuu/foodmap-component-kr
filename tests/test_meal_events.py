import unittest

from foodmap_parser.meal_events import parse_meal_events


class MealEventParserTests(unittest.TestCase):
    def test_splits_foods_by_meal_time(self):
        result = parse_meal_events("아침에 라떼 마시고 점심에 김치찌개랑 밥 먹었어.")

        self.assertEqual(len(result["events"]), 2)
        self.assertEqual(result["events"][0]["meal_time"], "breakfast")
        self.assertEqual(result["events"][0]["foods"][0]["food_id"], "F0004")
        self.assertEqual(result["events"][1]["meal_time"], "lunch")
        self.assertEqual(
            [food["food_id"] for food in result["events"][1]["foods"]],
            ["F0005", "F0010"],
        )

    def test_keeps_negated_food_out_of_consumed_event(self):
        result = parse_meal_events("저녁에 치킨 먹었지만 콜라는 안 마셨어.")

        self.assertEqual(len(result["events"]), 2)
        self.assertEqual(result["events"][0]["consumption_status"], "consumed")
        self.assertEqual(result["events"][1]["consumption_status"], "not_consumed")
        self.assertEqual(result["events"][1]["foods"][0]["food_id"], "F0007")

    def test_extracts_quantity_and_modifier(self):
        result = parse_meal_events("오후 4시에 디카페인 라떼 한 잔 마심.")
        food = result["events"][0]["foods"][0]

        self.assertEqual(result["events"][0]["meal_time"], "snack")
        self.assertEqual(food["quantity"], 1)
        self.assertEqual(food["unit"], "cup")
        self.assertIn("decaf", food["modifiers"])

    def test_marks_future_intent_as_planned(self):
        result = parse_meal_events("내일 아침에 그릭요거트랑 바나나 먹을 거야.")

        self.assertEqual(result["events"][0]["consumption_status"], "planned")
        self.assertEqual(result["events"][0]["meal_time"], "breakfast")

    def test_rejects_non_food_homonym(self):
        result = parse_meal_events("오늘 사과할 일이 있었지만 과일은 안 먹었어.")

        self.assertEqual(result["events"], [])

    def test_recognizes_new_compound_food_without_extracting_rice_alias(self):
        result = parse_meal_events("저녁에 비빔밥 먹었어.")

        foods = result["events"][0]["foods"]
        self.assertEqual([food["food_id"] for food in foods], ["F0031"])
        self.assertNotIn("F0010", [food["food_id"] for food in foods])

    def test_does_not_mistake_duyu_for_the_quantity_word_two(self):
        food = parse_meal_events("아침에 두유 먹었어.")["events"][0]["foods"][0]

        self.assertEqual(food["food_id"], "F0067")
        self.assertIsNone(food["quantity"])

    def test_adds_role_aware_representative_components_for_generic_sandwich(self):
        result = parse_meal_events("점심에 샌드위치 먹었어.")
        components = result["events"][0]["foods"][0]["representative_components"]
        by_id = {component["component_id"]: component for component in components}

        self.assertEqual(by_id["C060"]["role"], "core")
        self.assertEqual(by_id["C065"]["role"], "typical")
        self.assertEqual(by_id["C067"]["role"], "variant")
        self.assertIn("language_model", by_id["C067"]["inference_method"])
        self.assertIn("model_probability", by_id["C060"])

    def test_promotes_explicit_sandwich_variant_and_removes_meat_for_vegan_context(self):
        result = parse_meal_events("점심에 비건 햄치즈 샌드위치 먹었어.")
        foods = result["events"][0]["foods"]
        self.assertEqual([food["food_id"] for food in foods], ["F0018"])
        components = foods[0]["representative_components"]
        by_id = {component["component_id"]: component for component in components}

        self.assertEqual(by_id["C059"]["role"], "explicit")
        self.assertNotIn("C067", by_id)
        self.assertNotIn("C044", by_id)
        self.assertEqual(by_id["C048"]["role"], "typical")


if __name__ == "__main__":
    unittest.main()
