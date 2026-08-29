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

    def test_does_not_extract_rice_from_unknown_compound_food(self):
        result = parse_meal_events("저녁에 비빔밥 먹었어.")

        self.assertEqual(result["events"], [])


if __name__ == "__main__":
    unittest.main()
