# FoodMap MealEvent Annotation Guideline

## Goal

Annotate what the text states about eating or drinking. Do not infer exact
ingredients, quantities, or timestamps that are not supported by the text.

## Record Structure

Each JSONL line contains one `input` and one or more `events`. Each event has:

- `event_index`: order of the event in the text, starting at zero.
- `time_expression`: exact phrase from the input, or `null` when absent.
- `meal_time`: `breakfast`, `lunch`, `dinner`, `snack`, `night`, `late_night`, or `unknown`.
- `consumption_status`: `consumed`, `not_consumed`, `planned`, or `unknown`.
- `foods`: normalized food mentions. A `not_consumed` event can use an empty list
  when the text says only that a meal was skipped.

Each food mention contains `raw_text`, `food_id`, `quantity`, `unit`, and
`modifiers`. Use `null` for an unknown quantity or unit and `[]` when there are
no modifiers.

## Core Rules

1. Create a separate event whenever the text assigns different foods to different meal times.
2. Preserve the original phrase in `raw_text`; use `food_id` for normalization.
3. Attach a time expression only to the food or event it modifies.
4. Record quantities only when explicitly stated. "조금" is a modifier, not a numeric quantity.
5. Mark "안 먹었어" as `not_consumed` and "먹을 거야" as `planned`; neither is a consumed event.
6. Record product and preparation detail as modifiers: `zero_sugar`, `decaf`, `iced`, `fried`, or `spicy`.
7. Do not infer a food mention from a non-food homonym such as "사과할" or "마라톤".
8. When the text cannot establish a time or amount, preserve the food and use `unknown` or `null` rather than guessing.
9. An order without an explicit eating or drinking statement is `unknown`, not `consumed`.
10. Do not create an implicit food mention from an omitted subject. For example, "오후엔 안 마셨어" does not identify a drink unless the text names it.

## Food and Event Examples

```json
{"input":"아침에 디카페인 라떼 한 잔 마셨어.","events":[{"event_index":0,"time_expression":"아침","meal_time":"breakfast","consumption_status":"consumed","foods":[{"raw_text":"디카페인 라떼","food_id":"F0004","quantity":1,"unit":"cup","modifiers":["decaf"]}]}]}
{"input":"저녁엔 치킨 먹었지만 콜라는 안 마셨어.","events":[{"event_index":0,"time_expression":"저녁","meal_time":"dinner","consumption_status":"consumed","foods":[{"raw_text":"치킨","food_id":"F0006","quantity":null,"unit":null,"modifiers":[]}]},{"event_index":1,"time_expression":"저녁","meal_time":"dinner","consumption_status":"not_consumed","foods":[{"raw_text":"콜라","food_id":"F0007","quantity":null,"unit":null,"modifiers":[]}]}]}
```

## Corpus Expansion Plan

The corpus must grow through diverse, independently written examples rather
than near-duplicate templates. The next milestone is 600 examples:

| Split | Target | Purpose |
| --- | ---: | --- |
| Train | 360 | Parser and prompt development |
| Development | 120 | Error analysis and design choices |
| Test | 120 | Final frozen evaluation only |

Each split should include the same broad phenomena, but an alias, exact wording,
or near-duplicate sentence must not appear in more than one split. At least 20
percent of the 600 examples should contain a difficult condition: multiple meal
events, quantities, negation, future intent, modifier, vague time, or non-food
homonym.

The seed set in `data/v0.2/meal_event_examples.jsonl` is a starting point for
schema and parser development. It is not large enough to report generalization
performance.
