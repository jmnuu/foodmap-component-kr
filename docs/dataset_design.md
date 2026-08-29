# FoodMap Dataset Design

## Purpose

FoodMap uses different data representations for different stages of the
research pipeline. The design avoids forcing all data into CSV while keeping
small reference data easy to inspect and correct.

```text
Natural-language meal log
  -> JSONL annotation and parser output
  -> normalized MealEvent and food rows
  -> Parquet event and wearable tables
  -> DuckDB joins, feature creation, and association analysis
```

## Storage Choices

| Data layer | Format | Why this format is used now | Use another format when |
| --- | --- | --- | --- |
| Food and component reference data | CSV | The current tables are small, curated, diff-friendly, and easy to review in Git. | Move a copy to Parquet once joins or full-table scans become frequent. |
| Text annotation corpus | JSONL | A single sentence can contain several events, foods, quantities, and modifiers. JSONL preserves this nesting and supports one-record-at-a-time review. | Export flattened tables for SQL analysis or model training. |
| Raw wearable export | Original vendor file, read-only | Preserves provenance and enables reprocessing when normalization rules improve. | Never overwrite raw data; create a new derived version instead. |
| Normalized wearable measurements | Parquet | Timestamped measurements become large quickly; Parquet compresses well and reads only requested columns. | CSV remains acceptable for small illustrative samples. |
| Meal-event and response features | Parquet | Event-level analysis repeatedly scans selected columns across many rows. | A relational database is useful only when concurrent application writes or access control become necessary. |
| Local analytical joins | DuckDB | It queries Parquet directly, needs no server, and is sufficient for a single-researcher or small-team workflow. | Use PostgreSQL or a managed warehouse for multi-user production services. |

## Versioned Data Layout

```text
data/
  v0.1/                         small curated food reference tables
  v0.2/                         MealEvent annotation data and corpus plan
  raw/                           immutable source exports; excluded from Git if sensitive
  processed/
    wearable_measurements/       normalized Parquet partitions
    meal_events/                 parser output in Parquet
    response_windows/            meal-aligned analysis features in Parquet
```

`raw/` must not contain personal health data in a public repository. Real users
should be represented by a pseudonymous `user_id`; direct identifiers are kept
outside the research dataset.

## Meal Log Annotation Record

The unit of annotation is one input text record. It may contain one or more
MealEvents, because a user can describe breakfast and lunch in one sentence.

```json
{
  "id": "ME0001",
  "input": "아침 8시에 아아 한 잔 마시고 점심에 김밥 먹었어.",
  "events": [
    {
      "event_index": 0,
      "time_expression": "아침 8시",
      "meal_time": "breakfast",
      "consumption_status": "consumed",
      "foods": [
        {
          "raw_text": "아아",
          "food_id": "F0003",
          "quantity": 1,
          "unit": "cup",
          "modifiers": ["iced"]
        }
      ]
    }
  ]
}
```

`consumed_at` is intentionally not required in the annotation corpus. Natural
language often gives only a relative expression such as "점심". Exact timestamps
are added only when the source record provides a date, time, and time zone.

## Analytical Tables

JSONL events are flattened into two linked tables before association analysis.

### `meal_events`

| Column | Meaning |
| --- | --- |
| `event_id` | Stable meal-event identifier |
| `user_id` | Pseudonymous user identifier |
| `event_time` | Time-zone-aware event time when known |
| `meal_time` | Breakfast, lunch, dinner, snack, night, or unknown |
| `consumption_status` | Consumed, not consumed, planned, or unknown |
| `source_text` | Original user text, access-controlled when personal |
| `parser_version` | Parser version that generated the record |

### `meal_event_foods`

| Column | Meaning |
| --- | --- |
| `event_id` | Link to `meal_events` |
| `food_position` | Food order within the event |
| `food_id` | Canonical FoodMap food ID |
| `quantity` and `unit` | Observed amount when expressed |
| `modifier_json` | Product, cooking, or dietary modifiers |
| `parse_confidence` | Extraction confidence |

The split prevents duplicated event metadata for each food while still making
food-level component joins efficient.

## Wearable Data and Response Windows

Raw wearable measurements should first be normalized to a long table:

```text
user_id, timestamp, metric_name, value, unit, device, source_file_version
```

Long format accommodates different devices without adding sparse columns. For a
specific analysis, derive a wide feature table such as:

```text
event_id, caffeine_present, caffeine_event_time, sleep_onset,
sleep_duration_minutes, sleep_efficiency, overnight_hrv, activity_before_event
```

The first FoodMap study will define a fixed window for late caffeine events and
same-night sleep. The exact window must be written in the study protocol before
examining outcomes, rather than selected after inspecting results.

## Evidence and Uncertainty

Food-component links remain explicit about uncertainty. `confirmed` means the
link is supported for the represented food definition; `possible` means it
depends on a variant, brand, recipe, or portion. Future annotation versions
should add `evidence_source`, `source_id`, and `evidence_note` to retain whether
the link comes from an official database, a label, literature, or a documented
curation decision.

## Synthetic Data Policy

Synthetic people and wearable records may be used to test schema, time joins,
feature engineering, and analysis code. They cannot establish real food-body
effects. Synthetic response parameters must record their literature source or
explicit assumption; an LLM must not invent physiological outcomes without this
traceability.
