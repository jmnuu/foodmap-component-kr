# FoodMap Literature Evidence Model

## Purpose

`data/v0.3/literature_evidence_relations.json` records how literature should
inform FoodMap. It does not convert a paper into a rule such as "this food will
cause this outcome for this person." The file separates the exposure definition,
the required FoodMap features, the outcome scope, the study design, and the
interpretation limit.

This distinction is essential because most cited nutrition umbrella reviews
summarize observational evidence. They are strong sources for deciding which
variables to preserve, but they do not by themselves identify a causal effect
for an individual user.

## Why Features Are More Detailed Than Tags

The evidence model intentionally avoids broad boolean tags such as
`sugar=true`, `protein_rich=true`, `dairy=true`, or `fiber_rich=true`.

| Broad tag to avoid | FoodMap fields to retain | Reason |
| --- | --- | --- |
| `sugar=true` | `added_sugar_g`, `fructose_g`, `beverage_type`, `serving_ml`, `food_source` | Added sugar, fructose, and sugar-sweetened beverages are distinct literature exposures. |
| `protein_rich=true` | `protein_g`, `protein_source`, `food_source` | Total, animal, and plant protein have different exposure definitions. |
| `dairy=true` | `dairy_product_type`, `fat_content`, `portion_g`, `region` | Milk, yogurt, cheese, and butter cannot be assumed to share one association pattern. |
| `high_sodium=true` | `sodium_mg`, `portion_g`, `measurement_window` | Sodium is dose-bearing and needs an aligned blood-pressure measurement. |
| `fiber_rich=true` | `fiber_g`, `fiber_type`, `food_source` | Total, soluble, insoluble, and source-specific fiber must remain distinguishable when possible. |
| `ultra_processed=true` | `processing_classification`, `classification_system`, `portion_g`, `frequency` | Processing is a classification system, not a nutrient amount; the classification method must be known. |

`unknown` is a valid value for optional detail. It is better to preserve missing
information explicitly than to infer plant protein, fat content, or processing
class from a food name without evidence.

## Data Structure Choice

Evidence rows are stored as JSON because one paper can name several required
features, outcomes, and allowed values. Storing these lists in one CSV cell
would make validation and queries fragile. The output of a future analysis is
still flattened into relational or Parquet tables, for example:

```text
event_id, food_id, sodium_mg, event_time, bp_measurement_id, window_id
```

This keeps reference knowledge expressive while keeping analysis data efficient
for DuckDB joins and aggregation.

## Using the Evidence Model

1. Use `SRC_GBD_DIET_2019` to decide which dietary-risk concepts should receive
   early ontology coverage.
2. Use each exposure-specific row to decide which food or nutrient fields must
   be collected before an association analysis.
3. Create an analysis protocol with the outcome measurement, time window,
   confounders, and missing-data policy before reading result values.
4. Keep `causal_claim_allowed: false` for the current literature rows unless a
   future study design supplies causal identification assumptions and evidence.

## Current Limits

The table is a research-design layer, not a clinical recommendation engine.
It does not contain effect sizes, risk scores, or individualized thresholds.
Those should be added only after recording the exact comparison, unit, outcome,
population, certainty assessment, and compatible measurement protocol.
