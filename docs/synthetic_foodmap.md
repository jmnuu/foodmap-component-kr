# Synthetic FoodMap and Personalization

## Model Structure

```text
Literature-backed population prior
  + synthetic or real MealEvent observations
  -> population FoodMap
  -> individual posterior FoodMap
```

The population FoodMap is stored in `data/v0.3/population_foodmap_priors.json`.
It records a component trigger, an observable outcome, the required instrument,
direction, evidence, and a synthetic-only standardized prior parameter.

All external and internal references are registered in
`data/v0.3/source_registry.json`. Every population prior names one registry
`source_id`; generated relationships preserve their input and reference source
identifiers in `source_ledger.jsonl` and `event_relations.jsonl`.

Synthetic response observations and personalized FoodMap rows also retain
`source_ids`. These identify the input event, the population-prior evidence, and
the synthetic generator. A personal estimate therefore remains auditable even
after it has been updated from many observations.

The first three priors intentionally cover different measurement capabilities:

| Trigger | Outcome | Required instrument | Why it is limited this way |
| --- | --- | --- | --- |
| Late caffeine | Sleep-disruption score | Sleep-capable wearable | Caffeine and subsequent sleep can be temporally aligned. |
| Sugar or refined carbohydrate | Postprandial glucose-response score | CGM | A standard watch cannot measure glucose response. |
| High sodium | Blood-pressure response score | Validated BP measurement | A standard watch cannot reliably supply the needed BP outcome. |

The standardized means are not medical effect sizes. They make a synthetic data
generator behave consistently while preserving the direction and measurement
scope supported by the cited evidence.

## Personalization Update

For every user and prior relationship, FoodMap uses empirical-Bayes shrinkage:

```text
personal estimate =
  (prior strength * population mean + sum of observed response scores)
  / (prior strength + observation count)
```

This choice is deliberate. With one or two observations, an individual's map
remains close to the population prior. As repeated observations accumulate, the
individual's own records have more influence. It prevents a single unusual night
from becoming a strong dietary conclusion.

## Generate a Synthetic Cohort

```bash
source .venv/bin/activate
PYTHONPATH=src python src/foodmap_parser/synthetic_foodmap.py \
  --profiles 30 \
  --days 14 \
  --seed 42 \
  --output-dir data/processed/synthetic_foodmap
```

The output includes synthetic profiles, meal-log text, association tables,
synthetic response observations, personalized FoodMap rows, and a run manifest.
Every generated record is marked `data_origin: synthetic`.

## Food Dictionary Strategy

The current 80-food reference table is a reviewed coverage expansion, not a
production-complete food dictionary. Expand it using three distinct layers:

| Layer | Source and purpose |
| --- | --- |
| Internal FoodMap IDs and aliases | Stable Korean normalization and parser behavior. |
| Official nutrient link | Link canonical foods to the MFDS Food Nutrient Database for nutrient values and source-version tracking. |
| Ontology link | Map food category, ingredient, processing, and cooking facets to FoodOn terms. |

FoodOn is a vocabulary and relation system; it does not replace Korean
food-specific nutrient values. MFDS links and FoodOn links therefore belong in
separate tables. New foods first enter `unresolved_food_candidates`, then become
an alias or canonical food only after review and evidence collection.
