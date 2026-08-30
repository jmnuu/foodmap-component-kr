# FoodMap-Component-KR

FoodMap-Component-KR is a Korean food component dataset for converting natural Korean meal descriptions into structured food component tokens.

## Goal

The goal of this project is to build a dataset and parser that can transform Korean food logs into structured food components such as:

- refined_carbohydrate
- high_sodium
- caffeine
- dairy
- gluten_possible
- ultra_processed
- spicy
- fried

This dataset will later be connected with wearable data such as Apple Watch sleep, heart rate, HRV, and activity metrics to study personalized food-body response patterns.

## Current Version

v0.1 baseline plus a reviewed v0.3 dictionary-expansion layer, with a v0.2
MealEvent dataset design.

## Documentation

- [Research roadmap](docs/research_roadmap.md)
- [Dataset design](docs/dataset_design.md)
- [Annotation guideline](docs/annotation_guideline.md)
- [Literature notes](docs/literature_notes.md)
- [Literature evidence model](docs/evidence_model.md)
- [General FoodMap paper summaries](docs/paper_summaries_general_foodmap.md)
- [Research architecture and algorithm choices](docs/research_architecture.md)
- [Korean text component language model](docs/component_language_model.md)
- [General probability data expansion](docs/general_probability_data_expansion.md)
- [v0.3 dictionary expansion](docs/dictionary_expansion_v0.3.md)
- [Food component uncertainty inventory](docs/food_uncertainty_inventory.md)
- [Daily result: 2026-08-29](reports/2026-08-29_foodmap_result.md)
- [Association extraction pipeline](docs/association_pipeline.md)
- [Synthetic FoodMap and personalization](docs/synthetic_foodmap.md)

## Dataset Files

```text
data/v0.1/
├── component_vocab.csv
├── food_items.csv
├── food_component_annotations.csv
├── food_aliases.csv
└── sentence_examples.jsonl

data/v0.3/
├── food_items_extension.csv
├── food_aliases_extension.csv
├── component_vocab_extension.csv
├── component_aliases_extension.csv
└── food_component_annotations_extension.csv
```

## Current Scope

This version combines a rule-based MealEvent baseline with an auditable,
probability-based representative-component model for known foods.

The parser performs:

- food expression detection
- food alias normalization
- food ID linking
- food-component lookup
- explicit ingredient/component detection
- final component merging
- dataset validation
- parser evaluation
- missing alias mining

The v0.2 `MealEvent Parser` builds on this baseline to group detected foods by
time expression and extract explicit quantities, modifiers, and consumption
status. It remains a rule-based research baseline and is evaluated separately
from the closed-set v0.1 food-ID score. Its primary metric is food-time-status
matching, which requires every extracted food to be assigned the correct time
bucket and consumption state.

## Current Dataset Size

- Foods: 80
- Components: 94
- Food-component annotations: 373
- Food aliases: 142
- Component aliases: 52
- Sentence examples: 53
- General component priors: 80 foods
- Direct human-curated component-language examples: 14
- Weakly supervised component-language examples: 143

## Current Evaluation

On the current closed-set sentence examples:

- Food ID precision: 1.000
- Food ID recall: 1.000
- Food ID F1: 1.000
- Component precision: 1.000
- Component recall: 1.000
- Component F1: 1.000

## MealEvent Baseline Evaluation

On the current 36-example v0.2 development seed set:

- Food ID F1: 1.000
- Food-time-status F1: 1.000

This is a closed development seed set used for schema and rule regression
testing. It is not evidence of generalization; the planned frozen evaluation
set will use independently written, unseen meal logs.

## Important Limitation

The current evaluation is a closed-set test.  
Most food expressions in the test examples are already covered by the alias tables.

The current parser does not yet handle:

- negation
- future intent
- quantity
- food-time relation
- contrastive expressions
- zero-sugar or decaf modifiers
- non-food homonyms such as 사과하다 or 마라톤

## Next Version

The next version will implement a Meal Event Parser that extracts:

- meal time
- meal type
- food-time relation
- quantity and unit
- consumption status
- modifiers
- negation and future intent

The first association study will focus on the relationship between late caffeine
events and same-night wearable sleep metrics. It is an association study, not a
medical or causal claim.

## Data Architecture

FoodMap deliberately uses more than one storage format because each layer has a
different job:

- CSV: small, curated reference tables that researchers need to inspect and edit.
- JSONL: nested text annotations where one meal log can contain several meal events.
- Parquet: large, immutable, analysis-ready event and wearable tables.
- DuckDB: local joins and aggregate analysis over Parquet without operating a server.

The `data/v0.2/` directory contains the MealEvent annotation seed set and the
corpus expansion plan. `data/v0.3/` is an additive, reviewed coverage layer;
the loaders merge it with the immutable v0.1 baseline. The detailed schema and
selection criteria are in `docs/dataset_design.md`.

## Association Pipeline

The association pipeline creates relationship-rich MealEvent, food mention, and
typed relation data from meal-log JSONL. It writes an unresolved-food review
queue so the food dictionary expands from real input instead of guessed entries.
It materializes both auditable JSONL and DuckDB-backed Parquet outputs.

## Synthetic FoodMap

The v0.3 synthetic FoodMap separates literature-backed directional priors from
synthetic response scores. It uses an empirical-Bayes update so individual data
gradually adjusts a population map instead of overreacting to a small number of
records. Synthetic outputs are pipeline-validation data, not clinical evidence.
