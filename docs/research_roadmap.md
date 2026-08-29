# Research Roadmap

## Summary

FoodMap is a long-term research program that starts with Korean natural-language meal logs and eventually aims to support personalized food-body response modeling.

This repository currently covers the first foundation layer:

- a baseline parser for Korean meal descriptions
- a reference data system for foods, aliases, and components
- validation and evaluation utilities for structured extraction

The current goal is not yet to predict body responses. The current goal is to convert free-text food logs into normalized food-component events that can later be aligned with wearable time-series data.

## Three Model Types

The broader FoodMap study should distinguish three different model layers.

### 1. Parser Model

The parser model converts language into structured records.

Example input:

```text
eoje bam-e ramyeon meokgo kolla masyeosseo
```

Example target structure:

```json
{
  "meal_time": "night",
  "foods": ["ramyeon", "cola"],
  "components": [
    "high_sodium",
    "refined_carbohydrate",
    "ultra_processed",
    "carbonated",
    "sugar"
  ]
}
```

Its responsibilities include:

- detecting food expressions
- normalizing mentions to canonical food IDs
- linking foods to component IDs
- extracting explicit component mentions
- supporting meal-event structure such as time, quantity, modifiers, and consumption status

### 2. Association Model

The association model connects food components to population-level wearable response patterns.

Typical research questions include:

- Is `high_sodium` associated with post-meal heart-rate change?
- Is `caffeine` associated with sleep metrics later in the day?
- Is a late meal associated with next-morning HRV change?

At this stage the interpretation should be association, not causation.

### 3. Personalization Model

The personalization model starts from population-level patterns and adapts them to each user.

The intended progression is:

- begin with a population prior
- accumulate user-specific meal and wearable history
- update the model toward individual baselines and sensitivities

This makes it possible to represent differences such as caffeine sensitivity or late-meal sensitivity at the individual level.

## Current Scope of This Repository

This repository is in the early-to-middle part of the parser foundation stage.

Current assets include:

- `data/v0.1/food_items.csv`
- `data/v0.1/food_aliases.csv`
- `data/v0.1/component_vocab.csv`
- `data/v0.1/component_aliases.csv`
- `data/v0.1/food_component_annotations.csv`
- `data/v0.1/mfds_food_links.csv`
- `data/v0.1/sentence_examples.jsonl`
- `src/foodmap_parser/parser.py`
- `src/foodmap_parser/validate.py`
- `src/foodmap_parser/evaluate.py`

The core question for the current repository is:

How reliably can natural Korean meal descriptions be transformed into normalized food, component, and meal-event records?

## Research Phases

### Phase 1. Structure Natural-Language Meal Logs

Goal:

- transform user-written meal logs into structured data

Why it matters:

- downstream analysis depends on consistent event records
- wearable alignment is only possible after meal events are normalized
- large-scale analysis breaks down if equivalent foods remain fragmented across many free-text forms

Key outputs:

- canonical food IDs
- component IDs
- meal-event records
- parser evaluation metrics

Important limitations to address in later parser versions:

- negation
- future intent
- quantity and unit
- contrastive expressions
- food-time relation
- zero-sugar or decaf modifiers
- non-food homonyms

### Phase 2. Build and Expand the Reference Data System

This phase formalizes the ontology and lookup tables used by the parser.

Main tables:

- `food_items.csv`: canonical food inventory
- `food_aliases.csv`: user-expression normalization
- `component_vocab.csv`: canonical component vocabulary
- `component_aliases.csv`: surface-form normalization for components
- `food_component_annotations.csv`: food-to-component mapping
- `mfds_food_links.csv`: linkage to external reference data

Why it matters:

- many user expressions must collapse to the same internal IDs
- consistent normalization is required for aggregation and longitudinal analysis
- external data sources become easier to join once canonical IDs exist

### Phase 3. Build a Scalable Data Pipeline

Once the reference data is stable enough, the next step is large-scale ingestion and retrieval.

Planned data-engineering work:

- convert MFDS raw spreadsheets into analysis-friendly tables
- store reference data in Parquet
- query and join with DuckDB
- use SQL or Polars for batch processing
- add stronger validation and QA checks

The storage choice is intentional. Small, curated reference tables remain CSV
because they are easy to inspect and version. Nested text annotations use JSONL.
Large, append-oriented wearable and event tables move to Parquet, while DuckDB
provides local analytical joins without adding a database server.

This is the bridge between a small curated dataset and large-scale structured meal-event generation.

### Phase 4. Model Population-Level Food-Body Associations

After meal events can be generated reliably, the next stage is to align them with wearable time-series data.

An example analytical table might include:

```text
user_id
meal_event_id
event_time
component_vector
context_features
wearable_before_meal
wearable_after_meal
response_delta
```

Potential analysis targets include:

- post-meal heart-rate change
- next-morning HRV change
- sleep disruption
- activity changes after meals

This phase should be framed as learning general associations between food components and physiological responses.

The first narrow study should focus on late caffeine events and same-night sleep
metrics. It has a clearer text signal and time alignment than broad food-body
claims, and it does not require treating consumer wearable data as clinical
ground truth.

### Phase 5. Personalize the FoodMap Model

The final stage adapts population-level patterns to each user.

A practical strategy is:

- learn a population model first
- use it as a prior for new users
- update it as user-specific data accumulates

This should lead to individualized sensitivity estimates rather than a one-size-fits-all response map.

## Literature Review Priorities

Literature review is not only background reading. It should support:

- problem framing
- gap analysis
- representation choices
- evaluation design

Priority topics:

- personalized nutrition and glycemic response
- wearable-based digital biomarkers
- food logging and dietary assessment
- LLM-based information extraction
- time-series health data analysis

Useful review questions:

- How is food data represented?
- How are user-specific differences handled?
- What time windows are used around meals?
- Are results framed as causal or associative?
- How does the method differ from the FoodMap pipeline?

## Current Position

As of the current `v0.1` scope, this repository mainly provides:

- food ontology draft
- component vocabulary
- food-component annotation
- alias-based baseline parsing
- validation and evaluation scripts
- MFDS linkage support

In short, the repository is building the parser model and reference data system, not yet the association model or the personalization model.

## Immediate Next Steps

1. Formalize a richer meal-event schema and annotation guideline.
2. Grow the annotation corpus to 600 independently written examples, with frozen development and test splits.
3. Extend the parser for time-to-food relations, quantity, modifiers, negation, and future intent.
4. Add provenance fields to food-component evidence and convert large external reference data to Parquet.
5. Normalize wearable exports into Parquet, then create DuckDB-based meal-to-sleep response windows.
6. Validate the analysis pipeline with documented, evidence-grounded synthetic data before using real user data.

## Working Definition

One concise description of the broader research direction is:

FoodMap begins by extracting foods and components from natural-language meal records and converting them into normalized food-component events. It then aligns large-scale meal-event data with wearable time-series signals to analyze general associations between food components and physiological responses. Finally, it adapts those general patterns to each user's baseline and sensitivities to build a personalized FoodMap model.
