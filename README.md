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

v0.1

## Dataset Files

```text
data/v0.1/
├── component_vocab.csv
├── food_items.csv
├── food_component_annotations.csv
├── food_aliases.csv
└── sentence_examples.jsonl

## Current Scope

This version implements a rule-based baseline parser for Korean meal logs.

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

## Current Dataset Size

- Foods: 30
- Components: 64
- Food-component annotations: 168
- Food aliases: 67
- Component aliases: 19
- Sentence examples: 47

## Current Evaluation

On the current closed-set sentence examples:

- Food ID precision: 1.000
- Food ID recall: 1.000
- Food ID F1: 1.000
- Component precision: 1.000
- Component recall: 1.000
- Component F1: 1.000

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