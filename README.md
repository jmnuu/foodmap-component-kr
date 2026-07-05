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