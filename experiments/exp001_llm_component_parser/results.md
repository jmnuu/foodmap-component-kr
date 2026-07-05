# Experiment 001: Rule-based FoodMap Parser

## Date
2026-07-05

## Dataset
- foods: 10
- components: 37
- aliases: 30
- sentence examples: 3

## Metrics
Food ID Matching:
- precision:
- recall:
- f1:

Component Matching:
- precision:
- recall:
- f1:

## Error Cases
- None

## Next Improvements
- Add more aliases
- Add more Korean meal sentences
- Add ambiguous food expressions

# Experiment 002: Rule-based FoodMap Parser

## Date
2026-07-05

## Dataset

- components: 37
- foods: 10
- annotations: 37
- aliases: 31
- sentence examples: 42

## Result Before Fix

Food ID Matching:

- precision: 0.913
- recall: 1.000
- f1: 0.948

Component Matching:

- precision: 0.977
- recall: 1.000
- f1: 0.988

## Problem

The parser detected short aliases inside longer food names.

Examples:

- 김밥 triggered both 김밥 and 밥
- 카페라떼 triggered both 카페라떼 and 라떼
- 치즈김밥 triggered both 치즈김밥 and 김밥 and 밥

## Fix

Updated parser to match longer aliases first and prevent overlapping text spans.

## Result After Fix

Food ID Matching:

- precision: 1.000
- recall: 1.000
- f1: 1.000

Component Matching:

- precision: 1.000
- recall: 1.000
- f1: 1.000

## Interpretation

The rule-based parser works correctly on the current v0.1-mini dataset.  
The next step is to test generalization by adding more foods, harder aliases, and more diverse meal sentences.