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

## Adversarial Alias Guard

### Problem

The parser could falsely detect short or ambiguous aliases inside non-food words.

Examples:

- 마라톤 → 마라
- 포기하지 말자 → 포
- 사과할 일이 있었어 → 사과

### Fix

- Removed risky short aliases such as `마라` and `포`.
- Added `should_accept_food_alias()` to block ambiguous aliases in non-food contexts.
- Added adversarial examples with empty `food_mentions`.
- Updated evaluation to treat empty gold and empty prediction as correct.

### Result

The parser correctly avoids false positives for adversarial non-food sentences while still detecting real food sentences such as:

- 오늘 사과 먹었어.
- 마라탕 먹었어.
- 쌀국수 먹었어.

---

## MFDS Food Database Linking v0.2.1

### Date

2026-07-12

### Goal

Connect the curated FoodMap food ontology with official MFDS food database entries.

The purpose of this step is not to import the entire MFDS database into FoodMap directly, but to create a link table between FoodMap food IDs and official MFDS food codes.

### Why This Step Matters

FoodMap currently uses a manually curated food ontology.  
By linking FoodMap food IDs to the MFDS food database, the project gains:

- official Korean food names
- official food codes
- external food category information
- future access to nutrition values
- a basis for expanding alias and component candidates

This makes the dataset more reliable and easier to extend.

### Data Used

The MFDS food database files were added locally under:

```text
data/raw/mfds/