# v0.3 Dictionary Expansion

## What Changed

The immutable v0.1 reference tables remain the baseline. This release adds a
separate v0.3 layer, loaded together at runtime, so expansion can be reviewed
or replaced without rewriting the original data.

| Table | v0.1 baseline | v0.3 merged total | Why it is separate |
| --- | ---: | ---: | --- |
| Canonical foods | 30 | 80 | One stable ID per food family keeps text and later wearable joins consistent. |
| Food aliases | 65 | 142 | Meal logs contain abbreviations and spelling variants such as `제육`, `샤브`, and `돈가스`. |
| Components | 68 | 94 | Broad nutrients are insufficient for ambiguous meals, so ingredient-level concepts such as garlic, tofu, leafy greens, and pork belly are retained. |
| Food-component annotations | 168 | 373 | Each connection has a status, confidence, evidence type, note, version, and source ID. |
| Component aliases | 23 | 52 | Explicitly written ingredients are stronger evidence than a food-family prior. |

The expansion includes representative Korean meal families: stews, pork and
fish dishes, noodles, rice dishes, vegetables, beverages, and common
side ingredients. It deliberately covers uncertain foods such as
`제육볶음`, `샤브샤브`, `비빔밥`, and `카레` with multiple candidate components
instead of claiming one fixed recipe.

## Data Choice

CSV remains appropriate here because the reference set is still small enough
for line-by-line review, Git diffs, and research annotation. Each extension
row is merged by `src/foodmap_parser/reference_data.py`; downstream parsers,
the uncertainty inventory, and weak-label generation therefore use one
consistent vocabulary.

The confidence values are **reviewed priors**, not measured Korean recipe
frequencies. For a generic food, `confirmed` means a stable defining feature;
`possible` preserves plausible but variable ingredients. The General FoodMap
later updates these Beta priors only with independently sourced recipe or
ingredient-list observations. Generated examples and LLM guesses can help
find candidates, but cannot be counted as population frequency evidence.

## Example: Why This Helps

For `제육볶음`, pork, spicy seasoning, and chili paste are high-probability
representative components. Garlic, green onion, and fat remain lower-probability
possibilities because household and restaurant recipes differ. For
`샤브샤브`, broth, leafy greens, and a general protein filling are kept as
representative categories; mushroom and noodle additions remain variable.

This is more useful than pretending to know every user's exact sodium grams.
When a person later writes an explicit ingredient, the parser records that
explicit text separately and it outranks the generic prior. When the user only
writes the food name, the system returns an auditable Top-k probability map and
can ask a clarification question when a health-relevant choice remains close.
