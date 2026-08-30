# General Probability Data Expansion

## What Has Expanded

FoodMap now constructs a representative-component prior for all 80 current food
IDs, rather than only using a hand-written sandwich prior. The v0.1 baseline
and v0.3 extension together provide 373 reviewed food-component annotations as
prior evidence. They do **not** pretend to be 373 independent recipes or
measured population frequencies.

The text model now trains from two clearly separated sources:

| Training source | Current size | Role | May be used for final evaluation? |
| --- | --- | --- | --- |
| Human-curated composite-food examples | 14 records | Teaches generic and contextual sandwich variants. | No; it is training data. |
| Weak records derived from reviewed food IDs and aliases | Generated at runtime; currently 143 records | Broadens known food and alias coverage. | No; labels are derived from the reference table. |

The combined training set currently contains 157 records across the 80 known food
IDs. This is a training-pipeline expansion, not evidence that the model has
already generalized to unseen Korean meal language.

Weak labels alone do not produce a language-model probability. FoodMap emits a
`model_probability` only for a food family with at least three directly
human-labeled phrases. For the other current foods, the system uses the explicit
population prior rather than falsely reporting near-certain machine-learning
scores derived from duplicated reference annotations.

## General Probability Model

For every food-component connection, FoodMap stores a Beta prior:

```text
component probability = alpha / (alpha + beta)
```

`confirmed` and `possible` reference annotations are mapped to different,
explicit prior strengths. A manual prior remains authoritative for foods such as
generic sandwiches where food-level annotations do not adequately represent
recipe variability.

The collection queue and posterior updater are now implemented in
`data/v0.3/high_uncertainty_recipe_collection_queue.csv` and
`src/foodmap_parser/recipe_observations.py`. When independently sourced recipe
observations are reviewed, the model can update the prior without replacing its
provenance:

```text
alpha_posterior = alpha_prior + weighted recipes containing the component
beta_posterior  = beta_prior  + weighted recipes not containing the component
posterior_mean  = alpha_posterior / (alpha_posterior + beta_posterior)
```

This is preferable to a raw frequency because a new or rare food does not jump
to 0% or 100% after one recipe. The prior weakens naturally as more independent
recipe observations accumulate.

## Required Recipe-Observation Import Format

External recipe data should be normalized into records like this before it can
update General FoodMap:

```json
{
  "recipe_observation_id": "RECIPE001",
  "food_id": "F0018",
  "recipe_variant": "ham_cheese_sandwich",
  "component_ids": ["C033", "C060", "C067", "C059"],
  "source_id": "SRC_RECIPE_PROVIDER_VERSION",
  "source_url": "https://source.example/recipe/123",
  "region": "KR",
  "retrieved_at": "2026-08-30",
  "quality_weight": 1.0
}
```

Only records with a stable source ID, a source URL or locator, and a documented
recipe definition should update the posterior. LLM-generated recipes may create
candidate training phrases, but they must not be counted as population recipe
observations.

## Next Data Collection Priority

1. Import independently sourced Korean recipe or product-ingredient records for
   high-frequency ambiguous food families.
2. Label unseen free-text meal phrases separately from all training sources.
3. Use active learning to prioritize food mentions where the prior and text
   model disagree or both have low confidence.
4. Compare the prior-only model, text model, and later retrieval-augmented LLM
   on the frozen human-labeled test set.
