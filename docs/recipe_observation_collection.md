# High-Uncertainty Recipe Collection

## Decision

FoodMap now prioritizes data quality over immediately adding more canonical food
names. The first collection wave covers the 25 current food families whose
representative components are most uncertain. Its saved review queue is
`data/v0.3/high_uncertainty_recipe_collection_queue.csv`.

The queue allocates 1,050 externally observed recipe or ingredient-list records:

| Queue tier | Foods | Target per food | Minimum independent sources | Purpose |
| --- | ---: | ---: | ---: | --- |
| Ranks 1-5 | 5 | 60 | 5 | Estimate the most variable, high-impact composite foods first. |
| Ranks 6-15 | 10 | 45 | 4 | Stabilize common Korean meal families with meaningful variation. |
| Ranks 16-25 | 10 | 30 | 3 | Establish an initial evidence base, then reassess uncertainty. |

`independent` means different publishers, providers, or recipe authors. Copies
of one recipe across several websites count as one source. Each food should be
sampled across household recipes, restaurant or delivery descriptions, and
commercial or meal-kit ingredient lists when that stratum exists.

## Observation Record

Use [recipe_observations.example.jsonl](../samples/recipe_observations.example.jsonl)
as the exact JSONL shape. Every row requires:

- A stable observation ID and a canonical FoodMap `food_id`.
- The observed component IDs, recipe/product variant, source ID, source URL or locator, retrieval date, and an author/provider-level `independence_group_id`.
- A `quality_weight` from greater than 0 to 1.0 and one observation type: `recipe`, `product_ingredient_list`, or `meal_kit`.

Do not add LLM-generated recipes, inferred ingredients, copied descriptions, or
uncited summaries as observations. They can suggest a search term but cannot
contribute evidence weight. New rows start as `pending_review`; only rows marked
`approved` can update a posterior. This makes the initial external collection
inspectable before it affects the General FoodMap.

## Initial Pilot Collection

`data/v0.3/recipe_observations_pilot.jsonl` contains 10 normalized observations
across the first five queue foods: poke, bibimbap, hamburger, curry, and
bossam. Each row points to its public recipe URL and an author-level
independence group. All rows have a conservative 0.35 weight and
`pending_review` status, so they are **not** yet used by the posterior updater.

This pilot tests whether the component vocabulary can represent actual recipe
variation. It does not satisfy the 1,050-observation target or the required
cross-provider diversity; the next collection wave must use official public
APIs or other independent providers as well.

## Posterior Update

Validate and apply a reviewed JSONL collection with:

```bash
PYTHONPATH=src python3 src/foodmap_parser/recipe_observations.py \
  --input data/private/reviewed_recipe_observations.jsonl \
  --output data/processed/component_posteriors.json
```

For every already reviewed food-component candidate, the module calculates:

```text
posterior_alpha = prior_alpha + sum(weight where component is present)
posterior_beta  = prior_beta  + sum(weight where component is absent)
posterior_probability = posterior_alpha / (posterior_alpha + posterior_beta)
```

An observed component that is known in the vocabulary but not a reviewed
candidate for that food is emitted under `unmodeled_component_ids`. It does not
silently enter the FoodMap. Review it first, then add a cited annotation to the
reference layer. This protects the model from treating a one-off recipe as a
universal component.

## The Other 30 Percent

Do not proactively add a large list of food names. Run the association pipeline
on real meal logs and review `unresolved_food_candidates`. Add an alias when it
maps cleanly to an existing food family; create a new canonical food only when
the expression is frequent and has a distinct component profile. This keeps
dictionary expansion grounded in observed user language.
