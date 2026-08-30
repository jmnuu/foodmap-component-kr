# Food Component Uncertainty Inventory

## Purpose

Not every food needs the same amount of recipe data. A banana is relatively
stable, while sandwiches, pasta, shabu-shabu, stews, and mixed meals can have
many plausible ingredient combinations. The uncertainty inventory ranks every
known FoodMap food so data collection begins where a fixed food-to-component
mapping would be least reliable.

Run it with:

```bash
PYTHONPATH=src python src/foodmap_parser/food_uncertainty.py \
  --output data/processed/food_uncertainty_inventory.json
```

## Score Components

The score is between 0 and 1 and is not a health-risk score.

For the current small ontology, scores of `0.50` or above are collected as
`high` priority. This intentionally includes pasta and other food families with
several plausible variants; the threshold should be recalibrated after external
recipe observations become available.

| Signal | Meaning | Why it matters |
| --- | --- | --- |
| Component entropy | How close component probabilities are to uncertain middle values | A food with many 40-60% candidates cannot be treated as one recipe. |
| Optional/variant ratio | Share of components that are not core or typical | Captures recipe variation directly. |
| Prior evidence deficit | How little curated evidence supports the current probability | Identifies food families that need independent recipe observations. |
| Composite-food category | Whether the food is a mixed meal, stew, or fast food | Gives priority to food types whose ingredients often vary by restaurant, household, or product. |

## How To Use the Output

1. Start recipe-observation collection with `high` uncertainty foods.
2. For each selected food, collect independent recipe or ingredient-list records
   with stable source IDs and normalize their components.
3. Update the Beta prior from these observations, then rerun the inventory.
4. Use clarification questions only when the remaining uncertain candidates are
   both health-relevant and close enough in probability to alter the Top-5.

The current inventory is only as strong as the initial curated priors. It is a
transparent collection queue, not proof that any food has a measured population
ingredient distribution.
