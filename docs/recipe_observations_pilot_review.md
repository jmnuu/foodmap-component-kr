# Pilot Recipe Observation Review

## Current Status

The pilot file contains 10 public recipe observations across the first five
high-uncertainty foods. Every row is valid JSONL with a source URL, source ID,
retrieval date, author-level independence group, and `pending_review` status.
It has **not** updated any FoodMap probability.

| Food family | FoodMap ID | Pilot observations | Source type | Current status |
| --- | --- | ---: | --- | --- |
| Poke | F0016 | 2 | Public user-authored recipe pages | Pending review |
| Bibimbap | F0031 | 2 | Public user-authored recipe pages | Pending review |
| Hamburger | F0012 | 2 | Public user-authored recipe pages | Pending review |
| Curry | F0050 | 2 | Public user-authored recipe pages | Pending review |
| Bossam | F0047 | 2 | Public user-authored recipe pages | Pending review |

The source URLs are stored row-by-row in
`data/v0.3/recipe_observations_pilot.jsonl`; recipe text and images were not
copied into this repository.

## Components Requiring Review

The following components were observed in the source recipes but are not yet
reviewed candidates for the corresponding food family. They are intentionally
reported rather than automatically added.

| Food family | Candidate components needing a decision |
| --- | --- |
| Poke | tomato (`C039`), lettuce/leafy greens (`C048`) |
| Bibimbap | beef (`C042`), carrot (`C051`), garlic (`C070`), green onion (`C071`), mushroom (`C072`), sesame oil (`C075`), bean sprout (`C084`) |
| Hamburger | onion (`C049`), sauce/spread (`C066`), egg (`C068`) |
| Curry | meat (`C034`), onion (`C049`), carrot (`C051`), processed meat (`C067`) |
| Bossam | onion (`C049`), garlic (`C070`), green onion (`C071`), kimchi (`C076`), soybean paste (`C077`) |

For each candidate, decide one of three actions:

1. Add it as a reviewed `possible` or `confirmed` component for the food family.
2. Keep it out because it is too recipe-specific for the General FoodMap.
3. Collect more independent observations before deciding.

## Approval Rule

Only after row-level review should `review_status` change from `pending_review`
to `approved`. The posterior updater then uses the stated 0.35 weight. These
ten rows all come from one platform, so even after approval they are an initial
pilot signal, not enough evidence to close the collection queue or claim a
population recipe frequency.
