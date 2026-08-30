# Association Extraction Pipeline

## Goal

The pipeline turns Korean meal-log text into relationship-rich data for later
wearable analysis. It does not infer a health effect. Its job is to preserve the
evidence needed to ask a valid association question later.

```text
Meal-log JSONL
  -> MealEvent Parser
  -> event, food mention, and relation rows
  -> unresolved-food review queue
  -> JSONL audit files and Parquet analytical files
```

## Output Tables

### Representative component inference for composite foods

Every recognized food mention can now carry `representative_components`. For an
ambiguous composite food such as a generic sandwich, these are not a claim that
the user ate a fixed recipe. They are population-level candidates with a role:
`core`, `typical`, `optional`, or `variant`. Text such as "햄치즈 샌드위치" can
promote a candidate to `explicit`; incompatible context such as "비건" can
remove meat variants.

The pipeline writes these as `HAS_REPRESENTATIVE_COMPONENT` relations. Their
`evidence_type`, `inference_method`, `inference_version`, confidence, and source
IDs distinguish them from confirmed reference annotations. The current grounded
prior is stored in `data/v0.3/representative_component_priors.json`. It is
combined with a small trainable Korean component language model documented in
`docs/component_language_model.md`. The model ranks reviewed candidates; it is
not an LLM-generated recipe or a nutrient-value estimate.

| Output | Relationship represented | Why it is separate |
| --- | --- | --- |
| `meal_events` | Event -> time expression and consumption state | One sentence can contain several meal events. |
| `meal_event_foods` | Event -> food mention -> quantity/modifier | Food-level facts must not duplicate event metadata. |
| `event_relations` | Event/food mention -> typed relation -> target | Preserves `OCCURS_AT`, `CONTAINS_FOOD`, `HAS_QUANTITY`, `HAS_MODIFIER`, and `HAS_COMPONENT` as queryable edges. |
| `unresolved_food_candidates` | Eating clause -> unrecognized candidate | Expands the ontology from real input without automatic, unsafe registration. |
| `coverage_report` | Input coverage summary | Shows whether the current ontology is keeping up with actual language. |
| `source_ledger` | Input source identifiers and origin | Allows every output relation to point back to a source record without copying private text into every row. |

Each relation includes `source_ids`, its source record, parser version,
confidence, and, when applicable, the original text span. A food-component
relation includes both the meal-log source and the reference annotation source.
This makes a downstream food-body result traceable to the text and
reference-data decision that produced it.

## Why JSONL and Parquet Both Exist

JSONL is the audit layer. It preserves nested values such as modifiers and lets a
researcher inspect exactly what was extracted from an individual meal log.

Parquet is the analysis layer. It is columnar, compressed, and supports fast
DuckDB joins once the project contains many meal events and wearable records.
The pipeline writes both from the same extraction result, avoiding separate
sources of truth.

## Run the Pipeline

```bash
source .venv/bin/activate
PYTHONPATH=src python src/foodmap_parser/association_pipeline.py \
  --input samples/association_pipeline_input.jsonl \
  --output-dir data/processed/association_pipeline
```

The generated `data/processed/` directory is intentionally excluded from Git.
It may later contain sensitive data and can always be regenerated from the
source records and a recorded parser version.

## Ontology Expansion Loop

1. Run the pipeline over newly collected meal logs.
2. Review `unresolved_food_candidates` rather than adding candidates automatically.
3. Decide whether each candidate is a new alias, a new canonical food, or not food.
4. Add the reviewed mapping with provenance to the reference tables.
5. Add representative sentences to the frozen-split corpus and rerun evaluation.

The merged v0.1/v0.3 vocabulary now covers 80 reviewed food families, but it is
still not expected to cover hundreds of real meal logs. Coverage reports and the
review queue make later expansion evidence-driven rather than speculative.
