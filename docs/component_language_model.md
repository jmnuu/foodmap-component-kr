# Korean Text Component Language Model

## What Is Implemented

`src/foodmap_parser/component_language_model.py` is a trainable multi-label
Korean text classifier. It learns token and character n-gram associations between
meal phrases and representative component labels. It receives a local text
context and a recognized food ID, then outputs a probability for every reviewed
component candidate of that food.

For example, a generic sandwich should retain high scores for bread and a broad
protein filling, while "햄치즈 샌드위치" should raise the scores of processed
meat and cheese. The parser combines those model probabilities with curated
population priors and explicit text overrides.

This is genuine supervised machine learning, but it is **not a pretrained LLM**.
The project currently has no sufficiently large labeled Korean meal corpus or
local pretrained language model dependency. Calling this seed model an LLM
would overstate what it does.

## Why This Is the Right First Model

| Choice | Why it is used now | Replace or extend when |
| --- | --- | --- |
| Character n-gram multi-label Naive Bayes | Korean compounds such as `햄치즈샌드위치` remain usable without tokenization or external packages; probabilities and learned features are inspectable. | A diverse, independently labeled corpus has hundreds or thousands of examples. |
| Curated candidate prior | Prevents a tiny model from inventing ingredients outside reviewed FoodMap concepts. | Retrieval from a large recipe and food-ontology index becomes available. |
| Explicit-text override | A directly written ingredient is stronger evidence than a population-level prediction. | Keep this rule even with an LLM; explicit mentions remain auditable evidence. |
| Future LLM reranker | Can use broader context and semantic similarity for unseen foods and paraphrases. | Introduce only with a fixed prompt, retrieved evidence, version logging, and independent evaluation. |

## Training Data and Provenance

The initial direct training set is
`data/v0.3/component_language_model_training.jsonl`. It contains 14 human-curated
seed examples of generic and variant sandwich phrases and is registered as
`SRC_COMPONENT_LANGUAGE_TRAINING_V03`. At runtime the model also derives weak
training examples from reviewed food IDs, aliases, and food-component annotations
for the other known foods. Their origin remains either
`SRC_FOODMAP_CURATED_V01` or `SRC_FOODMAP_CURATED_V03_EXPANSION`.

These sources are enough to exercise the training and prediction pipeline. They
are not a performance claim and must never be used as the final evaluation data.
A future study needs a separate human-labeled train, validation, and test split,
with food families rather than only sentence strings separated across splits.

Weak labels by themselves are not enough to emit a `model_probability`. The
current implementation requires at least three direct human-labeled phrases for
a food family; otherwise it returns the audited population prior only. This
prevents repeated copies of the same reference annotation from appearing as a
high-confidence learned result.

## Output and Interpretation

Each representative component can include:

```json
{
  "component_id": "C067",
  "role": "variant",
  "confidence": 0.35,
  "model_probability": 0.81,
  "model_name": "korean_char_ngram_multilabel_naive_bayes",
  "model_version": "v0.3-seed",
  "training_source_id": "SRC_COMPONENT_LANGUAGE_TRAINING_V03"
}
```

`model_probability` is a score from the seed model's learned training
distribution. It is not a probability that the user certainly ate that
ingredient, and it is not a nutritional amount. The association pipeline keeps
the model and training-source fields in every
`HAS_REPRESENTATIVE_COMPONENT` relation so later evaluations can compare this
model against the rule baseline and an LLM reranker.

## Next Research Experiment

1. Label at least several hundred independent Korean meal phrases with food IDs,
   representative components, roles, and `unknown` when evidence is insufficient.
2. Freeze an unseen test set before model tuning. Split by food family to avoid
   near-duplicate variants leaking from train to test.
3. Compare the rule prior, this language model, and a retrieval-augmented LLM on
   component Top-k precision, recall, calibration, and over-assertion rate.
4. Allow automatic use only above a calibrated threshold; send low-confidence or
   out-of-vocabulary predictions to the review queue.
