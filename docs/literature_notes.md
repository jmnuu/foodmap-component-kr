# FoodMap Literature Notes

## Core Reading Map

| Research need | Reference | Design implication for FoodMap |
| --- | --- | --- |
| Personal variability in food response | Zeevi et al., *Personalized Nutrition by Prediction of Glycemic Responses* (Cell, 2015) | The final model should adapt population patterns to individual history rather than assume one response for everyone. |
| Wearable digital biomarkers | Daniore et al., *From wearable sensor data to digital biomarker development* (npj Digital Medicine, 2024) | Wearable fields require provenance, validation, and a pre-specified interpretation. |
| Consumer wearable validity | Miller, Sargent, and Roach, *A Validation of Six Wearable Devices* (Sensors, 2022) | Consumer-device signals are measured proxies; FoodMap should preserve device metadata and avoid clinical claims. |
| Food ontology | Dooley et al., *FoodOn* (npj Science of Food, 2018) | Canonical food IDs, aliases, ingredients, and processing should be stored separately. |
| Food entity extraction | Popovski et al., *FoodBase corpus* (Database, 2019) | Parser evaluation needs annotated examples and a distinction between surface text and canonical food IDs. |
| Food NER robustness | *Revisiting named entity recognition in food computing* (Artificial Intelligence Review, 2024) | Evaluate ambiguity, unseen wording, and robustness instead of only closed-set accuracy. |
| LLM food linking | *Zero-shot evaluation of ChatGPT for food named-entity recognition and linking* (Frontiers in Nutrition, 2024) | Use LLMs as a measurable parser component or fallback, not as untraceable biomedical evidence generation. |

## Working Interpretation

The literature supports a staged program: normalize meal language, connect it to
structured food data, align it with wearable signals, study population-level
associations, then personalize as repeated user observations accumulate. It does
not justify treating synthetic data or consumer wearable outputs as clinical
ground truth.
