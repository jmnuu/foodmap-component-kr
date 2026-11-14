"""Grounded representative-component inference for ambiguous composite foods.

The module emits a population prior plus text-supported overrides. It deliberately
does not infer nutrient amounts or fabricate a recipe. A future LLM can propose
additional context overrides, but every accepted component must retain a source,
role, confidence, and reviewable prior record.
"""

import csv
import re
from pathlib import Path

try:
    from .component_language_model import predict_seed_components
    from .general_component_priors import load_general_component_priors
    from .reference_data import load_reference_csv
except ImportError:  # Allows direct module execution during early local use.
    from component_language_model import predict_seed_components
    from general_component_priors import load_general_component_priors
    from reference_data import load_reference_csv


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
INFERENCE_VERSION = "v0.4-partial-evidence"

ROLE_RANK = {
    "variant": 0,
    "optional": 1,
    "typical": 2,
    "core": 3,
    "explicit": 4,
}


def _load_component_metadata():
    return {
        row["component_id"]: row
        for row in load_reference_csv("component_vocab.csv")
    }


def _add_or_update_component(
    inferred,
    component_id,
    role,
    confidence,
    method,
    note=None,
    prior_metadata=None,
):
    existing = inferred.get(component_id)
    candidate = {
        "component_id": component_id,
        "role": role,
        "confidence": confidence,
        "inference_method": method,
    }
    if note:
        candidate["note"] = note
    if prior_metadata:
        candidate.update(prior_metadata)

    if existing is None:
        inferred[component_id] = candidate
        return

    if ROLE_RANK[role] > ROLE_RANK[existing["role"]]:
        existing["role"] = role
    existing["confidence"] = max(existing["confidence"], confidence)
    if method not in existing["inference_method"].split("+"):
        existing["inference_method"] = f"{existing['inference_method']}+{method}"
    if note and "note" not in existing:
        existing["note"] = note
    if prior_metadata:
        existing.update({
            key: value for key, value in prior_metadata.items()
            if value is not None
        })


def _append_method(component, method):
    if method not in component["inference_method"].split("+"):
        component["inference_method"] = f"{component['inference_method']}+{method}"


def _mention_context(text, food, window=24):
    """Use nearby language so another meal does not alter this food's inference."""
    start = food.get("context_start", food.get("start", 0))
    end = food.get("context_end", food.get("end", len(text)))
    return text[start:end]


def infer_representative_components(food, text, modifiers=(), component_priors=None):
    """Return role-aware representative components for one recognized food.

    Generic food names use curated population priors. Words in the original
    mention context can promote a variant to `explicit` or remove incompatible
    candidates. This is intentionally a deterministic, auditable baseline that
    can later be compared with an LLM reranker.
    """
    priors = load_general_component_priors() if component_priors is None else component_priors
    prior = priors.get(food["food_id"])
    if prior is None:
        return []

    component_metadata = _load_component_metadata()
    inferred = {}
    for component in prior["components"]:
        _add_or_update_component(
            inferred,
            component["component_id"],
            component["role"],
            float(component.get("posterior_probability", component["confidence"])),
            "recipe_posterior" if "posterior_probability" in component else "population_prior",
            component.get("note"),
            {
                "prior_probability": float(component["confidence"]),
                "prior_alpha": component.get("prior_alpha"),
                "prior_beta": component.get("prior_beta"),
                "prior_kind": component.get("prior_kind"),
                "prior_source_ids": component.get("source_ids", []),
                **{k: component[k] for k in ("posterior_probability", "posterior_alpha", "posterior_beta", "observation_ids") if k in component},
            },
        )

    context = _mention_context(text, food)

    # The nearby phrase preserves Korean compounds such as "햄치즈 샌드위치"
    # without letting a second, distant food alter this mention's components.
    excluded_component_ids = set()
    for override in prior.get("context_overrides", []):
        if not any(keyword in context for keyword in override["keywords"]):
            continue
        if any(re.search(re.escape(keyword) + r"\s*(?:없는|뺀|제외한)", context) for keyword in override["keywords"]):
            excluded_component_ids.add(override["component_id"])
            continue
        excluded_component_ids.update(override.get("exclude_component_ids", []))
        _add_or_update_component(
            inferred,
            override["component_id"],
            override["role"],
            float(override["confidence"]),
            "text_override",
            prior_metadata={
                "prior_source_ids": prior.get("components", [{}])[0].get("source_ids", []),
            },
        )

    if "without_cheese" in modifiers:
        excluded_component_ids.add("C059")

    for component_id in excluded_component_ids:
        inferred.pop(component_id, None)

    # The learned model ranks grounded candidates from Korean text features. It
    # cannot introduce a component outside the reviewed prior candidate space.
    model_predictions = {
        prediction["component_id"]: prediction
        for prediction in predict_seed_components(context, food["food_id"])
    }
    for component_id, component in inferred.items():
        prediction = model_predictions.get(component_id)
        if prediction is None:
            continue
        component["model_probability"] = prediction["probability"]
        component["training_data_fingerprint"] = prediction["training_data_fingerprint"]
        component["model_probability_kind"] = prediction["probability_kind"]
        component["model_name"] = prediction["model_name"]
        component["model_version"] = prediction["model_version"]
        component["training_source_id"] = prediction["training_source_id"]
        component["training_source_ids"] = prediction["training_source_ids"]
        _append_method(component, "language_model")

    results = []
    for component in inferred.values():
        metadata = component_metadata[component["component_id"]]
        results.append({
            **component,
            "component_name_en": metadata["component_name_en"],
            "component_name_ko": metadata["component_name_ko"],
            "source_ids": list(dict.fromkeys([
                *component.get("prior_source_ids", []),
                *component.get("training_source_ids", []),
            ])),
        })

    return sorted(
        results,
        key=lambda item: (
            -ROLE_RANK[item["role"]],
            -item["confidence"],
            item["component_id"],
        ),
    )
