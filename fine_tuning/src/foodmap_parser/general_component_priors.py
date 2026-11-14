"""Build auditable population component priors from FoodMap reference evidence.

The resulting probabilities are prior beliefs, not measured recipe frequencies.
They are designed to be updated later with independently sourced recipe records
using a Beta-Binomial update instead of treating a single food name as a fixed
recipe.
"""

import json
from functools import lru_cache
from pathlib import Path

try:
    from .reference_data import load_reference_csv, source_id_for_annotation
except ImportError:  # Allows direct module execution during early local use.
    from reference_data import load_reference_csv, source_id_for_annotation


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REFERENCE_SOURCE_ID = "SRC_FOODMAP_CURATED_V01"
MANUAL_PRIOR_SOURCE_ID = "SRC_FOODMAP_COMPONENT_PRIORS_V03"


def _load_csv(filename):
    return load_reference_csv(filename)


def _role_for_annotation(annotation):
    if annotation["status"] == "confirmed":
        if annotation["evidence_type"] in {"food_base", "food_structure", "cooking_method"}:
            return "core"
        return "typical"
    return "optional" if float(annotation["confidence"]) >= 0.7 else "variant"


def _beta_prior_for_annotation(annotation):
    """Translate review confidence into a weak, explicit Beta prior.

    These hyperparameters make the current curation useful without claiming that
    it represents an observed recipe sample. Future recipe records add directly
    to alpha (component present) or beta (component absent).
    """
    confidence = float(annotation["confidence"])
    if annotation["status"] == "confirmed":
        alpha = 1.0 + 8.0 * confidence
        beta = 1.0
    else:
        alpha = 1.0 + 3.0 * confidence
        beta = 2.0
    return round(alpha, 3), round(beta, 3)


@lru_cache(maxsize=1)
def load_general_component_priors():
    """Return one reviewed population-prior record per known food ID."""
    vocabulary = {
        row["component_id"]: row
        for row in _load_csv("component_vocab.csv")
    }
    priors = {row["food_id"]: {"components": [], "context_overrides": []} for row in _load_csv("food_items.csv")}

    for annotation in _load_csv("food_component_annotations.csv"):
        if annotation["status"] not in {"confirmed", "possible"}:
            continue
        metadata = vocabulary.get(annotation["component_id"])
        if metadata is None:
            continue
        alpha, beta = _beta_prior_for_annotation(annotation)
        priors.setdefault(annotation["food_id"], {"components": [], "context_overrides": []})["components"].append({
            "component_id": annotation["component_id"],
            "role": _role_for_annotation(annotation),
            "confidence": round(alpha / (alpha + beta), 4),
            "prior_alpha": alpha,
            "prior_beta": beta,
            "prior_kind": "curated_beta_prior",
            "source_ids": [source_id_for_annotation(annotation)],
            "note": annotation["notes"],
        })

    # Hand-reviewed priors replace derived candidates for composite foods where
    # the generic food name has especially high recipe variability.
    with open(DATA_DIR / "v0.3" / "representative_component_priors.json", "r", encoding="utf-8") as file:
        manual_data = json.load(file)
    for manual in manual_data["foods"]:
        components = []
        for component in manual["components"]:
            confidence = float(component["confidence"])
            strength = 8.0 if component["role"] in {"core", "typical"} else 3.0
            alpha = 1.0 + confidence * strength
            beta = 1.0 + (1.0 - confidence) * strength
            components.append({
                **component,
                "curation_confidence": confidence,
                "confidence": round(alpha / (alpha + beta), 4),
                "prior_alpha": round(alpha, 3),
                "prior_beta": round(beta, 3),
                "prior_kind": "manual_composite_food_prior",
                "source_ids": [MANUAL_PRIOR_SOURCE_ID],
            })
        priors[manual["food_id"]] = {
            "components": components,
            "context_overrides": manual.get("context_overrides", []),
        }

    return priors
