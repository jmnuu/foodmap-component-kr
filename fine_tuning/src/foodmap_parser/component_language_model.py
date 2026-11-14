"""A trainable Korean text model for representative food-component inference.

This dependency-free multi-label classifier is deliberately small enough to make
its evidence, labels, and probabilities inspectable. It is a machine-learning
baseline, not a pretrained LLM. Its role is to learn text-to-component
associations that can later be compared with a retrieval-augmented LLM reranker.
"""

import json
import hashlib
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

try:
    from .reference_data import load_reference_csv, source_id_for_annotation
    from .label_validation import validate_training_record
except ImportError:  # Allows direct module execution during early local use.
    from reference_data import load_reference_csv, source_id_for_annotation
    from label_validation import validate_training_record


ROOT = Path(__file__).resolve().parents[2]
TRAINING_PATH = ROOT / "data" / "v0.3" / "component_language_model_training.jsonl"
ACTIVE_LEARNING_TRAINING_PATH = ROOT / "data" / "processed" / "active_learning_approved.jsonl"
MODEL_NAME = "korean_char_ngram_multilabel_naive_bayes"
MODEL_VERSION = "v0.4-masked-naive-bayes"
TRAINING_SOURCE_ID = "SRC_COMPONENT_LANGUAGE_TRAINING_V03"


def _sigmoid(value):
    if value >= 35:
        return 1.0
    if value <= -35:
        return 0.0
    return 1 / (1 + math.exp(-value))


def text_features(text, food_id):
    """Create Korean token and character-ngram features without external models."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    features = {f"food:{food_id}"}

    for token in re.findall(r"[가-힣a-z0-9]+", normalized):
        features.add(f"word:{token}")
        for size in (2, 3, 4):
            for start in range(max(0, len(token) - size + 1)):
                ngram = token[start:start + size]
                if len(ngram) == size:
                    features.add(f"char:{ngram}")

    return features


def load_training_records(path=TRAINING_PATH, active_learning_path=ACTIVE_LEARNING_TRAINING_PATH):
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("text") or not record.get("food_id") or not record.get("components"):
                raise ValueError(f"{path}:{line_number} requires text, food_id, and components")
            record.setdefault("training_source_ids", [TRAINING_SOURCE_ID])
            records.append(record)
    if active_learning_path is not None and Path(active_learning_path).exists():
        records.extend(_load_approved_active_learning_records(active_learning_path))
    return records + load_weak_reference_records()


def _load_approved_active_learning_records(path):
    """Load only records produced by the review exporter, never raw queue rows."""
    records = []
    seen = set()
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_training_record(record)
            if record["id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate training ID")
            seen.add(record["id"])
            records.append(record)
    return records


def load_weak_reference_records():
    """Convert reviewed reference links into weak labels for all known foods.

    These records expand language coverage but remain distinct from human-written
    gold labels. They are used for training only, never for final evaluation.
    """
    foods = load_reference_csv("food_items.csv")
    aliases = load_reference_csv("food_aliases.csv")
    annotations = load_reference_csv("food_component_annotations.csv")

    components_by_food = defaultdict(list)
    source_ids_by_food = defaultdict(set)
    for annotation in annotations:
        if annotation["status"] not in {"confirmed", "possible"}:
            continue
        role = "typical" if annotation["status"] == "confirmed" else "variant"
        components_by_food[annotation["food_id"]].append({
            "component_id": annotation["component_id"],
            "role": role,
        })
        source_ids_by_food[annotation["food_id"]].add(
            source_id_for_annotation(annotation)
        )

    names_by_food = defaultdict(set)
    for food in foods:
        names_by_food[food["food_id"]].add(food["normalized_food_ko"])
    for alias in aliases:
        names_by_food[alias["food_id"]].add(alias["alias"])

    records = []
    for food_id, names in names_by_food.items():
        components = components_by_food.get(food_id, [])
        if not components:
            continue
        for index, name in enumerate(sorted(names)):
            records.append({
                "id": f"WEAK:{food_id}:{index:03d}",
                "text": f"점심에 {name} 먹었어.",
                "food_id": food_id,
                "components": components,
                "training_source_ids": sorted(source_ids_by_food[food_id]),
                "label_origin": "weak_reference_annotation",
            })
    return records


class ComponentLanguageModel:
    """Multi-label Naive Bayes with a probability per component candidate."""

    def __init__(self):
        self.document_count_by_food = Counter()
        self.feature_document_counts = defaultdict(Counter)
        self.label_document_counts = Counter()
        self.label_feature_counts = defaultdict(Counter)
        self.label_role_counts = defaultdict(Counter)
        self.labels_by_food = defaultdict(set)
        self.training_source_ids = set()
        self.human_document_count_by_food = Counter()
        self.known_counts = Counter()
        self.known_feature_counts = defaultdict(Counter)

    def fit(self, records):
        self.__init__()
        records = list(records)
        self.training_data_fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        seen = set()
        for record in records:
            if record.get("id") in seen:
                raise ValueError("duplicate training ID")
            seen.add(record.get("id"))
            food_id = record["food_id"]
            self.document_count_by_food[food_id] += 1
            self.training_source_ids.update(record.get("training_source_ids", []))
            if (
                TRAINING_SOURCE_ID in record.get("training_source_ids", [])
                or record.get("label_origin") == "active_learning_human_review"
            ):
                self.human_document_count_by_food[food_id] += 1
            features = text_features(record["text"], food_id)
            for feature in features:
                self.feature_document_counts[food_id][feature] += 1

            if record.get("label_origin") == "active_learning_human_review":
                validate_training_record(record)
                components = record["component_labels"]
            else:
                # Legacy recipe-dependent labels are priors, not observed truth.
                # Only a legacy explicit mention supports presence; omissions
                # never support absence. Weak dictionary templates are masked.
                components = [
                    {**c, "state": "present" if c["role"] == "explicit"
                     and record.get("label_origin") != "weak_reference_annotation"
                     else "unknown"}
                    for c in record["components"]
                ]
            for component in components:
                component_id = component["component_id"]
                key = (food_id, component_id)
                self.labels_by_food[food_id].add(component_id)
                if component["state"] == "unknown":
                    continue
                self.known_counts[key] += 1
                for feature in features:
                    self.known_feature_counts[key][feature] += 1
                if component["state"] == "present":
                    self.label_role_counts[key][component["role"]] += 1
                    self.label_document_counts[key] += 1
                    for feature in features:
                        self.label_feature_counts[key][feature] += 1

        if not self.document_count_by_food:
            raise ValueError("Training records must not be empty")
        return self

    def predict(self, text, food_id):
        """Return model scores for reviewed component candidates of one food."""
        # Weak labels expand known vocabulary, but they are not independent
        # language evidence. Do not emit a learned probability until this food
        # family has enough directly human-labeled phrases to support one.
        if self.human_document_count_by_food[food_id] < 3:
            return []

        features = text_features(text, food_id)
        predictions = []
        labels = sorted(self.labels_by_food.get(food_id, ()))

        for component_id in labels:
            key = (food_id, component_id)
            document_count = self.known_counts[key]
            positive_documents = self.label_document_counts[key]
            negative_documents = document_count - positive_documents
            if document_count < 3 or not positive_documents or not negative_documents:
                continue
            role_counts = self.label_role_counts[key]
            role = max(role_counts, key=role_counts.get)

            score = math.log((positive_documents + 1) / (negative_documents + 1))

            # Compare feature likelihood under documents with and without this
            # component. Normalizing prevents long Korean sentences dominating.
            for feature in features:
                positive_feature_count = self.label_feature_counts[key][feature]
                total_feature_count = self.known_feature_counts[key][feature]
                negative_feature_count = total_feature_count - positive_feature_count
                positive_probability = (positive_feature_count + 1) / (positive_documents + 2)
                negative_probability = (negative_feature_count + 1) / (negative_documents + 2)
                score += math.log(positive_probability / negative_probability)

            score /= max(1.0, math.sqrt(len(features)))
            predictions.append({
                "component_id": component_id,
                "probability": round(_sigmoid(score), 4),
                "probability_kind": "uncalibrated_model_score",
                "known_label_count": document_count,
                "training_data_fingerprint": self.training_data_fingerprint,
                "predicted_role": role,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "training_source_id": TRAINING_SOURCE_ID,
                "training_source_ids": sorted(self.training_source_ids),
            })

        return sorted(predictions, key=lambda item: (-item["probability"], item["component_id"]))


@lru_cache(maxsize=1)
def load_seed_model():
    """Train once per process and cache the small, versioned seed model."""
    return ComponentLanguageModel().fit(load_training_records())


def predict_seed_components(text, food_id):
    return load_seed_model().predict(text, food_id)
