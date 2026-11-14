"""Shared validation for partial, reviewed component labels."""

from datetime import date, datetime

try:
    from .reference_data import load_reference_csv
except ImportError:
    from reference_data import load_reference_csv


STATES = {"present", "absent", "unknown"}
ROLES = {"core", "typical", "optional", "variant", "explicit"}


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def validate_review(review, allow_date=False):
    for field in ("reviewer", "reviewed_at", "rationale"):
        require_text(review.get(field), field)
    if allow_date and len(review["reviewed_at"]) == 10:
        date.fromisoformat(review["reviewed_at"])
        return
    try:
        stamp = datetime.fromisoformat(review["reviewed_at"])
    except ValueError as error:
        raise ValueError("reviewed_at must be an ISO timestamp") from error
    if stamp.utcoffset() is None:
        raise ValueError("reviewed_at requires a timezone")


def validate_labels(food_id, labels):
    if food_id not in {r["food_id"] for r in load_reference_csv("food_items.csv")}:
        raise ValueError(f"unknown food_id: {food_id}")
    known = {r["component_id"] for r in load_reference_csv("component_vocab.csv")}
    if not isinstance(labels, list) or not labels:
        raise ValueError("component_labels must be a non-empty list")
    seen = set()
    for label in labels:
        if not isinstance(label, dict):
            raise ValueError("component label must be an object")
        cid = label.get("component_id")
        if not isinstance(cid, str) or cid not in known or cid in seen:
            raise ValueError(f"unknown or duplicate component_id: {cid}")
        seen.add(cid)
        if label.get("state") not in STATES or label.get("role") not in ROLES:
            raise ValueError("invalid label state or role")
        if label["state"] != "unknown":
            sources = label.get("evidence_source_ids")
            if not isinstance(sources, list) or not sources:
                raise ValueError("known label requires evidence_source_ids")
            for source in sources:
                require_text(source, "evidence_source_id")


def validate_training_record(record):
    if record.get("schema_version") != "v0.4-partial-labels":
        raise ValueError("training record requires v0.4-partial-labels")
    if record.get("label_origin") != "active_learning_human_review":
        raise ValueError("not a human-reviewed training record")
    if record.get("review_status") != "approved" or record.get("split") != "train":
        raise ValueError("only approved train records may be fitted")
    for field in ("id", "text", "review_task_id", "input_record_id"):
        require_text(record.get(field), field)
    sources = record.get("training_source_ids")
    if not isinstance(sources, list) or not sources:
        raise ValueError("training_source_ids are required")
    for source in sources:
        require_text(source, "training_source_id")
    validate_review(record)
    validate_labels(record.get("food_id"), record.get("component_labels"))
