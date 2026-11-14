"""Apply auditable human review decisions to recipe observations."""

import argparse
import csv
import json
from datetime import date
from pathlib import Path

try:
    from .recipe_observations import load_recipe_observations, validate_recipe_observation
    from .label_validation import validate_review
except ImportError:  # Allows direct module execution during early local use.
    from recipe_observations import load_recipe_observations, validate_recipe_observation
    from label_validation import validate_review


VALID_DECISIONS = {"approved", "pending_review", "rejected"}
REQUIRED_REVIEW_FIELDS = {"recipe_observation_id", "decision"}


def load_review_decisions(path):
    """Load review CSV without allowing an approval to be anonymous."""
    decisions = {}
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_REVIEW_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            observation_id = row["recipe_observation_id"].strip()
            decision = (row.get("decision") or "pending_review").strip()
            if not observation_id:
                raise ValueError(f"{path}:{line_number}: recipe_observation_id is required")
            if observation_id in decisions:
                raise ValueError(f"{path}:{line_number}: duplicate observation ID {observation_id}")
            if decision not in VALID_DECISIONS:
                raise ValueError(f"{path}:{line_number}: invalid decision {decision}")

            reviewer = (row.get("reviewer") or "").strip()
            reviewed_at = (row.get("reviewed_at") or "").strip()
            rationale = (row.get("rationale") or "").strip()
            if decision in {"approved", "rejected"}:
                if not reviewer or not reviewed_at or not rationale:
                    raise ValueError(
                        f"{path}:{line_number}: {decision} requires reviewer, reviewed_at, and rationale"
                    )
                try:
                    date.fromisoformat(reviewed_at)
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line_number}: reviewed_at must be YYYY-MM-DD"
                    ) from error

            decisions[observation_id] = {
                "decision": decision,
                "reviewer": reviewer or None,
                "reviewed_at": reviewed_at or None,
                "rationale": rationale or None,
            }
            for field in ("component_ids", "absent_component_ids"):
                if (row.get(field) or "").strip():
                    decisions[observation_id][field] = [v.strip() for v in row[field].split("|")]
    return decisions


def apply_review_decisions(observations, decisions):
    """Return a reviewed copy while preserving the unmodified source records."""
    observation_ids = {record["recipe_observation_id"] for record in observations}
    if len(observation_ids) != len(observations):
        raise ValueError("duplicate recipe observation ID")
    unknown_ids = sorted(set(decisions) - observation_ids)
    if unknown_ids:
        raise ValueError("unknown observation IDs: " + ", ".join(unknown_ids))

    reviewed_records = []
    for record in observations:
        review = decisions.get(record["recipe_observation_id"])
        reviewed = dict(record)
        if review is None:
            reviewed_records.append(reviewed)
            continue
        if review["decision"] not in VALID_DECISIONS:
            raise ValueError("invalid review decision")
        if review["decision"] != "pending_review":
            validate_review(review, allow_date=True)
        reviewed["review_status"] = review["decision"]
        if review["decision"] != "pending_review":
            reviewed["review_metadata"] = {
                "reviewer": review["reviewer"],
                "reviewed_at": review["reviewed_at"],
                "rationale": review["rationale"],
            }
        for field in ("component_ids", "absent_component_ids"):
            if field in review:
                reviewed[field] = list(review[field])
        errors = validate_recipe_observation(reviewed)
        if errors:
            raise ValueError("; ".join(errors))
        reviewed_records.append(reviewed)
    return reviewed_records


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def main():
    argument_parser = argparse.ArgumentParser(
        description="Create a reviewed FoodMap recipe-observation JSONL file."
    )
    argument_parser.add_argument("--observations", type=Path, required=True)
    argument_parser.add_argument("--reviews", type=Path, required=True)
    argument_parser.add_argument("--output", type=Path, required=True)
    args = argument_parser.parse_args()

    observations = load_recipe_observations(args.observations)
    decisions = load_review_decisions(args.reviews)
    records = apply_review_decisions(observations, decisions)
    write_jsonl(args.output, records)

    counts = {decision: sum(record["review_status"] == decision for record in records)
              for decision in sorted(VALID_DECISIONS)}
    print(f"review_counts={counts}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
