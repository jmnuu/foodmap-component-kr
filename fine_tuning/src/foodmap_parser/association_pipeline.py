"""Build association-ready FoodMap datasets from Korean meal-log JSONL input."""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

try:
    from .active_learning import build_active_learning_queue
    from .component_inference import INFERENCE_VERSION
    from .meal_events import parse_meal_events
    from .parser import build_component_lookup, parse_food_log
    from .reference_data import load_reference_csv
except ImportError:
    from active_learning import build_active_learning_queue
    from component_inference import INFERENCE_VERSION
    from meal_events import parse_meal_events
    from parser import build_component_lookup, parse_food_log
    from reference_data import load_reference_csv


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PARSER_VERSION = "v0.2-rule-based"
REFERENCE_SOURCE_ID = "SRC_FOODMAP_CURATED_V01"

MODIFIER_WORDS = {
    "제로", "디카페인", "아이스", "매운", "작은", "큰", "뜨거운", "차가운",
}
NON_FOOD_WORDS = {
    "오늘", "어제", "내일", "아침", "점심", "저녁", "오전", "오후", "밤",
    "야식", "간식", "후식", "식사", "음식", "그냥", "조금", "많이", "하나",
}
CONSUMPTION_PATTERN = re.compile(
    r"(?P<phrase>[^.!?]{0,60}?)(?:먹었|먹음|먹고|먹을|마셨|마심|마시고|마실|주문했)"
)
PARTICLE_SUFFIXES = ("이랑", "으로", "에는", "에서", "에게", "랑", "과", "와", "을", "를", "은", "는", "이", "가", "에", "엔", "만", "도")


def load_jsonl(path):
    records = []
    seen = set()
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record.get("input"), str) or not record["input"].strip():
                raise ValueError(f"{path}:{line_number} requires a non-empty input field")
            record.setdefault("id", f"line-{line_number:06d}")
            if str(record["id"]) in seen:
                raise ValueError(f"{path}:{line_number}: duplicate input ID")
            seen.add(str(record["id"]))
            if record.get("split", "unassigned") not in {"train", "validation", "test", "unassigned"}:
                raise ValueError("unsupported dataset split")
            records.append(record)
    return records


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def load_component_metadata():
    return {
        row["component_name_en"]: row
        for row in load_reference_csv("component_vocab.csv")
    }


def clean_candidate(token):
    token = token.strip()
    for suffix in PARTICLE_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix):
            token = token[:-len(suffix)]
            break
    return token.strip()


def find_unresolved_food_candidates(text, parsed_foods):
    """Find review-only candidates when an eating clause contains no known food.

    This is intentionally conservative. A candidate is not automatically added to
    the ontology because incorrect normalization would corrupt later associations.
    """
    candidates = []
    known_spans = [(food["start"], food["end"]) for food in parsed_foods]

    for match in CONSUMPTION_PATTERN.finditer(text):
        phrase_start = match.start("phrase")
        phrase_end = match.end("phrase")
        phrase = text[phrase_start:phrase_end]
        masked = list(phrase)

        for start, end in known_spans:
            if end <= phrase_start or start >= phrase_end:
                continue
            for index in range(max(start, phrase_start) - phrase_start, min(end, phrase_end) - phrase_start):
                masked[index] = " "

        masked_phrase = "".join(masked)
        for token_match in re.finditer(r"[가-힣A-Za-z]{2,}", masked_phrase):
            candidate = clean_candidate(token_match.group(0))
            if candidate in NON_FOOD_WORDS or candidate in MODIFIER_WORDS or len(candidate) < 2:
                continue
            if any(candidate in word for word in NON_FOOD_WORDS):
                continue

            start = phrase_start + token_match.start()
            candidates.append({
                "candidate_text": candidate,
                "source_start": start,
                "source_end": start + len(candidate),
                "source_text": text,
                "discovery_method": "unrecognized_consumption_clause",
                "review_status": "pending",
                "confidence": 0.2,
            })

    deduplicated = []
    seen = set()
    for candidate in candidates:
        key = (candidate["candidate_text"], candidate["source_start"])
        if key not in seen:
            deduplicated.append(candidate)
            seen.add(key)
    return deduplicated


def build_pipeline(records, component_priors=None):
    component_lookup = build_component_lookup()
    component_metadata = load_component_metadata()
    meal_events = []
    meal_event_foods = []
    relations = []
    unresolved_candidates = []
    component_counts = Counter()
    representative_component_counts = Counter()
    source_ledger = []

    for record in records:
        source_record_id = str(record["id"])
        text = record["input"]
        input_source_id = f"INPUT:{source_record_id}"
        source_ledger.append({
            "source_id": input_source_id,
            "source_type": "meal_log_input",
            "source_record_id": source_record_id,
            "data_origin": record.get("data_origin", "unknown"),
            "recorded_at": record.get("recorded_at"),
            "quality_note": "Original input text is stored in the event table and may require access control.",
        })
        parsed_events = parse_meal_events(text, component_priors=component_priors)["events"]
        parsed_foods = parse_food_log(text)["foods"]

        for candidate_index, candidate in enumerate(find_unresolved_food_candidates(text, parsed_foods)):
            candidate["candidate_id"] = f"{source_record_id}:U{candidate_index:02d}"
            candidate["source_record_id"] = source_record_id
            candidate["source_ids"] = [input_source_id]
            candidate["split"] = record.get("split", "unassigned")
            unresolved_candidates.append(candidate)

        for event in parsed_events:
            event_id = f"{source_record_id}:E{event['event_index']:02d}"
            event_row = {
                "event_id": event_id,
                "source_record_id": source_record_id,
                "source_text": text,
                "user_id": record.get("user_id"),
                "recorded_at": record.get("recorded_at"),
                "data_origin": record.get("data_origin", "unknown"),
                "split": record.get("split", "unassigned"),
                "source_ids": [input_source_id],
                "time_expression": event["time_expression"],
                "meal_time": event["meal_time"],
                "consumption_status": event["consumption_status"],
                "parser_version": PARSER_VERSION,
            }
            meal_events.append(event_row)

            relations.append({
                "relation_id": f"{event_id}:R00",
                "subject_type": "meal_event",
                "subject_id": event_id,
                "relation_type": "OCCURS_AT",
                "object_type": "time_expression",
                "object_id": event["meal_time"],
                "object_value": event["time_expression"],
                "evidence_type": "observed_text",
                "confidence": 1.0,
                "source_record_id": source_record_id,
                "source_ids": [input_source_id],
                "parser_version": PARSER_VERSION,
            })

            for food_position, food in enumerate(event["foods"]):
                food_mention_id = f"{event_id}:F{food_position:02d}"
                food_row = {
                    "food_mention_id": food_mention_id,
                    "event_id": event_id,
                    "food_position": food_position,
                    "food_id": food["food_id"],
                    "raw_text": food["raw_text"],
                    "source_start": food["source_start"],
                    "source_end": food["source_end"],
                    "quantity": food["quantity"],
                    "unit": food["unit"],
                    "modifiers": food["modifiers"],
                    "mention_context": food["mention_context"],
                    "representative_components": food["representative_components"],
                    "parser_version": PARSER_VERSION,
                    "source_ids": [input_source_id],
                }
                meal_event_foods.append(food_row)

                base_relation = {
                    "subject_type": "meal_event",
                    "subject_id": event_id,
                    "relation_type": "CONTAINS_FOOD",
                    "object_type": "food_mention",
                    "object_id": food_mention_id,
                    "object_value": food["food_id"],
                    "evidence_type": "observed_text",
                    "confidence": 1.0,
                    "source_record_id": source_record_id,
                    "source_start": food["source_start"],
                    "source_end": food["source_end"],
                    "parser_version": PARSER_VERSION,
                    "source_ids": [input_source_id],
                }
                base_relation["relation_id"] = f"{food_mention_id}:R00"
                relations.append(base_relation)

                if food["quantity"] is not None or food["unit"] is not None:
                    relations.append({
                        "relation_id": f"{food_mention_id}:R01",
                        "subject_type": "food_mention",
                        "subject_id": food_mention_id,
                        "relation_type": "HAS_QUANTITY",
                        "object_type": "quantity",
                        "object_id": None,
                        "object_value": {
                            "quantity": food["quantity"],
                            "unit": food["unit"],
                        },
                        "evidence_type": "observed_text",
                        "confidence": 1.0,
                        "source_record_id": source_record_id,
                        "source_start": food["source_start"],
                        "source_end": food["source_end"],
                        "parser_version": PARSER_VERSION,
                        "source_ids": [input_source_id],
                    })

                for modifier_index, modifier in enumerate(food["modifiers"]):
                    relations.append({
                        "relation_id": f"{food_mention_id}:M{modifier_index:02d}",
                        "subject_type": "food_mention",
                        "subject_id": food_mention_id,
                        "relation_type": "HAS_MODIFIER",
                        "object_type": "modifier",
                        "object_id": modifier,
                        "object_value": modifier,
                        "evidence_type": "observed_text",
                        "confidence": 1.0,
                        "source_record_id": source_record_id,
                        "source_start": food["source_start"],
                        "source_end": food["source_end"],
                        "parser_version": PARSER_VERSION,
                        "source_ids": [input_source_id],
                    })

                for component in component_lookup.get(food["food_id"], []):
                    metadata = component_metadata[component["component"]]
                    component_counts[metadata["component_id"]] += 1
                    relations.append({
                        "relation_id": f"{food_mention_id}:{metadata['component_id']}",
                        "subject_type": "food_mention",
                        "subject_id": food_mention_id,
                        "relation_type": "HAS_COMPONENT",
                        "object_type": "component",
                        "object_id": metadata["component_id"],
                        "object_value": component["component"],
                        "component_status": component["status"],
                        "evidence_type": "reference_annotation",
                        "reference_record_id": f"{food['food_id']}:{metadata['component_id']}",
                        "confidence": component["confidence"],
                        "source_record_id": source_record_id,
                        "source_ids": [input_source_id, *component["source_ids"]],
                        "source_start": food["source_start"],
                        "source_end": food["source_end"],
                        "parser_version": PARSER_VERSION,
                    })

                for component in food["representative_components"]:
                    representative_component_counts[component["component_id"]] += 1
                    relations.append({
                        "relation_id": f"{food_mention_id}:P{component['component_id']}",
                        "subject_type": "food_mention",
                        "subject_id": food_mention_id,
                        "relation_type": "HAS_REPRESENTATIVE_COMPONENT",
                        "object_type": "component",
                        "object_id": component["component_id"],
                        "object_value": component["component_name_en"],
                        "component_role": component["role"],
                        "evidence_type": "text_supported_component_inference" if "text_override" in component["inference_method"] else
                        "recipe_component_posterior" if "recipe_posterior" in component["inference_method"] else "population_component_prior",
                        "confidence": component["confidence"],
                        "source_record_id": source_record_id,
                        "source_ids": [input_source_id, *component["source_ids"]],
                        "source_start": food["source_start"],
                        "source_end": food["source_end"],
                        "parser_version": PARSER_VERSION,
                        "inference_version": INFERENCE_VERSION,
                        "inference_method": component["inference_method"],
                        "model_probability": component.get("model_probability"),
                        "model_probability_kind": component.get("model_probability_kind"),
                        "training_data_fingerprint": component.get("training_data_fingerprint"),
                        "prior_probability": component.get("prior_probability"),
                        "posterior_probability": component.get("posterior_probability"),
                        "observation_ids": component.get("observation_ids", []),
                        "model_name": component.get("model_name"),
                        "model_version": component.get("model_version"),
                        "training_source_id": component.get("training_source_id"),
                    })

    result = {
        "meal_events": meal_events,
        "meal_event_foods": meal_event_foods,
        "event_relations": relations,
        "unresolved_food_candidates": unresolved_candidates,
        "source_ledger": source_ledger,
    }
    active_learning_queue = build_active_learning_queue(result)
    coverage_report = {
        "input_records": len(records),
        "meal_events": len(meal_events),
        "food_mentions": len(meal_event_foods),
        "relation_rows": len(relations),
        "unresolved_food_candidates": len(unresolved_candidates),
        "active_learning_review_tasks": len(active_learning_queue),
        "unique_food_ids_observed": len({row["food_id"] for row in meal_event_foods}),
        "unique_components_linked": len(component_counts),
        "component_mention_counts": dict(sorted(component_counts.items())),
        "representative_component_relation_count": sum(representative_component_counts.values()),
        "representative_component_mention_counts": dict(sorted(representative_component_counts.items())),
        "source_record_count": len(source_ledger),
        "parser_version": PARSER_VERSION,
    }

    return {**result, "active_learning_queue": active_learning_queue, "coverage_report": coverage_report}


def materialize_parquet(output_dir, output_paths):
    try:
        import duckdb
    except ImportError as error:
        raise RuntimeError(
            "DuckDB is required for Parquet output. Install it with `pip install duckdb` "
            "or run with --skip-parquet."
        ) from error

    connection = duckdb.connect()
    for name in ("meal_events", "meal_event_foods", "event_relations", "unresolved_food_candidates", "active_learning_queue", "source_ledger"):
        source_path = output_paths[name]
        if source_path.stat().st_size == 0:
            continue
        target_path = output_dir / f"{name}.parquet"
        source_sql_path = str(source_path).replace("'", "''")
        target_sql_path = str(target_path).replace("'", "''")
        connection.execute(
            f"COPY (SELECT * FROM read_json_auto('{source_sql_path}')) "
            f"TO '{target_sql_path}' (FORMAT PARQUET)"
        )
    connection.close()


def run_pipeline(input_path, output_dir, write_parquet=True, posterior_path=None):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory is not empty; use a new run directory")
    records = load_jsonl(input_path)
    component_priors = None
    if posterior_path is not None:
        try:
            from .recipe_observations import load_posterior_snapshot
        except ImportError:
            from recipe_observations import load_posterior_snapshot
        component_priors = load_posterior_snapshot(posterior_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_pipeline(records, component_priors=component_priors)
    output_paths = {}

    for name in ("meal_events", "meal_event_foods", "event_relations", "unresolved_food_candidates", "active_learning_queue", "source_ledger"):
        output_path = output_dir / f"{name}.jsonl"
        write_jsonl(output_path, result[name])
        output_paths[name] = output_path

    report_path = output_dir / "coverage_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(result["coverage_report"], file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")

    if write_parquet:
        materialize_parquet(output_dir, output_paths)

    return result


def main():
    argument_parser = argparse.ArgumentParser(
        description="Create association-ready FoodMap tables from meal-log JSONL input."
    )
    argument_parser.add_argument("--input", type=Path, required=True)
    argument_parser.add_argument("--component-posteriors", type=Path, help="Validated v0.4 recipe posterior snapshot")
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "processed" / "association_pipeline",
    )
    argument_parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Write auditable JSONL only when DuckDB is unavailable.",
    )
    args = argument_parser.parse_args()

    result = run_pipeline(args.input, args.output_dir, write_parquet=not args.skip_parquet, posterior_path=args.component_posteriors)
    print("FoodMap Association Pipeline")
    print("============================")
    for key, value in result["coverage_report"].items():
        if key != "component_mention_counts":
            print(f"{key}: {value}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
