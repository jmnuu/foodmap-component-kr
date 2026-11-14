"""Create an evidence-aware synthetic FoodMap and personalize it from observations."""

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

try:
    from .association_pipeline import build_pipeline, write_jsonl
except ImportError:
    from association_pipeline import build_pipeline, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PRIORS_PATH = ROOT / "data" / "v0.3" / "population_foodmap_priors.json"
SYNTHETIC_GENERATOR_SOURCE_ID = "SRC_SYNTHETIC_GENERATOR_V03"

MEAL_OPTIONS = {
    "breakfast": [
        ("아아 한 잔", "아침 8시에"),
        ("라떼 한 잔", "아침 8시에"),
        ("그릭요거트랑 바나나", "아침에"),
        ("쌀밥이랑 김치찌개", "아침에"),
    ],
    "lunch": [
        ("김밥이랑 콜라", "점심에"),
        ("비빔밥", "점심에"),
        ("쌀국수", "점심에"),
        ("샐러드랑 포케", "점심에"),
        ("떡볶이랑 김밥", "점심에"),
    ],
    "dinner": [
        ("치킨이랑 콜라", "저녁 7시에"),
        ("파스타", "저녁에"),
        ("김치찌개랑 쌀밥", "저녁에"),
        ("햄버거", "저녁에"),
        ("피자 두 조각", "저녁에"),
    ],
    "night": [
        ("아이스 아메리카노 한 잔", "밤 10시에"),
        ("제로 콜라 작은 캔 하나", "밤 11시에"),
        ("컵라면 하나", "야식으로"),
        ("탕후루 하나", "늦은 밤에"),
    ],
}


def load_priors(path=PRIORS_PATH):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_profiles(count, rng):
    profiles = []
    chronotypes = ["early", "intermediate", "late"]

    for index in range(1, count + 1):
        profile = {
            "user_id": f"SYN{index:03d}",
            "data_origin": "synthetic",
            "chronotype": rng.choices(chronotypes, weights=[0.25, 0.5, 0.25])[0],
            "caffeine_preference": rng.choice(["low", "moderate", "high"]),
            "sensitivity_multipliers": {},
        }
        profiles.append(profile)

    return profiles


def add_sensitivity_multipliers(profiles, priors, rng):
    for profile in profiles:
        for prior in priors:
            profile["sensitivity_multipliers"][prior["prior_id"]] = round(
                rng.uniform(0.5, 1.5), 3
            )


def choose_daily_meal_slots(profile, rng):
    slots = ["breakfast", "lunch", "dinner"]
    late_weight = 0.55 if profile["chronotype"] == "late" else 0.2
    if rng.random() < late_weight:
        slots.append("night")
    return slots


def generate_meal_logs(profiles, days, start_date, rng):
    logs = []
    record_index = 1

    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        for profile in profiles:
            for meal_slot in choose_daily_meal_slots(profile, rng):
                food_text, time_text = rng.choice(MEAL_OPTIONS[meal_slot])
                input_text = f"{time_text} {food_text} 먹었어."
                hour_by_slot = {
                    "breakfast": 8,
                    "lunch": 12,
                    "dinner": 19,
                    "night": 22,
                }
                recorded_at = datetime.combine(
                    current_date,
                    time(hour=hour_by_slot[meal_slot], minute=rng.choice([0, 15, 30, 45])),
                )
                logs.append({
                    "id": f"SLOG{record_index:06d}",
                    "user_id": profile["user_id"],
                    "recorded_at": recorded_at.isoformat() + "+09:00",
                    "data_origin": "synthetic",
                    "input": input_text,
                })
                record_index += 1

    return logs


def event_components(pipeline_result):
    components_by_event = defaultdict(set)
    event_by_food_mention = {
        food["food_mention_id"]: food["event_id"]
        for food in pipeline_result["meal_event_foods"]
    }

    for relation in pipeline_result["event_relations"]:
        if relation["relation_type"] != "HAS_COMPONENT":
            continue
        event_id = event_by_food_mention[relation["subject_id"]]
        components_by_event[event_id].add(relation["object_id"])

    return components_by_event


def prior_matches_event(prior, event, component_ids):
    trigger = prior["trigger"]
    return (
        event["meal_time"] in trigger["meal_times"]
        and bool(set(trigger["component_ids_any"]) & component_ids)
    )


def generate_response_observations(pipeline_result, profiles_by_id, priors, rng):
    observations = []
    components_by_event = event_components(pipeline_result)
    outcomes_by_id = {
        outcome["outcome_id"]: outcome
        for outcome in load_priors()["outcomes"]
    }

    for event in pipeline_result["meal_events"]:
        user_id = event.get("user_id")
        if user_id not in profiles_by_id:
            continue
        profile = profiles_by_id[user_id]

        for prior in priors:
            if not prior_matches_event(prior, event, components_by_event[event["event_id"]]):
                continue
            sensitivity = profile["sensitivity_multipliers"][prior["prior_id"]]
            expected_score = prior["population_prior_mean"] * sensitivity
            observed_score = expected_score + rng.gauss(0, 0.2)
            observations.append({
                "observation_id": f"{event['event_id']}:{prior['prior_id']}",
                "event_id": event["event_id"],
                "user_id": user_id,
                "prior_id": prior["prior_id"],
                "outcome_id": prior["outcome_id"],
                "required_instrument": outcomes_by_id[prior["outcome_id"]]["required_instrument"],
                "expected_effect_score": round(expected_score, 4),
                "observed_response_score": round(observed_score, 4),
                "data_origin": "synthetic",
                "evidence_prior_id": prior["prior_id"],
                "source_ids": [
                    f"INPUT:{event['source_record_id']}",
                    prior["evidence"]["source_id"],
                    SYNTHETIC_GENERATOR_SOURCE_ID,
                ],
            })

    return observations


def build_personal_foodmaps(profiles, priors, observations):
    """Use a simple empirical-Bayes update to avoid overreacting to few records."""
    observed_scores = defaultdict(list)
    observed_source_ids = defaultdict(set)
    for observation in observations:
        key = (observation["user_id"], observation["prior_id"])
        observed_scores[key].append(
            observation["observed_response_score"]
        )
        observed_source_ids[key].update(observation["source_ids"])

    foodmaps = []
    for profile in profiles:
        for prior in priors:
            key = (profile["user_id"], prior["prior_id"])
            values = observed_scores[key]
            source_ids = observed_source_ids[key]
            prior_weight = prior["prior_strength"]
            posterior_mean = (
                prior_weight * prior["population_prior_mean"] + sum(values)
            ) / (prior_weight + len(values))
            foodmaps.append({
                "user_id": profile["user_id"],
                "prior_id": prior["prior_id"],
                "outcome_id": prior["outcome_id"],
                "population_prior_mean": prior["population_prior_mean"],
                "individual_effect_estimate": round(posterior_mean, 4),
                "observation_count": len(values),
                "confidence_weight": prior_weight + len(values),
                "update_method": "empirical_bayes_shrinkage",
                "data_origin": "synthetic",
                "source_ids": sorted(source_ids | {
                    prior["evidence"]["source_id"],
                    SYNTHETIC_GENERATOR_SOURCE_ID,
                }),
            })
    return foodmaps


def generate_synthetic_foodmap(output_dir, profile_count=30, days=14, seed=42, write_parquet=False):
    rng = random.Random(seed)
    prior_document = load_priors()
    priors = prior_document["priors"]
    profiles = generate_profiles(profile_count, rng)
    add_sensitivity_multipliers(profiles, priors, rng)
    meal_logs = generate_meal_logs(profiles, days, date(2026, 8, 1), rng)
    pipeline_result = build_pipeline(meal_logs)
    profiles_by_id = {profile["user_id"]: profile for profile in profiles}
    observations = generate_response_observations(pipeline_result, profiles_by_id, priors, rng)
    personal_foodmaps = build_personal_foodmaps(profiles, priors, observations)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "synthetic_person_profiles.jsonl", profiles)
    write_jsonl(output_dir / "synthetic_meal_logs.jsonl", meal_logs)
    write_jsonl(output_dir / "synthetic_response_observations.jsonl", observations)
    write_jsonl(output_dir / "personal_foodmaps.jsonl", personal_foodmaps)

    association_dir = output_dir / "association"
    association_dir.mkdir(exist_ok=True)
    for name in ("meal_events", "meal_event_foods", "event_relations", "unresolved_food_candidates", "active_learning_queue", "source_ledger"):
        write_jsonl(association_dir / f"{name}.jsonl", pipeline_result[name])

    if write_parquet:
        try:
            from .association_pipeline import materialize_parquet
        except ImportError:
            from association_pipeline import materialize_parquet
        output_paths = {
            name: association_dir / f"{name}.jsonl"
            for name in ("meal_events", "meal_event_foods", "event_relations", "unresolved_food_candidates", "active_learning_queue", "source_ledger")
        }
        materialize_parquet(association_dir, output_paths)

    manifest = {
        "seed": seed,
        "profile_count": profile_count,
        "days": days,
        "meal_log_count": len(meal_logs),
        "meal_event_count": len(pipeline_result["meal_events"]),
        "response_observation_count": len(observations),
        "personal_foodmap_row_count": len(personal_foodmaps),
        "data_origin": "synthetic",
        "source_registry": "data/v0.3/source_registry.json",
        "warning": "Synthetic response scores test the pipeline only and do not establish real food-body effects.",
    }
    with open(output_dir / "synthetic_run_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")

    return {
        "profiles": profiles,
        "meal_logs": meal_logs,
        "pipeline_result": pipeline_result,
        "observations": observations,
        "personal_foodmaps": personal_foodmaps,
        "manifest": manifest,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate an evidence-aware synthetic FoodMap and personalized updates."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "synthetic_foodmap")
    parser.add_argument("--profiles", type=int, default=30)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-parquet", action="store_true")
    args = parser.parse_args()

    result = generate_synthetic_foodmap(
        args.output_dir,
        profile_count=args.profiles,
        days=args.days,
        seed=args.seed,
        write_parquet=not args.skip_parquet,
    )
    print("Synthetic FoodMap")
    print("=================")
    for key, value in result["manifest"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
