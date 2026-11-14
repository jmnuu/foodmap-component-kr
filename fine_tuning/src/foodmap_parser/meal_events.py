"""Rule-based extraction of meal events from Korean meal-log text.

This module deliberately separates event structure from food normalization.
`parser.parse_food_log` owns food ID detection; this module assigns those foods
to time expressions and extracts only details explicitly stated in the text.
"""

import re

try:
    from .component_inference import infer_representative_components
    from .consumption_status import infer_consumption_status
    from .parser import parse_food_log
except ImportError:  # Allows `python meal_events.py` during early local use.
    from component_inference import infer_representative_components
    from consumption_status import infer_consumption_status
    from parser import parse_food_log


KOREAN_NUMBERS = {
    "반": 0.5,
    "한": 1,
    "하나": 1,
    "두": 2,
    "세": 3,
    "네": 4,
    "다섯": 5,
}

UNIT_NAMES = {
    "잔": "cup",
    "줄": "roll",
    "캔": "can",
    "스푼": "scoop",
    "개": "piece",
    "조각": "slice",
    "그릇": "bowl",
    "공기": "bowl",
    "모금": "sip",
    "봉지": "package",
    "팩": "package",
}

TIME_PATTERNS = [
    (r"(?:어제|오늘|내일)\s*늦은\s*밤", "late_night"),
    (r"늦은\s*밤", "late_night"),
    (r"(?:(?:어제|오늘|내일)\s*)?밤\s*\d{1,2}시(?:\s*반)?(?:쯤|에)?", "night"),
    (r"(?:아침|점심|저녁)\s*\d{1,2}시(?:\s*반)?(?:쯤|에|엔|은|으로)?", None),
    (r"(?:오전|오후)\s*\d{1,2}시(?:\s*반)?(?:쯤|에)?", "snack"),
    (r"\d{1,2}시(?:\s*반)?(?:쯤|에)?", None),
    (r"(?:어제|오늘|내일)\s*아침", "breakfast"),
    (r"(?:어제|오늘|내일)\s*점심", "lunch"),
    (r"(?:어제|오늘|내일)\s*저녁", "dinner"),
    (r"(?:아침|점심|저녁|오전|오후)\s*간식", "snack"),
    (r"야식", "late_night"),
    (r"후식", "snack"),
    (r"간식", "snack"),
    (r"아침", "breakfast"),
    (r"점심", "lunch"),
    (r"저녁", "dinner"),
    (r"오전", "snack"),
    (r"오후", "snack"),
    (r"밤", "night"),
]


def _meal_time_for_match(text, match, default_meal_time):
    """Resolve a time phrase while keeping bare clock time intentionally coarse."""
    phrase = match.group(0)

    if "밤" in phrase and re.search(r"1[1-2]시", phrase):
        return "late_night"
    if default_meal_time is not None:
        return default_meal_time
    if "아침" in phrase:
        return "breakfast"
    if "점심" in phrase:
        return "lunch"
    if "저녁" in phrase:
        return "dinner"

    hour_match = re.search(r"(\d{1,2})시", phrase)
    if not hour_match:
        return "unknown"

    hour = int(hour_match.group(1))
    if 5 <= hour <= 10:
        return "breakfast"
    if 11 <= hour <= 14:
        return "lunch"
    if 17 <= hour <= 20:
        return "dinner"
    if 21 <= hour or hour <= 4:
        return "night"
    return "snack"


def find_time_expressions(text):
    """Return non-overlapping time expressions ordered by their text position."""
    candidates = []

    for pattern, default_meal_time in TIME_PATTERNS:
        for match in re.finditer(pattern, text):
            candidates.append({
                "start": match.start(),
                "end": match.end(),
                "time_expression": match.group(0),
                "meal_time": _meal_time_for_match(text, match, default_meal_time),
            })

    candidates.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))
    results = []

    for candidate in candidates:
        overlaps = any(
            not (candidate["end"] <= existing["start"] or candidate["start"] >= existing["end"])
            for existing in results
        )
        if not overlaps:
            results.append(candidate)

    # A bare "9시" after dinner is ordinarily a night expression, whereas a
    # bare clock time without context stays a coarse clock-based classification.
    for index, result in enumerate(results):
        if not re.match(r"^\d{1,2}시", result["time_expression"]):
            continue
        hour = int(re.match(r"^(\d{1,2})시", result["time_expression"]).group(1))
        earlier_times = results[:index]
        if 8 <= hour <= 11 and any(
            item["meal_time"] in {"dinner", "night", "late_night"}
            for item in earlier_times
        ):
            result["meal_time"] = "night"

    return results


def _closest_time_expression(food, time_expressions):
    if not time_expressions:
        return {
            "start": -1,
            "end": -1,
            "time_expression": None,
            "meal_time": "unknown",
        }

    earlier_times = [
        time_expression
        for time_expression in time_expressions
        if time_expression["end"] <= food["start"]
    ]
    if earlier_times:
        return max(earlier_times, key=lambda item: item["end"])

    later_times = [
        time_expression
        for time_expression in time_expressions
        if time_expression["start"] >= food["end"]
    ]
    if later_times:
        return min(later_times, key=lambda item: item["start"])

    return time_expressions[0]


def _food_clause(text, food, foods):
    """Use the span up to the next food to avoid applying later negation backward."""
    next_starts = [item["start"] for item in foods if item["start"] > food["end"]]
    end = min(next_starts) if next_starts else len(text)
    return text[food["start"]:end]


def _status_clause(text, food, time_expressions):
    """Keep coordinated foods together, but stop before contrastive clauses."""
    boundaries = []
    boundary = re.search(r"하지만|지만|그런데|대신|결국|,|\.|\s+\S+는\s+(?:안|남겼)", text[food["start"]:])
    if boundary:
        boundaries.append(food["start"] + boundary.start())
    boundaries.extend(
        time_expression["start"]
        for time_expression in time_expressions
        if time_expression["start"] > food["end"]
    )
    end = min(boundaries) if boundaries else len(text)
    return text[food["start"]:end]


def _consumption_status(clause):
    if re.search(r"안\s*(먹|마시|마셨)|남겼|거르|아무것도\s*안", clause):
        return "not_consumed"
    if re.search(r"먹을까|마실까|먹을\s*거|마실\s*거|먹을\s*예정|마실\s*예정", clause):
        return "planned"
    if re.search(r"주문", clause) and not re.search(r"먹|마시", clause):
        return "unknown"
    return "consumed"


def _number_value(token):
    if token in KOREAN_NUMBERS:
        return KOREAN_NUMBERS[token]
    return int(token) if token.isdigit() else None


def _quantity_and_unit(food, text, foods):
    clause = _food_clause(text, food, foods)
    # Search after the matched food name so a food such as "두유" is not
    # mistaken for the Korean quantity word "두" (two).
    clause = clause[len(food["raw_text"]):]
    number_pattern = "|".join(sorted(KOREAN_NUMBERS, key=len, reverse=True)) + r"|\d+"
    unit_pattern = "|".join(UNIT_NAMES)

    # Korean usually places the count before the unit: "두 잔".
    match = re.search(rf"({number_pattern})\s*({unit_pattern})", clause)
    if match:
        quantity = _number_value(match.group(1))
        unit = UNIT_NAMES[match.group(2)]
        if match.group(2) == "조각" and food["food_id"] != "F0013":
            unit = "piece"
        if unit == "sip":
            return None, unit
        return quantity, unit

    # Product descriptions can use the reverse order: "작은 캔 하나".
    match = re.search(rf"(?:작은\s+)?({unit_pattern})\s*({number_pattern})", clause)
    if match:
        return _number_value(match.group(2)), UNIT_NAMES[match.group(1)]

    match = re.search(rf"({number_pattern})(?:만)?", clause)
    if match:
        raw_text = food["raw_text"]
        default_unit = "package" if "라면" in raw_text else "piece"
        return _number_value(match.group(1)), default_unit

    return None, None


def _modifiers(food, text, time_expression, foods):
    previous_food_ends = [item["end"] for item in foods if item["end"] <= food["start"]]
    start = max(previous_food_ends, default=time_expression["end"])
    context = text[start:food["end"]]
    clause = _food_clause(text, food, foods)
    modifiers = []

    if food["food_id"] == "F0003" and ("아이스" in context or "아아" in context):
        modifiers.append("iced")
    if "디카페인" in context or "디카페인" in clause:
        modifiers.append("decaf")
    if "제로" in context or "제로" in clause:
        modifiers.append("zero_sugar")
    if "매운" in context or "매운" in clause:
        modifiers.append("spicy")
    if "작은" in context or "작은" in clause:
        modifiers.append("small")
    if "치즈 없는" in context or "치즈 없는" in clause:
        modifiers.append("without_cheese")
    if "새우 없는" in context:
        modifiers.append("without_shrimp")
    if "컵라면" in food["raw_text"]:
        modifiers.append("cup")

    return modifiers


def mention_context(text, food, foods, time_expressions):
    """Use the current food's prefix, bounded by time and food mentions.

    This conservative baseline does not attach a later food's ingredients or
    post-meal descriptions to an earlier mention.
    """
    starts = [0]
    starts.extend(f["end"] for f in foods if f["end"] <= food["start"])
    starts.extend(t["end"] for t in time_expressions if t["end"] <= food["start"])
    start = max(starts)
    prefix = text[start:food["start"]]
    boundaries = list(re.finditer(r"[.!?,]|먹\S*|마셨\S*|마시\S*|그리고|대신|이랑|랑", prefix))
    if boundaries:
        start += boundaries[-1].end()
    return start, food["end"]


def parse_meal_events(text, component_priors=None, *, status_strategy="legacy"):
    """Extract normalized meal events from one Korean meal-log text.

    The result contains no fabricated timestamps. `time_expression` preserves the
    original phrase and `meal_time` is a coarse normalized label for analysis.
    `scoped` enables the experimental predicate resolver; `unscoped` is its
    food-scope ablation. The legacy default stays until human validation.
    """
    if status_strategy not in {"legacy", "scoped", "unscoped"}:
        raise ValueError("unsupported status_strategy")
    food_result = parse_food_log(text)
    foods = food_result["foods"]
    time_expressions = find_time_expressions(text)
    grouped = {}

    for food in foods:
        time_expression = _closest_time_expression(food, time_expressions)
        evidence = None
        if status_strategy == "legacy":
            clause = _status_clause(text, food, time_expressions)
            status = _consumption_status(clause)
        else:
            evidence = infer_consumption_status(
                text, food, foods, time_expressions,
                isolate_foods=status_strategy == "scoped",
            )
            status = evidence["status"]
        key = (time_expression["start"], status)

        event = grouped.setdefault(key, {
            "time_expression": time_expression["time_expression"],
            "meal_time": time_expression["meal_time"],
            "consumption_status": status,
            "foods": [],
        })
        quantity, unit = _quantity_and_unit(food, text, foods)
        modifiers = _modifiers(food, text, time_expression, foods)
        context_start, context_end = mention_context(text, food, foods, time_expressions)
        event["foods"].append({
            "raw_text": food["raw_text"],
            "food_id": food["food_id"],
            "source_start": food["start"],
            "source_end": food["end"],
            "quantity": quantity,
            "unit": unit,
            "modifiers": modifiers,
            "mention_context": text[context_start:context_end].strip(),
            "representative_components": infer_representative_components(
                {**food, "context_start": context_start, "context_end": context_end},
                text,
                modifiers,
                component_priors=component_priors,
            ),
        })
        if evidence is not None:
            event["foods"][-1]["consumption_evidence"] = evidence

    events = sorted(grouped.values(), key=lambda event: (
        next(
            (item["start"] for item in time_expressions if item["time_expression"] == event["time_expression"]),
            -1,
        ),
        event["consumption_status"],
    ))

    for index, event in enumerate(events):
        event["event_index"] = index

    return {
        "input": text,
        "events": events,
        "unmatched_time_expressions": time_expressions if not foods else [],
    }
