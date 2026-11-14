import json
from pathlib import Path

try:
    from .reference_data import load_reference_csv, source_id_for_annotation
except ImportError:  # Allows direct module execution during early local use.
    from reference_data import load_reference_csv, source_id_for_annotation

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "v0.1"


def load_csv(filename):
    return load_reference_csv(filename)


def build_food_lookup():
    foods = load_csv("food_items.csv")
    aliases = load_csv("food_aliases.csv")
    categories_by_food_id = {
        row["food_id"]: row["food_category"]
        for row in foods
    }

    lookup = {}

    for row in foods:
        name = row["normalized_food_ko"]
        lookup[name] = {
            "food_id": row["food_id"],
            "normalized_food_ko": row["normalized_food_ko"],
            "food_category": row["food_category"],
        }

    for row in aliases:
        lookup[row["alias"]] = {
            "food_id": row["food_id"],
            "normalized_food_ko": row["normalized_food_ko"],
            "food_category": categories_by_food_id[row["food_id"]],
        }

    return lookup

def build_component_alias_lookup():
    component_aliases = load_csv("component_aliases.csv")

    lookup = {}

    for row in component_aliases:
        lookup[row["alias"]] = {
            "component_id": row["component_id"],
            "component_name_en": row["component_name_en"],
            "confidence": float(row["confidence"]),
        }

    return lookup

def build_component_lookup():
    components = load_csv("component_vocab.csv")
    annotations = load_csv("food_component_annotations.csv")

    component_name_by_id = {
        row["component_id"]: row["component_name_en"]
        for row in components
    }

    component_lookup = {}

    for row in annotations:
        if row["status"] not in {"confirmed", "possible"}:
            continue

        food_id = row["food_id"]
        component_id = row["component_id"]
        component_name = component_name_by_id.get(component_id)

        if component_name is None:
            continue

        component_lookup.setdefault(food_id, []).append({
            "component": component_name,
            "status": row["status"],
            "confidence": float(row["confidence"]),
            "source_ids": [source_id_for_annotation(row)],
        })

    return component_lookup

def merge_final_components(found_foods, explicit_components):
    merged = {}

    status_rank = {
        "absent": 0,
        "unknown": 1,
        "possible": 2,
        "confirmed": 3,
    }

    def add_component(component, status, confidence, source, component_id=None):
        if component not in merged:
            merged[component] = {
                "component_id": component_id,
                "component": component,
                "status": status,
                "confidence": confidence,
                "sources": [source],
            }
            return

        existing = merged[component]

        # status는 confirmed가 possible보다 우선
        if status_rank.get(status, 0) > status_rank.get(existing["status"], 0):
            existing["status"] = status

        # confidence는 더 높은 값을 사용
        existing["confidence"] = max(existing["confidence"], confidence)

        # component_id가 새로 있으면 보존
        if existing.get("component_id") is None and component_id is not None:
            existing["component_id"] = component_id

        if source not in existing["sources"]:
            existing["sources"].append(source)

    # 1. 음식에서 추론된 components
    for food in found_foods:
        food_id = food["food_id"]

        for item in food.get("components", []):
            add_component(
                component=item["component"],
                status=item["status"],
                confidence=item["confidence"],
                source=f"food:{food_id}",
            )

    # 2. 문장에 직접 나온 components
    for item in explicit_components:
        add_component(
            component=item["component"],
            status=item["status"],
            confidence=item["confidence"],
            source=item["source"],
            component_id=item.get("component_id"),
        )

    return sorted(
        merged.values(),
        key=lambda x: (
            {"confirmed": 0, "possible": 1, "unknown": 2, "absent": 3}.get(x["status"], 9),
            -x["confidence"],
            x["component"],
        )
    )

AMBIGUOUS_FOOD_ALIASES = {
    "마라",
    "포",
    "사과",
}

# A one-syllable food alias must not be recovered from the tail of an unknown
# compound such as "비빔밥". Longer aliases such as "공기밥" are matched first.
SHORT_ALIAS_BOUNDARY_REQUIRED = {"밥"}
ALLOWED_PREVIOUS_CHARS = {"랑", "과", "와", "은", "는", "을", "를", "에"}

BLOCKED_NEXT_CHARS = {
    "마라": ["톤"],
    "포": ["기", "장", "함"],
    "사과": ["하", "할", "해", "했", "합니다", "했다"],
}

MEAL_CONTEXT_CUES = [
    "먹", "먹었", "먹음", "먹고", "먹는", "먹을",
    "마시", "마셨", "마심", "마시고",
    "주문", "시켰", "배달",
    "아침", "점심", "저녁", "야식", "간식", "후식",
    "식사", "밥", "카페", "디저트",
]


def has_meal_context(text, start, end, window=12):
    left = max(0, start - window)
    right = min(len(text), end + window)
    context = text[left:right]

    return any(cue in context for cue in MEAL_CONTEXT_CUES)


def should_accept_food_alias(text, alias, start, end):
    next_text = text[end:end + 5]

    if alias in SHORT_ALIAS_BOUNDARY_REQUIRED and start > 0:
        previous_char = text[start - 1]
        if "가" <= previous_char <= "힣" and previous_char not in ALLOWED_PREVIOUS_CHARS:
            return False

    # 예: 마라톤, 포기, 사과하다 차단
    for blocked in BLOCKED_NEXT_CHARS.get(alias, []):
        if next_text.startswith(blocked):
            return False

    # 짧고 애매한 alias는 식사 문맥이 있을 때만 허용
    if alias in AMBIGUOUS_FOOD_ALIASES:
        return has_meal_context(text, start, end)

    return True


COMPOSITE_FOOD_CATEGORIES = {"mixed_meal", "stew", "fast_food", "noodles"}


def _remove_embedded_ingredient_foods(found_foods, component_alias_lookup, text):
    """Keep '치즈 샌드위치' as one meal food plus an explicit component.

    A food word can also be an ingredient word. When it appears immediately
    before a composite food with no connector, treating it as a second meal food
    duplicates the event. The component-alias pass still preserves the explicit
    evidence. A connector such as '랑' prevents this removal.
    """
    retained = []
    for food in found_foods:
        is_explicit_component = food["raw_text"] in component_alias_lookup
        is_embedded = any(
            other["food_category"] in COMPOSITE_FOOD_CATEGORIES
            and food["end"] <= other["start"]
            and text[food["end"]:other["start"]].strip() in {"", "없는", "뺀", "제외한"}
            and other["start"] - food["end"] <= 8
            for other in found_foods
        )
        if is_explicit_component and is_embedded:
            continue
        retained.append(food)
    return retained

def parse_food_log(text):
    food_lookup = build_food_lookup()
    component_lookup = build_component_lookup()
    component_alias_lookup = build_component_alias_lookup()

    found_foods = []
    explicit_components = []
    occupied_spans = []

    food_aliases = sorted(food_lookup.items(), key=lambda x: len(x[0]), reverse=True)

    for alias, info in food_aliases:
        start = 0

        while True:
            index = text.find(alias, start)

            if index == -1:
                break

            span = (index, index + len(alias))

            overlaps = any(
                not (span[1] <= existing[0] or span[0] >= existing[1])
                for existing in occupied_spans
            )

            if not overlaps:
                if not should_accept_food_alias(text, alias, span[0], span[1]):
                    start = index + len(alias)
                    continue

                food_id = info["food_id"]

                found_foods.append({
                    "raw_text": alias,
                    "food_id": food_id,
                    "normalized_food": info["normalized_food_ko"],
                    "food_category": info["food_category"],
                    "start": span[0],
                    "end": span[1],
                    "components": component_lookup.get(food_id, []),
                })

                occupied_spans.append(span)

            start = index + len(alias)

    found_foods = _remove_embedded_ingredient_foods(
        found_foods,
        component_alias_lookup,
        text,
    )
    for food in found_foods:
        food.pop("food_category", None)

    # component aliases는 음식 span과 겹쳐도 일부 허용해야 함
    # 예: 토마토파스타에서 토마토 + 파스타 둘 다 잡기
    component_aliases = sorted(
        component_alias_lookup.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    component_spans = []

    for alias, info in component_aliases:
        start = 0

        while True:
            index = text.find(alias, start)

            if index == -1:
                break

            span = (index, index + len(alias))

            overlaps_component = any(
                not (span[1] <= existing[0] or span[0] >= existing[1])
                for existing in component_spans
            )

            if not overlaps_component:
                explicit_components.append({
                    "raw_text": alias,
                    "component_id": info["component_id"],
                    "component": info["component_name_en"],
                    "status": "confirmed",
                    "confidence": info["confidence"],
                    "source": "explicit_component_alias",
                    "start": span[0],
                    "end": span[1],
                })

                component_spans.append(span)

            start = index + len(alias)

    found_foods = sorted(found_foods, key=lambda x: x["start"])
    explicit_components = sorted(explicit_components, key=lambda x: x["start"])

    final_components = merge_final_components(found_foods, explicit_components)

    return {
    "input": text,
    "foods": found_foods,
    "explicit_components": explicit_components,
    "final_components": final_components,
}


if __name__ == "__main__":
    test_sentences = [
        "토마토 파스타 먹었어.",
        "새우버거랑 콜라 마셨어.",
        "딸기탕후루 먹었어.",
        "크림파스타랑 샐러드 먹었어.",
        "아보카도 포케 먹었어.",
    ]

    for text in test_sentences:
        result = parse_food_log(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("-" * 50)
