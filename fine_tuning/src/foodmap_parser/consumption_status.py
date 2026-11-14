"""Experimental predicate-scoped consumption labels, without learned confidence.

This is a deliberately limited Korean rule baseline for comparison with a future
trained extractor. Evidence offsets refer to the unchanged input string.
"""

import re


COORDINATOR = re.compile(
    r"\s*(?:(?:한|두|세|네|반|\d+)\s*(?:잔|줄|개|캔|조각|그릇|공기)\s*)?"
    r"(?:이랑|랑|과|와|하고|,)\s*"
)

# Intentions precede actual-action patterns so '안 마시려고' does not assert
# non-consumption and '먹고 싶어' does not assert a completed meal.
PREDICATES = re.compile(
    r"(?P<intent>(?:안\s*)?(?:"
    r"먹(?:으려고|으려|을까|을\s*(?:거|예정|계획)|고\s*싶)|"
    r"마시(?:려고|려|고\s*싶)|마실(?:까|\s*(?:거|예정|계획)))\S*)"
    r"|(?P<long_negative>(?:먹|마시)지\s*(?:않|못)\S*)"
    r"|(?P<short_negative>(?:안|못)\s*(?:먹|마셨|마시|마셔)\S*)"
    r"|(?P<actual>(?:먹(?:었|음|고|어(?!야)|습니다)|마(?:셨|심|시고|셔(?!야)))\S*)"
)


def status_scope(text, food, foods, time_expressions, isolate_foods=True):
    start = food["start"]
    end = len(text)
    sentence_end = re.search(r"[.!?;\n]", text[food["end"]:])
    if sentence_end:
        end = food["end"] + sentence_end.end()

    # A first, postposed time can still belong to this food. Resolving multiple
    # postposed times requires a separate temporal attachment experiment.
    if any(t["end"] <= start for t in time_expressions):
        end = min([end] + [t["start"] for t in time_expressions if t["start"] >= food["end"]])

    if isolate_foods:
        previous_end = food["end"]
        for following in sorted(foods, key=lambda item: item["start"]):
            if following["start"] < previous_end or following["start"] >= end:
                continue
            # Coordinated nouns share their predicate. A completed action or
            # another topic breaks the chain before the next food's negation.
            if not COORDINATOR.fullmatch(text[previous_end:following["start"]]):
                end = following["start"]
                break
            previous_end = following["end"]
        # The next topic may not be in the food dictionary (e.g. '커피는').
        # Split only after a predicate, preserving coordinated noun phrases.
        for topic in re.finditer(r"\s+[가-힣]+(?:은|는)\s+", text[food["end"]:end]):
            boundary = food["end"] + topic.start()
            if PREDICATES.search(text[food["end"]:boundary]):
                end = boundary
                break
    return start, end


def classify_consumption(text, start, end):
    clause = text[start:end]

    def decision(status, rule, match=None):
        left, right = (start + match.start(), start + match.end()) if match else (start, end)
        return {"status": status, "rule": rule, "source_start": left,
                "source_end": right, "text": text[left:right],
                "scope_start": start, "scope_end": end,
                "method": "predicate_scope_rules_v1"}

    uncertainty = re.search(r"(?:먹었|마셨|먹|마시)는지", clause)
    if uncertainty:
        return decision("unknown", "uncertain_recollection", uncertainty)

    candidates = list(PREDICATES.finditer(clause))
    actual = [match for match in candidates if match.lastgroup != "intent"]
    if actual:
        match = actual[-1]
        status = "consumed" if match.lastgroup == "actual" else "not_consumed"
        return decision(status, match.lastgroup, match)
    if candidates:
        match = candidates[-1]
        negative_intent = match.group().startswith("안")
        return decision("unknown" if negative_intent else "planned",
                        "negative_intention_only" if negative_intent else "intention_only", match)

    leftovers = re.search(r"남겼|거르", clause)
    if leftovers:
        return decision("not_consumed", "legacy_leftover_convention", leftovers)
    order = re.search(r"주문", clause)
    if order:
        return decision("unknown", "order_without_consumption", order)
    return decision("consumed", "legacy_meal_log_shorthand_default")


def infer_consumption_status(text, food, foods, time_expressions, isolate_foods=True):
    start, end = status_scope(text, food, foods, time_expressions, isolate_foods)
    return classify_consumption(text, start, end)
