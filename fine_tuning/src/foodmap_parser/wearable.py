"""Vendor-neutral wearable contracts and conservative temporal links, not effects."""

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCHEMA_VERSION = "wearable-v0.1"
ROOT = Path(__file__).resolve().parents[2]


def load_metric_registry():
    catalog = json.loads((ROOT / "schemas/wearable_metrics.json").read_text(encoding="utf-8"))
    return {row["metric_id"]: row for row in catalog["metrics"]}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp requires an explicit UTC offset")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class MealAnchor:
    event_id: str
    user_id: str
    consumption_status: str
    consumed_at: datetime | None
    time_basis: str
    time_source_id: str | None
    data_origin: str

    def __post_init__(self):
        for name in ("event_id", "user_id"):
            _text(getattr(self, name), name)
        if self.consumption_status not in {"consumed", "not_consumed", "planned", "unknown"}:
            raise ValueError("invalid consumption status")
        if self.data_origin not in {"real", "synthetic", "unknown"}:
            raise ValueError("invalid data origin")
        if self.consumed_at is None:
            if self.time_basis != "unresolved" or self.time_source_id is not None:
                raise ValueError("unresolved time must not claim a resolution source")
        else:
            if self.time_basis not in {"user_confirmed", "explicit_datetime"}:
                raise ValueError("only explicitly resolved meal times can anchor a window")
            _text(self.time_source_id, "time_source_id")
            object.__setattr__(self, "consumed_at", utc(self.consumed_at))

    @classmethod
    def from_parser_event(cls, event, *, consumed_at=None, time_basis="unresolved", time_source_id=None):
        # recorded_at is deliberately never used as a fallback consumption time.
        return cls(event["event_id"], event.get("user_id"), event["consumption_status"],
                   consumed_at, time_basis, time_source_id, event.get("data_origin", "unknown"))


@dataclass(frozen=True)
class WearableObservation:
    observation_id: str
    user_id: str
    source_record_id: str
    metric_id: str
    start_at: datetime
    end_at: datetime
    value: float | None
    unit: str
    quality_status: str
    missing_reason: str | None
    data_origin: str

    def __post_init__(self):
        for name in ("observation_id", "user_id", "source_record_id", "metric_id", "unit"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "start_at", utc(self.start_at))
        object.__setattr__(self, "end_at", utc(self.end_at))
        if self.end_at < self.start_at:
            raise ValueError("observation end precedes start")
        if self.quality_status not in {"valid", "unknown", "invalid", "missing"}:
            raise ValueError("invalid quality status")
        if self.data_origin not in {"real", "synthetic", "unknown"}:
            raise ValueError("invalid data origin")
        if self.value is None:
            if self.quality_status != "missing":
                raise ValueError("null value requires missing status")
            _text(self.missing_reason, "missing_reason")
        else:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(self.value):
                raise ValueError("value must be a finite number")
            if self.quality_status == "missing" or self.missing_reason is not None:
                raise ValueError("non-null value cannot have missing metadata")

    def validate_metric(self, registry):
        spec = registry.get(self.metric_id)
        if spec is None or self.unit != spec["unit"]:
            raise ValueError("unknown metric or incompatible canonical unit")
        if spec["temporal_kind"] == "point" and self.start_at != self.end_at:
            raise ValueError("point metric cannot represent an interval summary")
        if spec["temporal_kind"] == "interval" and self.start_at >= self.end_at:
            raise ValueError("interval metric requires positive duration")


@dataclass(frozen=True)
class ResponseWindow:
    protocol_id: str
    metric_id: str
    start_offset_minutes: float
    end_offset_minutes: float

    def __post_init__(self):
        _text(self.protocol_id, "protocol_id")
        _text(self.metric_id, "metric_id")
        offsets = (self.start_offset_minutes, self.end_offset_minutes)
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in offsets):
            raise ValueError("window offsets must be finite numbers")
        if self.start_offset_minutes >= self.end_offset_minutes:
            raise ValueError("response window must have positive duration")


def link_observations(meal, observations, window, *, registry=None):
    """One-meal reference join; [start, end), no averaging or causal attribution."""
    registry = load_metric_registry() if registry is None else registry
    if window.metric_id not in registry:
        raise ValueError("window uses an unregistered metric")
    observations = list(observations)
    seen = set()
    for obs in observations:
        obs.validate_metric(registry)
        if obs.observation_id in seen:
            raise ValueError("duplicate observation ID; reconcile source revisions before joining")
        seen.add(obs.observation_id)
    result = {"schema_version": SCHEMA_VERSION, "event_id": meal.event_id,
              "protocol_id": window.protocol_id, "metric_id": window.metric_id,
              "status": "ready", "links": [], "coverage_fraction": None}
    if meal.consumption_status != "consumed":
        result["status"] = "not_consumed"
        return result
    if meal.consumed_at is None:
        result["status"] = "unresolved_meal_time"
        return result
    if meal.data_origin == "unknown":
        result["status"] = "unknown_data_origin"
        return result
    start = meal.consumed_at + timedelta(minutes=window.start_offset_minutes)
    end = meal.consumed_at + timedelta(minutes=window.end_offset_minutes)
    result.update(window_start_at=start.isoformat(), window_end_at=end.isoformat())
    for obs in observations:
        if (obs.user_id != meal.user_id or obs.metric_id != window.metric_id
                or obs.data_origin != meal.data_origin):
            continue
        point = obs.start_at == obs.end_at
        overlaps = start <= obs.start_at < end if point else obs.start_at < end and obs.end_at > start
        if not overlaps:
            continue
        contained = point or (start <= obs.start_at and obs.end_at <= end)
        # A daily steps total overlapping an hour cannot be divided proportionally.
        eligible = contained and obs.quality_status == "valid" and obs.value is not None
        reason = "eligible" if eligible else ("partial_interval" if not contained else "quality_not_valid")
        result["links"].append({
            "observation_id": obs.observation_id,
            "relation_type": "TEMPORALLY_OVERLAPS", "eligible": eligible,
            "exclusion_reason": None if eligible else reason,
            "lag_seconds": (obs.start_at - meal.consumed_at).total_seconds(),
            "overlap_seconds": 0.0 if point else (min(end, obs.end_at) - max(start, obs.start_at)).total_seconds(),
        })
    result["links"].sort(key=lambda row: row["observation_id"])
    if not result["links"]:
        result["status"] = "no_observations"
    elif not any(row["eligible"] for row in result["links"]):
        result["status"] = "no_eligible_observations"
    return result


def initialize_database(connection):
    """Create a new DuckDB namespace; existing tables are not overwritten."""
    registry = load_metric_registry()
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute((ROOT / "schemas/wearable.sql").read_text(encoding="utf-8"))
        for spec in registry.values():
            connection.execute("INSERT INTO wearable.metric_definitions VALUES (?, ?, ?, ?, ?, ?)",
                               [spec["metric_id"], spec["unit"], spec["temporal_kind"],
                                spec["definition"], json.dumps(spec["source_ids"]), SCHEMA_VERSION])
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
