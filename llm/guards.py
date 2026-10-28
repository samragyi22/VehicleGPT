"""
llm/guards.py - Safety Guards, Vocabulary Gates, and Grounding Checks for VehicleGPT.

Implements:
1. Control-command guard: Regex & keyword guard that intercepts actuator/control requests
   before reaching any tool or LLM, ensuring the system remains strictly read-only.
2. Vehicle-vocabulary gate: Ensures queries pertain to vehicle data/telemetry based on
   vocabularies derived from schema.py enums plus synonym tables.
3. Grounding check: Verifies that tool arguments (e.g., event types, metrics, segments)
   are directly supported by keywords in the user's question.
4. ENUM_SYNONYMS table: Comprehensive mappings for Metric, EventType, Segment,
   BucketSize, and ChangeDirection with automated coverage test.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from schema import BucketSize, ChangeDirection, EventType, Metric, Segment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Closed Vocabulary Synonym Tables
# Derived from schema.py enums. Every enum member MUST have at least one synonym.
# ---------------------------------------------------------------------------

METRIC_SYNONYMS: dict[Metric, list[str]] = {
    Metric.energy_wh_per_km: [
        "energy",
        "efficiency",
        "consumption",
        "wh/km",
        "wh_per_km",
        "wh per km",
        "fuel efficiency",
        "power use",
        "energy use",
        "economy",
        "fuel economy",
        "fuel_consumption_litre_per_100km",
    ],
    Metric.estimated_range_km: [
        "range",
        "driving range",
        "distance",
        "km remaining",
        "battery range",
        "mileage",
        "full-charge range",
    ],
    Metric.pack_temp_c: [
        "pack temp",
        "pack temperature",
        "battery temperature",
        "battery temp",
        "thermal",
        "cell temp",
        "temperature_c",
        "temperature",
        "pack_temperature_c",
    ],
    Metric.soc_pct: [
        "soc",
        "state of charge",
        "battery level",
        "battery percent",
        "charge level",
        "percentage",
    ],
    Metric.speed_kmh: [
        "speed",
        "velocity",
        "km/h",
        "kmh",
        "driving speed",
        "fast",
        "slow",
        "speed_km_per_hour",
    ],
    Metric.hvac_power_kw: [
        "hvac",
        "air conditioning",
        "ac power",
        "climate control",
        "heater",
        "heating",
        "cooling",
        "cabin cooling",
        "aircon",
    ],
}

EVENT_TYPE_SYNONYMS: dict[EventType, list[str]] = {
    EventType.BATTERY_OVERHEAT_WARNING: [
        "overheat",
        "overheating",
        "thermal warning",
        "hot battery",
        "battery warning",
        "temperature warning",
        "battery overheat",
        "battery overheating warning",
        "battery_overheat_warning",
    ],
    EventType.ADAS_DISENGAGEMENT: [
        "adas",
        "disengagement",
        "disengagements",
        "disengage",
        "autopilot",
        "driver assist",
        "driver assistance",
        "lane assist",
        "cruise control",
        "handover",
        "auto_pilot_disengagement",
        "autopilot_disengagement",
        "adas_event",
        "adas_events",
        "adas event",
    ],
    EventType.LOW_SOC_WARNING: [
        "low soc",
        "low battery",
        "low charge",
        "charge warning",
        "empty battery",
        "critical battery",
        "low battery warning",
        "low_battery_warning",
        "low_soc_warning",
    ],
}

SEGMENT_SYNONYMS: dict[Segment, list[str]] = {
    Segment.city: [
        "city",
        "urban",
        "stop and go",
        "town",
        "local roads",
        "street",
        "traffic",
    ],
    Segment.highway: [
        "highway",
        "freeway",
        "motorway",
        "interstate",
        "expressway",
    ],
}

BUCKET_SIZE_SYNONYMS: dict[BucketSize, list[str]] = {
    BucketSize.hour: ["hour", "hourly"],
    BucketSize.day: ["day", "daily"],
    BucketSize.week: ["week", "weekly"],
}

CHANGE_DIRECTION_SYNONYMS: dict[ChangeDirection, list[str]] = {
    ChangeDirection.drop: [
        "drop",
        "decrease",
        "reduction",
        "reduced",
        "fell",
        "loss",
        "lower",
        "worst",
        "decline",
    ],
    ChangeDirection.rise: [
        "rise",
        "increase",
        "jump",
        "increased",
        "rose",
        "spike",
        "higher",
        "growth",
    ],
}

# Aggregate all terms for vocabulary gating.
# Excludes generic temporal words (yesterday, history, etc.) that would cause
# off-topic questions like "tell me what the stock market did yesterday" to
# slip through the gate.
ALL_VEHICLE_TERMS: set[str] = {
    "vehicle",
    "car",
    "ev",
    "electric",
    "drive",
    "driving",
    "trip",
    "trips",
    "telemetry",
    "battery",
    "pack",
    "charge",
    "range",
    "energy",
    "efficiency",
    "power",
    "speed",
    "adas",
    "temperature",
    "temp",
    "hvac",
    "climate",
    "warning",
    "warnings",
    "event",
    "events",
    "overheat",
    "overheating",
    "disengagement",
    "disengagements",
    "city",
    "highway",
    "soc",
    "consumption",
    "signals",
    # "yesterday" and "history" deliberately omitted — they are too generic
    # and let off-topic questions ("stock market yesterday") past the gate.
    "trend",
    "compare",
}

for _d in (
    METRIC_SYNONYMS,
    EVENT_TYPE_SYNONYMS,
    SEGMENT_SYNONYMS,
    BUCKET_SIZE_SYNONYMS,
    CHANGE_DIRECTION_SYNONYMS,
):
    for _syns in _d.values():  # type: ignore[attr-defined]
        for _s in _syns:
            ALL_VEHICLE_TERMS.add(_s.lower())

# Stem prefixes: any token starting with one of these prefixes is treated
# as a vehicle term. This catches "speedup", "speeding", etc.
_VEHICLE_STEM_PREFIXES: tuple[str, ...] = (
    "speed",
    "effici",      # efficiency, efficient
    "discharg",    # discharging, discharge
    "consum",      # consumption, consuming
    "temperatur",  # temperature, temperatures
    "batterie",    # batteries
)


def _edit_distance_1(a: str, b: str) -> bool:
    """Return True if Levenshtein distance between a and b is exactly 1."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(ca != cb for ca, cb in zip(a, b)) == 1
    # One char inserted/deleted
    short, long_ = (a, b) if la < lb else (b, a)
    i = j = diffs = 0
    while i < len(short) and j < len(long_):
        if short[i] != long_[j]:
            diffs += 1
            if diffs > 1:
                return False
            j += 1
        else:
            i += 1
            j += 1
    return True


def _fuzzy_match_vehicle_term(token: str) -> bool:
    """
    Conservative fuzzy matching: accepts a token if its edit distance
    to any vehicle term is ≤ 1, but only for tokens with ≥ 5 characters
    (to avoid short-word false positives like "car" matching "bar").
    """
    if len(token) < 5:
        return False
    for term in ALL_VEHICLE_TERMS:
        if len(term) < 5:
            continue
        if _edit_distance_1(token, term):
            return True
    return False



# ---------------------------------------------------------------------------
# 2. Control Command Guard
# Intercepts commands that try to actuate or alter vehicle state.
# ---------------------------------------------------------------------------

_CONTROL_ACTION_PATTERN = re.compile(
    r"\b("
    r"turn\s+(on|off|up|down)|switch\s+(on|off|to)|shut\s+off|"
    r"open|close|unlock|lock|"
    r"start|stop|"
    r"set|adjust|change|"
    r"increase|decrease|raise|lower|"
    r"enable|disable|activate|deactivate|"
    r"put(\s+(on|up|down))?|"
    r"roll\s+(up|down)|"
    r"brake|accelerate|speed\s+up|slow\s+down|"
    r"honk|flash|steer|drive\s+to"
    r")\b",
    re.IGNORECASE,
)

_CONTROL_TARGET_PATTERN = re.compile(
    r"\b("
    r"air\s+conditioning|ac|hvac|climate|heater|heating|cooling|fan|temperature|temp|"
    r"door|doors|window|windows|trunk|frunk|hood|sunroof|"
    r"lock|locks|"
    r"light|lights|headlight|headlights|high\s+beam|wipers|"
    r"horn|"
    r"seat|seats|radio|music|volume|"
    r"speed|cruise\s+control|autopilot|adas|"
    r"charging|charger|charge\s+port|"
    r"brake|parking\s+brake|mirror|mirrors|"
    r"sport\s+mode|eco\s+mode|comfort\s+mode|drive\s+mode|"
    r"motor|engine|car"
    r")\b",
    re.IGNORECASE,
)

# Polite request openers: "could you ...", "I'd like the ...", "please ...".
# These are requests to actuate, not questions, so when such an opener is
# followed by a control target (and no question intent anywhere), the guard
# must refuse instead of letting a disguised command reach the tools.
_REQUEST_OPENER_PATTERN = re.compile(
    r"^(could\s+you|would\s+you|can\s+you|will\s+you|i'd\s+like|i\s+would\s+like|please)\b",
    re.IGNORECASE,
)

# Query-intent markers anywhere in the question mean the user is asking about
# data, not commanding the vehicle; those must NOT be refused by the guard.
_QUERY_INTENT_PATTERN = re.compile(
    r"\b(why|when|what|how|show|list|compare|tell|explain|did|does|"
    r"is|are|was|were|which|where)\b",
    re.IGNORECASE,
)

CONTROL_REFUSAL_MESSAGE = (
    "Vehicle control commands are refused. VehicleGPT operates strictly in read-only "
    "telemetry mode with no command path to vehicle actuators or safety-critical ECUs."
)


def is_control_command(question: str) -> tuple[bool, str | None]:
    """
    Checks if a question is an actuation or vehicle control command.

    Returns:
        (True, refusal_message) if it is a control command, else (False, None).
    """
    q = question.strip()

    # Direct action check
    has_action = bool(_CONTROL_ACTION_PATTERN.search(q))
    has_target = bool(_CONTROL_TARGET_PATTERN.search(q))

    # Explicit phrases
    imperative_starts = (
        "please unlock",
        "unlock the",
        "lock the",
        "turn off",
        "turn on",
        "roll down",
        "roll up",
        "open the",
        "close the",
        "set the temperature",
        "set climate",
        "honk the",
    )
    lower_q = q.lower()
    if any(lower_q.startswith(p) for p in imperative_starts):
        return True, CONTROL_REFUSAL_MESSAGE

    # Direct actuation commands that don't need a separate target
    standalone_actions = ("accelerate", "brake", "honk", "steer")
    if any(lower_q.startswith(act) for act in standalone_actions):
        return True, CONTROL_REFUSAL_MESSAGE

    # Polite request opener + control target and no question intent anywhere:
    # a disguised command ("I'd like the windows down."), not a data question.
    if (
        _REQUEST_OPENER_PATTERN.search(q)
        and has_target
        and not _QUERY_INTENT_PATTERN.search(q)
    ):
        return True, CONTROL_REFUSAL_MESSAGE

    if has_action and has_target:
        # Exceptions: questions asking about data or events, e.g., "when did it start?"
        if _QUERY_INTENT_PATTERN.search(q):
            return False, None
        return True, CONTROL_REFUSAL_MESSAGE

    return False, None


# ---------------------------------------------------------------------------
# 3. Vehicle Vocabulary Gate
# Filters out questions completely outside the vehicle domain.
# ---------------------------------------------------------------------------

OFF_TOPIC_MESSAGE = (
    "This question is outside the scope of VehicleGPT. I can only assist with "
    "vehicle telemetry, driving efficiency, events, battery status, and historical trips."
)


def has_vehicle_terms(question: str) -> bool:
    """
    Checks if the question contains at least one vehicle-related term or synonym.
    Uses three matching strategies:
      1. Exact token match against ALL_VEHICLE_TERMS.
      2. Stem-prefix match against _VEHICLE_STEM_PREFIXES (catches speedup, eficiency).
      3. Conservative fuzzy match (edit distance ≤ 1 for tokens ≥ 5 chars).
      4. Multi-word synonym check for phrases like "stop and go".
    """
    q_lower = question.lower()

    # Recognise specific event IDs like EV-0002
    if re.search(r"\bev-\d+\b", q_lower):
        return True

    tokens = set(re.findall(r"[a-z0-9_/%-]+", q_lower))

    # 1. Exact token match
    if tokens & ALL_VEHICLE_TERMS:
        return True

    # 2. Stem-prefix match
    for token in tokens:
        for prefix in _VEHICLE_STEM_PREFIXES:
            if token.startswith(prefix):
                return True

    # 3. Fuzzy match (edit distance ≤ 1, tokens ≥ 5 chars)
    for token in tokens:
        if _fuzzy_match_vehicle_term(token):
            return True

    # 4. Multi-word synonyms
    for term in ALL_VEHICLE_TERMS:
        if " " in term and term in q_lower:
            return True

    return False


def normalize_tool_args(args: dict[str, Any], question: str = "", tool_name: str = "") -> dict[str, Any]:
    """
    Normalizes synonymous or colloquial argument values into canonical schema enums
    and grounds missing arguments against question phrasing strictly per tool.
    """
    normalized = dict(args)
    q_lower = question.lower()

    # Normalize metric
    if "metric" in normalized and isinstance(normalized["metric"], str):
        raw_m = normalized["metric"].lower().replace("-", "_").strip()
        matched = False
        for metric_enum, syns in METRIC_SYNONYMS.items():
            if raw_m == metric_enum.value.lower() or raw_m in [s.lower().replace("-", "_").replace(" ", "_") for s in syns]:
                normalized["metric"] = metric_enum.value
                matched = True
                break
        if not matched:
            if "fuel" in raw_m or "efficien" in raw_m or "kwh" in raw_m or "energy" in raw_m:
                normalized["metric"] = Metric.energy_wh_per_km.value
            elif "speed" in raw_m or "kmph" in raw_m or "km_per_hour" in raw_m:
                normalized["metric"] = Metric.speed_kmh.value
            elif "temp" in raw_m:
                normalized["metric"] = Metric.pack_temp_c.value
            elif "range" in raw_m:
                normalized["metric"] = Metric.estimated_range_km.value

    # Normalize event type (only for get_events)
    if tool_name == "get_events" or (not tool_name and "type" in normalized):
        if "type" in normalized and isinstance(normalized["type"], str):
            raw_t = normalized["type"].lower().replace("-", "_").strip()
            for ev_enum, syns in EVENT_TYPE_SYNONYMS.items():
                if raw_t == ev_enum.value.lower() or raw_t in [s.lower().replace("-", "_").replace(" ", "_") for s in syns]:
                    normalized["type"] = ev_enum.value
                    break
    elif "type" in normalized and tool_name:
        normalized.pop("type", None)

    # Normalize event_id from question if explicit ID is mentioned
    ev_match = re.search(r"\b(ev-\d+)\b", q_lower)
    if ev_match:
        normalized["event_id"] = ev_match.group(1).upper()
    elif "event_id" in normalized and isinstance(normalized["event_id"], str):
        raw_id = normalized["event_id"].strip()
        sub_m = re.search(r"(ev[-_]?\d+)", raw_id, re.IGNORECASE)
        if sub_m and not raw_id.startswith("latest_"):
            formatted = sub_m.group(1).upper().replace("_", "-")
            if not formatted.startswith("EV-"):
                formatted = "EV-" + formatted[2:]
            normalized["event_id"] = formatted

    # Normalize bucket (ONLY for trend tool)
    if tool_name == "trend" or (not tool_name and "bucket" in normalized):
        if "bucket" in normalized and isinstance(normalized["bucket"], str):
            raw_b = normalized["bucket"].lower().strip()
            for b_enum, syns in BUCKET_SIZE_SYNONYMS.items():
                if raw_b == b_enum.value.lower() or raw_b in [s.lower() for s in syns]:
                    normalized["bucket"] = b_enum.value
                    break
        elif question:
            if "hour" in q_lower:
                normalized["bucket"] = BucketSize.hour.value
            elif "week" in q_lower:
                normalized["bucket"] = BucketSize.week.value
            elif "day" in q_lower:
                normalized["bucket"] = BucketSize.day.value
    elif "bucket" in normalized and tool_name:
        normalized.pop("bucket", None)

    # Normalize direction (ONLY for find_largest_change tool)
    if tool_name == "find_largest_change" or (not tool_name and "direction" in normalized):
        if "direction" in normalized and isinstance(normalized["direction"], str):
            raw_d = normalized["direction"].lower().strip()
            for d_enum, syns in CHANGE_DIRECTION_SYNONYMS.items():
                if raw_d == d_enum.value.lower() or raw_d in [s.lower() for s in syns]:
                    normalized["direction"] = d_enum.value
                    break
        elif question:
            if any(w in q_lower for w in ("rise", "increase", "jump", "higher", "spike")):
                normalized["direction"] = ChangeDirection.rise.value
            elif any(w in q_lower for w in ("drop", "decrease", "reduction", "fell", "loss", "lower")):
                normalized["direction"] = ChangeDirection.drop.value
    elif "direction" in normalized and tool_name:
        normalized.pop("direction", None)

    return normalized


# ---------------------------------------------------------------------------
# 4. Grounding Check
# Verifies that tool arguments are grounded in the user's question.
# ---------------------------------------------------------------------------


def is_grounded(question: str, tool_name: str, args: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Verifies that tool call arguments are supported by the user question.

    Returns:
        (True, None) if grounded, or (False, reason) if hallucinated.
    """
    q_lower = question.lower()

    if tool_name == "get_events":
        ev_type_raw = args.get("type")
        if ev_type_raw:
            try:
                ev_type = EventType(ev_type_raw)
                syns = EVENT_TYPE_SYNONYMS.get(ev_type, [ev_type.value.lower()])
                # Must match at least one synonym for this specific event type
                if not any(s in q_lower for s in syns):
                    return False, f"Event type {ev_type_raw} is not mentioned in question."
            except ValueError:
                return False, f"Unknown event type: {ev_type_raw}"

    elif tool_name == "compare":
        # Compare tool requires comparison context
        comp_terms = {"compare", "versus", "vs", "difference", "between", "city", "highway"}
        if not any(t in q_lower for t in comp_terms):
            return False, "Compare tool selected without comparison context in question."

    elif tool_name == "context_before":
        # Context before requires event or pre-event context
        ctx_terms = {"before", "prior to", "leading up to", "preceding", "warning", "what happened"}
        if not any(t in q_lower for t in ctx_terms):
            return False, "context_before selected without pre-event temporal context."

    elif tool_name == "find_largest_change":
        change_terms = {"largest", "biggest", "drop", "rise", "increase", "decrease", "when did", "most"}
        if not any(t in q_lower for t in change_terms):
            return False, "find_largest_change selected without change/extremum context."

    elif tool_name == "explain_factors":
        explain_terms = {"why", "reason", "cause", "factors", "reduced", "dropped", "fell", "loss"}
        if not any(t in q_lower for t in explain_terms):
            return False, "explain_factors selected without causal/why context."

    return True, None
