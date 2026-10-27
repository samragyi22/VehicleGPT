"""
llm/grammar.py - GBNF Grammar Builder for VehicleGPT.

build_gbnf() generates a flat, non-recursive GBNF that constrains the SLM to
emit exactly one valid tool call.  Key design decisions (per handoff §6 item 2):

- "range" is NOT in the grammar.  The time parser fills it before the model is
  called; the model never needs to emit it.
- out_of_scope.reason is one of three fixed string literals, never free text.
- compare allows only city->highway or highway->city (two orderings, not four).
- All enum fields (metric, event type, direction, bucket, segment) are inline
  alternations of quoted literals generated from schema.py at import time.
- event_id accepts EV-<digits> or latest_<EVENT_TYPE> via explicit rules.
- No recursive rules: object/array helpers are not used.

build_json_schema() is kept for test inspection only; it is not used at
inference time.
"""

from __future__ import annotations

from typing import Any

from schema import (
    BucketSize,
    ChangeDirection,
    EventType,
    Metric,
    Segment,
    TimeRangeKind,
    ToolName,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enum(cls) -> list[str]:
    """Return sorted list of enum string values for JSON schema 'enum' keyword."""
    return sorted(v.value for v in cls)


def _q(s: str) -> str:
    """
    Wrap a value string in GBNF-escaped double quotes.

    In GBNF, a terminal that matches the JSON string "foo" is written:
        "\"foo\""
    which in a Python string literal is:
        '"\\\"foo\\\""'
    """
    return '"\\\"' + s + '\\\""'


def _alt(values: list[str]) -> str:
    """Build a GBNF alternation of quoted string literals."""
    return " | ".join(_q(v) for v in sorted(values))


def _kv(key: str, val: str) -> str:
    """
    GBNF terminal matching a JSON key:value pair with trailing comma.
    e.g. _kv("tool", "get_events") matches: "tool": "get_events",
    In GBNF: "\"tool\": \"get_events\","
    Python:  '"\\\"tool\\\": \\\"get_events\\\","'
    """
    return '"\\\"' + key + '\\\": \\\"' + val + '\\\","'


def _key(k: str) -> str:
    """
    GBNF terminal matching a JSON key prefix (no value).
    e.g. _key("type") matches: "type":
    In GBNF: "\"type\": "
    Python:  '"\\\"type\\\": "'
    """
    return '"\\\"' + k + '\\\": "'


# ---------------------------------------------------------------------------
# GBNF builder
# ---------------------------------------------------------------------------

# Fixed reason phrases for out_of_scope — never free text, never shown to user.
OUT_OF_SCOPE_REASONS = [
    "not_vehicle_data",
    "vehicle_control_prohibited",
    "ambiguous_query",
]


def build_gbnf() -> str:
    """
    Build a flat, non-recursive GBNF grammar for VehicleGPT tool calls.

    The grammar enforces:
    - Exactly one of 7 tool branches (search_logs excluded).
    - All enum fields constrained to schema.py values.
    - "range" key absent — filled by the time parser before inference.
    - out_of_scope.reason is one of three fixed literals.
    - compare segments: only city->highway or highway->city.
    - event_id: EV-<digits> or latest_<EVENT_TYPE>.
    """
    metrics  = _alt(_enum(Metric))
    ev_types = _alt(_enum(EventType))
    dirs     = _alt(_enum(ChangeDirection))
    buckets  = _alt(_enum(BucketSize))
    reasons  = _alt(OUT_OF_SCOPE_REASONS)

    # event_id: latest_<TYPE> literals or EV-DDDD
    # latest_X matches the bare string latest_BATTERY_OVERHEAT_WARNING etc.
    # (no surrounding quotes in the rule — the quotes come from _q)
    ev_latest = " | ".join(_q("latest_" + t.value) for t in EventType)
    # EV-DDDD: matches "EV-" followed by 4 digits, surrounded by JSON quotes
    ev_id_digits = '"\\\"EV-" [0-9] [0-9] [0-9] [0-9] "\\\"" '
    event_id_rule = "event-id ::= " + ev_latest + " | " + ev_id_digits

    lines = [
        (
            "root ::= get-events-call | context-before-call | compare-call"
            " | find-largest-change-call | trend-call | explain-factors-call"
            " | out-of-scope-call"
        ),
        "",
        # ---- get_events ----
        (
            'get-events-call ::= "{" ws '
            + _kv("tool", "get_events")
            + ' ws "\"args\": {" ws '
            + _key("type")
            + " event-type ws \"}\" ws \"}\""
        ),
        "event-type ::= " + ev_types,
        "",
        # ---- context_before ----
        (
            'context-before-call ::= "{" ws '
            + _kv("tool", "context_before")
            + ' ws "\"args\": {" ws '
            + _key("event_id")
            + " event-id cb-opt ws \"}\" ws \"}\""
        ),
        event_id_rule,
        'cb-opt ::= ("," ws ' + _key("seconds") + " cb-seconds)?",
        'cb-seconds ::= "60" | "120" | "180" | "240" | "300"',
        "",
        # ---- compare (only city->highway or highway->city) ----
        (
            'compare-call ::= "{" ws '
            + _kv("tool", "compare")
            + ' ws "\"args\": {" ws '
            + _key("metric")
            + " metric ws \",\" ws"
            + ' "\"segment_a\": \"city\"," ws "\"segment_b\": \"highway\"" ws "}" ws "}"'
            + ' | "{" ws '
            + _kv("tool", "compare")
            + ' ws "\"args\": {" ws '
            + _key("metric")
            + " metric ws \",\" ws"
            + ' "\"segment_a\": \"highway\"," ws "\"segment_b\": \"city\"" ws "}" ws "}"'
        ),
        "metric ::= " + metrics,
        "",
        # ---- find_largest_change ----
        (
            'find-largest-change-call ::= "{" ws '
            + _kv("tool", "find_largest_change")
            + ' ws "\"args\": {" ws '
            + _key("metric")
            + " metric ws \",\" ws "
            + _key("direction")
            + " direction ws \"}\" ws \"}\""
        ),
        "direction ::= " + dirs,
        "",
        # ---- trend ----
        (
            'trend-call ::= "{" ws '
            + _kv("tool", "trend")
            + ' ws "\"args\": {" ws '
            + _key("metric")
            + " metric ws \",\" ws "
            + _key("bucket")
            + " bucket ws \"}\" ws \"}\""
        ),
        "bucket ::= " + buckets,
        "",
        # ---- explain_factors ----
        (
            'explain-factors-call ::= "{" ws '
            + _kv("tool", "explain_factors")
            + ' ws "\"args\": {" ws '
            + _key("metric")
            + " metric ws \"}\" ws \"}\""
        ),
        "",
        # ---- out_of_scope ----
        (
            'out-of-scope-call ::= "{" ws '
            + _kv("tool", "out_of_scope")
            + ' ws "\"args\": {" ws '
            + _key("reason")
            + " oos-reason ws \"}\" ws \"}\""
        ),
        "oos-reason ::= " + reasons,
        "",
        "ws ::= [ \\t\\n\\r]*",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON Schema (used by tests only, not at inference time)
# ---------------------------------------------------------------------------

def _symbolic_time_range() -> dict[str, Any]:
    fixed_kinds = [
        k.value for k in TimeRangeKind
        if k not in (TimeRangeKind.last_n_days, TimeRangeKind.last_n_hours)
    ]
    return {
        "anyOf": [
            {
                "type": "object",
                "properties": {"kind": {"enum": fixed_kinds}},
                "required": ["kind"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "kind": {"const": "last_n_days"},
                    "n": {"type": "integer", "minimum": 1, "maximum": 28},
                },
                "required": ["kind", "n"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "kind": {"const": "last_n_hours"},
                    "n": {"type": "integer", "minimum": 1, "maximum": 72},
                },
                "required": ["kind", "n"],
                "additionalProperties": False,
            },
        ]
    }


def _branch_get_events() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool": {"const": ToolName.get_events.value},
            "args": {
                "type": "object",
                "properties": {"type": {"enum": _enum(EventType)}},
                "required": ["type"],
                "additionalProperties": False,
            },
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


def _branch_context_before() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool": {"const": ToolName.context_before.value},
            "args": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "pattern": "^(EV-\\d+|latest_(BATTERY_OVERHEAT_WARNING|ADAS_DISENGAGEMENT|LOW_SOC_WARNING))$",
                    },
                    "seconds": {"type": "integer", "minimum": 1, "maximum": 300},
                },
                "required": ["event_id"],
                "additionalProperties": False,
            },
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


def _branch_compare() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool": {"const": ToolName.compare.value},
            "args": {
                "type": "object",
                "properties": {
                    "metric": {"enum": _enum(Metric)},
                    "segment_a": {"enum": _enum(Segment)},
                    "segment_b": {"enum": _enum(Segment)},
                },
                "required": ["metric"],
                "additionalProperties": False,
            },
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


def _branch_find_largest_change() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool": {"const": ToolName.find_largest_change.value},
            "args": {
                "type": "object",
                "properties": {
                    "metric": {"enum": _enum(Metric)},
                    "direction": {"enum": _enum(ChangeDirection)},
                },
                "required": ["metric"],
                "additionalProperties": False,
            },
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


def _branch_trend() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool": {"const": ToolName.trend.value},
            "args": {
                "type": "object",
                "properties": {
                    "metric": {"enum": _enum(Metric)},
                    "bucket": {"enum": _enum(BucketSize)},
                },
                "required": ["metric"],
                "additionalProperties": False,
            },
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


def _branch_explain_factors() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool": {"const": ToolName.explain_factors.value},
            "args": {
                "type": "object",
                "properties": {"metric": {"enum": _enum(Metric)}},
                "required": ["metric"],
                "additionalProperties": False,
            },
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


def _branch_out_of_scope() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool": {"const": ToolName.out_of_scope.value},
            "args": {
                "type": "object",
                "properties": {"reason": {"enum": OUT_OF_SCOPE_REASONS}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


def build_json_schema() -> dict[str, Any]:
    """
    Build the JSON Schema for a VehicleGPT tool call (test inspection only).
    Note: range is absent from args — it is injected by the time parser.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "VehicleGPT Tool Call",
        "anyOf": [
            _branch_get_events(),
            _branch_context_before(),
            _branch_compare(),
            _branch_find_largest_change(),
            _branch_trend(),
            _branch_explain_factors(),
            _branch_out_of_scope(),
        ],
    }


def load_grammar():
    """
    Load and return a LlamaGrammar object from the built GBNF.
    This is the object passed to llama_cpp Llama.create_chat_completion().
    """
    try:
        from llama_cpp import LlamaGrammar  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required for grammar loading."
        ) from exc

    return LlamaGrammar.from_string(build_gbnf())
