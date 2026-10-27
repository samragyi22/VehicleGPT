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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
from pathlib import Path
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

_TMPL_PATH = Path(__file__).parent / "tool_grammar.gbnf.tmpl"


def _enum(cls) -> list[str]:
    """Return sorted list of enum string values for JSON schema 'enum' keyword."""
    return sorted(v.value for v in cls)


def _q(s: str) -> str:
    """
    Wrap a value string in GBNF-escaped double quotes.
    In GBNF, a terminal matching literal "foo" is written: "\"foo\""
    """
    return "\"\\\"" + s + "\\\"\""


def _alt(values: list[str]) -> str:
    """Build a GBNF alternation of quoted string literals."""
    return " | ".join(_q(v) for v in sorted(values))


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
    Build a flat, non-recursive GBNF grammar for VehicleGPT tool calls by
    substituting schema enums into llm/tool_grammar.gbnf.tmpl.

    The grammar enforces:
    - Exactly one of 7 tool branches (search_logs excluded).
    - All enum fields constrained to schema.py values.
    - "range" key absent — filled by the time parser before inference.
    - out_of_scope.reason is one of three fixed literals.
    - compare segments: only city->highway or highway->city.
    - event_id: EV-<digits> or latest_<EVENT_TYPE>.
    """
    tmpl = _TMPL_PATH.read_text(encoding="utf-8")

    substitutions = {
        "{{EVENT_TYPES}}": _alt([e.value for e in EventType]),
        "{{EVENT_IDS}}": _alt(["latest_" + e.value for e in EventType]),
        "{{METRICS}}": _alt([m.value for m in Metric]),
        "{{DIRECTIONS}}": _alt([d.value for d in ChangeDirection]),
        "{{BUCKETS}}": _alt([b.value for b in BucketSize]),
        "{{REASONS}}": _alt(OUT_OF_SCOPE_REASONS),
    }

    for placeholder, val in substitutions.items():
        tmpl = tmpl.replace(placeholder, val)

    return tmpl


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
