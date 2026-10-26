"""
llm/grammar.py - JSON Schema and GBNF Grammar Builder for VehicleGPT.

Generates the grammar that constrains the SLM to emit exactly one valid tool call
as a JSON object: {"tool": "<name>", "args": {...}}.

The schema is derived entirely from schema.py enums and arg models so that
adding a new enum value in schema.py automatically updates the grammar.

Design notes:
- We use JSON Schema (anyOf over all tool branches) and convert to GBNF via
  llama_cpp.llama_grammar.json_schema_to_gbnf.
- Each branch has "tool": {"const": "<name>"} and an "args" object tailored to
  that tool's required and optional arguments.
- LlamaGrammar.from_json_schema is the clean public API; build_json_schema()
  is exposed so tests can inspect the schema without loading a model.
- SymbolicTimeRange is inlined as an object with a "kind" enum, so the grammar
  rejects unknown kinds at generation time.
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


def _symbolic_time_range() -> dict[str, Any]:
    """
    Inline JSON Schema for SymbolicTimeRange.

    Using anyOf so that:
    - Kinds that require 'n' (last_n_days, last_n_hours) must include it.
    - All other kinds must NOT include 'n'.
    """
    fixed_kinds = [
        k.value for k in TimeRangeKind
        if k not in (TimeRangeKind.last_n_days, TimeRangeKind.last_n_hours)
    ]
    return {
        "anyOf": [
            # Fixed kinds: no 'n' field allowed
            {
                "type": "object",
                "properties": {
                    "kind": {"enum": fixed_kinds},
                },
                "required": ["kind"],
                "additionalProperties": False,
            },
            # last_n_days: n required in [1, 28]
            {
                "type": "object",
                "properties": {
                    "kind": {"const": "last_n_days"},
                    "n": {"type": "integer", "minimum": 1, "maximum": 28},
                },
                "required": ["kind", "n"],
                "additionalProperties": False,
            },
            # last_n_hours: n required in [1, 72]
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


# ---------------------------------------------------------------------------
# Per-tool branches
# ---------------------------------------------------------------------------

def _branch_get_events() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool": {"const": ToolName.get_events.value},
            "args": {
                "type": "object",
                "properties": {
                    "type": {"enum": _enum(EventType)},
                    "range": _symbolic_time_range(),
                },
                "required": ["type", "range"],
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
                        # Rough pattern accepted by the model; code validates exactly.
                        "pattern": "^(EV-\\d+|latest_(BATTERY_OVERHEAT_WARNING|ADAS_DISENGAGEMENT|LOW_SOC_WARNING))$",
                    },
                    "seconds": {"type": "integer", "minimum": 1, "maximum": 300},
                    "signals": {
                        "type": "array",
                        "items": {
                            "enum": [
                                "adas_active",
                                "ambient_temp_c",
                                "hvac_power_kw",
                                "pack_current_a",
                                "pack_temp_c",
                                "pack_voltage_v",
                                "power_kw",
                                "road_segment",
                                "soc_pct",
                                "speed_kmh",
                            ]
                        },
                    },
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
                    "range": _symbolic_time_range(),
                    "segment_a": {"enum": _enum(Segment)},
                    "segment_b": {"enum": _enum(Segment)},
                },
                "required": ["metric", "range"],
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
                    "range": _symbolic_time_range(),
                    "direction": {"enum": _enum(ChangeDirection)},
                },
                "required": ["metric", "range"],
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
                    "range": _symbolic_time_range(),
                    "bucket": {"enum": _enum(BucketSize)},
                },
                "required": ["metric", "range"],
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
                "properties": {
                    "metric": {"enum": _enum(Metric)},
                    "range": _symbolic_time_range(),
                },
                "required": ["metric", "range"],
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
                "properties": {
                    "reason": {"type": "string"},
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_json_schema() -> dict[str, Any]:
    """
    Build the complete JSON Schema for a VehicleGPT tool call.

    The schema is an anyOf over all enabled tools. The grammar enforcer will
    reject any output that does not match one of these branches at token level.
    search_logs is omitted because the FTS5 table is not yet built.
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


def build_gbnf() -> str:
    """
    Convert the JSON schema to a GBNF grammar string using llama_cpp's built-in
    converter. Returns the GBNF string suitable for LlamaGrammar.from_string().

    Note: json_schema_to_gbnf in llama_cpp 0.3.x expects a JSON *string*,
    not a dict, so we serialize first.

    Raises ImportError if llama_cpp is not installed (it is an optional dep).
    """
    import json as _json

    try:
        from llama_cpp.llama_grammar import json_schema_to_gbnf  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required for grammar generation. "
            "Install it with: uv pip install llama-cpp-python"
        ) from exc

    schema = build_json_schema()
    # llama_cpp 0.3.x expects a JSON string, not a dict.
    return json_schema_to_gbnf(_json.dumps(schema))


def load_grammar():
    """
    Load and return a LlamaGrammar object from the built GBNF.
    This is the object passed to llama_cpp Llama.create_chat_completion().

    Note: LlamaGrammar.from_json_schema in llama_cpp 0.3.x also expects a
    JSON string, not a dict.
    """
    import json as _json

    try:
        from llama_cpp import LlamaGrammar  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required for grammar loading."
        ) from exc

    # Serialize to string — llama_cpp 0.3.x API requires a JSON string here.
    return LlamaGrammar.from_json_schema(_json.dumps(build_json_schema()))
