"""
llm/router.py - SLM Tool Router for VehicleGPT.

Routes natural-language vehicle questions to deterministic query tools via
grammar-constrained JSON decoding, then executes, renders, and verifies the answer.

Pipeline per PROJECT_CONTEXT.md Section 3:
  question
    -> SLM (grammar-constrained JSON tool call)
    -> validate_tool_call (schema.py Pydantic; extra='forbid')
    -> prepare_executable_args (date resolution)
    -> tool execution (read-only engine)
    -> render_answer (template, no model text for numbers)
    -> verify (numbers in answer must appear in tool output)
    -> [optional] SLM short explanation (≤ 2 sentences, run after evidence is secured)
    -> verify_or_fallback (discard model text if any number is hallucinated)

Security boundaries maintained throughout:
- Model output never reaches SQL or a shell.
- DB is opened read-only via config.get_readonly_db_uri().
- Parameterized queries and row limits are enforced inside engine/tools.py.
- Grammar rejects tool names and enum values not in schema.py at generation time.
- Code re-validates after decoding (grammar alone is not trusted).

Timing: latency is split and returned as model_ms, query_ms, render_ms.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are VehicleGPT, an on-device assistant for a single electric vehicle.
You route the driver's question to exactly one tool and return only a JSON tool call.

Available tools:
- get_events: list battery overheating warnings, ADAS disengagements, or low-charge warnings.
- context_before: show signals in the seconds before a specific event.
- compare: compare a metric between city and highway driving.
- find_largest_change: find when a metric dropped or rose the most.
- trend: show how a metric changed over time (hourly / daily / weekly).
- explain_factors: explain why a metric changed, ranking possible contributing factors.
- out_of_scope: use for ANY question not answerable from vehicle telemetry data.

Supported metrics: energy_wh_per_km, estimated_range_km, pack_temp_c, soc_pct, speed_kmh, hvac_power_kw
Supported event types: BATTERY_OVERHEAT_WARNING, ADAS_DISENGAGEMENT, LOW_SOC_WARNING
Supported directions: drop, rise
Supported buckets: hour, day, week

Do NOT include a "range" field. The time range is handled separately.

Rules:
1. Reply with ONLY one JSON object, no other text.
2. Use out_of_scope for: general knowledge, chit-chat, vehicle control, weather, navigation.
3. For context_before: use exact event ID (e.g. "EV-0042") or "latest_<EVENT_TYPE>".
4. out_of_scope reason must be exactly one of: not_vehicle_data, vehicle_control_prohibited, ambiguous_query.

Examples:
Q: List events where the driver assistance system disengaged.
A: {"tool": "get_events", "args": {"type": "ADAS_DISENGAGEMENT"}}

Q: Why has my driving range reduced?
A: {"tool": "explain_factors", "args": {"metric": "estimated_range_km"}}

Q: Which point in time had the biggest efficiency decrease?
A: {"tool": "find_largest_change", "args": {"metric": "energy_wh_per_km", "direction": "drop"}}

Q: Show signal values leading up to the last overheat warning.
A: {"tool": "context_before", "args": {"event_id": "latest_BATTERY_OVERHEAT_WARNING", "seconds": 300}}

Q: Compute the energy usage difference on city roads versus motorway.
A: {"tool": "compare", "args": {"metric": "energy_wh_per_km", "segment_a": "city", "segment_b": "highway"}}

Q: Tell me the population of France.
A: {"tool": "out_of_scope", "args": {"reason": "not_vehicle_data"}}

Q: Lower the driver side glass.
A: {"tool": "out_of_scope", "args": {"reason": "vehicle_control_prohibited"}}
"""


_EXPLANATION_PROMPT = """\
In one to two sentences, explain this vehicle data result to the driver.
Do NOT introduce any numbers or facts that are not in the evidence below.
Evidence:
{evidence}

Answer (1-2 sentences only):"""

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_llm_instance: Any | None = None
_loaded_model_path: str | None = None


def _get_llm(model_path: Path | str | None = None) -> Any:
    """
    Return a cached Llama instance.  Loads the model on first call.
    Thread-safety note: this is a prototype; production use would use a lock.
    """
    global _llm_instance, _loaded_model_path

    resolved = str(Path(model_path).resolve() if model_path else config.DEFAULT_MODEL_PATH.resolve())

    if _llm_instance is not None and _loaded_model_path == resolved:
        return _llm_instance

    try:
        from llama_cpp import Llama  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required for the SLM router. "
            "Install with: uv pip install llama-cpp-python"
        ) from exc

    logger.info("Loading model: %s", resolved)
    _llm_instance = Llama(
        model_path=resolved,
        n_ctx=1024,
        n_batch=128,
        n_threads=2,
        n_gpu_layers=0,   # CPU-only; edge device target
        verbose=False,
    )
    _loaded_model_path = resolved
    logger.info("Model loaded.")
    return _llm_instance


def unload_model() -> None:
    """Release the cached model (useful for testing or memory-constrained envs)."""
    global _llm_instance, _loaded_model_path
    _llm_instance = None
    _loaded_model_path = None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Module ends. run_pipeline (engine/pipeline.py) is the ONLY pipeline entry
# point; route() was retired on freeze-prep.

# ---------------------------------------------------------------------------
# Deterministic pre-model routing rules (handoff §6 items 3, 6)
# ---------------------------------------------------------------------------

import re as _re

_WHY_PATTERN = _re.compile(
    r"\b(why|reason|cause|factors?|what\s+caused|how\s+come)\b", _re.IGNORECASE
)
_WHEN_PATTERN = _re.compile(
    r"\b(when|biggest|largest|most|highest|lowest)\b", _re.IGNORECASE
)

# Metrics that pair naturally with explain_factors vs find_largest_change
_RANGE_EFFICIENCY_PATTERN = _re.compile(
    r"\b(range|efficiency|consumption|energy|wh)\b", _re.IGNORECASE
)


def _deterministic_tool_override(question: str) -> str | None:
    """
    Return a forced tool name string when the question matches a deterministic
    rule, or None to let the model decide.

    Rules (handoff §6 item 6):
    - "why" + range/efficiency keyword  -> explain_factors
    - "when" / "biggest" / "largest"    -> find_largest_change
    """
    q = question
    if _WHY_PATTERN.search(q) and _RANGE_EFFICIENCY_PATTERN.search(q):
        return "explain_factors"
    if _WHEN_PATTERN.search(q):
        return "find_largest_change"
    return None


# ---------------------------------------------------------------------------

def _call_model_for_explanation(
    llm: Any,
    evidence_text: str,
) -> tuple[str, float]:
    """
    Run a second (unconstrained) inference call to generate a short explanation.
    No grammar — but the verifier will check the output before it reaches the user.
    Returns (explanation_text, elapsed_ms).
    """
    prompt = _EXPLANATION_PROMPT.format(evidence=evidence_text)
    t0 = time.perf_counter()
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=120,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    raw = response["choices"][0]["message"]["content"].strip()
    return raw, elapsed_ms

