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

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config
from engine.render import render_answer, render_evidence
from engine.tools import (
    compare,
    context_before,
    explain_factors,
    find_largest_change,
    get_events,
    out_of_scope,
    trend,
)
from engine.validate import prepare_executable_args, validate_tool_call
from engine.verify import verify_or_fallback
from schema import ToolName, ToolOutput

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
Q: Show all ADAS disengagement events.
A: {"tool": "get_events", "args": {"type": "ADAS_DISENGAGEMENT"}}

Q: Why has my driving range reduced?
A: {"tool": "explain_factors", "args": {"metric": "estimated_range_km"}}

Q: When did energy efficiency show the largest drop?
A: {"tool": "find_largest_change", "args": {"metric": "energy_wh_per_km", "direction": "drop"}}

Q: What happened before the battery overheating warning?
A: {"tool": "context_before", "args": {"event_id": "latest_BATTERY_OVERHEAT_WARNING", "seconds": 300}}

Q: Compare energy consumption between city and highway driving.
A: {"tool": "compare", "args": {"metric": "energy_wh_per_km", "segment_a": "city", "segment_b": "highway"}}

Q: What is the capital of France?
A: {"tool": "out_of_scope", "args": {"reason": "not_vehicle_data"}}

Q: Turn off the air conditioning.
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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RouterResult:
    """
    Full pipeline output for one question.

    Attributes:
        question:       Original user question.
        tool_call_raw:  Raw JSON string emitted by the model.
        tool_name:      Resolved ToolName (or None on parse failure).
        tool_args_dict: Validated + resolved executable args as dict.
        tool_output:    ToolOutput from engine (or None on execution error).
        answer:         Final rendered answer shown to the user.
        explanation:    Optional 1-2 sentence model explanation (post-evidence).
        verified:       True if all numbers in the answer appear in tool output.
        error:          Non-empty string if any step failed.
        model_ms:       Milliseconds spent in the SLM (routing + optional explanation).
        query_ms:       Milliseconds spent in engine tool execution.
        render_ms:      Milliseconds spent in rendering + verification.
    """
    question: str
    tool_call_raw: str = ""
    tool_name: ToolName | None = None
    tool_args_dict: dict[str, Any] = field(default_factory=dict)
    tool_output: ToolOutput | None = None
    answer: str = ""
    explanation: str = ""
    verified: bool = False
    error: str = ""
    model_ms: float = 0.0
    query_ms: float = 0.0
    render_ms: float = 0.0


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
# Core pipeline steps
# ---------------------------------------------------------------------------

def _call_model_for_tool(
    llm: Any,
    question: str,
    grammar,
) -> tuple[str, float]:
    """
    Run grammar-constrained inference to get the tool call JSON string.
    Returns (raw_json_string, elapsed_ms).
    """
    t0 = time.perf_counter()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        grammar=grammar,
        temperature=0,
        max_tokens=200,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    raw = response["choices"][0]["message"]["content"].strip()
    return raw, elapsed_ms


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


def _execute_tool(tool_name: ToolName, exec_args) -> tuple[ToolOutput, float]:
    """
    Dispatch the validated executable args to the correct engine function.
    Returns (ToolOutput, elapsed_ms).
    """
    TOOL_FN = {
        ToolName.get_events: get_events,
        ToolName.context_before: context_before,
        ToolName.compare: compare,
        ToolName.find_largest_change: find_largest_change,
        ToolName.trend: trend,
        ToolName.explain_factors: explain_factors,
        ToolName.out_of_scope: out_of_scope,
    }
    fn = TOOL_FN.get(tool_name)
    if fn is None:
        raise ValueError(f"No executor registered for tool: {tool_name}")

    args_dict = exec_args.model_dump()
    t0 = time.perf_counter()
    output = fn(**args_dict)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return output, elapsed_ms


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def route(
    question: str,
    model_path: Path | str | None = None,
    with_explanation: bool = True,
) -> RouterResult:
    """
    Full pipeline: question → tool call → engine → rendered + verified answer.

    Args:
        question:         Natural-language user question.
        model_path:       Optional override for model GGUF path.
                          Defaults to config.DEFAULT_MODEL_PATH.
        with_explanation: If True, ask the model for a 1-2 sentence explanation
                          after the evidence is secured. Adds ~model latency.

    Returns:
        RouterResult with all pipeline fields populated.
    """
    result = RouterResult(question=question)

    # --- Step 1: Load model and grammar ---
    try:
        llm = _get_llm(model_path)
        from llm.grammar import load_grammar
        grammar = load_grammar()
    except ImportError as exc:
        result.error = f"Model/grammar load failed: {exc}"
        return result

    # --- Step 1.5: Time parser runs BEFORE the model ---
    from llm.timeparse import UNSUPPORTED_TIME_MESSAGE, parse_time_range
    authoritative_range, _ = parse_time_range(question, model_range=None)
    if authoritative_range is None:
        result.answer = UNSUPPORTED_TIME_MESSAGE
        result.error = "unsupported_time"
        return result

    # --- Step 1.6: Check control commands and vocabulary gates BEFORE model ---
    from llm.guards import OFF_TOPIC_MESSAGE, has_vehicle_terms, is_control_command
    is_ctrl, _ = is_control_command(question)
    if is_ctrl:
        result.tool_name = ToolName.out_of_scope
        result.answer = OFF_TOPIC_MESSAGE
        result.error = "control_refused"
        result.verified = True
        return result
    
    has_terms = has_vehicle_terms(question)
    if not has_terms:
        result.tool_name = ToolName.out_of_scope
        result.answer = OFF_TOPIC_MESSAGE
        result.error = "off_topic"
        result.verified = True
        return result

    # --- Step 1.7: Deterministic tool override (why -> explain_factors, when -> find_largest_change) ---
    # --- Step 1.7: Deterministic tool override (why -> explain_factors, when -> find_largest_change) ---
    forced_tool = _deterministic_tool_override(question)

    # --- Step 2: Grammar-constrained tool call ---
    try:
        raw_json, model_ms = _call_model_for_tool(llm, question, grammar)
        result.tool_call_raw = raw_json
        result.model_ms = model_ms
    except BaseException as exc:  # noqa: BLE001 — any model error surfaces as result.error
        result.error = f"Model inference failed: {exc}"
        return result

    # --- Step 3: Parse JSON ---
    try:
        call_dict = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        result.error = f"Model output is not valid JSON: {exc}. Raw: {raw_json!r}"
        return result

    # Apply deterministic tool override if triggered
    if forced_tool is not None and call_dict.get("tool") != forced_tool:
        logger.info(
            "Deterministic override: %s -> %s for question: %s",
            call_dict.get("tool"), forced_tool, question,
        )
        call_dict["tool"] = forced_tool
        # Replace args with clean args for the new tool (old args may not match)
        call_dict["args"] = {}
        # Ensure metric is present for explain_factors / find_largest_change
        call_dict.setdefault("args", {})["metric"] = "estimated_range_km"

    # Inject the authoritative time range (model never emits range)
    if call_dict.get("tool") not in ("context_before", "out_of_scope"):
        call_dict.setdefault("args", {})["range"] = authoritative_range

    # Early exit: model returned out_of_scope — use fixed message, never show reason
    if call_dict.get("tool") == "out_of_scope":
        from llm.guards import OFF_TOPIC_MESSAGE
        result.tool_name = ToolName.out_of_scope
        result.answer = OFF_TOPIC_MESSAGE
        result.verified = True
        return result

    # --- Step 3.5: Normalize tool args ---
    from llm.guards import normalize_tool_args
    call_dict["args"] = normalize_tool_args(
        call_dict.get("args", {}),
        question=question,
        tool_name=call_dict.get("tool", ""),
    )

    # --- Step 4: Schema validation (second gate after grammar) ---
    try:
        tool_name, validated_sym_args = validate_tool_call(call_dict)
        result.tool_name = tool_name
    except BaseException as exc:  # noqa: BLE001
        result.error = f"Tool call validation failed: {exc}. Call: {call_dict}"
        return result

    # --- Step 5: Date resolution ---
    try:
        exec_args = prepare_executable_args(tool_name, validated_sym_args)
        result.tool_args_dict = exec_args.model_dump()
    except BaseException as exc:  # noqa: BLE001
        result.error = f"Argument preparation failed: {exc}"
        return result

    # --- Step 6: Tool execution ---
    try:
        tool_output, query_ms = _execute_tool(tool_name, exec_args)
        result.query_ms = query_ms
        result.tool_output = tool_output
    except BaseException as exc:  # noqa: BLE001
        result.error = f"Tool execution failed: {exc}"
        return result

    # --- Step 7: Template rendering ---
    t_render0 = time.perf_counter()
    template_answer = render_answer(tool_name, tool_output)
    evidence_text = render_evidence(tool_output)
    output_dict = tool_output.model_dump()

    # --- Step 8: Optional model explanation ---
    explanation = ""
    if with_explanation and tool_name != ToolName.out_of_scope:
        try:
            raw_explanation, exp_ms = _call_model_for_explanation(llm, evidence_text)
            result.model_ms += exp_ms
        except BaseException as exc:  # noqa: BLE001
            logger.warning("Explanation model call failed: %s", exc)
            raw_explanation = ""

        # Verify: every number the model introduces must appear in the tool output.
        if raw_explanation:
            checked_exp, exp_verified = verify_or_fallback(
                raw_explanation, output_dict, fallback=""
            )
            if exp_verified and checked_exp:
                explanation = checked_exp
            else:
                logger.info("Model explanation failed verification; discarding.")
    result.explanation = explanation

    # --- Step 9: Final answer assembly ---
    # Template answer is always shown; explanation appended if verified.
    final_answer = template_answer
    if explanation:
        final_answer = f"{template_answer}\n\n{explanation}"

    # Verify the final answer (template numbers must all be in tool output).
    final_answer, verified = verify_or_fallback(final_answer, output_dict, fallback=template_answer)
    result.answer = final_answer
    result.verified = verified
    result.render_ms = (time.perf_counter() - t_render0) * 1000

    return result


def route_batch(
    questions: list[str],
    model_path: Path | str | None = None,
    with_explanation: bool = False,
) -> list[RouterResult]:
    """
    Route a list of questions sharing one model instance.
    Explanation is off by default for batch runs (eval/benchmark).
    """
    llm = _get_llm(model_path)  # warm up once
    _ = llm  # used implicitly via _get_llm singleton
    return [route(q, model_path=model_path, with_explanation=with_explanation) for q in questions]
