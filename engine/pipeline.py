"""
engine/pipeline.py - End-to-End Query Pipeline for VehicleGPT.

Sequences:
1. Guard: Control-command interception (refuses actuation/control commands).
2. Gate: Vehicle-vocabulary check (rejects completely off-topic queries).
3. Router: Grammar-constrained SLM inference producing a valid tool call.
4. Validation: Schema & enum validation with extra='forbid'.
5. Time Parser: Deterministic NL time expression parsing with model override & disagreement logging.
6. Grounding: Verifies that tool arguments are grounded in the user's question.
7. Date Resolver: Maps symbolic time expressions to day-aligned ISO ranges.
8. Tool Execution: Safe, read-only parameterized SQLite queries.
9. Renderer: Deterministic answer and evidence rendering from templates.
10. Verifier: Number-vs-evidence verification to guarantee zero hallucinations.

Returns:
- answer: Final human-readable response (with friendly messages for refusals)
- tool_call: Validated tool call dict
- evidence: Evidence rows/identifiers
- timings: Per-stage timings (ms)
- status: Status code ("ok", "control_refused", "off_topic", "out_of_scope", "validation_error", "ungrounded", "query_error")
- verified: True if all numbers in the answer are strictly backed by tool output.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
from llm.guards import (
    CONTROL_REFUSAL_MESSAGE,
    OFF_TOPIC_MESSAGE,
    has_vehicle_terms,
    is_control_command,
    is_grounded,
    normalize_tool_args,
)
from llm.timeparse import parse_time_range
from schema import ToolName, ToolOutput

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Standard output structure for the VehicleGPT pipeline.
    """
    answer: str
    tool_call: dict[str, Any] | None = None
    evidence: list[Any] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    status: str = "ok"
    verified: bool = False
    tokens: dict[str, int] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TOOL_MAP = {
    ToolName.get_events: get_events,
    ToolName.context_before: context_before,
    ToolName.compare: compare,
    ToolName.find_largest_change: find_largest_change,
    ToolName.trend: trend,
    ToolName.explain_factors: explain_factors,
    ToolName.out_of_scope: out_of_scope,
}


def run_pipeline(
    question: str,
    model_path: Path | str | None = None,
    with_explanation: bool = False,
) -> PipelineResult:
    """
    Executes the complete VehicleGPT pipeline on a natural language question.
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {
        "guard_ms": 0.0,
        "gate_ms": 0.0,
        "model_ms": 0.0,
        "validation_ms": 0.0,
        "timeparse_ms": 0.0,
        "query_ms": 0.0,
        "render_ms": 0.0,
        "verify_ms": 0.0,
        "total_ms": 0.0,
    }
    tokens: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    # =========================================================================
    # Stage 1: Control-Command Guard
    # =========================================================================
    t0 = time.perf_counter()
    is_ctrl, ctrl_msg = is_control_command(question)
    timings["guard_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    if is_ctrl:
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return PipelineResult(
            answer=ctrl_msg or CONTROL_REFUSAL_MESSAGE,
            tool_call={"tool": "out_of_scope", "args": {"reason": "vehicle_control_prohibited"}},
            evidence=[],
            timings=timings,
            status="control_refused",
            verified=True,
            tokens=tokens,
        )

    # =========================================================================
    # Stage 2: Vehicle-Vocabulary Gate
    # =========================================================================
    t0 = time.perf_counter()
    in_vocab = has_vehicle_terms(question)
    timings["gate_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    if not in_vocab:
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return PipelineResult(
            answer=OFF_TOPIC_MESSAGE,
            tool_call={"tool": "out_of_scope", "args": {"reason": "off_topic"}},
            evidence=[],
            timings=timings,
            status="off_topic",
            verified=True,
            tokens=tokens,
        )

    # =========================================================================
    # Stage 3: SLM Router (Grammar-constrained JSON decoding)
    # =========================================================================
    from llm.grammar import load_grammar
    from llm.router import _SYSTEM_PROMPT, _get_llm

    try:
        llm = _get_llm(model_path)
        grammar = load_grammar()
    except Exception as exc:  # noqa: BLE001
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return PipelineResult(
            answer="Assistant initialization error: unable to load language model.",
            status="error",
            error=str(exc),
            timings=timings,
            tokens=tokens,
        )

    t0 = time.perf_counter()
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            grammar=grammar,
            temperature=0,
            max_tokens=200,
        )
        timings["model_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        raw_json = response["choices"][0]["message"]["content"].strip()
        usage = response.get("usage", {})
        tokens["prompt_tokens"] = usage.get("prompt_tokens", 0)
        tokens["completion_tokens"] = usage.get("completion_tokens", 0)
        call_dict = json.loads(raw_json)
    except Exception as exc:  # noqa: BLE001
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return PipelineResult(
            answer="Inference error: failed to produce structured tool call.",
            status="error",
            error=str(exc),
            timings=timings,
            tokens=tokens,
        )

    # If the model explicitly routed to out_of_scope
    if call_dict.get("tool") == "out_of_scope":
        # Always use the fixed off-topic message; never expose the model's
        # free-text "reason" field to the user.
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return PipelineResult(
            answer=OFF_TOPIC_MESSAGE,
            tool_call=call_dict,
            evidence=[],
            timings=timings,
            status="out_of_scope",
            verified=True,
            tokens=tokens,
        )

    # =========================================================================
    # Stage 4: Deterministic Time Parsing Override
    # =========================================================================
    from llm.timeparse import UNSUPPORTED_TIME_MESSAGE
    t0 = time.perf_counter()
    args = call_dict.get("args", {})
    if "range" in args or call_dict.get("tool") in (
        "get_events",
        "compare",
        "find_largest_change",
        "trend",
        "explain_factors",
    ):
        model_range = args.get("range")
        authoritative_range, _ = parse_time_range(question, model_range=model_range)
        if authoritative_range is None:
            # Unresolvable temporal phrase — ask user to rephrase.
            timings["timeparse_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return PipelineResult(
                answer=UNSUPPORTED_TIME_MESSAGE,
                tool_call=call_dict,
                evidence=[],
                timings=timings,
                status="unsupported_time",
                verified=True,
                tokens=tokens,
            )
        args["range"] = authoritative_range
    args = normalize_tool_args(args, question=question, tool_name=call_dict.get("tool", ""))
    call_dict["args"] = args
    timings["timeparse_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # =========================================================================
    # Stage 5: Schema Validation
    # =========================================================================
    t0 = time.perf_counter()
    try:
        tool_name, validated_sym_args = validate_tool_call(call_dict)
    except Exception as exc:  # noqa: BLE001
        timings["validation_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return PipelineResult(
            answer="The query arguments could not be validated against the vehicle schema.",
            tool_call=call_dict,
            status="validation_error",
            error=str(exc),
            timings=timings,
            tokens=tokens,
        )
    timings["validation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # =========================================================================
    # Stage 6: Grounding Check
    # =========================================================================
    is_grnd, grnd_reason = is_grounded(question, tool_name.value, args)
    if not is_grnd:
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return PipelineResult(
            answer=f"Telemetry grounding failure: {grnd_reason}",
            tool_call=call_dict,
            status="ungrounded",
            error=grnd_reason or "ungrounded",
            timings=timings,
            tokens=tokens,
        )

    # =========================================================================
    # Stage 7: Date Resolution
    # =========================================================================
    try:
        exec_args = prepare_executable_args(tool_name, validated_sym_args)
    except Exception as exc:  # noqa: BLE001
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return PipelineResult(
            answer="Date resolution error: unable to resolve time range.",
            tool_call=call_dict,
            status="validation_error",
            error=str(exc),
            timings=timings,
            tokens=tokens,
        )

    # =========================================================================
    # Stage 8: Tool Execution (Read-only Database Query)
    # =========================================================================
    t0 = time.perf_counter()
    fn = _TOOL_MAP.get(tool_name)
    if fn is None:
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return PipelineResult(
            answer=f"Unknown tool executor for {tool_name}.",
            tool_call=call_dict,
            status="query_error",
            timings=timings,
            tokens=tokens,
        )

    try:
        tool_output: ToolOutput = fn(**exec_args.model_dump())
    except Exception as exc:  # noqa: BLE001
        timings["query_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return PipelineResult(
            answer="Database execution failed while querying vehicle telemetry.",
            tool_call=call_dict,
            status="query_error",
            error=str(exc),
            timings=timings,
            tokens=tokens,
        )
    timings["query_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # =========================================================================
    # Stage 9: Template Rendering
    # =========================================================================
    t0 = time.perf_counter()
    template_answer = render_answer(tool_name, tool_output)
    timings["render_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Optional model explanation
    final_text = template_answer
    if with_explanation and tool_name != ToolName.out_of_scope:
        try:
            evidence_text = render_evidence(tool_output)
            from llm.router import _call_model_for_explanation
            explanation, exp_ms = _call_model_for_explanation(llm, evidence_text)
            timings["model_ms"] += round(exp_ms, 2)
            if explanation:
                final_text = f"{template_answer}\n\nExplanation: {explanation}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Explanation generation failed: %s", exc)

    # =========================================================================
    # Stage 10: Verifier (Number-vs-Evidence check)
    # =========================================================================
    t0 = time.perf_counter()
    tool_out_dict = tool_output.model_dump()
    verified_answer, is_verified = verify_or_fallback(
        rendered_answer=final_text,
        tool_output_dict=tool_out_dict,
        fallback=template_answer,
    )
    timings["verify_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

    return PipelineResult(
        answer=verified_answer,
        tool_call=call_dict,
        evidence=tool_output.evidence,
        timings=timings,
        status="ok",
        verified=is_verified,
        tokens=tokens,
    )
