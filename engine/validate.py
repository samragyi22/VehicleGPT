"""
engine/validate.py - Tool Call Validation and Preparation.

Enforces strict verification of SLM tool selection outputs:
1. Validates ToolCall against schema.py Pydantic definitions (extra='forbid').
2. Rejects unknown tools, unknown arguments, invalid enums, and out-of-range bounds.
3. Resolves symbolic time ranges into concrete ISO timestamps for executable tool execution.
"""

from typing import Any

from pydantic import BaseModel

from engine.dates import resolve_symbolic_range
from schema import (
    CompareArgs,
    CompareExecArgs,
    ContextBeforeArgs,
    ContextBeforeExecArgs,
    ExplainFactorsArgs,
    ExplainFactorsExecArgs,
    FindLargestChangeArgs,
    FindLargestChangeExecArgs,
    GetEventsArgs,
    GetEventsExecArgs,
    OutOfScopeArgs,
    OutOfScopeExecArgs,
    SearchLogsArgs,
    SearchLogsExecArgs,
    ToolCall,
    ToolName,
    TrendArgs,
    TrendExecArgs,
)

TOOL_ARG_MODELS: dict[ToolName, type[BaseModel]] = {
    ToolName.get_events: GetEventsArgs,
    ToolName.context_before: ContextBeforeArgs,
    ToolName.compare: CompareArgs,
    ToolName.find_largest_change: FindLargestChangeArgs,
    ToolName.trend: TrendArgs,
    ToolName.explain_factors: ExplainFactorsArgs,
    ToolName.out_of_scope: OutOfScopeArgs,
    ToolName.search_logs: SearchLogsArgs,
}


def validate_tool_call(call: ToolCall | dict[str, Any]) -> tuple[ToolName, BaseModel]:
    """
    Validates that a tool call structure adheres strictly to the frozen schema.

    Args:
        call: ToolCall instance or raw dictionary {'tool': ..., 'args': {...}}

    Returns:
        tuple[ToolName, BaseModel]: Validated tool name and validated symbolic arguments model.

    Raises:
        ValueError / ValidationError: If tool is unknown or arguments violate constraints.
    """
    if isinstance(call, dict):
        call = ToolCall.model_validate(call)

    tool_name = call.tool
    model_cls = TOOL_ARG_MODELS.get(tool_name)
    if model_cls is None:
        raise ValueError(f"Unknown or unsupported tool: {tool_name}")

    validated_args = model_cls.model_validate(call.args)
    if tool_name == ToolName.compare and validated_args.segment_a == validated_args.segment_b:
        raise ValueError(
            f"segment_a and segment_b must be different; both are '{validated_args.segment_a.value}'"
        )
    return tool_name, validated_args


def prepare_executable_args(
    tool_name: ToolName,
    args: BaseModel,
) -> BaseModel:
    """
    Converts validated symbolic arguments into concrete executable arguments
    with resolved ISO timestamps.
    """
    if tool_name == ToolName.get_events:
        assert isinstance(args, GetEventsArgs)
        start, end = resolve_symbolic_range(args.range)
        return GetEventsExecArgs(type=args.type, start=start, end=end)

    if tool_name == ToolName.context_before:
        assert isinstance(args, ContextBeforeArgs)
        return ContextBeforeExecArgs(
            event_id=args.event_id,
            seconds=args.seconds,
            signals=args.signals,
        )

    if tool_name == ToolName.compare:
        assert isinstance(args, CompareArgs)
        start, end = resolve_symbolic_range(args.range)
        return CompareExecArgs(
            metric=args.metric,
            start=start,
            end=end,
            segment_a=args.segment_a,
            segment_b=args.segment_b,
        )

    if tool_name == ToolName.find_largest_change:
        assert isinstance(args, FindLargestChangeArgs)
        start, end = resolve_symbolic_range(args.range)
        return FindLargestChangeExecArgs(
            metric=args.metric,
            start=start,
            end=end,
            direction=args.direction,
        )

    if tool_name == ToolName.trend:
        assert isinstance(args, TrendArgs)
        start, end = resolve_symbolic_range(args.range)
        return TrendExecArgs(
            metric=args.metric,
            start=start,
            end=end,
            bucket=args.bucket,
        )

    if tool_name == ToolName.explain_factors:
        assert isinstance(args, ExplainFactorsArgs)
        start, end = resolve_symbolic_range(args.range)
        return ExplainFactorsExecArgs(
            metric=args.metric,
            start=start,
            end=end,
        )

    if tool_name == ToolName.out_of_scope:
        assert isinstance(args, OutOfScopeArgs)
        return OutOfScopeExecArgs(reason=args.reason)

    if tool_name == ToolName.search_logs:
        assert isinstance(args, SearchLogsArgs)
        start, end = resolve_symbolic_range(args.range)
        return SearchLogsExecArgs(text=args.text, start=start, end=end)

    raise ValueError(f"Unhandled tool preparation for: {tool_name}")
