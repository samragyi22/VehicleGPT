"""
tests/test_llm.py - Phase 3 SLM Router and Grammar Tests.

Test strategy:
1. Grammar/schema tests (no model needed): verify the JSON schema structure,
   that all schema.py enums appear correctly, and that GBNF can be built.
2. Verifier unit tests (no model needed): check number extraction, normalization,
   and pass/fail behavior with synthetic tool outputs.
3. Validate-and-route integration tests (no model needed): call validate_tool_call +
   prepare_executable_args directly to confirm the pipeline handles all tools.
4. Slow end-to-end router tests (@pytest.mark.slow): load the real SLM and send
   the 5 case-study questions, asserting the correct tool is selected.

All slow tests are guarded with @pytest.mark.slow so that pytest -m "not slow"
skips them during fast development cycles (Phase 2 target: < 30 s suite).
"""

import json

import pytest
from pydantic import ValidationError

import config
from engine.verify import _extract_numbers, _normalize, verify, verify_or_fallback
from llm.grammar import _enum, build_gbnf, build_json_schema
from schema import (
    BucketSize,
    ChangeDirection,
    EventType,
    Metric,
    Segment,
    ToolName,
)

# ===========================================================================
# 1. Grammar / JSON Schema Tests (no model required)
# ===========================================================================


class TestGrammarSchema:
    """Verify that the JSON schema produced by grammar.py is correctly structured."""

    def test_schema_is_dict_with_anyof(self):
        schema = build_json_schema()
        assert isinstance(schema, dict)
        assert "anyOf" in schema

    def test_all_enabled_tools_present(self):
        schema = build_json_schema()
        tool_consts = {
            branch["properties"]["tool"]["const"]
            for branch in schema["anyOf"]
        }
        # All tools except search_logs (not yet enabled) must appear
        expected = {
            "get_events",
            "context_before",
            "compare",
            "find_largest_change",
            "trend",
            "explain_factors",
            "out_of_scope",
        }
        assert expected == tool_consts

    def test_search_logs_excluded(self):
        """search_logs is excluded because the FTS5 table is not yet built."""
        schema = build_json_schema()
        tool_consts = {
            branch["properties"]["tool"]["const"]
            for branch in schema["anyOf"]
        }
        assert "search_logs" not in tool_consts

    def test_metric_enum_matches_schema(self):
        """All Metric enum values appear in compare and other metric-bearing branches."""
        schema = build_json_schema()
        expected_metrics = set(_enum(Metric))

        compare_branch = next(
            b for b in schema["anyOf"]
            if b["properties"]["tool"]["const"] == "compare"
        )
        schema_metrics = set(compare_branch["properties"]["args"]["properties"]["metric"]["enum"])
        assert expected_metrics == schema_metrics

    def test_event_type_enum_matches_schema(self):
        schema = build_json_schema()
        ge_branch = next(
            b for b in schema["anyOf"]
            if b["properties"]["tool"]["const"] == "get_events"
        )
        schema_types = set(ge_branch["properties"]["args"]["properties"]["type"]["enum"])
        assert schema_types == set(_enum(EventType))

    def test_segment_enum_matches_schema(self):
        schema = build_json_schema()
        compare_branch = next(
            b for b in schema["anyOf"]
            if b["properties"]["tool"]["const"] == "compare"
        )
        args_props = compare_branch["properties"]["args"]["properties"]
        assert set(args_props["segment_a"]["enum"]) == set(_enum(Segment))
        assert set(args_props["segment_b"]["enum"]) == set(_enum(Segment))

    def test_time_range_kind_enum_in_schema(self):
        """JSON schema no longer includes range; this test verifies that."""
        schema = build_json_schema()
        ge_branch = next(
            b for b in schema["anyOf"]
            if b["properties"]["tool"]["const"] == "get_events"
        )
        # range must NOT be present — it is injected by the time parser
        assert "range" not in ge_branch["properties"]["args"]["properties"]

    def test_bucket_size_enum_matches_schema(self):
        schema = build_json_schema()
        trend_branch = next(
            b for b in schema["anyOf"]
            if b["properties"]["tool"]["const"] == "trend"
        )
        buckets = set(trend_branch["properties"]["args"]["properties"]["bucket"]["enum"])
        assert buckets == set(_enum(BucketSize))

    def test_change_direction_enum_matches_schema(self):
        schema = build_json_schema()
        flc_branch = next(
            b for b in schema["anyOf"]
            if b["properties"]["tool"]["const"] == "find_largest_change"
        )
        dirs = set(flc_branch["properties"]["args"]["properties"]["direction"]["enum"])
        assert dirs == set(_enum(ChangeDirection))

    def test_no_additional_properties_on_tool_branches(self):
        """Every tool branch forbids extra properties at the top level."""
        schema = build_json_schema()
        for branch in schema["anyOf"]:
            assert branch.get("additionalProperties") is False, (
                f"Branch for {branch['properties']['tool']} allows extra properties"
            )

    def test_out_of_scope_reason_is_enum_not_free_text(self):
        """out_of_scope reason must be one of the three fixed literals, not free text."""
        from llm.grammar import OUT_OF_SCOPE_REASONS
        schema = build_json_schema()
        oos_branch = next(
            b for b in schema["anyOf"]
            if b["properties"]["tool"]["const"] == "out_of_scope"
        )
        reason_schema = oos_branch["properties"]["args"]["properties"]["reason"]
        assert "enum" in reason_schema, "reason must be an enum, not free text"
        assert set(reason_schema["enum"]) == set(OUT_OF_SCOPE_REASONS)

    def test_gbnf_is_nonempty_string(self):
        """GBNF string can be built without errors."""
        gbnf = build_gbnf()
        assert isinstance(gbnf, str)
        assert len(gbnf) > 100, "GBNF string looks suspiciously short"
        assert "root" in gbnf

    def test_gbnf_contains_no_range_key(self):
        """range must not appear in the GBNF — it is injected by the time parser."""
        gbnf = build_gbnf()
        assert '"range"' not in gbnf

    def test_gbnf_all_metric_values_present(self):
        gbnf = build_gbnf()
        for m in _enum(Metric):
            assert m in gbnf, f"Metric {m!r} missing from GBNF"

    def test_gbnf_all_event_types_present(self):
        gbnf = build_gbnf()
        for et in _enum(EventType):
            assert et in gbnf, f"EventType {et!r} missing from GBNF"

    def test_gbnf_all_directions_present(self):
        gbnf = build_gbnf()
        for d in _enum(ChangeDirection):
            assert d in gbnf, f"Direction {d!r} missing from GBNF"

    def test_gbnf_all_buckets_present(self):
        gbnf = build_gbnf()
        for b in _enum(BucketSize):
            assert b in gbnf, f"Bucket {b!r} missing from GBNF"

    def test_gbnf_compare_only_two_segment_orderings(self):
        """compare must only allow city->highway and highway->city."""
        gbnf = build_gbnf()
        # Both orderings must be present
        assert 'city' in gbnf and 'highway' in gbnf
        # city->city and highway->highway must not appear as adjacent literals
        assert '"city"," ws "\\"segment_b\\": \\"city' not in gbnf
        assert '"highway"," ws "\\"segment_b\\": \\"highway' not in gbnf

    def test_gbnf_out_of_scope_fixed_reasons(self):
        """GBNF must contain all three fixed reason literals."""
        from llm.grammar import OUT_OF_SCOPE_REASONS
        gbnf = build_gbnf()
        for r in OUT_OF_SCOPE_REASONS:
            assert r in gbnf, f"Fixed reason {r!r} missing from GBNF"

    def test_schema_serializable_to_json(self):
        """schema dict must be JSON-serializable (no Python-only types)."""
        schema = build_json_schema()
        dumped = json.dumps(schema)
        assert json.loads(dumped) == schema


# ===========================================================================
# 2. Verifier Unit Tests (no model required)
# ===========================================================================


class TestVerifier:
    """Unit tests for engine/verify.py number extraction and matching."""

    def test_normalize_integer(self):
        assert _normalize("42") == "42"

    def test_normalize_float(self):
        assert _normalize("367.8") == "367.8"

    def test_normalize_trailing_zeros(self):
        # 367.80 should normalize the same as 367.8
        assert _normalize("367.80") == _normalize("367.8")

    def test_normalize_scientific(self):
        assert _normalize("3.678e2") == _normalize("367.8")

    def test_extract_numbers_basic(self):
        nums = _extract_numbers("Range is 367.8 km, using 158.2 Wh/km.")
        assert "367.8" in nums
        assert "158.2" in nums

    def test_extract_numbers_no_duplicates(self):
        nums = _extract_numbers("42 events in 42 days")
        assert nums.count(_normalize("42")) == 1

    def test_verify_pass_exact_match(self):
        answer = "Efficiency is 158.3 Wh/km."
        output = {"result": {"efficiency": 158.3}, "evidence": [], "meta": {"row_count": 10}}
        assert verify(answer, output) is True

    def test_verify_pass_different_format(self):
        # 158.30 in answer, 158.3 in output — should still pass after normalization
        answer = "Efficiency is 158.30 Wh/km."
        output = {"result": {"efficiency": 158.3}, "evidence": [], "meta": {"row_count": 10}}
        assert verify(answer, output) is True

    def test_verify_fail_hallucinated_number(self):
        answer = "Efficiency improved by 42.0 Wh/km."
        output = {"result": {"efficiency": 158.3}, "evidence": [], "meta": {"row_count": 10}}
        # 42 is not in the output
        assert verify(answer, output) is False

    def test_verify_pass_no_numbers_in_answer(self):
        answer = "No events were found in the selected window."
        output = {"result": {"count": 0}, "evidence": [], "meta": {"row_count": 0}}
        assert verify(answer, output) is True

    def test_verify_or_fallback_passes(self):
        output = {"result": {"value": 99.5}, "evidence": [], "meta": {}}
        answer, passed = verify_or_fallback("Value is 99.5.", output, fallback="fallback text")
        assert passed is True
        assert answer == "Value is 99.5."

    def test_verify_or_fallback_fails(self):
        output = {"result": {"value": 99.5}, "evidence": [], "meta": {}}
        answer, passed = verify_or_fallback("Value is 123.4.", output, fallback="fallback text")
        assert passed is False
        assert answer == "fallback text"

    def test_verify_numbers_in_evidence(self):
        """Numbers that appear in evidence rows should also pass verification."""
        answer = "Event EV-0007 recorded at speed 72.5 km/h."
        output = {
            "result": {"count": 1},
            "evidence": [{"event_id": "EV-0007", "speed_kmh": 72.5}],
            "meta": {"row_count": 1},
        }
        # 72.5 is in evidence; 1 is in meta; 7 appears in EV-0007
        assert verify(answer, output) is True

    def test_verify_numbers_in_meta(self):
        """Meta row counts (e.g. '200') should satisfy the verifier."""
        answer = "Showing 200 rows."
        output = {"result": {}, "evidence": [], "meta": {"row_count": 200}}
        assert verify(answer, output) is True


# ===========================================================================
# 3. Validate-and-Route Integration (no model)
# ===========================================================================


class TestValidateAndRoute:
    """
    Test the validate -> prepare_executable_args pipeline for all tools,
    including out_of_scope, without loading the SLM.
    """

    from engine.validate import prepare_executable_args, validate_tool_call

    def _run(self, call_dict: dict) -> tuple:
        from engine.validate import prepare_executable_args, validate_tool_call
        tool_name, sym_args = validate_tool_call(call_dict)
        exec_args = prepare_executable_args(tool_name, sym_args)
        return tool_name, exec_args

    def test_get_events_yesterday(self):
        tool_name, exec_args = self._run({
            "tool": "get_events",
            "args": {"type": "ADAS_DISENGAGEMENT", "range": {"kind": "yesterday"}},
        })
        assert tool_name == ToolName.get_events
        assert exec_args.start == "2026-10-27T00:00:00"
        assert exec_args.end == "2026-10-28T00:00:00"
        assert exec_args.type.value == "ADAS_DISENGAGEMENT"

    def test_context_before_latest(self):
        tool_name, exec_args = self._run({
            "tool": "context_before",
            "args": {"event_id": "latest_BATTERY_OVERHEAT_WARNING", "seconds": 60},
        })
        assert tool_name == ToolName.context_before
        assert exec_args.event_id == "latest_BATTERY_OVERHEAT_WARNING"
        assert exec_args.seconds == 60

    def test_context_before_specific_id(self):
        _tool_name, exec_args = self._run({
            "tool": "context_before",
            "args": {"event_id": "EV-0042", "seconds": 300},
        })
        assert exec_args.event_id == "EV-0042"
        assert exec_args.seconds == 300

    def test_compare_city_highway(self):
        tool_name, exec_args = self._run({
            "tool": "compare",
            "args": {
                "metric": "energy_wh_per_km",
                "range": {"kind": "last_14_days"},
                "segment_a": "city",
                "segment_b": "highway",
            },
        })
        assert tool_name == ToolName.compare
        assert exec_args.metric.value == "energy_wh_per_km"
        assert exec_args.start == "2026-10-14T00:00:00"

    def test_find_largest_change(self):
        tool_name, exec_args = self._run({
            "tool": "find_largest_change",
            "args": {
                "metric": "energy_wh_per_km",
                "range": {"kind": "last_28_days"},
                "direction": "drop",
            },
        })
        assert tool_name == ToolName.find_largest_change
        assert exec_args.direction.value == "drop"

    def test_trend_daily(self):
        tool_name, exec_args = self._run({
            "tool": "trend",
            "args": {
                "metric": "pack_temp_c",
                "range": {"kind": "last_7_days"},
                "bucket": "day",
            },
        })
        assert tool_name == ToolName.trend
        assert exec_args.bucket.value == "day"

    def test_explain_factors(self):
        tool_name, exec_args = self._run({
            "tool": "explain_factors",
            "args": {
                "metric": "estimated_range_km",
                "range": {"kind": "last_14_days"},
            },
        })
        assert tool_name == ToolName.explain_factors
        assert exec_args.metric.value == "estimated_range_km"

    def test_out_of_scope(self):
        tool_name, exec_args = self._run({
            "tool": "out_of_scope",
            "args": {"reason": "general knowledge question"},
        })
        assert tool_name == ToolName.out_of_scope
        assert exec_args.reason == "general knowledge question"

    def test_unknown_tool_rejected(self):
        with pytest.raises((ValueError, ValidationError)):
            self._run({"tool": "delete_data", "args": {}})

    def test_unknown_metric_rejected(self):
        with pytest.raises((ValueError, ValidationError)):
            self._run({
                "tool": "compare",
                "args": {"metric": "fuel_level", "range": {"kind": "yesterday"}},
            })

    def test_unknown_event_type_rejected(self):
        with pytest.raises((ValueError, ValidationError)):
            self._run({
                "tool": "get_events",
                "args": {"type": "ENGINE_FAILURE", "range": {"kind": "yesterday"}},
            })

    def test_extra_field_rejected(self):
        with pytest.raises((ValueError, ValidationError)):
            self._run({
                "tool": "out_of_scope",
                "args": {"reason": "ok", "extra_field": "bad"},
            })

    def test_invalid_event_id_format_rejected(self):
        with pytest.raises((ValueError, ValidationError)):
            self._run({
                "tool": "context_before",
                "args": {"event_id": "BAD-ID-FORMAT", "seconds": 60},
            })

    def test_last_n_days_bounds(self):
        """n must be in [1, 28]."""
        with pytest.raises((ValueError, ValidationError)):
            self._run({
                "tool": "explain_factors",
                "args": {
                    "metric": "energy_wh_per_km",
                    "range": {"kind": "last_n_days", "n": 0},
                },
            })
        with pytest.raises((ValueError, ValidationError)):
            self._run({
                "tool": "explain_factors",
                "args": {
                    "metric": "energy_wh_per_km",
                    "range": {"kind": "last_n_days", "n": 29},
                },
            })

    def test_last_n_hours_bounds(self):
        """n must be in [1, 72]."""
        with pytest.raises((ValueError, ValidationError)):
            self._run({
                "tool": "trend",
                "args": {
                    "metric": "speed_kmh",
                    "range": {"kind": "last_n_hours", "n": 73},
                },
            })

    def test_context_before_seconds_bounds(self):
        """seconds must be in [1, 300]."""
        with pytest.raises((ValueError, ValidationError)):
            self._run({
                "tool": "context_before",
                "args": {"event_id": "EV-0001", "seconds": 0},
            })
        with pytest.raises((ValueError, ValidationError)):
            self._run({
                "tool": "context_before",
                "args": {"event_id": "EV-0001", "seconds": 301},
            })


# ===========================================================================
# 4. End-to-End Router Tests (slow; require the real SLM)
# ===========================================================================


@pytest.mark.slow
class TestRouterE2E:
    """
    End-to-end tests that load the 0.5B GGUF model and route each of the
    5 case-study questions from PROJECT_CONTEXT.md Section 1.

    These are skipped in the fast suite (pytest -m 'not slow').
    """

    @classmethod
    @pytest.fixture(scope="class", autouse=True)
    def model_available(cls):
        """Skip if the default GGUF is not present."""
        if not config.DEFAULT_MODEL_PATH.exists():
            pytest.skip(f"Model not found: {config.DEFAULT_MODEL_PATH}")

    def _route(self, question: str):
        from llm.router import route
        return route(question, with_explanation=False)

    def test_case_1_range_drop_routes_to_explain_factors(self):
        """Q: 'Why has my driving range reduced over the last two weeks?'"""
        r = self._route("Why has my driving range reduced over the last two weeks?")
        assert r.error == "", f"Router error: {r.error}"
        assert r.tool_name == ToolName.explain_factors, (
            f"Expected explain_factors, got {r.tool_name}. Raw: {r.tool_call_raw}"
        )

    def test_case_2_efficiency_drop_routes_to_find_largest_change(self):
        """Q: 'When did fuel efficiency show the largest drop?'"""
        r = self._route("When did energy efficiency show the largest drop?")
        assert r.error == "", f"Router error: {r.error}"
        assert r.tool_name == ToolName.find_largest_change, (
            f"Expected find_largest_change, got {r.tool_name}. Raw: {r.tool_call_raw}"
        )

    def test_case_3_adas_yesterday_routes_to_get_events(self):
        """Q: 'Show all ADAS disengagement events from yesterday.'"""
        r = self._route("Show all ADAS disengagement events from yesterday.")
        assert r.error == "", f"Router error: {r.error}"
        assert r.tool_name == ToolName.get_events, (
            f"Expected get_events, got {r.tool_name}. Raw: {r.tool_call_raw}"
        )
        assert r.tool_args_dict.get("type") in ("ADAS_DISENGAGEMENT", "adas_disengagement"), (
            f"Wrong event type: {r.tool_args_dict}"
        )

    def test_case_4_battery_context_routes_to_context_before(self):
        """Q: 'What happened before the battery overheating warning?'"""
        r = self._route("What happened before the battery overheating warning?")
        assert r.error == "", f"Router error: {r.error}"
        assert r.tool_name == ToolName.context_before, (
            f"Expected context_before, got {r.tool_name}. Raw: {r.tool_call_raw}"
        )

    def test_case_5_city_highway_routes_to_compare(self):
        """Q: 'Compare energy consumption between city and highway driving.'"""
        r = self._route("Compare energy consumption between city and highway driving.")
        assert r.error == "", f"Router error: {r.error}"
        assert r.tool_name == ToolName.compare, (
            f"Expected compare, got {r.tool_name}. Raw: {r.tool_call_raw}"
        )

    def test_out_of_scope_general_knowledge(self):
        """General knowledge questions must route to out_of_scope."""
        r = self._route("What is the capital of France?")
        assert r.error == "", f"Router error: {r.error}"
        assert r.tool_name == ToolName.out_of_scope, (
            f"Expected out_of_scope, got {r.tool_name}. Raw: {r.tool_call_raw}"
        )

    def test_out_of_scope_vehicle_control(self):
        """Vehicle control requests must route to out_of_scope."""
        r = self._route("Please unlock the car doors.")
        assert r.error == "", f"Router error: {r.error}"
        assert r.tool_name == ToolName.out_of_scope, (
            f"Expected out_of_scope, got {r.tool_name}. Raw: {r.tool_call_raw}"
        )

    def test_answer_is_nonempty(self):
        """Router must always return a non-empty answer."""
        r = self._route("Show ADAS events from the last 7 days.")
        assert r.error == "", f"Router error: {r.error}"
        assert r.answer.strip() != "", "Answer is empty"

    def test_verified_flag_set(self):
        """For tool calls with numeric output, verified must be True."""
        r = self._route("Compare energy consumption between city and highway driving.")
        assert r.error == "", f"Router error: {r.error}"
        # Template numbers come from tool output, so verification must pass
        assert r.verified is True

    def test_latency_within_target(self):
        """
        Soft latency check: model_ms + query_ms should be under 10 s on the
        laptop (target is 2 s on the constrained container; reported separately).
        """
        r = self._route("Show all ADAS disengagement events from yesterday.")
        total_ms = r.model_ms + r.query_ms + r.render_ms
        assert total_ms < 10_000, f"Total latency {total_ms:.0f} ms exceeds 10 s soft limit"
