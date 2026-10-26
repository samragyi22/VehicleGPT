"""
tests/test_pipeline.py - Unit and Integration Tests for engine/pipeline.py.
"""

from unittest.mock import MagicMock, patch

from engine.pipeline import run_pipeline


def test_pipeline_control_command_refusal():
    res = run_pipeline("Turn off the air conditioning.")
    assert res.status == "control_refused"
    assert "refused" in res.answer.lower()
    assert res.tool_call == {"tool": "out_of_scope", "args": {"reason": "vehicle_control_prohibited"}}
    assert res.timings["guard_ms"] >= 0.0
    assert res.timings["total_ms"] >= 0.0


def test_pipeline_off_topic_refusal():
    res = run_pipeline("What is the capital of France?")
    assert res.status == "off_topic"
    assert "outside the scope" in res.answer.lower()
    assert res.tool_call == {"tool": "out_of_scope", "args": {"reason": "off_topic"}}
    assert res.timings["gate_ms"] >= 0.0


def test_pipeline_out_of_scope_from_model():
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"tool": "out_of_scope", "args": {"reason": "weather data unavailable"}}'
                }
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 15},
    }

    with (
        patch("llm.router._get_llm", return_value=mock_llm),
        patch("llm.grammar.load_grammar", return_value=MagicMock()),
    ):
        res = run_pipeline("Is it raining outside the car?")
        assert res.status == "out_of_scope"
        assert "outside the scope of VehicleGPT" in res.answer


def test_pipeline_successful_execution_mocked():
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"tool": "get_events", "args": {"type": "ADAS_DISENGAGEMENT", "range": {"kind": "yesterday"}}}'
                }
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 25},
    }

    with (
        patch("llm.router._get_llm", return_value=mock_llm),
        patch("llm.grammar.load_grammar", return_value=MagicMock()),
    ):
        res = run_pipeline("Show all ADAS disengagement events from yesterday.")
        assert res.status == "ok"
        assert res.verified is True
        assert res.tool_call["tool"] == "get_events"
        assert res.timings["model_ms"] >= 0.0
        assert res.timings["query_ms"] >= 0.0
        assert res.timings["render_ms"] >= 0.0
        assert res.timings["verify_ms"] >= 0.0
        assert res.tokens["prompt_tokens"] == 120
