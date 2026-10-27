"""
tests/test_guards.py - Tests for Control-Command Guard, Vocab Gate, and Grounding.

Ensures:
1. Every enum member in schema.py (Metric, EventType, Segment, BucketSize, ChangeDirection)
   has at least one synonym in llm/guards.py (fails automatically if a new enum is added).
2. The 3 control commands are reliably caught and refused.
3. The 15-question prototype list passes the vehicle vocabulary gate.
4. Out-of-scope off-topic questions fail the vehicle vocabulary gate.
5. Grounding check validates arguments against user question tokens.
"""

import pytest

from llm.guards import (
    BUCKET_SIZE_SYNONYMS,
    CHANGE_DIRECTION_SYNONYMS,
    CONTROL_REFUSAL_MESSAGE,
    EVENT_TYPE_SYNONYMS,
    METRIC_SYNONYMS,
    SEGMENT_SYNONYMS,
    has_vehicle_terms,
    is_control_command,
    is_grounded,
)
from schema import BucketSize, ChangeDirection, EventType, Metric, Segment

# ---------------------------------------------------------------------------
# 1. Enum Synonym Coverage Test (Fails if any enum member lacks synonyms)
# ---------------------------------------------------------------------------


def test_all_metric_enum_members_have_synonyms():
    """Fails if any Metric enum member has no synonyms."""
    for m in Metric:
        assert m in METRIC_SYNONYMS, f"Missing synonyms for Metric.{m.name}"
        syns = METRIC_SYNONYMS[m]
        assert len(syns) > 0, f"Empty synonyms list for Metric.{m.name}"
        for s in syns:
            assert isinstance(s, str) and len(s.strip()) > 0


def test_all_event_type_enum_members_have_synonyms():
    """Fails if any EventType enum member has no synonyms."""
    for ev in EventType:
        assert ev in EVENT_TYPE_SYNONYMS, f"Missing synonyms for EventType.{ev.name}"
        syns = EVENT_TYPE_SYNONYMS[ev]
        assert len(syns) > 0, f"Empty synonyms list for EventType.{ev.name}"
        for s in syns:
            assert isinstance(s, str) and len(s.strip()) > 0


def test_all_segment_enum_members_have_synonyms():
    """Fails if any Segment enum member has no synonyms."""
    for seg in Segment:
        assert seg in SEGMENT_SYNONYMS, f"Missing synonyms for Segment.{seg.name}"
        syns = SEGMENT_SYNONYMS[seg]
        assert len(syns) > 0, f"Empty synonyms list for Segment.{seg.name}"
        for s in syns:
            assert isinstance(s, str) and len(s.strip()) > 0


def test_all_bucket_size_enum_members_have_synonyms():
    """Fails if any BucketSize enum member has no synonyms."""
    for b in BucketSize:
        assert b in BUCKET_SIZE_SYNONYMS, f"Missing synonyms for BucketSize.{b.name}"
        syns = BUCKET_SIZE_SYNONYMS[b]
        assert len(syns) > 0, f"Empty synonyms list for BucketSize.{b.name}"


def test_all_change_direction_enum_members_have_synonyms():
    """Fails if any ChangeDirection enum member has no synonyms."""
    for d in ChangeDirection:
        assert d in CHANGE_DIRECTION_SYNONYMS, f"Missing synonyms for ChangeDirection.{d.name}"
        syns = CHANGE_DIRECTION_SYNONYMS[d]
        assert len(syns) > 0, f"Empty synonyms list for ChangeDirection.{d.name}"


# ---------------------------------------------------------------------------
# 2. Control-Command Guard Regression Tests
# ---------------------------------------------------------------------------

THREE_CONTROL_COMMANDS = [
    "Turn off the air conditioning.",
    "Please unlock the car doors.",
    "Roll down the windows.",
]


@pytest.mark.parametrize("cmd", THREE_CONTROL_COMMANDS)
def test_three_control_commands_intercepted(cmd: str):
    """The three canonical control commands must be intercepted and refused."""
    is_ctrl, msg = is_control_command(cmd)
    assert is_ctrl is True, f"Failed to intercept control command: {cmd}"
    assert msg == CONTROL_REFUSAL_MESSAGE


@pytest.mark.parametrize(
    "cmd",
    [
        "Lock the car",
        "Set climate temperature to 21 degrees",
        "Open trunk",
        "Start the heater",
        "Honk the horn",
        "Accelerate to 100 km/h",
    ],
)
def test_additional_control_commands_intercepted(cmd: str):
    """Additional actuator and control attempts must also be intercepted."""
    is_ctrl, _ = is_control_command(cmd)
    assert is_ctrl is True, f"Failed to intercept: {cmd}"


# ---------------------------------------------------------------------------
# 3. 15-Question Prototype List Regression Tests
# ---------------------------------------------------------------------------

FIFTEEN_PROTOTYPE_QUESTIONS = [
    # 5 Case Study Core Questions
    "Why has my driving range reduced over the last two weeks?",
    "When did fuel efficiency show the largest drop?",
    "Show all ADAS disengagement events from yesterday.",
    "What happened before the battery overheating warning?",
    "Compare energy consumption between city and highway driving.",
    # Paraphrases and Variations
    "When did energy efficiency show the largest drop?",
    "Show ADAS disengagements from yesterday.",
    "What happened before EV-0002 battery overheating warning?",
    "Compare energy consumption for city vs highway driving.",
    "Why has vehicle driving range decreased over the past 14 days?",
    # Additional Diagnostic Queries
    "Show the daily trend in energy consumption for the last 7 days.",
    "List all low battery warnings in the last 14 days.",
    "What happened before the latest ADAS disengagement?",
    "Compare pack temperature between city and highway driving.",
    "When did pack temperature show the largest rise?",
]


@pytest.mark.parametrize("q", FIFTEEN_PROTOTYPE_QUESTIONS)
def test_prototype_questions_pass_vocab_gate_and_not_control(q: str):
    """All 15 prototype questions must pass vocabulary gate and NOT be flagged as control."""
    is_ctrl, _ = is_control_command(q)
    assert is_ctrl is False, f"Question falsely flagged as control command: {q}"
    assert has_vehicle_terms(q) is True, f"Question failed vehicle vocabulary gate: {q}"


# ---------------------------------------------------------------------------
# 4. Off-Topic / Out-of-Scope Questions
# ---------------------------------------------------------------------------

OFF_TOPIC_QUESTIONS = [
    "What is the capital of France?",
    "Write a poem about the sunrise.",
    "Who won the World Cup in 2022?",
    "How do I bake a chocolate cake?",
    "What is the stock price of Apple today?",
]


@pytest.mark.parametrize("q", OFF_TOPIC_QUESTIONS)
def test_off_topic_questions_fail_vocab_gate(q: str):
    """Non-vehicle general knowledge questions must be rejected by vehicle vocab gate."""
    assert has_vehicle_terms(q) is False, f"Off-topic question passed vocab gate: {q}"


# ---------------------------------------------------------------------------
# 5. Grounding Check Tests
# ---------------------------------------------------------------------------


def test_grounding_valid():
    ok, _ = is_grounded("Show ADAS disengagements from yesterday", "get_events", {"type": "ADAS_DISENGAGEMENT"})
    assert ok is True

    ok, _ = is_grounded("Compare city and highway energy efficiency", "compare", {"segment_a": "city", "segment_b": "highway"})
    assert ok is True


def test_grounding_hallucinated_event():
    # Model hallucinated BATTERY_OVERHEAT_WARNING when user asked about ADAS
    ok, reason = is_grounded("Show ADAS disengagements from yesterday", "get_events", {"type": "BATTERY_OVERHEAT_WARNING"})
    assert ok is False
    assert "not mentioned" in str(reason)


# ---------------------------------------------------------------------------
# 6. Fuzzy Matching, Stem Matching, and Yesterday Removal Tests
# ---------------------------------------------------------------------------


def test_fuzzy_match_typo_passes_gate():
    """'enregy eficiency' should pass via fuzzy match (edit distance ≤ 1 to 'energy')."""
    assert has_vehicle_terms("show me enregy eficiency for the last wek") is True


def test_stem_match_speedup_passes_gate():
    """'speedup' starts with stem prefix 'speed' and should pass."""
    assert has_vehicle_terms("biggest speedup ever seen") is True


def test_yesterday_alone_does_not_pass_gate():
    """'yesterday' was removed from vocabulary — off-topic questions should fail."""
    assert has_vehicle_terms("tell me what the stock market did yesterday") is False


def test_yesterday_with_vehicle_terms_still_passes():
    """A question with 'yesterday' AND vehicle terms should still pass."""
    assert has_vehicle_terms("Show ADAS events from yesterday") is True


# ---------------------------------------------------------------------------
# 7. segment_a == segment_b Rejection Test
# ---------------------------------------------------------------------------

def test_compare_same_segment_rejected():
    """CompareArgs.segments_must_differ must reject segment_a == segment_b."""
    from pydantic import ValidationError

    from engine.validate import validate_tool_call
    with pytest.raises((ValueError, ValidationError), match="segment_a and segment_b must be different"):
        validate_tool_call({
            "tool": "compare",
            "args": {
                "metric": "speed_kmh",
                "range": {"kind": "last_7_days"},
                "segment_a": "city",
                "segment_b": "city",
            },
        })


# ---------------------------------------------------------------------------
# 8. Out-of-Scope Fixed Message Test
# ---------------------------------------------------------------------------

def test_out_of_scope_uses_fixed_message():
    """Pipeline must use fixed OFF_TOPIC_MESSAGE, never exposing model reason text."""
    from llm.guards import OFF_TOPIC_MESSAGE
    # Import the message used in pipeline's out_of_scope path
    assert "VehicleGPT" in OFF_TOPIC_MESSAGE
    assert "telemetry" in OFF_TOPIC_MESSAGE


# ---------------------------------------------------------------------------
# 9. Unknown Key ("hack") Rejection Test
# ---------------------------------------------------------------------------

def test_unknown_key_in_args_rejected():
    """Unknown keys like 'hack' must be rejected with extra_forbidden."""
    from pydantic import ValidationError

    from engine.validate import validate_tool_call
    with pytest.raises((ValueError, ValidationError), match="Extra inputs are not permitted|extra_forbidden"):
        validate_tool_call({
            "tool": "compare",
            "args": {
                "metric": "speed_kmh",
                "range": {"kind": "last_7_days"},
                "segment_a": "city",
                "segment_b": "highway",
                "hack": "malicious_payload",
            },
        })


# ---------------------------------------------------------------------------
# Comprehensive Control Command Tests (Guard Gap Fix)
# ---------------------------------------------------------------------------


class TestControlCommandComprehensive:
    """Test that all variants of control commands are properly caught."""

    @pytest.mark.parametrize("cmd", [
        "Accelerate to 100 km/h.",
        "Could you put on the heater?",
        "I'd like the windows down.",
        "Please start charging.",
        "Turn off the air conditioning.",
        "Unlock the car doors.",
        "Set the temperature to 20 degrees.",
        "Open the sunroof.",
        "Close all windows.",
        "Lock the vehicle.",
        "Can you start the engine?",
        "Please turn on the headlights.",
        "Disable the parking brake.",
        "Could you adjust the mirrors?",
        "I'd like cruise control on.",
        "Put the seat heater on.",
    ])
    def test_16_control_commands_caught(self, cmd: str):
        """All control command variants must be caught by is_control_command guard."""
        is_ctrl, reason = is_control_command(cmd)
        assert is_ctrl is True, f"Failed to catch control command: {cmd}"
        assert "control" in reason.lower() or "command" in reason.lower(), f"Bad reason: {reason}"


# ---------------------------------------------------------------------------
# Gate Tightening Tests: Off-Topic vs Vehicle Questions
# ---------------------------------------------------------------------------


class TestGateTightening:
    """Test that off-topic questions fail the gate, vehicle questions pass."""

    @pytest.mark.parametrize("off_topic_q", [
        "How do I bake a chocolate cake?",
        "What is the capital of France?",
        "Tell me a joke.",
        "How do I tie a necktie?",
        "What is the weather today?",
        "Who won the World Cup?",
        "How do I solve a quadratic equation?",
        "What is quantum physics?",
        "How do I cook pasta?",
        "Tell me about ancient Rome.",
        "What is the best pizza recipe?",
        "How do I play chess?",
        "What are the rules of cricket?",
        "Tell me about famous painters.",
        "How do I knit a sweater?",
        "What is the stock market?",
        "How do I meditate?",
        "What are the Olympics?",
        "Tell me about astronomy.",
        "How do I make sushi?",
    ])
    def test_20_off_topic_questions_fail_gate(self, off_topic_q: str):
        """Off-topic questions must fail the has_vehicle_terms gate."""
        assert has_vehicle_terms(off_topic_q) is False, f"Off-topic question passed gate: {off_topic_q}"

    @pytest.mark.parametrize("typo_vehicle_q", [
        "Why has my driving range redused?",  # typo: redused→reduced
        "When did enregy eficiency change?",  # typos: enregy→energy, eficiency→efficiency
        "Show all ADAS disenganement events.",  # typo: disenganement→disengagement
        "What happend before the battry warning?",  # typos: happend→happened, battry→battery
        "Compare energy consumtion between city and highway.",  # typo: consumtion→consumption
        "Show the daily trnd in power.",  # typo: trnd→trend
        "When did pack temp rise the most?",  # abbrev: temp, temp→temperature
        "List all low SOC warrnings.",  # typo: warrnings→warnings
        "Show hvac consuption trends.",  # typo: consuption→consumption
        "What happend before the latest ADAS event?",  # typo: happend→happened
    ])
    def test_10_vehicle_questions_with_typos_pass_gate(self, typo_vehicle_q: str):
        """Vehicle questions with typos must still pass the fuzzy matching gate."""
        assert has_vehicle_terms(typo_vehicle_q) is True, f"Vehicle question with typos failed gate: {typo_vehicle_q}"
