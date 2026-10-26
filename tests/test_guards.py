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
