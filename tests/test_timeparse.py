"""
tests/test_timeparse.py - Tests for Deterministic Natural-Language Time Parser.
"""

from llm.timeparse import interpretation_text, parse_time_range


def test_yesterday_parsed():
    r, dis = parse_time_range("Show all ADAS disengagement events from yesterday.")
    assert r == {"kind": "yesterday"}
    assert dis is False


def test_last_two_weeks_parsed():
    r, dis = parse_time_range("Why has my driving range reduced over the last two weeks?")
    assert r == {"kind": "last_14_days"}
    assert dis is False


def test_past_14_days_parsed():
    r, dis = parse_time_range("Why has my driving range reduced over the past 14 days?")
    assert r == {"kind": "last_14_days"}
    assert dis is False


def test_last_7_days_parsed():
    r, dis = parse_time_range("Show trend of speed for the last 7 days.")
    assert r == {"kind": "last_7_days"}
    assert dis is False


def test_past_week_parsed():
    r, dis = parse_time_range("Show energy trend over the past week.")
    assert r == {"kind": "last_7_days"}
    assert dis is False


def test_last_month_parsed():
    r, dis = parse_time_range("Compare efficiency over the past month.")
    assert r == {"kind": "last_28_days"}
    assert dis is False


def test_last_n_days_arbitrary():
    r, dis = parse_time_range("Show events over the last 10 days.")
    assert r == {"kind": "last_n_days", "n": 10}
    assert dis is False


def test_last_n_hours():
    r, dis = parse_time_range("Show telemetry in the last 6 hours.")
    assert r == {"kind": "last_n_hours", "n": 6}
    assert dis is False


def test_no_time_defaults_to_all_history():
    r, dis = parse_time_range("When did fuel efficiency show the largest drop?")
    assert r == {"kind": "all_history"}
    assert dis is False


def test_disagreement_logged_and_overridden():
    # Model proposed last_7_days for a question saying yesterday
    model_range = {"kind": "last_7_days"}
    r, dis = parse_time_range("Show all ADAS disengagements from yesterday.", model_range=model_range)
    assert r == {"kind": "yesterday"}
    assert dis is True


def test_fortnight_maps_to_14_days():
    r, _ = parse_time_range("in the past fortnight did range go up or down")
    assert r == {"kind": "last_14_days"}


def test_two_days_ago_returns_none():
    """Unresolvable time phrase returns None (unsupported_time)."""
    r, _ = parse_time_range("two days ago how was the range?")
    assert r is None


def test_this_week_maps_to_7_days():
    """'this week' is now deterministic: the last 7 days (updated from clarification)."""
    r, _ = parse_time_range("this week energy numbers")
    assert r == {"kind": "last_7_days"}


def test_past_week_maps_to_7_days():
    r, _ = parse_time_range("show energy numbers for the past week")
    assert r == {"kind": "last_7_days"}


def test_last_month_maps_to_28_days():
    r, _ = parse_time_range("how was consumption over the last month?")
    assert r == {"kind": "last_28_days"}


def test_n_weeks_ago_returns_none():
    """'N weeks ago' stays a clarification request."""
    r, _ = parse_time_range("how was range 2 weeks ago?")
    assert r is None


def test_since_last_tuesday_returns_none():
    r, _ = parse_time_range("since last tuesday what happened to the battery temp?")
    assert r is None


def test_current_week_returns_none():
    r, _ = parse_time_range("pull up adas events for the current week please")
    assert r is None


def test_no_time_phrase_still_defaults_all_history():
    """When no temporal phrase is present at all, all_history is used."""
    r, _ = parse_time_range("biggest speedup ever seen")
    assert r == {"kind": "all_history"}


def test_interpretation_text_states_phrase_and_dates():
    text = interpretation_text("this week energy numbers", {"kind": "last_7_days"})
    assert text == "Interpreted 'this week' as the last 7 days (21 Oct to 28 Oct)."


def test_interpretation_text_no_phrase():
    text = interpretation_text("Compare energy consumption between city and highway.", {"kind": "all_history"})
    assert text.startswith("No time phrase found; interpreting as all history")
    assert "30 Sep" in text and "28 Oct" in text

