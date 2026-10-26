"""
tests/test_timeparse.py - Tests for Deterministic Natural-Language Time Parser.
"""

from llm.timeparse import parse_time_range


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
