"""
tests/test_schema.py - Unit tests verifying strict schema validation.

Verifies:
1. Extra/unknown fields are rejected across argument models.
2. Bad event_ids (not matching EV-<digits> or latest_<EVENT_TYPE>) are rejected.
3. Invalid signal names (including signals not in SNAPSHOT_SIGNALS like odometer_km) are rejected.
4. SymbolicTimeRange strictly validates n (required for last_n_days/hours, rejected for others).
"""

import pytest
from pydantic import ValidationError

from schema import (
    ContextBeforeArgs,
    EventType,
    GetEventsArgs,
    SymbolicTimeRange,
    TimeRangeKind,
)


def test_unknown_field_rejected():
    """Ensure extra/unknown fields cause a ValidationError."""
    # Extra field on GetEventsArgs
    with pytest.raises(ValidationError) as exc_info:
        GetEventsArgs(
            type=EventType.BATTERY_OVERHEAT_WARNING,
            range=SymbolicTimeRange(kind=TimeRangeKind.yesterday),
            unknown_extra_param="sneaky_injection",
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)

    # Extra field on SymbolicTimeRange
    with pytest.raises(ValidationError) as exc_info:
        SymbolicTimeRange(
            kind=TimeRangeKind.yesterday,
            bogus_key=123,
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_event_id_validation():
    """Ensure event_id accepts only EV-<digits> or latest_<EVENT_TYPE>."""
    # Valid forms
    cb_id = ContextBeforeArgs(event_id="EV-0042", seconds=30)
    assert cb_id.event_id == "EV-0042"

    cb_latest = ContextBeforeArgs(event_id="latest_BATTERY_OVERHEAT_WARNING", seconds=60)
    assert cb_latest.event_id == "latest_BATTERY_OVERHEAT_WARNING"

    cb_latest_adas = ContextBeforeArgs(event_id="latest_ADAS_DISENGAGEMENT", seconds=60)
    assert cb_latest_adas.event_id == "latest_ADAS_DISENGAGEMENT"

    # Invalid forms
    with pytest.raises(ValidationError) as exc_info:
        ContextBeforeArgs(event_id="random_event_123", seconds=30)
    assert "Invalid event_id" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        ContextBeforeArgs(event_id="latest_UNKNOWN_WARNING", seconds=30)
    assert "Invalid event_id" in str(exc_info.value)


def test_signal_validation():
    """Ensure signals must be valid members of SNAPSHOT_SIGNALS."""
    # Valid signals
    cb = ContextBeforeArgs(
        event_id="EV-100",
        seconds=60,
        signals=["speed_kmh", "pack_temp_c", "power_kw"],
    )
    assert len(cb.signals) == 3

    # Invalid signal (e.g. non-existent)
    with pytest.raises(ValidationError) as exc_info:
        ContextBeforeArgs(
            event_id="EV-100",
            seconds=60,
            signals=["speed_kmh", "non_existent_telemetry_signal"],
        )
    assert "Invalid signal 'non_existent_telemetry_signal'" in str(exc_info.value)

    # Disallowed identifiers (ts and trip_id should be rejected as signals)
    with pytest.raises(ValidationError) as exc_info:
        ContextBeforeArgs(
            event_id="EV-100",
            seconds=60,
            signals=["trip_id"],
        )
    assert "Invalid signal 'trip_id'" in str(exc_info.value)


def test_snapshot_signals_odometer_rejected_and_pack_temp_accepted():
    """Ensure odometer_km is rejected (not in SNAPSHOT_SIGNALS) and pack_temp_c is accepted."""
    # signals=["pack_temp_c"] is accepted
    cb_valid = ContextBeforeArgs(event_id="EV-0042", signals=["pack_temp_c"])
    assert cb_valid.signals == ["pack_temp_c"]

    # signals=["odometer_km"] is rejected
    with pytest.raises(ValidationError) as exc_info:
        ContextBeforeArgs(event_id="EV-0042", signals=["odometer_km"])
    assert "Invalid signal 'odometer_km'" in str(exc_info.value)


def test_symbolic_time_range_n_bounds():
    """Ensure n is strictly required for last_n_days/hours and forbidden otherwise."""
    # last_n_days requires 1 <= n <= 28
    r_days = SymbolicTimeRange(kind=TimeRangeKind.last_n_days, n=14)
    assert r_days.n == 14

    with pytest.raises(ValidationError) as exc_info:
        SymbolicTimeRange(kind=TimeRangeKind.last_n_days, n=None)
    assert "n is required" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        SymbolicTimeRange(kind=TimeRangeKind.last_n_days, n=30)
    assert "between 1 and 28" in str(exc_info.value)

    # last_n_hours requires 1 <= n <= 72
    r_hours = SymbolicTimeRange(kind=TimeRangeKind.last_n_hours, n=48)
    assert r_hours.n == 48

    with pytest.raises(ValidationError) as exc_info:
        SymbolicTimeRange(kind=TimeRangeKind.last_n_hours, n=100)
    assert "between 1 and 72" in str(exc_info.value)

    # yesterday must NOT have n
    with pytest.raises(ValidationError) as exc_info:
        SymbolicTimeRange(kind=TimeRangeKind.yesterday, n=5)
    assert "n must be None when kind is 'yesterday'" in str(exc_info.value)

    # last_7_days must NOT have n
    with pytest.raises(ValidationError) as exc_info:
        SymbolicTimeRange(kind=TimeRangeKind.last_7_days, n=7)
    assert "n must be None when kind is 'last_7_days'" in str(exc_info.value)
