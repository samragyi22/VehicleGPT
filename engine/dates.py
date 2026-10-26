"""
engine/dates.py - Deterministic Symbolic Time Range Resolver for VehicleGPT.

Resolves symbolic time specifications (emitted by the SLM or test suite)
into concrete, day-aligned ISO 8601 timestamp boundaries [start, end)
against config.SIMULATED_NOW.

Resolution Rules:
1. yesterday:
   Full previous calendar day: [now.date - 1 day at 00:00:00, now.date at 00:00:00).
   For SIMULATED_NOW = 2026-10-28 18:00:00, this resolves to 2026-10-27T00:00:00 to 2026-10-28T00:00:00.
2. last_7_days, last_14_days, last_28_days, last_n_days:
   Day-aligned window starting at midnight (00:00:00) n days before now.date(),
   and extending up to now.isoformat().
   - last_14_days: [2026-10-14T00:00:00, 2026-10-28T18:00:00)
   - last_7_days:  [2026-10-21T00:00:00, 2026-10-28T18:00:00)
   - last_28_days: [2026-09-30T00:00:00, 2026-10-28T18:00:00)
3. last_n_hours:
   Sub-day rolling window: [now - n hours, now].
4. all_history:
   Full dataset history: from data start (28 days before now at 00:00:00) to now.
"""

from datetime import datetime, time, timedelta

import config
from schema import SymbolicTimeRange, TimeRangeKind


def resolve_symbolic_range(
    time_range: SymbolicTimeRange | dict,
    simulated_now: datetime = config.SIMULATED_NOW,
) -> tuple[str, str]:
    """
    Deterministically resolves a symbolic time specification to (start_iso, end_iso).

    Args:
        time_range: A SymbolicTimeRange instance or equivalent dict {"kind": ..., "n": ...}.
        simulated_now: Reference datetime (defaults to config.SIMULATED_NOW).

    Returns:
        tuple[str, str]: (start_iso_timestamp, end_iso_timestamp).
    """
    if isinstance(time_range, dict):
        time_range = SymbolicTimeRange.model_validate(time_range)

    kind = time_range.kind
    now_date = simulated_now.date()
    now_iso = simulated_now.isoformat()

    if kind == TimeRangeKind.yesterday:
        yesterday_date = now_date - timedelta(days=1)
        start = datetime.combine(yesterday_date, time.min).isoformat()
        end = datetime.combine(now_date, time.min).isoformat()
        return start, end

    if kind == TimeRangeKind.last_7_days:
        start_date = now_date - timedelta(days=7)
        start = datetime.combine(start_date, time.min).isoformat()
        return start, now_iso

    if kind == TimeRangeKind.last_14_days:
        start_date = now_date - timedelta(days=14)
        start = datetime.combine(start_date, time.min).isoformat()
        return start, now_iso

    if kind == TimeRangeKind.last_28_days:
        start_date = now_date - timedelta(days=28)
        start = datetime.combine(start_date, time.min).isoformat()
        return start, now_iso

    if kind == TimeRangeKind.last_n_days:
        n = time_range.n if time_range.n is not None else 1
        start_date = now_date - timedelta(days=n)
        start = datetime.combine(start_date, time.min).isoformat()
        return start, now_iso

    if kind == TimeRangeKind.last_n_hours:
        n = time_range.n if time_range.n is not None else 1
        start_dt = simulated_now - timedelta(hours=n)
        return start_dt.isoformat(), now_iso

    if kind == TimeRangeKind.all_history:
        start_date = now_date - timedelta(days=config.HISTORY_DAYS)
        start = datetime.combine(start_date, time.min).isoformat()
        return start, now_iso

    raise ValueError(f"Unsupported time range kind: {kind}")
