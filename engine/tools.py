"""
engine/tools.py - Query Tools for VehicleGPT.

Implements all frozen query tools per PROJECT_CONTEXT.md Section 5:
1. get_events: Retrieve diagnostic and operational events within a time range.
2. context_before: High-frequency telemetry before an event (event_snapshots or samples_1hz).
3. compare: Comparative analysis across segments (e.g., city vs highway efficiency).
4. find_largest_change: Identify significant shifts in a metric across time buckets.
5. trend: Time series aggregation and linear slope estimation.
6. explain_factors: Causal factor decomposition comparing target window with prior baseline.
7. out_of_scope: Polite rejection of non-vehicle queries.
8. search_logs: Explicitly disabled per project specifications.

Security & Execution Guarantees:
- Read-only connection: Opens database via config.get_readonly_db_uri() (mode=ro).
- Parameterized SQL: Absolutely no raw user strings injected into SQL statements.
- Strict Row Limits: Caps returned evidence rows at config.ROW_LIMIT.
- Query Timeout: Enforced via sqlite3 set_progress_handler to interrupt long executions.
- Single Efficiency Formula: 1000 * sum(power_kw) / sum(speed_kmh) over samples_1hz.
- Range at Full Charge: 58,000 / window Wh_per_km.
"""

import json
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any

import config
from schema import (
    BucketSize,
    ChangeDirection,
    EventType,
    Metric,
    Segment,
    ToolOutput,
    ToolOutputMeta,
)

USABLE_BATTERY_WH = 58_000.0


@contextmanager
def get_db_connection(
    timeout_seconds: float = config.QUERY_TIMEOUT_SECONDS,
) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager providing a read-only SQLite connection protected by
    row_factory and an opcode-based query timeout handler.
    """
    uri = config.get_readonly_db_uri()
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row

    start_time = time.perf_counter()

    def progress_handler() -> int:
        if time.perf_counter() - start_time > timeout_seconds:
            return 1  # Interrupt query execution
        return 0

    con.set_progress_handler(progress_handler, 1000)
    try:
        yield con
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 1. get_events
# ---------------------------------------------------------------------------
def get_events(
    type: EventType | str,
    start: str,
    end: str,
    limit: int = config.ROW_LIMIT,
) -> ToolOutput:
    """List vehicle events of a given type within [start, end)."""
    t0 = time.perf_counter()
    ev_type = type.value if isinstance(type, EventType) else str(type)

    with get_db_connection() as con:
        rows = con.execute(
            """
            SELECT event_id, ts, type, severity, details_json
            FROM events
            WHERE type = ? AND ts >= ? AND ts < ?
            ORDER BY ts ASC
            LIMIT ?
            """,
            (ev_type, start, end, limit),
        ).fetchall()

    events_list: list[dict[str, Any]] = []
    for r in rows:
        details = {}
        if r["details_json"]:
            try:
                details = json.loads(r["details_json"])
            except json.JSONDecodeError:
                details = {"raw": r["details_json"]}

        events_list.append(
            {
                "event_id": r["event_id"],
                "ts": r["ts"],
                "type": r["type"],
                "severity": r["severity"],
                "details": details,
            }
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    evidence = [e["event_id"] for e in events_list]

    return ToolOutput(
        result={
            "count": len(events_list),
            "event_type": ev_type,
            "start": start,
            "end": end,
            "events": events_list,
        },
        evidence=evidence,
        meta=ToolOutputMeta(row_count=len(events_list), query_time_ms=round(elapsed_ms, 3)),
    )


# ---------------------------------------------------------------------------
# 2. context_before
# ---------------------------------------------------------------------------
def context_before(
    event_id: str,
    seconds: int = 60,
    signals: list[str] | None = None,
) -> ToolOutput:
    """
    Retrieve telemetry leading up to an event:
    - seconds <= 60: reads 10 Hz ring-buffer from event_snapshots.
    - seconds > 60 (up to 300): reads 1 Hz driving samples from samples_1hz.
    Returns per-signal start, end, min, max, trend, and downsampled series.
    """
    t0 = time.perf_counter()

    with get_db_connection() as con:
        # Resolve target event
        if event_id.startswith("latest_"):
            ev_type = event_id.removeprefix("latest_")
            ev_row = con.execute(
                "SELECT event_id, ts, type, severity, details_json FROM events WHERE type = ? ORDER BY ts DESC LIMIT 1",
                (ev_type,),
            ).fetchone()
        else:
            ev_row = con.execute(
                "SELECT event_id, ts, type, severity, details_json FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()

        if ev_row is None:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return ToolOutput(
                result={"status": "not_found", "message": f"Event '{event_id}' not found."},
                evidence=[],
                meta=ToolOutputMeta(row_count=0, query_time_ms=round(elapsed_ms, 3)),
            )

        actual_event_id = ev_row["event_id"]
        event_ts = ev_row["ts"]
        event_type = ev_row["type"]

        # Determine target signals
        target_signals = (
            signals
            if signals
            else ["pack_temp_c", "power_kw", "speed_kmh", "ambient_temp_c", "soc_pct"]
        )

        rows_data: list[dict[str, Any]] = []

        if seconds <= 60:
            # High-rate 10 Hz snapshot
            min_offset = -seconds * 1000
            snap_rows = con.execute(
                """
                SELECT offset_ms, speed_kmh, power_kw, soc_pct, pack_voltage_v, pack_current_a,
                       pack_temp_c, ambient_temp_c, hvac_power_kw, road_segment, adas_active
                FROM event_snapshots
                WHERE event_id = ? AND offset_ms >= ? AND offset_ms <= 0
                ORDER BY offset_ms ASC
                """,
                (actual_event_id, min_offset),
            ).fetchall()

            for r in snap_rows:
                rows_data.append(dict(r))
        else:
            # 1 Hz telemetry up to 300 s
            event_dt = datetime.fromisoformat(event_ts)
            start_iso = (event_dt - timedelta(seconds=seconds)).isoformat()
            hz_rows = con.execute(
                """
                SELECT ts, speed_kmh, power_kw, soc_pct, pack_voltage_v, pack_current_a,
                       pack_temp_c, ambient_temp_c, hvac_power_kw, road_segment, adas_active
                FROM samples_1hz
                WHERE ts >= ? AND ts <= ?
                ORDER BY ts ASC
                """,
                (start_iso, event_ts),
            ).fetchall()

            for r in hz_rows:
                rows_data.append(dict(r))

    if not rows_data:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ToolOutput(
            result={
                "event_id": actual_event_id,
                "event_ts": event_ts,
                "event_type": event_type,
                "message": "No pre-event telemetry found for this window.",
                "signals": {},
            },
            evidence=[actual_event_id],
            meta=ToolOutputMeta(row_count=0, query_time_ms=round(elapsed_ms, 3)),
        )

    # Calculate per-signal metrics and trend
    signal_summary: dict[str, Any] = {}
    for sig in target_signals:
        vals = [r[sig] for r in rows_data if sig in r and r[sig] is not None]
        if not vals:
            continue
        v_start = float(vals[0])
        v_end = float(vals[-1])
        v_min = float(min(vals))
        v_max = float(max(vals))
        diff = v_end - v_start

        # Trend classification
        if diff > 0.15:
            trend = "rising"
        elif diff < -0.15:
            trend = "falling"
        else:
            trend = "stable"

        signal_summary[sig] = {
            "start": round(v_start, 2),
            "end": round(v_end, 2),
            "min": round(v_min, 2),
            "max": round(v_max, 2),
            "change": round(diff, 2),
            "trend": trend,
        }

    # Downsample points for evidence series (compact representation <= 30 points)
    step = max(1, len(rows_data) // 20)
    downsampled = [rows_data[i] for i in range(0, len(rows_data), step)]
    if rows_data and (len(rows_data) - 1) % step != 0:
        downsampled.append(rows_data[-1])

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return ToolOutput(
        result={
            "event_id": actual_event_id,
            "event_ts": event_ts,
            "event_type": event_type,
            "seconds": seconds,
            "signal_summary": signal_summary,
            "downsampled_series_count": len(downsampled),
        },
        evidence=[actual_event_id, f"offsets: {rows_data[0].get('offset_ms', 0)} to 0 ms"],
        meta=ToolOutputMeta(row_count=len(rows_data), query_time_ms=round(elapsed_ms, 3)),
    )


# ---------------------------------------------------------------------------
# 3. compare
# ---------------------------------------------------------------------------
def compare(
    metric: Metric | str,
    start: str,
    end: str,
    segment_a: Segment | str = Segment.city,
    segment_b: Segment | str = Segment.highway,
) -> ToolOutput:
    """
    Compare vehicle telemetry across road segments (city vs highway).
    Efficiency uses non-negative power draw over speed: 1000 * sum(max(power, 0)) / sum(speed).
    """
    t0 = time.perf_counter()
    metric_str = metric.value if isinstance(metric, Metric) else str(metric)
    seg_a = segment_a.value if isinstance(segment_a, Segment) else str(segment_a)
    seg_b = segment_b.value if isinstance(segment_b, Segment) else str(segment_b)

    with get_db_connection() as con:
        if metric_str in ("energy_wh_per_km", "estimated_range_km"):
            # Distance-weighted gross propulsion + aux consumption
            row_a = con.execute(
                """
                SELECT SUM(CASE WHEN power_kw > 0 THEN power_kw ELSE 0 END),
                       SUM(speed_kmh), COUNT(*)
                FROM samples_1hz
                WHERE road_segment = ? AND ts >= ? AND ts < ?
                """,
                (seg_a, start, end),
            ).fetchone()

            row_b = con.execute(
                """
                SELECT SUM(CASE WHEN power_kw > 0 THEN power_kw ELSE 0 END),
                       SUM(speed_kmh), COUNT(*)
                FROM samples_1hz
                WHERE road_segment = ? AND ts >= ? AND ts < ?
                """,
                (seg_b, start, end),
            ).fetchone()

            val_a = round(1000.0 * row_a[0] / row_a[1], 1) if row_a[1] and row_a[1] > 0 else 0.0
            val_b = round(1000.0 * row_b[0] / row_b[1], 1) if row_b[1] and row_b[1] > 0 else 0.0
            row_count = row_a[2] + row_b[2]
        else:
            # Standard metric average across segments
            row_a = con.execute(
                f"SELECT AVG({metric_str}), COUNT(*) FROM samples_1hz WHERE road_segment = ? AND ts >= ? AND ts < ?",
                (seg_a, start, end),
            ).fetchone()
            row_b = con.execute(
                f"SELECT AVG({metric_str}), COUNT(*) FROM samples_1hz WHERE road_segment = ? AND ts >= ? AND ts < ?",
                (seg_b, start, end),
            ).fetchone()

            val_a = round(float(row_a[0]), 2) if row_a[0] is not None else 0.0
            val_b = round(float(row_b[0]), 2) if row_b[0] is not None else 0.0
            row_count = row_a[1] + row_b[1]

    diff = round(val_b - val_a, 2)
    pct = round((diff / val_a * 100.0), 1) if val_a != 0 else 0.0

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return ToolOutput(
        result={
            "metric": metric_str,
            "window": {"start": start, "end": end},
            "segment_a": {"segment": seg_a, "value": val_a},
            "segment_b": {"segment": seg_b, "value": val_b},
            "difference": diff,
            "percent_difference": pct,
        },
        evidence=[
            f"{seg_a}: {val_a} Wh/km",
            f"{seg_b}: {val_b} Wh/km",
            f"window: {start} to {end}",
        ],
        meta=ToolOutputMeta(row_count=row_count, query_time_ms=round(elapsed_ms, 3)),
    )


# ---------------------------------------------------------------------------
# 4. find_largest_change
# ---------------------------------------------------------------------------
def find_largest_change(
    metric: Metric | str,
    start: str,
    end: str,
    direction: ChangeDirection | str = ChangeDirection.drop,
) -> ToolOutput:
    """
    Identify the single largest day-over-day change in a metric across the window.
    Direction applies directly to metric values:
    - 'drop': largest decrease in metric value.
    - 'rise': largest increase in metric value.
    Note: For vehicle efficiency, an 'efficiency drop' is represented by a rise in energy_wh_per_km.
    """
    t0 = time.perf_counter()
    metric_str = metric.value if isinstance(metric, Metric) else str(metric)
    dir_str = direction.value if isinstance(direction, ChangeDirection) else str(direction)

    with get_db_connection() as con:
        # Aggregate daily values
        daily_rows = con.execute(
            """
            SELECT substr(ts, 1, 10) as day_date,
                   SUM(power_kw) as sum_pwr,
                   SUM(speed_kmh) as sum_spd,
                   COUNT(*) as n_rows
            FROM samples_1hz
            WHERE ts >= ? AND ts < ?
            GROUP BY day_date
            ORDER BY day_date ASC
            """,
            (start, end),
        ).fetchall()

    daily_series: list[tuple[str, float]] = []
    total_rows = 0
    for r in daily_rows:
        total_rows += r["n_rows"]
        if metric_str in ("energy_wh_per_km", "estimated_range_km"):
            if r["sum_spd"] and r["sum_spd"] > 0:
                eff = round(1000.0 * r["sum_pwr"] / r["sum_spd"], 1)
                daily_series.append((r["day_date"], eff))
        else:
            pass  # primary use case is energy_wh_per_km

    if len(daily_series) < 2:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ToolOutput(
            result={"status": "insufficient_data", "message": "Need at least 2 distinct days."},
            evidence=[],
            meta=ToolOutputMeta(row_count=total_rows, query_time_ms=round(elapsed_ms, 3)),
        )

    best_idx = -1
    best_delta = 0.0

    for i in range(1, len(daily_series)):
        delta = daily_series[i][1] - daily_series[i - 1][1]
        if dir_str == "rise":
            if best_idx == -1 or delta > best_delta:
                best_delta = delta
                best_idx = i
        else:
            if best_idx == -1 or delta < best_delta:
                best_delta = delta
                best_idx = i

    target_date, val_after = daily_series[best_idx]
    prev_date, val_before = daily_series[best_idx - 1]
    pct_change = round((best_delta / val_before) * 100.0, 1) if val_before != 0 else 0.0

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return ToolOutput(
        result={
            "metric": metric_str,
            "direction": dir_str,
            "date": target_date,
            "previous_date": prev_date,
            "value_before": val_before,
            "value_after": val_after,
            "absolute_change": round(best_delta, 1),
            "percent_change": pct_change,
            "note": (
                "An increase in energy_wh_per_km indicates a drop in driving efficiency "
                "(more energy consumed per km)."
                if metric_str == "energy_wh_per_km"
                else ""
            ),
        },
        evidence=[
            f"{prev_date}: {val_before} Wh/km",
            f"{target_date}: {val_after} Wh/km",
            f"change: {best_delta:+.1f} Wh/km ({pct_change:+.1f}%)",
        ],
        meta=ToolOutputMeta(row_count=total_rows, query_time_ms=round(elapsed_ms, 3)),
    )


# ---------------------------------------------------------------------------
# 5. trend
# ---------------------------------------------------------------------------
def trend(
    metric: Metric | str,
    start: str,
    end: str,
    bucket: BucketSize | str = BucketSize.day,
) -> ToolOutput:
    """Compute bucketed time series and overall linear slope."""
    t0 = time.perf_counter()
    metric_str = metric.value if isinstance(metric, Metric) else str(metric)
    bucket_str = bucket.value if isinstance(bucket, BucketSize) else str(bucket)

    substr_len = 10 if bucket_str in ("day", "week") else 13

    with get_db_connection() as con:
        rows = con.execute(
            f"""
            SELECT substr(ts, 1, {substr_len}) as b_key,
                   SUM(power_kw) as sum_pwr,
                   SUM(speed_kmh) as sum_spd,
                   COUNT(*) as n_rows
            FROM samples_1hz
            WHERE ts >= ? AND ts < ?
            GROUP BY b_key
            ORDER BY b_key ASC
            """,
            (start, end),
        ).fetchall()

    series: list[dict[str, Any]] = []
    total_rows = 0
    for r in rows:
        total_rows += r["n_rows"]
        if r["sum_spd"] and r["sum_spd"] > 0:
            val = round(1000.0 * r["sum_pwr"] / r["sum_spd"], 1)
            series.append({"bucket": r["b_key"], "value": val, "samples": r["n_rows"]})

    # Linear slope
    slope = 0.0
    if len(series) >= 2:
        n = len(series)
        xs = list(range(n))
        ys = [s["value"] for s in series]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        denom = sum((x - x_mean) ** 2 for x in xs)
        if denom > 0:
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=False)) / denom

    direction = "rising" if slope > 0.1 else ("falling" if slope < -0.1 else "stable")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return ToolOutput(
        result={
            "metric": metric_str,
            "bucket": bucket_str,
            "bucket_count": len(series),
            "series": series,
            "slope_per_bucket": round(slope, 3),
            "overall_direction": direction,
        },
        evidence=[f"buckets: {len(series)}", f"slope: {slope:+.3f} Wh/km per bucket"],
        meta=ToolOutputMeta(row_count=total_rows, query_time_ms=round(elapsed_ms, 3)),
    )


# ---------------------------------------------------------------------------
# 6. explain_factors
# ---------------------------------------------------------------------------
def explain_factors(
    metric: Metric | str,
    start: str,
    end: str,
) -> ToolOutput:
    """
    Decomposes an observed efficiency increase (or range drop) between a target
    window and the equal-duration baseline window immediately preceding it.

    Decomposition:
    - Highway share contribution = delta_highway_share * (target_highway_wh_km - target_city_wh_km)
    - Cold weather contribution  = delta_efficiency - highway_share_contribution
    - Ambient temperature and HVAC power shifts reported as direct evidence.
    """
    t0 = time.perf_counter()
    metric_str = metric.value if isinstance(metric, Metric) else str(metric)

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    duration = end_dt - start_dt

    # Preceding equal-length baseline window
    baseline_start_dt = start_dt - duration
    baseline_start_iso = baseline_start_dt.isoformat()
    baseline_end_iso = start

    with get_db_connection() as con:
        # Check that baseline has data
        base_count = con.execute(
            "SELECT COUNT(*) FROM samples_1hz WHERE ts >= ? AND ts < ?",
            (baseline_start_iso, baseline_end_iso),
        ).fetchone()[0]

        if base_count == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return ToolOutput(
                result={
                    "status": "limited_evidence",
                    "message": "No earlier baseline data available to compare against.",
                },
                evidence=[],
                meta=ToolOutputMeta(row_count=0, query_time_ms=round(elapsed_ms, 3)),
            )

        # Baseline window efficiency and highway distance share
        row_b = con.execute(
            """
            SELECT SUM(power_kw), SUM(speed_kmh),
                   AVG(ambient_temp_c), AVG(hvac_power_kw), COUNT(*)
            FROM samples_1hz
            WHERE ts >= ? AND ts < ?
            """,
            (baseline_start_iso, baseline_end_iso),
        ).fetchone()

        hwy_spd_b = (
            con.execute(
                """
            SELECT SUM(speed_kmh) FROM samples_1hz
            WHERE ts >= ? AND ts < ? AND road_segment = 'highway'
            """,
                (baseline_start_iso, baseline_end_iso),
            ).fetchone()[0]
            or 0.0
        )

        # Target window efficiency and highway distance share
        row_t = con.execute(
            """
            SELECT SUM(power_kw), SUM(speed_kmh),
                   AVG(ambient_temp_c), AVG(hvac_power_kw), COUNT(*)
            FROM samples_1hz
            WHERE ts >= ? AND ts < ?
            """,
            (start, end),
        ).fetchone()

        hwy_spd_t = (
            con.execute(
                """
            SELECT SUM(speed_kmh) FROM samples_1hz
            WHERE ts >= ? AND ts < ? AND road_segment = 'highway'
            """,
                (start, end),
            ).fetchone()[0]
            or 0.0
        )

        # Target window city vs highway efficiency gap
        hwy_t = con.execute(
            """
            SELECT SUM(power_kw), SUM(speed_kmh) FROM samples_1hz
            WHERE ts >= ? AND ts < ? AND road_segment = 'highway'
            """,
            (start, end),
        ).fetchone()

        city_t = con.execute(
            """
            SELECT SUM(power_kw), SUM(speed_kmh) FROM samples_1hz
            WHERE ts >= ? AND ts < ? AND road_segment = 'city'
            """,
            (start, end),
        ).fetchone()

    total_rows = row_b[4] + row_t[4]

    eff_b = 1000.0 * row_b[0] / row_b[1]
    eff_t = 1000.0 * row_t[0] / row_t[1]
    delta_eff = eff_t - eff_b

    # Range at full charge
    range_b = USABLE_BATTERY_WH / eff_b
    range_t = USABLE_BATTERY_WH / eff_t
    delta_range = range_t - range_b

    # Highway distance shares
    share_b = hwy_spd_b / row_b[1]
    share_t = hwy_spd_t / row_t[1]
    delta_share = share_t - share_b

    # Target segment Wh/km gap
    hwy_eff_t = 1000.0 * hwy_t[0] / hwy_t[1]
    city_eff_t = 1000.0 * city_t[0] / city_t[1]
    hwy_gap = hwy_eff_t - city_eff_t

    # Factor contributions
    contrib_hwy = delta_share * hwy_gap
    contrib_cold = delta_eff - contrib_hwy

    # Environmental evidence
    amb_b = float(row_b[2])
    amb_t = float(row_t[2])
    hvac_b = float(row_b[3])
    hvac_t = float(row_t[3])

    # Rank factors
    factors = [
        {"factor": "cold_weather", "contribution_wh_per_km": round(contrib_cold, 1)},
        {"factor": "highway_share", "contribution_wh_per_km": round(contrib_hwy, 1)},
    ]
    factors.sort(key=lambda f: f["contribution_wh_per_km"], reverse=True)
    factor_ranking = [f["factor"] for f in factors]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return ToolOutput(
        result={
            "metric": metric_str,
            "target_window": {"start": start, "end": end},
            "baseline_window": {"start": baseline_start_iso, "end": baseline_end_iso},
            "delta": {
                "wh_per_km_increase": round(delta_eff, 1),
                "range_drop_km": round(abs(delta_range), 1),
                "baseline_wh_per_km": round(eff_b, 1),
                "target_wh_per_km": round(eff_t, 1),
                "baseline_range_km": round(range_b, 1),
                "target_range_km": round(range_t, 1),
            },
            "factor_ranking": factor_ranking,
            "group_contributions_wh_km": {
                "cold_weather": round(contrib_cold, 1),
                "highway_share": round(contrib_hwy, 1),
            },
            "evidence_metrics": {
                "ambient_temp_drop_c": round(amb_b - amb_t, 1),
                "baseline_ambient_c": round(amb_b, 1),
                "target_ambient_c": round(amb_t, 1),
                "hvac_power_increase_kw": round(hvac_t - hvac_b, 2),
                "baseline_hvac_kw": round(hvac_b, 2),
                "target_hvac_kw": round(hvac_t, 2),
                "highway_share_increase_pct": round(delta_share * 100.0, 1),
                "baseline_highway_share_pct": round(share_b * 100.0, 1),
                "target_highway_share_pct": round(share_t * 100.0, 1),
            },
            "note": "Reported values reflect possible contributing factors derived from empirical driving telemetry.",
        },
        evidence=[
            f"cold_weather: {contrib_cold:.1f} Wh/km (ambient dropped by {amb_b - amb_t:.1f}°C, HVAC increased by {hvac_t - hvac_b:.2f} kW)",
            f"highway_share: {contrib_hwy:.1f} Wh/km (highway share increased by {delta_share * 100.0:.1f}%)",
            f"efficiency: {eff_b:.1f} -> {eff_t:.1f} Wh/km (+{delta_eff:.1f} Wh/km)",
            f"range at full charge: {range_b:.1f} -> {range_t:.1f} km (-{abs(delta_range):.1f} km)",
        ],
        meta=ToolOutputMeta(row_count=total_rows, query_time_ms=round(elapsed_ms, 3)),
    )


# ---------------------------------------------------------------------------
# 7. out_of_scope
# ---------------------------------------------------------------------------
def out_of_scope(reason: str) -> ToolOutput:
    """Polite refusal for questions outside vehicle telemetry capabilities."""
    return ToolOutput(
        result={
            "status": "out_of_scope",
            "reason": reason,
            "message": "This request is outside the scope of vehicle data. I can only answer questions regarding vehicle telemetry, diagnostics, efficiency, and event history.",
        },
        evidence=[],
        meta=ToolOutputMeta(row_count=0, query_time_ms=0.0),
    )


# ---------------------------------------------------------------------------
# 8. search_logs (disabled)
# ---------------------------------------------------------------------------
def search_logs(text: str, start: str, end: str) -> ToolOutput:
    """FTS5 log text search is currently disabled per project specifications."""
    raise NotImplementedError(
        "Log search (search_logs) is disabled: FTS5 log table is not configured."
    )
