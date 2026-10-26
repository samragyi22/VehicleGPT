"""
tests/test_engine.py - Phase 2 Deterministic Query Engine Unit Tests.

Validates the full deterministic engine without LLM dependencies:
1. Answers all 5 Case Study questions and validates results against data/ground_truth.json:
   - Case 1: Driving range drop over 2 weeks & causal factor ranking (cold_weather, highway_share).
   - Case 2: Date and magnitude of largest efficiency drop (2026-10-14).
   - Case 3: All ADAS disengagement events on 'yesterday' (EV-0007, EV-0008, EV-0009).
   - Case 4: Pre-event telemetry before battery overheating (pack_temp_c rising).
   - Case 5: City vs Highway energy consumption comparison (within 1% of GT).
2. Verifies security and execution controls:
   - Unknown/malformed tool arguments are rejected by validation.
   - Database row limits are strictly respected.
   - Opcode progress handler query timeouts interrupt runaway queries.
   - Disabled tools (search_logs) raise clear errors.
3. Benchmarks and prints per-tool median and p95/worst query times.
"""

import json
import sqlite3
import statistics
import time

import pytest
from pydantic import ValidationError

import config
from engine.dates import resolve_symbolic_range
from engine.render import render_answer, render_evidence
from engine.tools import (
    compare,
    context_before,
    explain_factors,
    find_largest_change,
    get_db_connection,
    get_events,
    out_of_scope,
    search_logs,
    trend,
)
from engine.validate import prepare_executable_args, validate_tool_call
from schema import (
    BucketSize,
    ChangeDirection,
    EventType,
    Metric,
    Segment,
    ToolName,
)


@pytest.fixture(scope="module")
def ground_truth() -> dict:
    with open(config.GROUND_TRUTH_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Date Resolution Tests
# ---------------------------------------------------------------------------
def test_dates_resolution():
    """Verify deterministic day-aligned date resolution against config.SIMULATED_NOW."""
    now = config.SIMULATED_NOW

    # Yesterday: full day 2026-10-27
    s_y, e_y = resolve_symbolic_range({"kind": "yesterday"})
    assert s_y == "2026-10-27T00:00:00"
    assert e_y == "2026-10-28T00:00:00"

    # Last 14 days: 2026-10-14T00:00:00 to now
    s_14, e_14 = resolve_symbolic_range({"kind": "last_14_days"})
    assert s_14 == "2026-10-14T00:00:00"
    assert e_14 == now.isoformat()

    # Last 7 days: 2026-10-21T00:00:00 to now
    s_7, e_7 = resolve_symbolic_range({"kind": "last_7_days"})
    assert s_7 == "2026-10-21T00:00:00"
    assert e_7 == now.isoformat()

    # Last 28 days: 2026-09-30T00:00:00 to now
    s_28, e_28 = resolve_symbolic_range({"kind": "last_28_days"})
    assert s_28 == "2026-09-30T00:00:00"
    assert e_28 == now.isoformat()

    # Last n days (n=5)
    s_5, e_5 = resolve_symbolic_range({"kind": "last_n_days", "n": 5})
    assert s_5 == "2026-10-23T00:00:00"
    assert e_5 == now.isoformat()

    # Last n hours (n=12)
    s_h, e_h = resolve_symbolic_range({"kind": "last_n_hours", "n": 12})
    assert s_h == "2026-10-28T06:00:00"
    assert e_h == now.isoformat()

    # All history: 28 days back to now
    s_all, e_all = resolve_symbolic_range({"kind": "all_history"})
    assert s_all == "2026-09-30T00:00:00"
    assert e_all == now.isoformat()


# ---------------------------------------------------------------------------
# 2. Case Question 1: Range Drop & Causal Factor Ranking
# ---------------------------------------------------------------------------
def test_case_1_range_drop_and_factors(ground_truth):
    """
    Q1: 'Why has my driving range reduced over the last two weeks?'
    - Window efficiency and full-charge range within 1% of ground truth.
    - cold_weather ranked #1, highway_share ranked #2.
    - Factor contributions within 3 Wh/km of 29.0 and 6.4.
    """
    rd = ground_truth["scenarios"]["range_drop"]
    start, end = resolve_symbolic_range({"kind": "last_14_days"})

    out = explain_factors(metric=Metric.estimated_range_km, start=start, end=end)
    res = out.result
    delta = res["delta"]

    # Verify baseline and target window metrics vs ground truth
    gt_base = rd["baseline_window"]
    gt_target = rd["target_window"]

    assert abs(delta["baseline_wh_per_km"] - gt_base["wh_per_km"]) / gt_base["wh_per_km"] < 0.01
    assert abs(delta["target_wh_per_km"] - gt_target["wh_per_km"]) / gt_target["wh_per_km"] < 0.01
    assert (
        abs(delta["baseline_range_km"] - gt_base["estimated_range_km"])
        / gt_base["estimated_range_km"]
        < 0.01
    )
    assert (
        abs(delta["target_range_km"] - gt_target["estimated_range_km"])
        / gt_target["estimated_range_km"]
        < 0.01
    )

    # Verify baseline truncation flag and window duration in days
    assert res["baseline_truncated"] is True
    assert res["window_durations_days"]["target"] == 14.75
    assert res["window_durations_days"]["baseline_nominal"] == 14.75
    assert res["window_durations_days"]["baseline_actual"] == 13.67

    # Verify factor ranking
    assert res["factor_ranking"] == ["cold_weather", "highway_share"]

    # Verify group contributions within 3 Wh/km of Ground Truth (29.0 and 6.4)
    contribs = res["group_contributions_wh_km"]
    assert abs(contribs["cold_weather"] - 29.0) <= 3.0
    assert abs(contribs["highway_share"] - 6.4) <= 3.0

    # Verify rendering uses 'estimated full-charge range' and explains truncation
    rendered = render_answer(ToolName.explain_factors, out)
    assert "estimated full-charge range" in rendered
    assert "Cold weather" in rendered
    assert "Highway driving share" in rendered
    assert "truncated" in rendered
    assert len(render_evidence(out)) > 0


# ---------------------------------------------------------------------------
# 3. Case Question 2: Largest Efficiency Drop
# ---------------------------------------------------------------------------
def test_case_2_largest_efficiency_drop(ground_truth):
    """
    Q2: 'When did fuel efficiency show the largest drop?'
    - Metric: energy_wh_per_km (direction: 'rise', since higher Wh/km = lower efficiency).
    - Date identified: 2026-10-14.
    - Efficiency before: ~143.7 Wh/km, after: ~190.2 Wh/km.
    """
    gt_drop = ground_truth["scenarios"]["largest_efficiency_drop"]
    start, end = resolve_symbolic_range({"kind": "all_history"})

    out = find_largest_change(
        metric=Metric.energy_wh_per_km,
        start=start,
        end=end,
        direction=ChangeDirection.rise,
    )
    res = out.result

    assert res["date"] == gt_drop["date"]  # 2026-10-14
    assert res["previous_date"] == gt_drop["previous_date"]  # 2026-10-13
    assert abs(res["value_before"] - gt_drop["efficiency_before_wh_per_km"]) <= 0.5
    assert abs(res["value_after"] - gt_drop["efficiency_after_wh_per_km"]) <= 0.5

    rendered = render_answer(ToolName.find_largest_change, out)
    assert "2026-10-14" in rendered


# ---------------------------------------------------------------------------
# 4. Case Question 3: ADAS Disengagements Yesterday
# ---------------------------------------------------------------------------
def test_case_3_adas_disengagements_yesterday(ground_truth):
    """
    Q3: 'Show all ADAS disengagement events from yesterday.'
    - Resolved range: 2026-10-27T00:00:00 to 2026-10-28T00:00:00.
    - Exactly 3 events: EV-0007, EV-0008, EV-0009.
    """
    gt_adas = ground_truth["scenarios"]["adas_disengagements"]
    start, end = resolve_symbolic_range({"kind": "yesterday"})

    out = get_events(type=EventType.ADAS_DISENGAGEMENT, start=start, end=end)
    res = out.result

    assert res["count"] == gt_adas["yesterday_count"]
    event_ids = [e["event_id"] for e in res["events"]]
    expected_ids = [e["event_id"] for e in gt_adas["yesterday_events"]]
    assert event_ids == expected_ids
    assert event_ids == ["EV-0007", "EV-0008", "EV-0009"]

    rendered = render_answer(ToolName.get_events, out)
    assert "EV-0007" in rendered
    assert "EV-0008" in rendered
    assert "EV-0009" in rendered


# ---------------------------------------------------------------------------
# 5. Case Question 4: Context Before Battery Overheating
# ---------------------------------------------------------------------------
def test_case_4_context_before_overheating():
    """
    Q4: 'What happened before the battery overheating warning?'
    - Defaults to 300 s at 1 Hz from samples_1hz plus last 60 s snapshot at 10 Hz.
    - Pack temperature must show a 'rising' trend reaching >= 45.6°C.
    """
    out = context_before(
        event_id="latest_BATTERY_OVERHEAT_WARNING",
        signals=["pack_temp_c", "power_kw", "speed_kmh", "ambient_temp_c"],
    )
    res = out.result

    assert res["event_id"] in ("EV-0001", "EV-0002")
    assert res["seconds"] == 300
    assert res["has_high_rate_snapshot_60s"] is True

    summary = res["signal_summary"]
    temp_stats = summary["pack_temp_c"]

    assert temp_stats["trend"] == "rising"
    assert temp_stats["start"] < temp_stats["end"]
    assert temp_stats["max"] >= 45.6

    rendered = render_answer(ToolName.context_before, out)
    assert "pack_temp_c" in rendered
    assert "rising" in rendered


# ---------------------------------------------------------------------------
# 6. Case Question 5: City vs Highway Energy Consumption
# ---------------------------------------------------------------------------
def test_case_5_city_vs_highway_comparison(ground_truth):
    """
    Q5: 'Compare energy consumption between city and highway driving.'
    - City Wh/km within 1% of ground truth (133.4 Wh/km net power).
    - Highway Wh/km within 1% of ground truth (188.6 Wh/km net power).
    """
    gt_eff = ground_truth["efficiency_metrics"]
    start, end = resolve_symbolic_range({"kind": "all_history"})

    out = compare(
        metric=Metric.energy_wh_per_km,
        start=start,
        end=end,
        segment_a=Segment.city,
        segment_b=Segment.highway,
    )
    res = out.result

    city_val = res["segment_a"]["value"]
    hwy_val = res["segment_b"]["value"]

    assert abs(city_val - gt_eff["city_avg_wh_per_km"]) / gt_eff["city_avg_wh_per_km"] < 0.01
    assert abs(hwy_val - gt_eff["highway_avg_wh_per_km"]) / gt_eff["highway_avg_wh_per_km"] < 0.01

    rendered = render_answer(ToolName.compare, out)
    assert "City" in rendered
    assert "Highway" in rendered


# ---------------------------------------------------------------------------
# 7. Distance-Weighted Segments & Global ROW_LIMIT Tests
# ---------------------------------------------------------------------------
def test_distance_weighted_segments_reproduce_overall():
    """
    Validates that city and highway net efficiency, weighted by distance,
    reproduces overall net efficiency within 0.5% (mathematically exact with net energy).
    """
    with get_db_connection() as con:
        row_c = con.execute(
            "SELECT SUM(power_kw), SUM(speed_kmh) FROM samples_1hz WHERE road_segment = 'city'"
        ).fetchone()
        row_h = con.execute(
            "SELECT SUM(power_kw), SUM(speed_kmh) FROM samples_1hz WHERE road_segment = 'highway'"
        ).fetchone()
        row_all = con.execute("SELECT SUM(power_kw), SUM(speed_kmh) FROM samples_1hz").fetchone()

    city_wh = 1000.0 * row_c[0] / row_c[1]
    city_dist = row_c[1] / 3600.0

    hwy_wh = 1000.0 * row_h[0] / row_h[1]
    hwy_dist = row_h[1] / 3600.0

    overall_wh = 1000.0 * row_all[0] / row_all[1]

    weighted_wh = (city_dist * city_wh + hwy_dist * hwy_wh) / (city_dist + hwy_dist)
    diff_pct = abs(weighted_wh - overall_wh) / overall_wh

    assert diff_pct < 0.005, (
        f"Distance-weighted mismatch: {weighted_wh:.4f} vs {overall_wh:.4f} ({diff_pct * 100:.4f}%)"
    )


def test_global_row_limit_cap(monkeypatch):
    """Verify that global ROW_LIMIT caps query results when results would exceed it."""
    start, end = resolve_symbolic_range({"kind": "all_history"})
    # When requesting limit=500 with monkeypatched ROW_LIMIT=3, at most 3 events returned
    monkeypatch.setattr(config, "ROW_LIMIT", 3)
    out_capped = get_events(type=EventType.ADAS_DISENGAGEMENT, start=start, end=end, limit=500)
    assert out_capped.meta.row_count <= 3
    assert len(out_capped.result["events"]) <= 3
    assert len(out_capped.result["events"]) == 3


# ---------------------------------------------------------------------------
# 7. Validation & Execution Security Tests
# ---------------------------------------------------------------------------
def test_unknown_arguments_rejected():
    """Verify that unknown arguments and invalid enums are rejected with ValueError or ValidationError."""
    # Unknown argument
    with pytest.raises((ValueError, ValidationError)):
        validate_tool_call(
            {
                "tool": "compare",
                "args": {
                    "metric": "energy_wh_per_km",
                    "range": {"kind": "yesterday"},
                    "bogus_argument": 42,
                },
            }
        )

    # Invalid enum value
    with pytest.raises((ValueError, ValidationError)):
        validate_tool_call(
            {
                "tool": "compare",
                "args": {
                    "metric": "non_existent_metric",
                    "range": {"kind": "yesterday"},
                },
            }
        )

    # Invalid event_id format
    with pytest.raises((ValueError, ValidationError)):
        validate_tool_call(
            {
                "tool": "context_before",
                "args": {
                    "event_id": "DROP TABLE events;--",
                },
            }
        )


def test_row_limit_enforced():
    """Verify get_events caps rows at the specified limit."""
    start, end = resolve_symbolic_range({"kind": "all_history"})
    out = get_events(type=EventType.ADAS_DISENGAGEMENT, start=start, end=end, limit=2)
    assert out.meta.row_count <= 2
    assert len(out.result["events"]) <= 2


def test_query_timeout():
    """Verify that query timeout handler interrupts long-running queries."""
    with (
        pytest.raises(sqlite3.OperationalError, match="interrupted"),
        get_db_connection(timeout_seconds=0.0001) as con,
    ):
        con.execute("SELECT COUNT(*) FROM samples_1hz s1, samples_1hz s2").fetchone()


def test_disabled_search_logs():
    """Verify that search_logs raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="disabled"):
        search_logs(text="error", start="2026-10-01T00:00:00", end="2026-10-02T00:00:00")


def test_validate_and_prepare_executable_args():
    """Verify tool call validation and executable argument preparation."""
    tool, args = validate_tool_call(
        {
            "tool": "get_events",
            "args": {"type": "ADAS_DISENGAGEMENT", "range": {"kind": "yesterday"}},
        }
    )
    exec_args = prepare_executable_args(tool, args)
    assert exec_args.start == "2026-10-27T00:00:00"
    assert exec_args.end == "2026-10-28T00:00:00"


def test_out_of_scope_tool():
    """Verify out_of_scope tool output and rendering."""
    out = out_of_scope(reason="User asked about weather forecast.")
    assert out.result["status"] == "out_of_scope"
    rendered = render_answer(ToolName.out_of_scope, out)
    assert "outside the scope" in rendered
    with pytest.raises(NotImplementedError, match="disabled"):
        search_logs(text="error", start="2026-10-01T00:00:00", end="2026-10-02T00:00:00")


# ---------------------------------------------------------------------------
# 8. Query Timing Benchmark (Prints Median & Worst Time per Tool)
# ---------------------------------------------------------------------------
def test_per_tool_timing_benchmark():
    """
    Executes each tool repeatedly to measure query latency.
    Prints a markdown table with median and worst execution times.
    """
    tools_to_benchmark = [
        (
            "get_events",
            lambda: get_events(
                type=EventType.ADAS_DISENGAGEMENT,
                start="2026-10-27T00:00:00",
                end="2026-10-28T00:00:00",
            ),
        ),
        (
            "context_before",
            lambda: context_before(event_id="latest_BATTERY_OVERHEAT_WARNING", seconds=60),
        ),
        (
            "compare",
            lambda: compare(
                metric=Metric.energy_wh_per_km,
                start="2026-09-30T00:00:00",
                end="2026-10-28T18:00:00",
            ),
        ),
        (
            "find_largest_change",
            lambda: find_largest_change(
                metric=Metric.energy_wh_per_km,
                start="2026-09-30T00:00:00",
                end="2026-10-28T18:00:00",
                direction=ChangeDirection.rise,
            ),
        ),
        (
            "trend",
            lambda: trend(
                metric=Metric.energy_wh_per_km,
                start="2026-10-14T00:00:00",
                end="2026-10-28T18:00:00",
                bucket=BucketSize.day,
            ),
        ),
        (
            "explain_factors",
            lambda: explain_factors(
                metric=Metric.estimated_range_km,
                start="2026-10-14T00:00:00",
                end="2026-10-28T18:00:00",
            ),
        ),
    ]

    print("\n--- PER-TOOL QUERY TIMING BENCHMARK ---")
    timings: dict[str, list[float]] = {}
    warmup_runs = 2
    measured_runs = 10

    for name, fn in tools_to_benchmark:
        for _ in range(warmup_runs):
            fn()
        durations: list[float] = []
        for _ in range(measured_runs):
            t_start = time.perf_counter()
            fn()
            durations.append((time.perf_counter() - t_start) * 1000.0)
        timings[name] = durations

    print("| Tool | Median (ms) | Worst (ms) | Target (<2000 ms) |")
    print("|---|---|---|---|")
    for name, dur_list in timings.items():
        med = statistics.median(dur_list)
        worst = max(dur_list)
        status = "PASS" if worst < 2000.0 else "FAIL"
        print(f"| `{name}` | {med:.2f} ms | {worst:.2f} ms | {status} |")
        assert worst < 2000.0, f"Tool {name} exceeded latency target: {worst:.2f} ms"
