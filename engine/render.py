"""
engine/render.py - Deterministic Answer Rendering for VehicleGPT.

Renders concise natural-language answers and evidence summaries from
schema.ToolOutput. Every numeric figure in the rendered answer is drawn
directly from the verified tool output.
"""

import json

from schema import ToolName, ToolOutput


def render_answer(tool_name: ToolName | str, output: ToolOutput) -> str:
    """
    Renders a human-readable text answer from a ToolOutput.

    Args:
        tool_name: The executed tool (ToolName enum or string).
        output: The ToolOutput containing result and evidence.

    Returns:
        str: Clean, formatted answer text backed by evidence.
    """
    name = tool_name.value if isinstance(tool_name, ToolName) else str(tool_name)
    res = output.result

    if not isinstance(res, dict):
        return str(res)

    # 1. get_events
    if name == "get_events":
        count = res.get("count", 0)
        ev_type = res.get("event_type", "event")
        start = res.get("start", "")
        end = res.get("end", "")
        events = res.get("events", [])

        if count == 0:
            return f"No {ev_type} events were recorded between {start} and {end}."

        lines = [f"Found {count} {ev_type} event(s) between {start} and {end}:"]
        for ev in events:
            ev_id = ev.get("event_id", "")
            ts = ev.get("ts", "")
            details = ev.get("details", {})
            reason = details.get("reason", "")
            seg = details.get("segment", "")
            speed = details.get("speed_kmh")
            extra = []
            if seg:
                extra.append(seg)
            if speed is not None:
                extra.append(f"{float(speed):.1f} km/h")
            if reason:
                extra.append(reason)
            extra_str = f" ({', '.join(extra)})" if extra else ""
            lines.append(f"- {ev_id} at {ts}{extra_str}")
        return "\n".join(lines)

    # 2. context_before
    if name == "context_before":
        if res.get("status") == "not_found":
            return res.get("message", "Event not found.")

        ev_id = res.get("event_id", "")
        ev_ts = res.get("event_ts", "")
        ev_type = res.get("event_type", "")
        secs = res.get("seconds", 60)
        signals = res.get("signal_summary", {})

        lines = [f"Telemetry in the {secs} seconds before {ev_type} ({ev_id} at {ev_ts}):"]
        for sig, stats in signals.items():
            start_v = stats.get("start")
            end_v = stats.get("end")
            trend = stats.get("trend")
            min_v = stats.get("min")
            max_v = stats.get("max")
            lines.append(
                f"- {sig}: {start_v} -> {end_v} (trend: {trend}, min: {min_v}, max: {max_v})"
            )
        return "\n".join(lines)

    # 3. compare
    if name == "compare":
        metric = res.get("metric", "")
        seg_a = res.get("segment_a", {})
        seg_b = res.get("segment_b", {})
        diff = res.get("difference", 0.0)
        pct = res.get("percent_difference", 0.0)

        name_a = seg_a.get("segment", "segment A")
        val_a = seg_a.get("value", 0.0)
        name_b = seg_b.get("segment", "segment B")
        val_b = seg_b.get("value", 0.0)

        lines = [
            f"Comparison of {metric} between {name_a} and {name_b} driving:",
            f"- {name_a.capitalize()}: {val_a} Wh/km",
            f"- {name_b.capitalize()}: {val_b} Wh/km",
            (
                f"{name_b.capitalize()} driving used {abs(diff):.1f} Wh/km ({pct:+.1f}%) "
                f"{'more' if diff > 0 else 'less'} energy than {name_a} driving."
            ),
        ]
        return "\n".join(lines)

    # 4. find_largest_change
    if name == "find_largest_change":
        if res.get("status") == "insufficient_data":
            return res.get("message", "Insufficient data.")

        metric = res.get("metric", "")
        target_date = res.get("date", "")
        prev_date = res.get("previous_date", "")
        val_before = res.get("value_before", 0.0)
        val_after = res.get("value_after", 0.0)
        chg = res.get("absolute_change", 0.0)
        pct = res.get("percent_change", 0.0)
        note = res.get("note", "")

        lines = [
            f"The largest change in {metric} occurred on {target_date}:",
            f"- Previous value ({prev_date}): {val_before} Wh/km",
            f"- New value ({target_date}): {val_after} Wh/km",
            f"- Shift: {chg:+.1f} Wh/km ({pct:+.1f}%)",
        ]
        if note:
            lines.append(note)
        return "\n".join(lines)

    # 5. trend
    if name == "trend":
        metric = res.get("metric", "")
        bucket = res.get("bucket", "")
        b_count = res.get("bucket_count", 0)
        slope = res.get("slope_per_bucket", 0.0)
        direction = res.get("overall_direction", "")
        series = res.get("series", [])
        latest_val = series[-1].get("value") if series else "N/A"

        return (
            f"Trend for {metric} across {b_count} {bucket} bucket(s) is {direction} "
            f"with a slope of {slope:+.3f} per {bucket}. Latest value: {latest_val} Wh/km."
        )

    # 6. explain_factors
    if name == "explain_factors":
        if res.get("status") == "limited_evidence":
            return res.get("message", "Limited evidence to explain factors.")

        delta = res.get("delta", {})
        delta_eff = delta.get("wh_per_km_increase", 0.0)
        delta_range = delta.get("range_drop_km", 0.0)
        base_eff = delta.get("baseline_wh_per_km", 0.0)
        tgt_eff = delta.get("target_wh_per_km", 0.0)
        base_range = delta.get("baseline_range_km", 0.0)
        tgt_range = delta.get("target_range_km", 0.0)

        contribs = res.get("group_contributions_wh_km", {})
        ranking = res.get("factor_ranking", [])
        evidence_m = res.get("evidence_metrics", {})

        cold_c = contribs.get("cold_weather", 0.0)
        hwy_c = contribs.get("highway_share", 0.0)

        amb_drop = evidence_m.get("ambient_temp_drop_c", 0.0)
        hvac_inc = evidence_m.get("hvac_power_increase_kw", 0.0)
        hwy_inc = evidence_m.get("highway_share_increase_pct", 0.0)

        lines = [
            (
                f"Over the last two weeks, estimated driving range at full charge dropped by {delta_range:.1f} km "
                f"(from {base_range:.1f} km to {tgt_range:.1f} km) as energy consumption increased "
                f"from {base_eff:.1f} Wh/km to {tgt_eff:.1f} Wh/km (+{delta_eff:.1f} Wh/km)."
            ),
            "",
            "Possible contributing factors ranked by impact:",
        ]

        rank_idx = 1
        for factor in ranking:
            if factor == "cold_weather":
                lines.append(
                    f"{rank_idx}. Cold weather: contributed approximately +{cold_c:.1f} Wh/km "
                    f"(ambient temperature dropped by {amb_drop:.1f}°C and HVAC power increased by {hvac_inc:.2f} kW)."
                )
            elif factor == "highway_share":
                lines.append(
                    f"{rank_idx}. Highway driving share: contributed approximately +{hwy_c:.1f} Wh/km "
                    f"(highway share increased by {hwy_inc:.1f}%)."
                )
            rank_idx += 1

        lines.append("")
        lines.append(
            "Note: Results represent possible contributing factors based on driving telemetry."
        )
        return "\n".join(lines)

    # 7. out_of_scope
    if name == "out_of_scope":
        reason = res.get("reason", "")
        msg = res.get("message", "Request out of scope.")
        return f"{msg} Reason: {reason}"

    return json.dumps(res, indent=2)


def render_evidence(output: ToolOutput) -> list[str]:
    """Returns the formatted list of evidence items backing the tool output."""
    return [str(e) for e in output.evidence]
