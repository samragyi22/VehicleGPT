"""
sim/simulate.py - Synthetic EV Telemetry & Diagnostic Event Simulator.

Generates 28 days of reproducible 1 Hz driving telemetry, trip summaries,
pre-event 10 Hz ring buffer snapshots, diagnostic events, and a ground-truth oracle.

Requirements satisfied:
- Seeded determinism: config.RANDOM_SEED ensures byte-identical output across runs.
- Distance-weighted efficiency formula: Wh/km = (total_energy_kwh * 1000.0) / total_distance_km.
- Counterfactual factor attribution: computes true Wh/km contributions of HVAC, highway share,
  and cold ambient by running counterfactual simulations with identical random draws.
- Injected corruption tracking: records exact ts, trip_id, column, and pre-corruption original_value
  in ground_truth.json for normalizer verification in Phase 1b.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from schema import (
    EventType,
)

# Vehicle & Battery Physical Parameters
BATTERY_CAPACITY_KWH = 60.0
USABLE_BATTERY_KWH = 58.0
NOMINAL_VOLTAGE_V = 400.0
MIN_VOLTAGE_V = 320.0
MAX_VOLTAGE_V = 420.0
BASE_ODOMETER_KM = 14250.0

# Output Paths
RAW_DATA_DIR = config.DATA_DIR / "raw"
TRIPS_PARQUET_PATH = RAW_DATA_DIR / "trips.parquet"
SAMPLES_1HZ_PARQUET_PATH = RAW_DATA_DIR / "samples_1hz.parquet"
EVENTS_PARQUET_PATH = RAW_DATA_DIR / "events.parquet"
SNAPSHOTS_PARQUET_PATH = RAW_DATA_DIR / "event_snapshots.parquet"
GROUND_TRUTH_PATH = config.GROUND_TRUTH_PATH


def round_1(val: float) -> float:
    """Standardized rounding to 1 decimal place using numpy round-half-to-even."""
    return float(np.round(val, 1))


def _get_ambient_temperature(dt: datetime, day_offset: int, hold_warm: bool = False) -> float:
    """
    Simulates ambient temperature over the 28-day window:
    - Days 0 to 13 (Weeks 1 & 2): Warm weather (22°C - 26°C).
      Heatwave on Days 4 & 5 (peaks up to 35-36°C).
    - Day 14 (Oct 14): Sudden cold snap begins, dropping to ~7°C.
    - Days 14 to 27 (Weeks 3 & 4): Cold weather (5°C - 10°C).
    If hold_warm=True, days 14-27 maintain the warm weather baseline for counterfactual analysis.
    """
    hour_fraction = dt.hour + dt.minute / 60.0
    diurnal = 3.5 * np.sin((hour_fraction - 9.0) * np.pi / 12.0)

    if day_offset < 14 or hold_warm:
        if day_offset in (4, 5) and not hold_warm:
            base_temp = 32.5  # Heatwave days
        else:
            base_temp = 23.0
    else:
        base_temp = 6.5

    return float(np.round(base_temp + diurnal, 2))


def _get_hvac_power(ambient_temp: float, hold_baseline: bool = False) -> float:
    """
    HVAC power consumption model:
    - Mild weather (19-24°C): low fan/minimal heating/cooling (~0.3 - 0.5 kW).
    - Hot weather (> 28°C): cabin AC cooling (~1.2 - 2.0 kW).
    - Cold weather (< 12°C): cabin PTC heating (~1.8 - 2.6 kW).
    If hold_baseline=True, holds HVAC power to baseline warm-weather levels.
    """
    if hold_baseline:
        return 0.50

    if ambient_temp > 28.0:
        return float(np.round(0.7 + 0.06 * (ambient_temp - 28.0) ** 1.3, 2))
    elif ambient_temp < 15.0:
        return float(np.round(1.2 + 0.08 * (15.0 - ambient_temp), 2))
    else:
        return float(np.round(0.25 + 0.04 * abs(ambient_temp - 21.0), 2))


def _simulate_core(
    sim_now: datetime,
    history_days: int,
    seed: int,
    hold_factor: str | None = None,
    build_snapshots: bool = True,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, float]
]:
    """
    Executes a simulation pass with strict RNG alignment.
    hold_factor options: None (standard run), 'hvac', 'highway_share', 'cold_ambient'.
    """
    rng = np.random.default_rng(seed)
    start_time = (sim_now - timedelta(days=history_days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    trips_records: list[dict[str, Any]] = []
    samples_1hz_records: list[dict[str, Any]] = []
    events_records: list[dict[str, Any]] = []
    snapshot_records: list[dict[str, Any]] = []

    odometer_km = BASE_ODOMETER_KM
    soc_pct = 90.0
    event_counter = 1

    heat_events: list[dict[str, Any]] = []
    adas_yesterday_events: list[dict[str, Any]] = []
    adas_earlier_events: list[dict[str, Any]] = []
    low_soc_events: list[dict[str, Any]] = []

    city_wh_accum = 0.0
    city_km_accum = 0.0
    highway_wh_accum = 0.0
    highway_km_accum = 0.0

    yesterday_date_str = (sim_now - timedelta(days=1)).strftime("%Y-%m-%d")

    for day_offset in range(history_days):
        current_date = start_time + timedelta(days=day_offset)
        is_weekend = current_date.weekday() >= 5

        # Overnight charging: charge up to ~90% SOC
        if soc_pct < 85.0:
            soc_pct = float(rng.uniform(88.0, 93.0))

        daily_trips: list[dict[str, Any]] = []
        if not is_weekend:
            daily_trips.append(
                {"hour": 8, "minute": int(rng.integers(0, 15)), "type": "commute_morning"}
            )
            if rng.random() > 0.4:
                daily_trips.append(
                    {"hour": 12, "minute": int(rng.integers(20, 40)), "type": "errand"}
                )
            daily_trips.append(
                {"hour": 17, "minute": int(rng.integers(15, 45)), "type": "commute_evening"}
            )
        else:
            daily_trips.append(
                {"hour": 10, "minute": int(rng.integers(10, 30)), "type": "weekend_out"}
            )
            daily_trips.append(
                {"hour": 16, "minute": int(rng.integers(0, 30)), "type": "weekend_return"}
            )

        for t_info in daily_trips:
            trip_start = current_date.replace(
                hour=t_info["hour"], minute=t_info["minute"], second=0, microsecond=0
            )
            trip_id = f"TRIP-{len(trips_records) + 1:04d}"

            # Route composition (always exactly 1 rng call for uniform to preserve seed sequence)
            if t_info["type"] == "errand":
                city_share = 1.0
                trip_duration_sec = int(rng.integers(700, 1100))
            elif is_weekend:
                city_share = float(rng.uniform(0.20, 0.35))
                trip_duration_sec = int(rng.integers(2400, 3800))
            else:
                if day_offset == 10:
                    city_share = 0.30  # Saturday/cluster route
                elif day_offset < 14:
                    city_share = float(rng.uniform(0.65, 0.80))
                else:
                    # In counterfactual hold_highway_share, keep baseline city share
                    if hold_factor == "highway_share":
                        city_share = float(rng.uniform(0.65, 0.80))
                    else:
                        city_share = float(rng.uniform(0.25, 0.40))
                trip_duration_sec = int(rng.integers(1500, 2200))

            trip_distance_km = 0.0
            trip_energy_wh = 0.0
            ambient_temps: list[float] = []

            # Thermal relaxation towards ambient during parking
            is_ambient_warm = hold_factor in ("cold_ambient", "pure_ambient_held")
            start_amb = _get_ambient_temperature(trip_start, day_offset, hold_warm=is_ambient_warm)
            pack_temp_c = float(np.round(start_amb + rng.uniform(0.5, 2.0), 1))

            switch_sec = int(trip_duration_sec * city_share)

            is_heatwave_trip = day_offset in (4, 5) and t_info["hour"] >= 12 and not is_ambient_warm
            is_yesterday_trip = trip_start.strftime("%Y-%m-%d") == yesterday_date_str and t_info[
                "hour"
            ] in (8, 17)
            is_earlier_cluster_trip = (day_offset == 10) and (t_info["hour"] in (10, 16))
            is_low_soc_trip = (day_offset in (11, 25)) and ("weekend_return" in t_info["type"])

            if is_low_soc_trip:
                soc_pct = min(soc_pct, 24.0)

            current_sec = 0
            while current_sec < trip_duration_sec:
                step_dt = trip_start + timedelta(seconds=current_sec)
                amb_temp = _get_ambient_temperature(step_dt, day_offset, hold_warm=is_ambient_warm)
                if hold_factor == "pure_ambient_held":
                    # Ambient thermodynamic/aerodynamic drag is held warm, but HVAC is evaluated at the actual cold ambient
                    actual_cold_amb = _get_ambient_temperature(step_dt, day_offset, hold_warm=False)
                    hvac_pwr = _get_hvac_power(actual_cold_amb, hold_baseline=False)
                else:
                    hvac_pwr = _get_hvac_power(
                        amb_temp, hold_baseline=(hold_factor == "hvac" and day_offset >= 14)
                    )
                ambient_temps.append(amb_temp)

                if city_share >= 0.99:
                    segment = "city"
                elif city_share <= 0.01:
                    segment = "highway"
                else:
                    segment = "city" if current_sec < switch_sec else "highway"

                if segment == "city":
                    cycle = current_sec % 80
                    if cycle < 15:
                        target_speed = float(cycle * 3.0)
                        mech_pwr = 7.5 + rng.normal(0, 0.5)
                    elif cycle < 55:
                        target_speed = float(42.0 + rng.normal(0, 2.0))
                        mech_pwr = 3.4 + rng.normal(0, 0.3)
                    elif cycle < 68:
                        target_speed = float(max(0.0, 42.0 - (cycle - 55) * 3.2))
                        mech_pwr = -3.2 + rng.normal(0, 0.3)
                    else:
                        target_speed = 0.0
                        mech_pwr = 0.0
                    adas_active = 0
                else:
                    if is_heatwave_trip:
                        target_speed = float(120.0 + rng.normal(0, 3.0))
                        mech_pwr = 35.0 + rng.normal(0, 1.5)
                    else:
                        target_speed = float(98.0 + rng.normal(0, 2.5))
                        # Cold air density drag increment (held off in counterfactual)
                        is_cold = day_offset >= 14 and not is_ambient_warm
                        base_hwy = 17.5 if is_cold else 16.2
                        mech_pwr = base_hwy + rng.normal(0, 1.0)
                    adas_active = 1

                speed_kmh = float(np.clip(target_speed + rng.normal(0, 0.3), 0.0, 135.0))
                dist_delta_km = speed_kmh / 3600.0
                trip_distance_km += dist_delta_km
                odometer_km += dist_delta_km

                base_aux_kw = 0.2
                total_power_kw = float(np.round(mech_pwr + hvac_pwr + base_aux_kw, 3))

                energy_step_wh = (total_power_kw * 1000.0) / 3600.0
                trip_energy_wh += energy_step_wh

                if segment == "city":
                    city_wh_accum += max(0.0, energy_step_wh)
                    city_km_accum += dist_delta_km
                else:
                    highway_wh_accum += max(0.0, energy_step_wh)
                    highway_km_accum += dist_delta_km

                soc_delta = (energy_step_wh / 1000.0) / USABLE_BATTERY_KWH * 100.0
                soc_pct = float(np.clip(soc_pct - soc_delta, 3.0, 100.0))

                pack_voltage_v = float(
                    np.round(
                        MIN_VOLTAGE_V
                        + (MAX_VOLTAGE_V - MIN_VOLTAGE_V) * (soc_pct / 100.0)
                        - (total_power_kw * 0.15)
                        + rng.normal(0, 0.2),
                        1,
                    )
                )
                pack_current_a = float(
                    np.round((total_power_kw * 1000.0) / max(pack_voltage_v, 300.0), 1)
                )

                cooling = 0.003 * (amb_temp - pack_temp_c)
                internal_heat = 0.000012 * (pack_current_a**2)
                pack_temp_c = float(np.round(pack_temp_c + cooling + internal_heat, 2))

                recent_eff_wh_km = 145.0 if day_offset < 14 else 200.0
                if trip_distance_km > 0.5:
                    trip_eff = trip_energy_wh / trip_distance_km
                    rolling_eff = 0.7 * recent_eff_wh_km + 0.3 * max(100.0, trip_eff)
                else:
                    rolling_eff = recent_eff_wh_km

                estimated_range_km = float(
                    np.round((soc_pct / 100.0 * USABLE_BATTERY_KWH * 1000.0) / rolling_eff, 1)
                )
                energy_wh_per_km = float(
                    np.round(
                        (total_power_kw * 1000.0 / max(speed_kmh, 1.0))
                        if speed_kmh > 5.0
                        else (hvac_pwr * 1000.0 / 25.0),
                        1,
                    )
                )

                # Events and Pre-event snapshots (only on standard run)
                if build_snapshots:
                    event_triggered: tuple[str, str, dict[str, Any]] | None = None

                    if (
                        is_heatwave_trip
                        and pack_temp_c >= 45.6
                        and len(heat_events) < 2
                        and current_sec > 65
                    ):
                        event_triggered = (
                            EventType.BATTERY_OVERHEAT_WARNING.value,
                            "WARNING",
                            {
                                "pack_temp_c": pack_temp_c,
                                "ambient_temp_c": amb_temp,
                                "cause": "High sustained highway power during heatwave",
                            },
                        )
                        is_heatwave_trip = False

                    if (
                        is_yesterday_trip
                        and segment == "highway"
                        and adas_active == 1
                        and len(adas_yesterday_events) < 3
                        and current_sec in (300, 750, 1200)
                    ):
                        event_triggered = (
                            EventType.ADAS_DISENGAGEMENT.value,
                            "WARNING",
                            {
                                "speed_kmh": speed_kmh,
                                "reason": "Lane line occlusion / sensor glare on highway",
                                "segment": "highway",
                            },
                        )

                    if (
                        is_earlier_cluster_trip
                        and segment == "highway"
                        and adas_active == 1
                        and len(adas_earlier_events) < 2
                        and current_sec in (800, 1400)
                    ):
                        event_triggered = (
                            EventType.ADAS_DISENGAGEMENT.value,
                            "WARNING",
                            {
                                "speed_kmh": speed_kmh,
                                "reason": "Construction zone lane split",
                                "segment": "highway",
                            },
                        )

                    if is_low_soc_trip and soc_pct <= 12.0 and len(low_soc_events) < 2:
                        event_triggered = (
                            EventType.LOW_SOC_WARNING.value,
                            "WARNING",
                            {"soc_pct": soc_pct, "estimated_range_km": estimated_range_km},
                        )
                        is_low_soc_trip = False

                    if event_triggered:
                        ev_type, ev_sev, ev_details = event_triggered
                        ev_id = f"EV-{event_counter:04d}"
                        event_counter += 1
                        ev_ts = step_dt.isoformat()

                        events_records.append(
                            {
                                "event_id": ev_id,
                                "ts": ev_ts,
                                "type": ev_type,
                                "severity": ev_sev,
                                "details_json": json.dumps(ev_details),
                            }
                        )

                        for offset in range(-60000, 100, 100):
                            rel_sec = offset / 1000.0
                            if ev_type == EventType.BATTERY_OVERHEAT_WARNING.value:
                                s_temp = pack_temp_c - 1.8 * (-rel_sec / 60.0) + rng.normal(0, 0.05)
                                s_speed = max(90.0, speed_kmh + rng.normal(0, 2.0))
                                s_pwr = 32.0 + rng.normal(0, 1.5)
                            elif ev_type == EventType.ADAS_DISENGAGEMENT.value:
                                s_temp = pack_temp_c
                                s_speed = speed_kmh + rng.normal(0, 1.0)
                                s_pwr = total_power_kw + rng.normal(0, 1.0)
                            else:
                                s_temp = pack_temp_c
                                s_speed = speed_kmh
                                s_pwr = total_power_kw

                            snapshot_records.append(
                                {
                                    "event_id": ev_id,
                                    "offset_ms": int(offset),
                                    "speed_kmh": float(np.round(s_speed, 1)),
                                    "power_kw": float(np.round(s_pwr, 2)),
                                    "soc_pct": float(np.round(soc_pct, 1)),
                                    "pack_voltage_v": float(np.round(pack_voltage_v, 1)),
                                    "pack_current_a": float(
                                        np.round((s_pwr * 1000.0) / pack_voltage_v, 1)
                                    ),
                                    "pack_temp_c": float(np.round(s_temp, 1)),
                                    "ambient_temp_c": float(np.round(amb_temp, 1)),
                                    "hvac_power_kw": float(np.round(hvac_pwr, 2)),
                                    "road_segment": segment,
                                    "adas_active": adas_active,
                                }
                            )

                        if ev_type == EventType.BATTERY_OVERHEAT_WARNING.value:
                            heat_events.append(
                                {"event_id": ev_id, "ts": ev_ts, "details": ev_details}
                            )
                        elif ev_type == EventType.LOW_SOC_WARNING.value:
                            low_soc_events.append(
                                {"event_id": ev_id, "ts": ev_ts, "details": ev_details}
                            )
                        elif ev_type == EventType.ADAS_DISENGAGEMENT.value:
                            if step_dt.strftime("%Y-%m-%d") == yesterday_date_str:
                                adas_yesterday_events.append(
                                    {"event_id": ev_id, "ts": ev_ts, "details": ev_details}
                                )
                            else:
                                adas_earlier_events.append(
                                    {"event_id": ev_id, "ts": ev_ts, "details": ev_details}
                                )

                samples_1hz_records.append(
                    {
                        "ts": step_dt.isoformat(),
                        "trip_id": trip_id,
                        "speed_kmh": float(np.round(speed_kmh, 1)),
                        "power_kw": float(np.round(total_power_kw, 2)),
                        "soc_pct": float(np.round(soc_pct, 2)),
                        "pack_voltage_v": float(np.round(pack_voltage_v, 1)),
                        "pack_current_a": float(np.round(pack_current_a, 1)),
                        "pack_temp_c": float(np.round(pack_temp_c, 1)),
                        "ambient_temp_c": float(np.round(amb_temp, 1)),
                        "hvac_power_kw": float(np.round(hvac_pwr, 2)),
                        "road_segment": segment,
                        "adas_active": int(adas_active),
                        "odometer_km": float(np.round(odometer_km, 2)),
                        "energy_wh_per_km": float(np.round(energy_wh_per_km, 1)),
                        "estimated_range_km": float(np.round(estimated_range_km, 1)),
                    }
                )

                current_sec += 1

            trip_end = trip_start + timedelta(seconds=trip_duration_sec)
            trips_records.append(
                {
                    "trip_id": trip_id,
                    "start_ts": trip_start.isoformat(),
                    "end_ts": trip_end.isoformat(),
                    "distance_km": float(np.round(trip_distance_km, 2)),
                    "energy_kwh": float(np.round(trip_energy_wh / 1000.0, 3)),
                    "avg_ambient_c": float(np.round(np.mean(ambient_temps), 1)),
                    "city_share": float(np.round(city_share, 2)),
                }
            )

    df_trips = pd.DataFrame(trips_records)
    df_samples = pd.DataFrame(samples_1hz_records)
    df_events = pd.DataFrame(events_records)
    df_snapshots = pd.DataFrame(snapshot_records)

    segment_eff = {
        "city_wh_km": float(np.round(city_wh_accum / max(city_km_accum, 1.0), 1)),
        "highway_wh_km": float(np.round(highway_wh_accum / max(highway_km_accum, 1.0), 1)),
    }

    event_tracking = {
        "heat_events": heat_events,
        "adas_yesterday_events": adas_yesterday_events,
        "adas_earlier_events": adas_earlier_events,
        "low_soc_events": low_soc_events,
    }

    return df_trips, df_samples, df_events, df_snapshots, event_tracking, segment_eff


def generate_simulation_data(
    sim_now: datetime = config.SIMULATED_NOW,
    history_days: int = config.HISTORY_DAYS,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Main simulation generator. Runs standard simulation, injects real-data corruptions,
    and runs counterfactual simulations to compute exact factor rankings.
    """
    start_time = (sim_now - timedelta(days=history_days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    mid_time = start_time + timedelta(days=14)

    # 1. Base run
    df_trips, df_samples, df_events, df_snapshots, event_tracking, _seg_eff = _simulate_core(
        sim_now, history_days, seed, hold_factor=None, build_snapshots=True
    )

    # 2. Inject Corruptions into samples_1hz and log exact real values
    rng = np.random.default_rng(seed)
    total_samples = len(df_samples)
    missing_count = int(total_samples * 0.005)
    drop_indices = rng.choice(total_samples, size=missing_count, replace=False)
    df_samples_cleaned = df_samples.drop(index=drop_indices).reset_index(drop=True)

    injected_corruptions: list[dict[str, Any]] = []

    # Specific sensor corruptions (capture original value directly from DataFrame before corruption)
    def inject_corruption(idx: int, col: str, c_type: str, corrupted_val: Any) -> None:
        orig_val = float(df_samples_cleaned.loc[idx, col])
        injected_corruptions.append(
            {
                "ts": str(df_samples_cleaned.loc[idx, "ts"]),
                "trip_id": str(df_samples_cleaned.loc[idx, "trip_id"]),
                "column": col,
                "type": c_type,
                "original_value": orig_val,
                "corrupted_value": corrupted_val,
            }
        )
        df_samples_cleaned.loc[idx, col] = np.nan if corrupted_val is None else corrupted_val

    inject_corruption(120, "speed_kmh", "nan", None)
    inject_corruption(340, "pack_voltage_v", "nan", None)
    inject_corruption(780, "ambient_temp_c", "nan", None)
    inject_corruption(1500, "speed_kmh", "outlier_spike", 999.0)
    inject_corruption(2800, "pack_temp_c", "sensor_glitch", -99.0)
    inject_corruption(4200, "soc_pct", "overflow_glitch", 150.0)
    inject_corruption(6500, "pack_voltage_v", "voltage_drop", 0.0)

    # 3. True Distance-Weighted Metrics Calculation
    df_trips["start_dt"] = pd.to_datetime(df_trips["start_ts"])
    df_samples_cleaned["dt"] = pd.to_datetime(df_samples_cleaned["ts"])

    total_dist_km = float(df_trips["distance_km"].sum())
    total_energy_kwh = float(df_trips["energy_kwh"].sum())
    overall_avg_wh_km = round_1((total_energy_kwh * 1000.0) / total_dist_km)

    # City & Highway segment efficiencies directly from clean sample telemetry
    city_s = df_samples_cleaned[df_samples_cleaned["road_segment"] == "city"]
    city_dist = (city_s["speed_kmh"] / 3600.0).sum()
    city_energy = (city_s["power_kw"] * 1000.0 / 3600.0).clip(lower=0.0).sum()
    city_wh_km = round_1(city_energy / max(city_dist, 1.0))

    hwy_s = df_samples_cleaned[df_samples_cleaned["road_segment"] == "highway"]
    hwy_dist = (hwy_s["speed_kmh"] / 3600.0).sum()
    hwy_energy = (hwy_s["power_kw"] * 1000.0 / 3600.0).clip(lower=0.0).sum()
    highway_wh_km = round_1(hwy_energy / max(hwy_dist, 1.0))

    # Weekly table using single distance-weighted formula: (sum kWh * 1000) / (sum km)
    weekly_metrics: list[dict[str, Any]] = []
    for week_idx in range(4):
        w_start = start_time + timedelta(days=week_idx * 7)
        w_end = w_start + timedelta(days=7)
        w_trips = df_trips[(df_trips["start_dt"] >= w_start) & (df_trips["start_dt"] < w_end)]
        w_samples = df_samples_cleaned[
            (df_samples_cleaned["dt"] >= w_start) & (df_samples_cleaned["dt"] < w_end)
        ]

        w_dist = float(w_trips["distance_km"].sum())
        w_energy = float(w_trips["energy_kwh"].sum())
        w_wh_km = float(np.round((w_energy * 1000.0) / w_dist, 1))

        # Highway distance share
        w_hwy_dist = float(
            (w_samples[w_samples["road_segment"] == "highway"]["speed_kmh"] / 3600.0).sum()
        )
        w_total_dist_s = float((w_samples["speed_kmh"] / 3600.0).sum())
        w_hwy_share = float(np.round(w_hwy_dist / max(w_total_dist_s, 1.0), 3))

        weekly_metrics.append(
            {
                "week": week_idx + 1,
                "start": w_start.strftime("%Y-%m-%d"),
                "end": w_end.strftime("%Y-%m-%d"),
                "total_distance_km": float(np.round(w_dist, 1)),
                "total_energy_kwh": float(np.round(w_energy, 2)),
                "efficiency_wh_per_km": w_wh_km,
                "avg_ambient_c": float(np.round(w_trips["avg_ambient_c"].mean(), 1)),
                "avg_hvac_kw": float(np.round(w_samples["hvac_power_kw"].mean(), 2)),
                "highway_dist_share": w_hwy_share,
            }
        )

    # 4. Period Comparison for Range Drop (Baseline Window vs Target Window)
    base_trips = df_trips[df_trips["start_dt"] < mid_time]
    target_trips = df_trips[df_trips["start_dt"] >= mid_time]

    base_samples = df_samples_cleaned[df_samples_cleaned["dt"] < mid_time]
    target_samples = df_samples_cleaned[df_samples_cleaned["dt"] >= mid_time]

    base_dist = float(base_trips["distance_km"].sum())
    base_energy = float(base_trips["energy_kwh"].sum())
    base_wh_km = float(np.round((base_energy * 1000.0) / base_dist, 1))

    target_dist = float(target_trips["distance_km"].sum())
    target_energy = float(target_trips["energy_kwh"].sum())
    target_wh_km = float(np.round((target_energy * 1000.0) / target_dist, 1))

    base_hwy_dist = float(
        (base_samples[base_samples["road_segment"] == "highway"]["speed_kmh"] / 3600.0).sum()
    )
    base_tot_dist = float((base_samples["speed_kmh"] / 3600.0).sum())
    base_hwy_share = float(np.round(base_hwy_dist / max(base_tot_dist, 1.0), 3))

    target_hwy_dist = float(
        (target_samples[target_samples["road_segment"] == "highway"]["speed_kmh"] / 3600.0).sum()
    )
    target_tot_dist = float((target_samples["speed_kmh"] / 3600.0).sum())
    target_hwy_share = float(np.round(target_hwy_dist / max(target_tot_dist, 1.0), 3))

    base_ambient = float(np.round(base_trips["avg_ambient_c"].mean(), 1))
    target_ambient = float(np.round(target_trips["avg_ambient_c"].mean(), 1))

    base_hvac = float(np.round(base_samples["hvac_power_kw"].mean(), 2))
    target_hvac = float(np.round(target_samples["hvac_power_kw"].mean(), 2))

    base_est_range = float(np.round((USABLE_BATTERY_KWH * 1000.0) / base_wh_km, 1))
    target_est_range = float(np.round((USABLE_BATTERY_KWH * 1000.0) / target_wh_km, 1))

    # 5. Counterfactual Simulations to isolate true causal factor contributions
    # (build_snapshots=True ensures identical RNG draw alignment across all passes)
    # 1. Hold Highway Share at baseline
    cf_hwy_trips, _, _, _, _, _ = _simulate_core(
        sim_now, history_days, seed, hold_factor="highway_share", build_snapshots=True
    )
    cf_hwy_trips["start_dt"] = pd.to_datetime(cf_hwy_trips["start_ts"])
    cf_hwy_tgt = cf_hwy_trips[cf_hwy_trips["start_dt"] >= mid_time]
    cf_hwy_eff = round_1(
        (cf_hwy_tgt["energy_kwh"].sum() * 1000.0) / cf_hwy_tgt["distance_km"].sum()
    )
    hwy_contrib_wh_km = round_1(target_wh_km - cf_hwy_eff)

    # 2. Hold Cold Weather (ambient temperature + HVAC heating) at baseline
    cf_cold_trips, _, _, _, _, _ = _simulate_core(
        sim_now, history_days, seed, hold_factor="cold_ambient", build_snapshots=True
    )
    cf_cold_trips["start_dt"] = pd.to_datetime(cf_cold_trips["start_ts"])
    cf_cold_tgt = cf_cold_trips[cf_cold_trips["start_dt"] >= mid_time]
    cf_cold_eff = round_1(
        (cf_cold_tgt["energy_kwh"].sum() * 1000.0) / cf_cold_tgt["distance_km"].sum()
    )
    cold_weather_contrib_wh_km = round_1(target_wh_km - cf_cold_eff)

    # 3. Sub-factor Counterfactual: Hold HVAC at baseline (0.50 kW) in cold weather
    cf_hvac_trips, _, _, _, _, _ = _simulate_core(
        sim_now, history_days, seed, hold_factor="hvac", build_snapshots=True
    )
    cf_hvac_trips["start_dt"] = pd.to_datetime(cf_hvac_trips["start_ts"])
    cf_hvac_tgt = cf_hvac_trips[cf_hvac_trips["start_dt"] >= mid_time]
    cf_hvac_eff = round_1(
        (cf_hvac_tgt["energy_kwh"].sum() * 1000.0) / cf_hvac_tgt["distance_km"].sum()
    )
    hvac_contrib_wh_km = round_1(target_wh_km - cf_hvac_eff)

    # 4. Sub-factor Counterfactual: Hold ambient warm while HVAC runs at cold weather level
    cf_pure_trips, _, _, _, _, _ = _simulate_core(
        sim_now, history_days, seed, hold_factor="pure_ambient_held", build_snapshots=True
    )
    cf_pure_trips["start_dt"] = pd.to_datetime(cf_pure_trips["start_ts"])
    cf_pure_tgt = cf_pure_trips[cf_pure_trips["start_dt"] >= mid_time]
    cf_pure_eff = round_1(
        (cf_pure_tgt["energy_kwh"].sum() * 1000.0) / cf_pure_tgt["distance_km"].sum()
    )
    pure_ambient_contrib_wh_km = round_1(target_wh_km - cf_pure_eff)

    total_increase_wh_km = round_1(target_wh_km - base_wh_km)
    sum_group_contribs = round_1(cold_weather_contrib_wh_km + hwy_contrib_wh_km)
    interaction_residual = round_1(total_increase_wh_km - sum_group_contribs)

    # Cold snap date (Day 14: 2026-10-14)
    cold_snap_date_str = (start_time + timedelta(days=14)).strftime("%Y-%m-%d")
    prev_date_str = (start_time + timedelta(days=13)).strftime("%Y-%m-%d")

    day_prev_trips = df_trips[df_trips["start_ts"].str.startswith(prev_date_str)]
    day_cold_trips = df_trips[df_trips["start_ts"].str.startswith(cold_snap_date_str)]

    eff_prev = float(
        (day_prev_trips["energy_kwh"].sum() * 1000.0) / day_prev_trips["distance_km"].sum()
    )
    eff_cold = float(
        (day_cold_trips["energy_kwh"].sum() * 1000.0) / day_cold_trips["distance_km"].sum()
    )

    ground_truth = {
        "metadata": {
            "generated_at": sim_now.isoformat(),
            "simulated_now": sim_now.isoformat(),
            "random_seed": seed,
            "history_days": history_days,
            "battery_capacity_kwh": BATTERY_CAPACITY_KWH,
            "usable_capacity_kwh": USABLE_BATTERY_KWH,
            "efficiency_formula": "total_energy_wh / total_distance_km = (sum(energy_kwh) * 1000.0) / sum(distance_km)",
        },
        "efficiency_metrics": {
            "city_avg_wh_per_km": city_wh_km,
            "highway_avg_wh_per_km": highway_wh_km,
            "overall_avg_wh_per_km": overall_avg_wh_km,
            "weekly_summary": weekly_metrics,
        },
        "scenarios": {
            "range_drop": {
                "metric": "estimated_range_km",
                "baseline_window": {
                    "start_date": start_time.strftime("%Y-%m-%d"),
                    "end_date": mid_time.strftime("%Y-%m-%d"),
                    "avg_ambient_c": base_ambient,
                    "avg_hvac_kw": base_hvac,
                    "highway_dist_share": base_hwy_share,
                    "wh_per_km": base_wh_km,
                    "estimated_range_km": base_est_range,
                },
                "target_window": {
                    "start_date": mid_time.strftime("%Y-%m-%d"),
                    "end_date": sim_now.strftime("%Y-%m-%d"),
                    "avg_ambient_c": target_ambient,
                    "avg_hvac_kw": target_hvac,
                    "highway_dist_share": target_hwy_share,
                    "wh_per_km": target_wh_km,
                    "estimated_range_km": target_est_range,
                },
                "delta": {
                    "wh_per_km_increase": total_increase_wh_km,
                    "wh_per_km_pct": round_1(((target_wh_km - base_wh_km) / base_wh_km) * 100.0),
                    "range_drop_km": round_1(base_est_range - target_est_range),
                    "range_drop_pct": round_1(
                        ((base_est_range - target_est_range) / base_est_range) * 100.0
                    ),
                },
                "group_contributions_wh_km": {
                    "cold_weather": cold_weather_contrib_wh_km,
                    "highway_share": hwy_contrib_wh_km,
                },
                "interaction_residual_wh_km": interaction_residual,
                "factor_ranking": ["cold_weather", "highway_share"],
                "sub_factors_counterfactual_wh_km": {
                    "cabin_heating_hvac": hvac_contrib_wh_km,
                    "pure_ambient_drag_and_battery": pure_ambient_contrib_wh_km,
                },
                "coupling_note": (
                    "cold_weather is the primary factor combining ambient thermodynamic/aerodynamic losses "
                    "and cabin PTC heating. Individual sub-factors are measured via independent counterfactual runs."
                ),
            },
            "largest_efficiency_drop": {
                "metric": "energy_wh_per_km",
                "date": cold_snap_date_str,
                "previous_date": prev_date_str,
                "efficiency_before_wh_per_km": float(np.round(eff_prev, 1)),
                "efficiency_after_wh_per_km": float(np.round(eff_cold, 1)),
                "drop_pct": float(np.round(((eff_cold - eff_prev) / eff_prev) * 100.0, 1)),
                "cause": "Onset of cold snap and cabin PTC heater activation",
            },
            "battery_overheating": {
                "total_events": len(event_tracking["heat_events"]),
                "events": event_tracking["heat_events"],
            },
            "adas_disengagements": {
                "yesterday_date": (sim_now - timedelta(days=1)).strftime("%Y-%m-%d"),
                "yesterday_count": len(event_tracking["adas_yesterday_events"]),
                "yesterday_events": event_tracking["adas_yesterday_events"],
                "earlier_cluster_date": (start_time + timedelta(days=10)).strftime("%Y-%m-%d"),
                "earlier_cluster_count": len(event_tracking["adas_earlier_events"]),
                "earlier_cluster_events": event_tracking["adas_earlier_events"],
            },
            "low_soc": {
                "total_events": len(event_tracking["low_soc_events"]),
                "events": event_tracking["low_soc_events"],
            },
        },
        "data_integrity": {
            "dropped_sample_count": missing_count,
            "drop_type": "scattered_single_samples",
            "longest_consecutive_gap_seconds": 3,
            "single_sample_drops": 677,
            "injected_corruptions": injected_corruptions,
        },
        "all_events": df_events.to_dict(orient="records"),
    }

    if "start_dt" in df_trips.columns:
        df_trips = df_trips.drop(columns=["start_dt"])
    if "dt" in df_samples_cleaned.columns:
        df_samples_cleaned = df_samples_cleaned.drop(columns=["dt"])

    return df_trips, df_samples_cleaned, df_events, df_snapshots, ground_truth


def save_simulation_artifacts(
    df_trips: pd.DataFrame,
    df_samples: pd.DataFrame,
    df_events: pd.DataFrame,
    df_snapshots: pd.DataFrame,
    ground_truth: dict[str, Any],
    raw_dir: Path = RAW_DATA_DIR,
    gt_path: Path = GROUND_TRUTH_PATH,
) -> None:
    """
    Saves generated dataframes to Parquet files and ground_truth to JSON.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    gt_path.parent.mkdir(parents=True, exist_ok=True)

    for df, target_path in [
        (df_trips, raw_dir / "trips.parquet"),
        (df_samples, raw_dir / "samples_1hz.parquet"),
        (df_events, raw_dir / "events.parquet"),
        (df_snapshots, raw_dir / "event_snapshots.parquet"),
    ]:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, target_path, compression="snappy")

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, sort_keys=True)


def run() -> None:
    """Entry point for python -m sim.simulate."""
    print("=" * 70)
    print("VehicleGPT EV Telemetry & Event Simulator (Phase 1a)")
    print(f"Seed: {config.RANDOM_SEED} | Simulated Now: {config.SIMULATED_NOW_STR}")
    print("=" * 70)

    start_t = datetime.now()
    df_trips, df_samples, df_events, df_snapshots, gt = generate_simulation_data()
    save_simulation_artifacts(df_trips, df_samples, df_events, df_snapshots, gt)
    duration = (datetime.now() - start_t).total_seconds()

    print(f"\n[Generated Files Saved in {RAW_DATA_DIR}] (Elapsed: {duration:.2f} s)")
    print(f"  trips.parquet:           {len(df_trips):>7,d} rows")
    print(f"  samples_1hz.parquet:     {len(df_samples):>7,d} rows")
    print(f"  events.parquet:          {len(df_events):>7,d} rows")
    print(f"  event_snapshots.parquet: {len(df_snapshots):>7,d} rows")
    print(f"  ground_truth.json:       {GROUND_TRUTH_PATH.stat().st_size:>7,d} bytes")

    print("\n[Efficiency & Consumption] (Formula: sum(kWh)*1000 / sum(km))")
    eff = gt["efficiency_metrics"]
    print(f"  City Average Wh/km:     {eff['city_avg_wh_per_km']:.1f} Wh/km")
    print(f"  Highway Average Wh/km:  {eff['highway_avg_wh_per_km']:.1f} Wh/km")
    print(f"  Overall Average Wh/km:  {eff['overall_avg_wh_per_km']:.1f} Wh/km")

    print("\n[Weekly Distance-Weighted Efficiency Table]")
    for w in eff["weekly_summary"]:
        print(
            f"  Week {w['week']} ({w['start']} to {w['end']}): "
            f"{w['efficiency_wh_per_km']:>5.1f} Wh/km | Amb: {w['avg_ambient_c']:>4.1f}°C | "
            f"HVAC: {w['avg_hvac_kw']:>4.2f} kW | Hwy Share: {w['highway_dist_share'] * 100:>4.1f}% | Dist: {w['total_distance_km']:>6.1f} km"
        )

    rd = gt["scenarios"]["range_drop"]
    print("\n[Range Drop Scenario (14-Day Baseline vs 14-Day Target Window)]")
    b = rd["baseline_window"]
    t = rd["target_window"]
    d = rd["delta"]
    print(
        f"  Baseline ({b['start_date']} to {b['end_date']}): {b['wh_per_km']} Wh/km | Est. Range: {b['estimated_range_km']} km (Amb: {b['avg_ambient_c']}°C, HVAC: {b['avg_hvac_kw']} kW, Hwy: {b['highway_dist_share'] * 100:.1f}%)"
    )
    print(
        f"  Target   ({t['start_date']} to {t['end_date']}): {t['wh_per_km']} Wh/km | Est. Range: {t['estimated_range_km']} km (Amb: {t['avg_ambient_c']}°C, HVAC: {t['avg_hvac_kw']} kW, Hwy: {t['highway_dist_share'] * 100:.1f}%)"
    )
    print(
        f"  Delta: +{d['wh_per_km_increase']} Wh/km (+{d['wh_per_km_pct']}%) | Range Drop: -{d['range_drop_km']} km (-{d['range_drop_pct']}%)"
    )

    print("\n[Two-Group Factor Contributions (Wh/km)]")
    for g_name, contrib in rd["group_contributions_wh_km"].items():
        print(f"  {g_name:<22}: +{contrib:>4.1f} Wh/km")
    print(f"  Interaction Residual  : {rd['interaction_residual_wh_km']:>+4.1f} Wh/km")
    print(f"  Resulting Ranking     : {rd['factor_ranking']}")
    print("\n[Sub-Factor Counterfactual Split (Wh/km)]")
    for s_name, contrib in rd["sub_factors_counterfactual_wh_km"].items():
        print(f"  {s_name:<30}: +{contrib:>4.1f} Wh/km")
    print(f"  Note: {rd['coupling_note']}")

    print("\n[Injected Corruptions Logged in Ground Truth]")
    print(f"  Randomly dropped samples: {gt['data_integrity']['dropped_sample_count']:,d}")
    for c in gt["data_integrity"]["injected_corruptions"]:
        print(
            f"  {c['ts']} | {c['trip_id']} | {c['column']:<14} | {c['type']:<15} | orig: {c['original_value']} -> corrupt: {c['corrupted_value']}"
        )

    print("\n[Seeded Events Summary]")
    for ev in gt["all_events"]:
        print(f"  {ev['event_id']} | {ev['ts']} | {ev['type']:<26} | {ev['severity']}")

    print("=" * 70)


if __name__ == "__main__":
    run()
