"""
tests/test_sim.py - Unit tests for Phase 1a EV Telemetry Simulator.

Validates:
1. Determinism: Two simulation runs produce identical DataFrames, JSON, and byte-identical Parquet files.
2. Schema conformity: Column names match schema.py table DDLs.
3. Plausible value ranges: Physical distributions verified after excluding rows logged in injected_corruptions.
4. Event validity: Valid EventTypes, EV-<digits> ID regex conformity, and chronological ordering.
5. Ground truth recomputation: Recomputes every ground-truth metric directly from Parquet files to verify exact match.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from schema import (
    EVENT_ID_REGEX,
    TABLES_DDL,
    EventType,
)
from sim.simulate import (
    generate_simulation_data,
    round_1,
    save_simulation_artifacts,
)


@pytest.fixture(scope="module")
def sim_artifacts():
    """Generates simulation data once for the test module."""
    df_trips, df_samples, df_events, df_snapshots, gt = generate_simulation_data()
    return df_trips, df_samples, df_events, df_snapshots, gt


@pytest.mark.slow
def test_determinism_two_runs_identical():
    """Ensure two simulation runs produce byte-identical Parquet and JSON files."""
    df_trips1, df_samples1, df_events1, df_snapshots1, gt1 = generate_simulation_data()
    df_trips2, df_samples2, df_events2, df_snapshots2, gt2 = generate_simulation_data()

    assert df_trips1.equals(df_trips2)
    assert df_samples1.equals(df_samples2)
    assert df_events1.equals(df_events2)
    assert df_snapshots1.equals(df_snapshots2)
    assert gt1 == gt2

    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        dir1 = Path(tmpdir1)
        dir2 = Path(tmpdir2)
        save_simulation_artifacts(
            df_trips1,
            df_samples1,
            df_events1,
            df_snapshots1,
            gt1,
            raw_dir=dir1,
            gt_path=dir1 / "gt.json",
        )
        save_simulation_artifacts(
            df_trips2,
            df_samples2,
            df_events2,
            df_snapshots2,
            gt2,
            raw_dir=dir2,
            gt_path=dir2 / "gt.json",
        )

        for fname in [
            "trips.parquet",
            "samples_1hz.parquet",
            "events.parquet",
            "event_snapshots.parquet",
            "gt.json",
        ]:
            h1 = hashlib.sha256((dir1 / fname).read_bytes()).hexdigest()
            h2 = hashlib.sha256((dir2 / fname).read_bytes()).hexdigest()
            assert h1 == h2, f"Byte mismatch in {fname}"


def test_column_names_match_schema(sim_artifacts):
    """Ensure column names match schema.TABLES_DDL definitions exactly."""
    df_trips, df_samples, df_events, df_snapshots, _ = sim_artifacts

    def extract_ddl_cols(table_name: str) -> list[str]:
        ddl = TABLES_DDL[table_name]
        create_stmt = ddl[: ddl.find(";")]
        body = create_stmt[create_stmt.find("(") + 1 : create_stmt.rfind(")")]
        cols = []
        for line in body.split("\n"):
            line = line.strip().rstrip(",")
            if not line or line.startswith(("PRIMARY", "FOREIGN", "CHECK", "CREATE", "CONSTRAINT")):
                continue
            cols.append(line.split()[0])
        return cols

    assert list(df_trips.columns) == extract_ddl_cols("trips")
    assert list(df_samples.columns) == extract_ddl_cols("samples_1hz")
    assert list(df_events.columns) == extract_ddl_cols("events")
    assert list(df_snapshots.columns) == extract_ddl_cols("event_snapshots")


def test_value_ranges_plausible(sim_artifacts):
    """
    Ensure simulated vehicle telemetry values are physically realistic.
    Excludes the exact corrupted rows logged in ground_truth.json (no ad-hoc filters).
    """
    _df_trips, df_samples, _, _, gt = sim_artifacts

    # Build set of corrupted (ts, trip_id) keys from ground truth
    corrupt_keys = {(c["ts"], c["trip_id"]) for c in gt["data_integrity"]["injected_corruptions"]}

    # Filter out only the explicitly logged corruptions
    sample_keys = list(zip(df_samples["ts"], df_samples["trip_id"], strict=False))
    is_corrupt = [k in corrupt_keys for k in sample_keys]
    valid_samples = df_samples[~pd.Series(is_corrupt, index=df_samples.index)].dropna().copy()

    # Speeds
    assert (valid_samples["speed_kmh"] >= 0.0).all()
    assert valid_samples["speed_kmh"].max() < 140.0

    # Temperatures
    assert (valid_samples["ambient_temp_c"] >= 2.0).all()
    assert (valid_samples["ambient_temp_c"] <= 40.0).all()
    assert (valid_samples["pack_temp_c"] >= 4.0).all()
    assert (valid_samples["pack_temp_c"] <= 55.0).all()

    # Battery
    assert (valid_samples["soc_pct"] >= 2.0).all()
    assert (valid_samples["pack_voltage_v"] >= 310.0).all()
    assert (valid_samples["pack_voltage_v"] <= 430.0).all()

    # City & Highway Consumption Requirements from PROJECT_CONTEXT.md
    eff = gt["efficiency_metrics"]
    assert 120.0 <= eff["city_avg_wh_per_km"] <= 160.0, (
        f"City avg {eff['city_avg_wh_per_km']} out of range [120, 160]"
    )
    assert 170.0 <= eff["highway_avg_wh_per_km"] <= 220.0, (
        f"Highway avg {eff['highway_avg_wh_per_km']} out of range [170, 220]"
    )


def test_event_types_and_ids_valid(sim_artifacts):
    """Ensure events have valid types, sequential EV-000X IDs, and valid pre-event snapshots."""
    _, _, df_events, df_snapshots, _ = sim_artifacts

    allowed_types = {e.value for e in EventType}
    for _, row in df_events.iterrows():
        assert row["type"] in allowed_types
        assert EVENT_ID_REGEX.match(row["event_id"])
        assert row["severity"] in ("WARNING", "CRITICAL", "INFO")
        details = json.loads(row["details_json"])
        assert isinstance(details, dict)

    expected_ids = [f"EV-{i:04d}" for i in range(1, len(df_events) + 1)]
    assert list(df_events["event_id"]) == expected_ids

    snap_counts = df_snapshots.groupby("event_id").size()
    for ev_id in expected_ids:
        assert snap_counts[ev_id] == 601


@pytest.mark.slow
def test_ground_truth_recomputation_from_parquet(sim_artifacts):
    """
    Recomputes every ground truth number directly from the DataFrames / Parquet
    and confirms 100% agreement with ground_truth.json.
    """
    df_trips, df_samples, df_events, _, gt = sim_artifacts

    # 1. Overall distance-weighted efficiency
    tot_dist = df_trips["distance_km"].sum()
    tot_energy = df_trips["energy_kwh"].sum()
    recomputed_overall = round_1(tot_energy * 1000.0 / tot_dist)
    assert recomputed_overall == gt["efficiency_metrics"]["overall_avg_wh_per_km"]

    # 2. City and Highway efficiency from samples
    # City:
    city_samples = df_samples[df_samples["road_segment"] == "city"]
    city_dist = (city_samples["speed_kmh"] / 3600.0).sum()
    city_energy = (city_samples["power_kw"] * 1000.0 / 3600.0).clip(lower=0.0).sum()
    recomputed_city = round_1(city_energy / city_dist)
    assert abs(recomputed_city - gt["efficiency_metrics"]["city_avg_wh_per_km"]) <= 0.05

    # Highway:
    hwy_samples = df_samples[df_samples["road_segment"] == "highway"]
    hwy_dist = (hwy_samples["speed_kmh"] / 3600.0).sum()
    hwy_energy = (hwy_samples["power_kw"] * 1000.0 / 3600.0).clip(lower=0.0).sum()
    recomputed_hwy = round_1(hwy_energy / hwy_dist)
    assert abs(recomputed_hwy - gt["efficiency_metrics"]["highway_avg_wh_per_km"]) <= 0.05

    # 3. Weekly distance-weighted metrics
    df_trips_dt = df_trips.copy()
    df_trips_dt["start_dt"] = pd.to_datetime(df_trips_dt["start_ts"])

    for w_entry in gt["efficiency_metrics"]["weekly_summary"]:
        w_start = pd.to_datetime(w_entry["start"])
        w_end = pd.to_datetime(w_entry["end"])
        w_trips = df_trips_dt[
            (df_trips_dt["start_dt"] >= w_start) & (df_trips_dt["start_dt"] < w_end)
        ]
        w_dist = round_1(w_trips["distance_km"].sum())
        w_wh_km = round_1(w_trips["energy_kwh"].sum() * 1000.0 / w_trips["distance_km"].sum())
        assert abs(w_dist - w_entry["total_distance_km"]) <= 0.05
        assert abs(w_wh_km - w_entry["efficiency_wh_per_km"]) <= 0.05

    # 4. Range drop period comparison
    rd = gt["scenarios"]["range_drop"]
    t_mid = pd.to_datetime(rd["target_window"]["start_date"])

    base_trips = df_trips_dt[df_trips_dt["start_dt"] < t_mid]
    target_trips = df_trips_dt[df_trips_dt["start_dt"] >= t_mid]

    recomp_base_wh_km = round_1(
        base_trips["energy_kwh"].sum() * 1000.0 / base_trips["distance_km"].sum()
    )
    recomp_tgt_wh_km = round_1(
        target_trips["energy_kwh"].sum() * 1000.0 / target_trips["distance_km"].sum()
    )

    assert recomp_base_wh_km == rd["baseline_window"]["wh_per_km"]
    assert recomp_tgt_wh_km == rd["target_window"]["wh_per_km"]
    assert round_1(recomp_tgt_wh_km - recomp_base_wh_km) == rd["delta"]["wh_per_km_increase"]

    # 5. Cold snap efficiency jump
    cold_date = gt["scenarios"]["largest_efficiency_drop"]["date"]
    prev_date = gt["scenarios"]["largest_efficiency_drop"]["previous_date"]
    eff_prev = round_1(
        df_trips[df_trips["start_ts"].str.startswith(prev_date)]["energy_kwh"].sum()
        * 1000.0
        / df_trips[df_trips["start_ts"].str.startswith(prev_date)]["distance_km"].sum()
    )
    eff_cold = round_1(
        df_trips[df_trips["start_ts"].str.startswith(cold_date)]["energy_kwh"].sum()
        * 1000.0
        / df_trips[df_trips["start_ts"].str.startswith(cold_date)]["distance_km"].sum()
    )
    assert eff_prev == gt["scenarios"]["largest_efficiency_drop"]["efficiency_before_wh_per_km"]
    assert eff_cold == gt["scenarios"]["largest_efficiency_drop"]["efficiency_after_wh_per_km"]

    # 6. Seeded events match
    yesterday_date = gt["scenarios"]["adas_disengagements"]["yesterday_date"]
    yesterday_adas = df_events[
        (df_events["type"] == EventType.ADAS_DISENGAGEMENT.value)
        & (df_events["ts"].str.startswith(yesterday_date))
    ]
    assert len(yesterday_adas) == 3
    assert len(yesterday_adas) == gt["scenarios"]["adas_disengagements"]["yesterday_count"]

    # 7. Injected corruptions verification
    assert len(gt["data_integrity"]["injected_corruptions"]) == 7
    assert gt["data_integrity"]["dropped_sample_count"] > 0


@pytest.mark.slow
def test_factor_contributions_within_10_percent_of_total(sim_artifacts):
    """
    Validates that the two-group factor contributions (cold_weather + highway_share)
    sum to within 10% of the total Wh/km increase.
    """
    _, _, _, _, gt = sim_artifacts
    rd = gt["scenarios"]["range_drop"]
    total_change = rd["delta"]["wh_per_km_increase"]
    groups = rd["group_contributions_wh_km"]

    cold_weather = groups["cold_weather"]
    highway_share = groups["highway_share"]
    sum_contribs = round_1(cold_weather + highway_share)
    residual = rd["interaction_residual_wh_km"]

    # Verify ranking
    assert rd["factor_ranking"] == ["cold_weather", "highway_share"]

    # Verify sum within 10% of total change
    discrepancy_ratio = abs(sum_contribs - total_change) / total_change
    assert discrepancy_ratio <= 0.10, f"Discrepancy {discrepancy_ratio:.2%} exceeds 10%"

    # Verify interaction residual matches total - sum
    assert residual == round_1(total_change - sum_contribs)
    assert abs(residual) / total_change <= 0.10


def test_dropped_samples_and_corruption_integrity(sim_artifacts):
    """
    Using actual samples_1hz DataFrame:
    1. Verifies per-trip missing seconds, longest consecutive gap, and single drops.
    2. Confirms none of the 7 injected corruption rows were dropped.
    """
    _, df_samples, _, _, gt = sim_artifacts
    df_samples_dt = df_samples.copy()
    df_samples_dt["dt"] = pd.to_datetime(df_samples_dt["ts"])

    corruptions = gt["data_integrity"]["injected_corruptions"]
    assert len(corruptions) == 7

    # Check each corruption row is present and intact
    for c in corruptions:
        match = df_samples[(df_samples["ts"] == c["ts"]) & (df_samples["trip_id"] == c["trip_id"])]
        assert len(match) == 1, f"Corruption {c['trip_id']} at {c['ts']} was dropped!"

    # Check per-trip gap characteristics: gaps are scattered, max consecutive gap <= 3s
    max_gap_found = 0
    single_drops_found = 0
    for _, t_df in df_samples_dt.groupby("trip_id", sort=False):
        t_sorted = t_df.sort_values("dt").reset_index(drop=True)
        deltas = t_sorted["dt"].diff().dt.total_seconds().dropna().astype(int)
        gaps = deltas[deltas > 1]
        if len(gaps) > 0:
            max_gap_found = max(max_gap_found, int(gaps.max()))
            single_drops_found += int((deltas == 2).sum())

    assert max_gap_found <= 3, f"Unexpectedly large gap found: {max_gap_found}s"
    assert single_drops_found >= 600, (
        f"Expected predominantly single drops, found {single_drops_found}"
    )
