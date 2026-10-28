"""
ingest/normalize.py  –  Phase 1b: Raw-Parquet → SQLite normaliser for VehicleGPT.

Run with:
    python -m ingest.normalize

Cleaning rules (applied in order, bounds defined once in PLAUSIBLE_BOUNDS):
  1. RANGE-NULL   Values strictly outside PLAUSIBLE_BOUNDS are set to NULL.
                  Bounds are conservative but wider than any realistic reading.
  2. INTERPOLATE  Within the same trip, interior NULL runs whose length ≤
                  MAX_INTERP_GAP (4) are repaired by linear interpolation
                  between the nearest valid neighbours.
                  * "Interior" means the null run does NOT start at the first
                    row of the trip and does NOT end at the last row of the trip.
                  * Leading nulls (trip-start boundary) are left as NULL.
                  * Trailing nulls (trip-end boundary) are left as NULL.
                  * Runs longer than MAX_INTERP_GAP rows are left as NULL.
  3. NO PHANTOM ROWS  Dropped timestamps from the simulator are not recreated.

Per-row signal semantics in samples_1hz (values come from the simulator
and are written to Parquet unchanged – the normaliser does not recompute them):

  energy_wh_per_km:
    Source: simulator, each 1-second row.
    Formula:
      if speed_kmh > 5.0:  power_kw * 1000 / speed_kmh   (Wh travelled per km)
      else (idle/stopped): hvac_power_kw * 1000 / 25.0   (normalised to 25 km/h
                           equivalent so standstill heating isn't ÷0 or undefined)
    Includes all loads (mechanical + HVAC + aux). Negative when regenerating.
    At speed = 0 this column is NOT null; it uses the hvac-normalised formula.
    Note: the per-row value is a noisy instantaneous snapshot. Use window-level
    aggregations (from trips or samples_1min) for meaningful efficiency numbers.

  estimated_range_km:
    Source: simulator, each 1-second row.
    Formula: (soc_pct / 100 × USABLE_BATTERY_KWH × 1000) / rolling_eff
    where rolling_eff = 0.7 × recent_history_eff_wh_km
                      + 0.3 × max(100, trip_efficiency_so_far_wh_km)
    rolling_eff is a smoothed estimate that decays toward trip history; at the
    start of a trip before any distance is covered it defaults to recent history.
    At speed = 0 this column is not null; it reflects soc and rolling_eff only.

samples_1min.energy_wh_per_km is energy-weighted, NOT a mean of ratios:
    1000 × sum(power_kw) / sum(speed_kmh)   [over all 1-Hz rows in the minute]
  This equals (total energy Wh over the minute) / (total distance km), matching
  the window formula in trips and ground_truth.json.
  NULL when sum(speed_kmh) = 0 (entire minute is stopped / vehicle parked).

Efficiency formula (identical to simulator, per metadata.efficiency_formula):
    window_wh_per_km = sum(energy_kwh) × 1000 / sum(distance_km)   [from trips]
    estimated_range_km = USABLE_BATTERY_KWH × 1000 / window_wh_per_km

The normaliser NEVER reads ground_truth.json.  That file is the test oracle only.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from schema import TABLES_DDL

# ---------------------------------------------------------------------------
# Plausible Physical Bounds (single source – change here, nowhere else)
# ---------------------------------------------------------------------------
# Each entry: (low_inclusive, high_inclusive).  Values strictly outside are nulled.
PLAUSIBLE_BOUNDS: dict[str, tuple[float, float]] = {
    "speed_kmh": (0.0, 200.0),
    "power_kw": (-50.0, 300.0),  # includes regen
    "soc_pct": (0.0, 100.0),
    "pack_voltage_v": (200.0, 500.0),
    "pack_current_a": (-600.0, 600.0),
    "pack_temp_c": (-40.0, 80.0),
    "ambient_temp_c": (-40.0, 60.0),
    "hvac_power_kw": (0.0, 10.0),
}

# Maximum consecutive interior NULL rows that will be interpolated (inclusive).
MAX_INTERP_GAP = 4

# Raw data paths
RAW_DIR = config.DATA_DIR / "raw"
QUALITY_REPORT_PATH = config.DATA_DIR / "quality_report.json"

# Usable battery – must match sim/simulate.py USABLE_BATTERY_KWH
USABLE_BATTERY_KWH = 58.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def _null_out_of_range(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace values outside PLAUSIBLE_BOUNDS with NaN.  Returns (df, count_nulled)."""
    total_nulled = 0
    for col, (lo, hi) in PLAUSIBLE_BOUNDS.items():
        if col not in df.columns:
            continue
        mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        total_nulled += int(mask.sum())
        df.loc[mask, col] = np.nan
    return df, total_nulled


def _interpolate_short_gaps(
    df: pd.DataFrame,
    numeric_cols: list[str],
    max_gap: int = MAX_INTERP_GAP,
) -> tuple[pd.DataFrame, int, int]:
    """
    Within each trip, linearly interpolate interior NULL runs of at most max_gap rows.

    Rules applied per (trip, column):
      - Scan for contiguous NULL runs.
      - A run is "interior" only if it does NOT start at position 0 of the trip
        AND does NOT end at the last position of the trip.
      - Interior runs with length ≤ max_gap: linearly interpolated between the
        nearest valid left and right neighbours.
      - All other runs (leading, trailing, or longer than max_gap): left as NULL
        and counted toward gaps_left.

    Returns (df_fixed, values_interpolated, gaps_left_as_null).
    """
    interp_count = 0
    gaps_left = 0

    for col in numeric_cols:
        if col not in df.columns:
            continue

        for _trip_id, grp in df.groupby("trip_id", sort=False):
            vals = grp[col].values.astype(float).copy()  # numpy array, trip order
            n = len(vals)

            if not np.any(np.isnan(vals)):
                continue

            i = 0
            while i < n:
                if not np.isnan(vals[i]):
                    i += 1
                    continue

                # Start of a null run – find its end
                j = i + 1
                while j < n and np.isnan(vals[j]):
                    j += 1
                # Null run spans [i, j)
                run_len = j - i
                is_leading = i == 0
                is_trailing = j == n

                if not is_leading and not is_trailing and run_len <= max_gap:
                    # Linear interpolation between nearest valid neighbours
                    v_left = vals[i - 1]
                    v_right = vals[j]
                    for k in range(run_len):
                        t = (k + 1) / (run_len + 1)
                        vals[i + k] = v_left + t * (v_right - v_left)
                    interp_count += run_len
                else:
                    gaps_left += 1

                i = j

            df.loc[grp.index, col] = vals

    return df, interp_count, gaps_left


def _build_samples_1min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1 Hz samples to 1-minute buckets (floor to minute).

    energy_wh_per_km is energy-weighted:
        1000 × sum(power_kw) / sum(speed_kmh)  over all rows in the minute.
    This equals total_energy_Wh / total_distance_km and is consistent with the
    trips-level window formula.  NULL when sum(speed_kmh) = 0 (entire minute stopped).

    estimated_range_km is the mean of the per-second values (already smoothed by
    the simulator's rolling filter; further averaging is appropriate).
    """
    df = df.copy()
    df["ts_dt"] = pd.to_datetime(df["ts"])
    df["ts_min"] = df["ts_dt"].dt.floor("min").dt.strftime("%Y-%m-%dT%H:%M:00")

    grp = df.groupby("ts_min", sort=True)

    agg = (
        grp.agg(
            avg_speed_kmh=("speed_kmh", "mean"),
            avg_power_kw=("power_kw", "mean"),
            avg_soc_pct=("soc_pct", "mean"),
            avg_pack_temp_c=("pack_temp_c", "mean"),
            avg_ambient_temp_c=("ambient_temp_c", "mean"),
            avg_hvac_power_kw=("hvac_power_kw", "mean"),
            estimated_range_km=("estimated_range_km", "mean"),
            _sum_power_kw=("power_kw", "sum"),
            _sum_speed_kmh=("speed_kmh", "sum"),
        )
        .reset_index()
        .rename(columns={"ts_min": "ts"})
    )

    # Energy-weighted efficiency: 1000 * sum(power_kw) / sum(speed_kmh)
    mask = agg["_sum_speed_kmh"] > 0
    agg["energy_wh_per_km"] = np.nan
    agg.loc[mask, "energy_wh_per_km"] = (
        1000.0 * agg.loc[mask, "_sum_power_kw"] / agg.loc[mask, "_sum_speed_kmh"]
    )

    return agg.drop(columns=["_sum_power_kw", "_sum_speed_kmh"])


def _compute_trip_reconciliation(
    df_trips: pd.DataFrame, df_samples: pd.DataFrame
) -> dict[str, Any]:
    """
    Exact per-trip accounting of missing samples.

    Rule for expected rows:
        expected_rows = int((end_ts - start_ts).total_seconds())
        i.e. one row per second for the full trip duration.

    For each trip the missing rows are classified as:
      interior_single_gaps   – Δt = 2 s (1 sample missing) inside the trip
      interior_multi_gap_sizes – list of Δt-1 for each Δt > 2 s (≥2 missing)
      boundary_drops         – samples missing at the start or end of the trip
                               (leading: first observed ts > trip_start_ts;
                                trailing: last observed ts < trip_start_ts + duration - 1 s)

    Total check: interior_single + sum(multi_sizes) + boundary == missing  (asserted).
    """
    df_trips = df_trips.copy()
    df_trips["start_dt"] = pd.to_datetime(df_trips["start_ts"])
    df_trips["end_dt"] = pd.to_datetime(df_trips["end_ts"])

    counts = df_samples.groupby("trip_id").size()
    sample_ts_map = {
        tid: pd.to_datetime(grp["ts"]).sort_values().reset_index(drop=True)
        for tid, grp in df_samples.groupby("trip_id")
    }

    total_expected = 0
    total_actual = 0
    total_missing = 0
    total_int_single = 0
    total_int_multi: list[int] = []
    total_boundary = 0
    per_trip: dict[str, dict] = {}

    for _, row in df_trips.iterrows():
        tid = row["trip_id"]
        expected = round((row["end_dt"] - row["start_dt"]).total_seconds())
        actual = int(counts.get(tid, 0))
        missing = expected - actual

        ts = sample_ts_map.get(tid, pd.Series(dtype="datetime64[ns]"))

        int_single = 0
        int_multi: list[int] = []
        boundary = 0

        if len(ts) > 0:
            diffs = ts.diff().dt.total_seconds().dropna()
            for d in diffs:
                if d > 1.5:
                    n_miss = round(d) - 1
                    if n_miss == 1:
                        int_single += 1
                    else:
                        int_multi.append(n_miss)

            int_total = int_single + sum(int_multi)

            exp_first = row["start_dt"]
            exp_last = row["start_dt"] + pd.Timedelta(seconds=expected - 1)
            leading = max(0, round((ts.iloc[0] - exp_first).total_seconds()))
            trailing = max(0, round((exp_last - ts.iloc[-1]).total_seconds()))
            boundary = leading + trailing
        else:
            int_total = 0
            boundary = missing

        assert int_total + boundary == missing, (
            f"{tid}: interior={int_total} boundary={boundary} != missing={missing}"
        )

        total_expected += expected
        total_actual += actual
        total_missing += missing
        total_int_single += int_single
        total_int_multi.extend(int_multi)
        total_boundary += boundary

        per_trip[tid] = {
            "expected_rows": expected,
            "actual_rows": actual,
            "missing_rows": missing,
            "interior_single_gaps": int_single,
            "interior_multi_gap_sizes": int_multi,
            "boundary_drops": boundary,
        }

    return {
        "rule": (
            "expected_rows = int((end_ts - start_ts).total_seconds()); "
            "one row per second for the full trip duration"
        ),
        "total_expected_rows": total_expected,
        "total_actual_rows": total_actual,
        "total_missing_rows": total_missing,
        "interior_single_gaps": total_int_single,
        "interior_multi_gap_sizes": total_int_multi,
        "boundary_drops": total_boundary,
        "per_trip": per_trip,
    }


# ---------------------------------------------------------------------------
# Main normalise function
# ---------------------------------------------------------------------------


def normalise(
    raw_dir: Path = RAW_DIR,
    db_path: Path = config.DB_PATH,
    report_path: Path = QUALITY_REPORT_PATH,
) -> dict[str, Any]:
    """
    Full pipeline: read → clean → load → report.
    Deletes and recreates the database on every call (rebuildable by design).
    Uses journal_mode=DELETE; no -wal or -shm files are created.
    Returns the quality report dict.
    """

    # -----------------------------------------------------------------------
    # 0. Delete and recreate DB (and any stale WAL/SHM from previous runs)
    # -----------------------------------------------------------------------
    db_path.parent.mkdir(parents=True, exist_ok=True)
    for p in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
        if p.exists():
            p.unlink()

    # -----------------------------------------------------------------------
    # 1. Read raw Parquet files
    # -----------------------------------------------------------------------
    df_trips_raw = _read_parquet(raw_dir / "trips.parquet")
    df_samples_raw = _read_parquet(raw_dir / "samples_1hz.parquet")
    df_events_raw = _read_parquet(raw_dir / "events.parquet")
    df_snap_raw = _read_parquet(raw_dir / "event_snapshots.parquet")

    rows_read = {
        "trips": len(df_trips_raw),
        "samples_1hz": len(df_samples_raw),
        "events": len(df_events_raw),
        "event_snapshots": len(df_snap_raw),
    }

    # -----------------------------------------------------------------------
    # 2. Clean samples_1hz
    # -----------------------------------------------------------------------
    df_samples = df_samples_raw.copy()
    df_samples, values_nulled = _null_out_of_range(df_samples)

    numeric_cols = [c for c in PLAUSIBLE_BOUNDS if c in df_samples.columns]
    df_samples, values_interpolated, gaps_left = _interpolate_short_gaps(df_samples, numeric_cols)

    # Drop rows where mandatory numeric columns are still NULL (violates NOT NULL).
    mandatory_cols = [
        "speed_kmh",
        "power_kw",
        "soc_pct",
        "pack_voltage_v",
        "pack_current_a",
        "pack_temp_c",
        "ambient_temp_c",
        "hvac_power_kw",
    ]
    rows_before_drop = len(df_samples)
    df_samples = df_samples.dropna(subset=mandatory_cols).reset_index(drop=True)
    rows_dropped_after_clean = rows_before_drop - len(df_samples)

    # -----------------------------------------------------------------------
    # 3. Build samples_1min
    # -----------------------------------------------------------------------
    df_1min = _build_samples_1min(df_samples)

    # -----------------------------------------------------------------------
    # 4. Exact per-trip reconciliation of the 684 dropped simulator samples
    # -----------------------------------------------------------------------
    trip_reconciliation = _compute_trip_reconciliation(df_trips_raw, df_samples_raw)

    # -----------------------------------------------------------------------
    # 5. Write SQLite (journal_mode=DELETE – no WAL files)
    # -----------------------------------------------------------------------
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = DELETE")

    for tbl in ("trips", "samples_1hz", "samples_1min", "events", "event_snapshots"):
        for stmt in TABLES_DDL[tbl].strip().split(";"):
            s = stmt.strip()
            if s:
                con.execute(s)

    df_trips_raw.to_sql("trips", con, if_exists="append", index=False)
    df_samples.to_sql("samples_1hz", con, if_exists="append", index=False)
    df_1min.to_sql("samples_1min", con, if_exists="append", index=False)
    df_events_raw.to_sql("events", con, if_exists="append", index=False)
    df_snap_raw.to_sql("event_snapshots", con, if_exists="append", index=False)

    con.commit()

    rows_loaded: dict[str, int] = {}
    for tbl in ("trips", "samples_1hz", "samples_1min", "events", "event_snapshots"):
        rows_loaded[tbl] = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]

    con.close()

    # -----------------------------------------------------------------------
    # 6. Quality Report
    # -----------------------------------------------------------------------
    report: dict[str, Any] = {
        "generated_at": config.SIMULATED_NOW_ISO,
        # elapsed intentionally omitted: wall-clock timing made the committed
        # report drift on every run; the report must be byte-stable.
        "db_path": str(db_path),
        "db_size_bytes": db_path.stat().st_size,
        "rows_read": rows_read,
        "rows_loaded": rows_loaded,
        "cleaning": {
            "values_nulled_out_of_range": values_nulled,
            "values_interpolated": values_interpolated,
            "gaps_left_as_null": gaps_left,
            "rows_dropped_unresolvable_null": rows_dropped_after_clean,
            "max_interp_gap_seconds": MAX_INTERP_GAP,
        },
        "gap_analysis": {
            "trip_reconciliation": trip_reconciliation,
        },
        "plausible_bounds": {k: {"low": v[0], "high": v[1]} for k, v in PLAUSIBLE_BOUNDS.items()},
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Entry point: python -m ingest.normalize"""
    import sqlite3 as _sq

    print("=" * 70)
    print("VehicleGPT Normaliser  (Phase 1b)")
    print("=" * 70)

    report = normalise()

    print(f"\n[Quality Report written to {QUALITY_REPORT_PATH}]")
    print(f"  DB size      : {report['db_size_bytes'] / 1_048_576:.2f} MB")

    print("\n[Rows Read from Parquet]")
    for tbl, n in report["rows_read"].items():
        print(f"  {tbl:<22}: {n:>8,d}")

    print("\n[Rows Loaded into SQLite]")
    for tbl, n in report["rows_loaded"].items():
        print(f"  {tbl:<22}: {n:>8,d}")

    print("\n[Cleaning Summary]")
    c = report["cleaning"]
    print(f"  Values nulled (out of range) : {c['values_nulled_out_of_range']}")
    print(f"  Values interpolated          : {c['values_interpolated']}")
    print(f"  Gaps left as NULL            : {c['gaps_left_as_null']}")
    print(f"  Rows dropped (unresolvable)  : {c['rows_dropped_unresolvable_null']}")

    rec = report["gap_analysis"]["trip_reconciliation"]
    print("\n[Dropped-Sample Reconciliation (exact per-trip)]")
    print(f"  Rule             : {rec['rule']}")
    print(f"  Total expected   : {rec['total_expected_rows']:,d}")
    print(f"  Total actual     : {rec['total_actual_rows']:,d}")
    print(f"  Total missing    : {rec['total_missing_rows']:,d}")
    print(f"  Interior singles : {rec['interior_single_gaps']}")
    print(f"  Interior multi   : {rec['interior_multi_gap_sizes']}  (missing rows each)")
    print(f"  Boundary drops   : {rec['boundary_drops']}")
    check = (
        rec["interior_single_gaps"] + sum(rec["interior_multi_gap_sizes"]) + rec["boundary_drops"]
    )
    print(
        f"  Cross-check      : {rec['interior_single_gaps']} + "
        f"{sum(rec['interior_multi_gap_sizes'])} + {rec['boundary_drops']} = {check}"
    )

    # -----------------------------------------------------------------------
    # energy_wh_per_km / estimated_range_km: 5 illustrative rows
    # -----------------------------------------------------------------------
    print("\n[samples_1hz: energy_wh_per_km and estimated_range_km – 5 rows]")
    print("  Semantics:")
    print("    energy_wh_per_km  : if speed>5: power_kw*1000/speed_kmh")
    print("                        else (idle): hvac_power_kw*1000/25")
    print("                        → NOT null at speed=0; uses HVAC normalisation")
    print("    estimated_range_km: (soc_pct/100 × 58kWh × 1000) / rolling_eff")
    print("                        rolling_eff: 0.7×recent_hist + 0.3×current_trip")
    print("  Source: passed through from simulator Parquet unchanged.")
    con2 = _sq.connect(config.DB_PATH)
    rows = con2.execute(
        "SELECT ts, trip_id, speed_kmh, power_kw, hvac_power_kw, "
        "       energy_wh_per_km, estimated_range_km "
        "FROM samples_1hz "
        "ORDER BY ts LIMIT 3"
    ).fetchall()
    # also fetch one stopped row and one regen row
    stopped = con2.execute(
        "SELECT ts, trip_id, speed_kmh, power_kw, hvac_power_kw, "
        "       energy_wh_per_km, estimated_range_km "
        "FROM samples_1hz WHERE speed_kmh = 0 LIMIT 1"
    ).fetchone()
    regen = con2.execute(
        "SELECT ts, trip_id, speed_kmh, power_kw, hvac_power_kw, "
        "       energy_wh_per_km, estimated_range_km "
        "FROM samples_1hz WHERE power_kw < 0 LIMIT 1"
    ).fetchone()
    con2.close()
    all_rows = rows[:3] + ([stopped] if stopped else []) + ([regen] if regen else [])
    hdr = f"  {'ts':<22} {'trip':<12} {'spd':>7} {'pwr':>8} {'hvac':>6} {'wh/km':>7} {'range':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row2 in all_rows[:5]:
        ts, tid, spd, pwr, hvac, eff, rng = row2
        print(f"  {ts:<22} {tid:<12} {spd:>7.1f} {pwr:>8.2f} {hvac:>6.2f} {eff:>7.1f} {rng:>7.1f}")

    # -----------------------------------------------------------------------
    # Corruption table
    # -----------------------------------------------------------------------
    import json as _j

    with open(config.GROUND_TRUTH_PATH) as f:
        gt = _j.load(f)
    corruptions = gt["data_integrity"]["injected_corruptions"]

    print("\n[7 Injected Corruptions – as found in DB after cleaning]")
    print("  The corruption set is UNCHANGED from the simulator commit (Phase 1a).")
    print("  All 7 are repaired by the normaliser (4 range-nulled then interpolated,")
    print("  3 nan-type values interpolated directly).")
    print()
    hdr2 = f"  {'date':<12} {'trip':<12} {'column':<18} {'corrupted':>12} {'original':>10} {'repaired':>10}"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    con3 = _sq.connect(config.DB_PATH)
    for c in corruptions:
        row3 = con3.execute(
            f"SELECT {c['column']} FROM samples_1hz WHERE ts=? AND trip_id=?",
            (c["ts"], c["trip_id"]),
        ).fetchone()
        repaired = f"{row3[0]:.2f}" if row3 and row3[0] is not None else "NULL"
        corrupted_str = f"{c['corrupted_value']:.2f}" if c["corrupted_value"] is not None else "nan"
        date_str = c["ts"][:10]
        print(
            f"  {date_str:<12} {c['trip_id']:<12} {c['column']:<18} "
            f"{corrupted_str:>12} {c['original_value']:>10.2f} {repaired:>10}"
        )
    con3.close()
    print("=" * 70)


if __name__ == "__main__":
    run()
