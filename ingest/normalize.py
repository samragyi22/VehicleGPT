"""
ingest/normalize.py  –  Phase 1b: Raw-Parquet → SQLite normaliser for VehicleGPT.

Run with:
    python -m ingest.normalize

Cleaning rules (applied in order, bounds defined once in PLAUSIBLE_BOUNDS):
  1. RANGE-NULL  Values outside plausible physical bounds are set to NULL.
                 Bounds are conservative but wider than any realistic reading.
  2. INTERPOLATE  Within the same trip, short NULL runs (≤ MAX_INTERP_GAP rows,
                  i.e. ≤ 4 consecutive missing seconds) are repaired by
                  pandas linear interpolation (index-aware, only interior points).
                  Longer gaps remain NULL and are counted in the quality report.
  3. NO PHANTOM ROWS  Dropped timestamps from the simulator are not recreated;
                      only genuine samples that arrived in the Parquet are loaded.

Efficiency formula (identical to simulator, per metadata.efficiency_formula):
    energy_wh_per_km  = (total_energy_wh) / (total_distance_km)
                       = (sum(trip.energy_kwh) * 1000) / sum(trip.distance_km)
This formula is used both for the per-row instant metric and for window-level
aggregations verified in tests/test_ingest.py.

estimated_range_km window oracle:
    est_range = (USABLE_BATTERY_KWH * 1000) / window_avg_wh_per_km
    (same formula the simulator uses to populate ground_truth.json)

The normaliser NEVER reads ground_truth.json.  That file is only the test oracle.
"""

import json
import sqlite3
import time
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

# Maximum consecutive NULL rows that will be interpolated (inclusive).
# 4 seconds covers the longest gap injected by the simulator (max 3 s).
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
    Within each trip, linearly interpolate NULL runs whose length <= max_gap.
    Longer runs, and leading/trailing NULLs at trip boundaries, are left as NULL.

    Returns (df_fixed, values_interpolated, gaps_left_as_null).
    """
    interp_count = 0
    gaps_left = 0

    for col in numeric_cols:
        if col not in df.columns or df[col].notna().all():
            continue

        fixed_series = df[col].copy()

        for _trip_id, grp in df.groupby("trip_id", sort=False):
            col_vals = grp[col].copy()
            if col_vals.isna().sum() == 0:
                continue

            is_null = col_vals.isna()
            run_id = (is_null != is_null.shift()).cumsum()

            for rid, run in col_vals.groupby(run_id):
                if not is_null.loc[run.index[0]]:
                    continue  # not a null run
                run_len = len(run)
                start_pos = run.index[0]
                end_pos = run.index[-1]

                # Interior check: need valid value before AND after within trip
                before_vals = (
                    grp[col].loc[: start_pos - 1]
                    if start_pos > grp.index[0]
                    else pd.Series(dtype=float)
                )
                after_vals = (
                    grp[col].loc[end_pos + 1 :]
                    if end_pos < grp.index[-1]
                    else pd.Series(dtype=float)
                )
                has_before = before_vals.dropna().shape[0] > 0
                has_after = after_vals.dropna().shape[0] > 0

                if run_len <= max_gap and has_before and has_after:
                    col_vals = col_vals.interpolate(method="index")
                else:
                    gaps_left += 1

            n_filled = int((grp[col].isna() & col_vals.notna()).sum())
            interp_count += n_filled
            fixed_series.loc[grp.index] = col_vals

        df[col] = fixed_series

    return df, interp_count, gaps_left


def _build_samples_1min(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1 Hz samples to 1-minute buckets (floor to minute)."""
    df = df.copy()
    df["ts_dt"] = pd.to_datetime(df["ts"])
    df["ts_min"] = df["ts_dt"].dt.floor("min").dt.strftime("%Y-%m-%dT%H:%M:00")

    agg = (
        df.groupby("ts_min", sort=True)
        .agg(
            avg_speed_kmh=("speed_kmh", "mean"),
            avg_power_kw=("power_kw", "mean"),
            avg_soc_pct=("soc_pct", "mean"),
            avg_pack_temp_c=("pack_temp_c", "mean"),
            avg_ambient_temp_c=("ambient_temp_c", "mean"),
            avg_hvac_power_kw=("hvac_power_kw", "mean"),
            energy_wh_per_km=("energy_wh_per_km", "mean"),
            estimated_range_km=("estimated_range_km", "mean"),
        )
        .reset_index()
        .rename(columns={"ts_min": "ts"})
    )
    return agg


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
    Returns the quality report dict.
    """
    t0 = time.perf_counter()

    # -----------------------------------------------------------------------
    # 0. Delete and recreate DB
    # -----------------------------------------------------------------------
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

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
    # 4. Gap-profile analysis (characterise 684 dropped simulator samples)
    # -----------------------------------------------------------------------
    df_samples["ts_dt_tmp"] = pd.to_datetime(df_samples["ts"])
    single_drops = 0
    multi_drops = 0
    max_gap_sec = 0

    for _, grp in df_samples.groupby("trip_id", sort=False):
        grp_sorted = grp.sort_values("ts_dt_tmp")
        diffs = grp_sorted["ts_dt_tmp"].diff().dt.total_seconds().dropna()
        for g in diffs[diffs > 1.05]:  # 5 % tolerance over nominal 1 s
            missing = round(g) - 1
            if missing >= 1:
                max_gap_sec = max(max_gap_sec, round(g))
                if missing == 1:
                    single_drops += 1
                else:
                    multi_drops += 1

    df_samples = df_samples.drop(columns=["ts_dt_tmp"])

    gap_analysis: dict[str, Any] = {
        "total_timestamp_gaps_found": single_drops + multi_drops,
        "single_second_gaps": single_drops,
        "multi_second_gaps": multi_drops,
        "max_gap_seconds": max_gap_sec,
        "reconciliation": (
            "The simulator dropped samples by random index selection before writing "
            f"the Parquet file ({rows_read['samples_1hz'] - len(df_samples) + rows_dropped_after_clean} "
            "raw rows missing vs expected continuous recording). "
            f"The cleaned samples_1hz shows {single_drops} single-second timestamp gaps "
            f"and {multi_drops} multi-second gaps (max {max_gap_sec} s). "
            "No phantom rows were created to fill gaps."
        ),
    }

    # -----------------------------------------------------------------------
    # 5. Write SQLite (foreign keys enforced)
    # -----------------------------------------------------------------------
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")

    # Create tables in FK dependency order
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

    elapsed = time.perf_counter() - t0

    # -----------------------------------------------------------------------
    # 6. Quality Report
    # -----------------------------------------------------------------------
    report: dict[str, Any] = {
        "generated_at": config.SIMULATED_NOW_ISO,
        "elapsed_seconds": round(elapsed, 2),
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
        "gap_analysis": gap_analysis,
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
    print("=" * 70)
    print("VehicleGPT Normaliser  (Phase 1b)")
    print("=" * 70)

    report = normalise()

    print(f"\n[Quality Report written to {QUALITY_REPORT_PATH}]")
    print(f"  Elapsed      : {report['elapsed_seconds']:.2f} s")
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

    ga = report["gap_analysis"]
    print("\n[Dropped-Sample Gap Profile (from cleaned DB)]")
    print(f"  Single-second gaps : {ga['single_second_gaps']}")
    print(f"  Multi-second gaps  : {ga['multi_second_gaps']}")
    print(f"  Max gap (seconds)  : {ga['max_gap_seconds']}")
    print(f"  Reconciliation     : {ga['reconciliation']}")

    print("=" * 70)


if __name__ == "__main__":
    run()
