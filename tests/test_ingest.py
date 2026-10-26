"""
tests/test_ingest.py  –  Phase 1b: Normaliser correctness tests.

Oracle: data/ground_truth.json  (read-only; never passed to normalize.py).
Database under test: data/vehicle.db built by ingest/normalize.normalise().

Tests:
  1. test_interpolate_short_gaps_synthetic
       Unit test on a synthetic DataFrame:
         - leading null (trip position 0)   → stays null
         - interior 2-row null run          → filled by linear interpolation
         - interior 6-row null run          → stays null (too long)
  2. test_corrupted_cells_repaired
       All 7 corrupted cells are repaired to within 1.0 of original_value.
  3. test_no_out_of_range_values
       No value outside PLAUSIBLE_BOUNDS remains after cleaning.
  4. test_row_counts
       Table row counts match expectations.
  5. test_readonly_db_uri
       DB opened via config.get_readonly_db_uri(); INSERT raises OperationalError.
  6. test_ts_index_used
       EXPLAIN QUERY PLAN shows idx_samples_1hz_ts for a time-bounded query.
  7. test_window_efficiency_matches_ground_truth
       trips-based window efficiency matches ground_truth.json within 1%.
  8. test_samples_1hz_window_efficiency
       samples_1hz-based window efficiency (1000*sum(power_kw)/sum(speed_kmh))
       matches trips-based figure AND ground truth within 0.5%.
  9. test_missing_rows_total_is_684
       Sum of (expected - actual) across all trips = 684 (exact per-trip check).
 10. test_no_wal_files
       journal_mode=DELETE: no vehicle.db-wal or vehicle.db-shm files present.
 11. test_samples_1min_energy_weighted
       samples_1min.energy_wh_per_km matches hand-computed 1000*sum(power_kw)/sum(speed_kmh).
 12. test_samples_1min_parked_null
       samples_1min.energy_wh_per_km is NULL (NaN) when all speed_kmh=0 in a minute.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import config
from ingest.normalize import (
    MAX_INTERP_GAP,
    PLAUSIBLE_BOUNDS,
    _build_samples_1min,
    _interpolate_short_gaps,
    normalise,
)

# ---------------------------------------------------------------------------
# Module-level fixture: run the normaliser once for the entire test session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report():
    """Return the quality report from a fresh normalise() run."""
    return normalise()


@pytest.fixture(scope="module")
def db_rw(report):
    """Read-write connection for inspection tests."""
    con = sqlite3.connect(config.DB_PATH)
    yield con
    con.close()


@pytest.fixture(scope="module")
def ground_truth():
    with open(config.GROUND_TRUTH_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Unit test: _interpolate_short_gaps on synthetic data
# ---------------------------------------------------------------------------


def test_interpolate_short_gaps_synthetic():
    """
    Trip T1, 25 rows (positions 0-24).

    Layout:
      Position  0        → NaN   leading null        → STAYS null
      Positions 1-4      → 1,2,3,4  (valid)
      Positions 5-6      → NaN   interior 2-run      → FILLED
      Positions 7-9      → 7,8,9  (valid)
      Positions 10-15    → NaN   interior 6-run      → STAYS null (>4)
      Positions 16-24    → 16..24  (valid)

    Expected:
      interp_count = 2  (positions 5 and 6)
      gaps_left    = 2  (leading null + 6-row run)
    Linear interpolation for 2-run gap:
      v_left = 4.0 (pos 4), v_right = 7.0 (pos 7)
      pos 5 → 4 + 1/3 × 3 = 5.0
      pos 6 → 4 + 2/3 × 3 = 6.0
    """
    assert MAX_INTERP_GAP == 4, "Test assumes MAX_INTERP_GAP = 4"

    vals = list(
        range(25),
    )  # 0..24
    vals[0] = float("nan")  # leading null
    vals[5] = float("nan")  # interior 2-run
    vals[6] = float("nan")
    for p in range(10, 16):  # interior 6-run
        vals[p] = float("nan")

    df = pd.DataFrame({"trip_id": ["T1"] * 25, "val": [float(v) for v in vals]})

    df_fixed, n_interp, n_gaps = _interpolate_short_gaps(df.copy(), ["val"], max_gap=4)

    # Leading null stays null
    assert pd.isna(df_fixed["val"].iloc[0]), "Leading null must remain null"

    # Interior 2-run is filled with correct linear values
    assert pd.notna(df_fixed["val"].iloc[5]), "Interior 2-run pos 5 must be filled"
    assert pd.notna(df_fixed["val"].iloc[6]), "Interior 2-run pos 6 must be filled"
    assert abs(df_fixed["val"].iloc[5] - 5.0) < 1e-9, (
        f"Expected 5.0 at pos 5, got {df_fixed['val'].iloc[5]}"
    )
    assert abs(df_fixed["val"].iloc[6] - 6.0) < 1e-9, (
        f"Expected 6.0 at pos 6, got {df_fixed['val'].iloc[6]}"
    )

    # Interior 6-run stays null
    for p in range(10, 16):
        assert pd.isna(df_fixed["val"].iloc[p]), f"Long gap pos {p} must remain null"

    # Valid tails are unchanged
    assert df_fixed["val"].iloc[24] == 24.0

    assert n_interp == 2, f"Expected 2 interpolated values, got {n_interp}"
    assert n_gaps == 2, f"Expected 2 gaps left (leading + 6-run), got {n_gaps}"


# ---------------------------------------------------------------------------
# 2. Corrupted cells repaired within tolerance
# ---------------------------------------------------------------------------
# Tolerance of 1.0 is tight but physically justified: linear interpolation
# between immediate 1-second neighbours can deviate by at most the half-step
# of the local gradient.  For all 7 corruptions the actual deviation is < 0.8.
REPAIR_TOLERANCE = 1.0


def test_corrupted_cells_repaired(db_rw, ground_truth):
    """Every corrupted cell exists in the DB and is within 1.0 of original_value."""
    corruptions = ground_truth["data_integrity"]["injected_corruptions"]
    assert len(corruptions) == 7, "Expected exactly 7 injected corruptions"

    for c in corruptions:
        col = c["column"]
        row = db_rw.execute(
            f"SELECT {col} FROM samples_1hz WHERE ts = ? AND trip_id = ?",
            (c["ts"], c["trip_id"]),
        ).fetchone()

        assert row is not None, f"Row missing: ts={c['ts']} trip={c['trip_id']}"
        db_val = row[0]
        assert db_val is not None, f"Column {col} still NULL after normalisation at ts={c['ts']}"
        orig = c["original_value"]
        diff = abs(db_val - orig)
        assert diff <= REPAIR_TOLERANCE, (
            f"Repair exceeds tolerance: col={col} ts={c['ts']} "
            f"original={orig} db_val={db_val:.4f} diff={diff:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. No out-of-range values after cleaning
# ---------------------------------------------------------------------------


def test_no_out_of_range_values(db_rw):
    """Every numeric column obeys PLAUSIBLE_BOUNDS after normalisation."""
    for col, (lo, hi) in PLAUSIBLE_BOUNDS.items():
        count = db_rw.execute(
            f"SELECT COUNT(*) FROM samples_1hz "
            f"WHERE {col} IS NOT NULL AND ({col} < ? OR {col} > ?)",
            (lo, hi),
        ).fetchone()[0]
        assert count == 0, f"Column {col} has {count} out-of-range value(s) outside [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# 4. Row counts
# ---------------------------------------------------------------------------


def test_row_counts(db_rw, ground_truth):
    """Tables have the expected number of rows."""
    assert db_rw.execute("SELECT COUNT(*) FROM trips").fetchone()[0] == 69
    expected_events = len(ground_truth["all_events"])
    assert db_rw.execute("SELECT COUNT(*) FROM events").fetchone()[0] == expected_events
    assert (
        db_rw.execute("SELECT COUNT(*) FROM event_snapshots").fetchone()[0] == expected_events * 601
    )
    assert db_rw.execute("SELECT COUNT(*) FROM samples_1hz").fetchone()[0] == 136_269
    assert db_rw.execute("SELECT COUNT(*) FROM samples_1min").fetchone()[0] > 2000


# ---------------------------------------------------------------------------
# 5. Read-only URI
# ---------------------------------------------------------------------------


def test_readonly_db_uri(report):
    """config.get_readonly_db_uri() opens DB read-only; INSERT raises OperationalError."""
    _ = report  # ensure normalise() has run
    uri = config.get_readonly_db_uri()
    con = sqlite3.connect(uri, uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            con.execute(
                "INSERT INTO events(event_id, ts, type, severity) "
                "VALUES('EV-TEST','2026-01-01T00:00:00','ADAS_DISENGAGEMENT','INFO')"
            )
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 6. Time-range query uses the ts index
# ---------------------------------------------------------------------------


def test_ts_index_used(db_rw):
    """EXPLAIN QUERY PLAN shows idx_samples_1hz_ts for a time-bounded query."""
    plan = db_rw.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT ts, speed_kmh FROM samples_1hz "
        "WHERE ts >= '2026-10-14T00:00:00' AND ts < '2026-10-28T00:00:00'"
    ).fetchall()
    plan_text = " ".join(str(r) for r in plan).lower()
    assert "idx_samples_1hz_ts" in plan_text, (
        f"Expected idx_samples_1hz_ts in EXPLAIN QUERY PLAN, got:\n{plan_text}"
    )


# ---------------------------------------------------------------------------
# 7. Trips-based window efficiency matches ground_truth.json within 1%
# ---------------------------------------------------------------------------
USABLE_BATTERY_WH = 58_000.0


def _trips_window_eff(con, start, end):
    row = con.execute(
        "SELECT SUM(energy_kwh)*1000.0 / SUM(distance_km), SUM(distance_km) "
        "FROM trips WHERE start_ts >= ? AND start_ts < ?",
        (start, end),
    ).fetchone()
    wh_km = row[0]
    return wh_km, USABLE_BATTERY_WH / wh_km


def test_window_efficiency_matches_ground_truth(db_rw, ground_truth):
    """DB trips-based efficiency and range are within 1% of ground_truth.json."""
    rd = ground_truth["scenarios"]["range_drop"]
    bw = rd["baseline_window"]
    tw = rd["target_window"]

    b_wh_km, b_range = _trips_window_eff(
        db_rw,
        bw["start_date"] + "T00:00:00",
        bw["end_date"] + "T00:00:00",
    )
    assert abs(b_wh_km - bw["wh_per_km"]) / bw["wh_per_km"] < 0.01, (
        f"Baseline efficiency: DB={b_wh_km:.2f} GT={bw['wh_per_km']}"
    )
    assert abs(b_range - bw["estimated_range_km"]) / bw["estimated_range_km"] < 0.01

    t_wh_km, t_range = _trips_window_eff(
        db_rw,
        tw["start_date"] + "T00:00:00",
        tw["end_date"] + "T18:00:01",
    )
    assert abs(t_wh_km - tw["wh_per_km"]) / tw["wh_per_km"] < 0.01, (
        f"Target efficiency: DB={t_wh_km:.2f} GT={tw['wh_per_km']}"
    )
    assert abs(t_range - tw["estimated_range_km"]) / tw["estimated_range_km"] < 0.01


# ---------------------------------------------------------------------------
# 8. samples_1hz-derived window efficiency matches trips and ground truth (≤2%)
# ---------------------------------------------------------------------------
# Formula: 1000 × sum(power_kw) / sum(speed_kmh)
# = total_energy_Wh / total_distance_km   (each row covers exactly 1 second)
# Slight deviation from trips formula expected (684 dropped rows, ~0.5%).


def _hz_window_eff(con, start, end):
    """Derive efficiency from samples_1hz power and speed."""
    row = con.execute(
        "SELECT SUM(power_kw), SUM(speed_kmh) FROM samples_1hz WHERE ts >= ? AND ts < ?",
        (start, end),
    ).fetchone()
    sum_pwr, sum_spd = row
    assert sum_spd is not None and sum_spd > 0, "No speed data in window"
    wh_km = 1000.0 * sum_pwr / sum_spd
    return wh_km, USABLE_BATTERY_WH / wh_km


def test_samples_1hz_window_efficiency(db_rw, ground_truth):
    """
    samples_1hz-based efficiency matches both trips-based figure and ground truth
    within 2% for baseline and target windows.
    """
    rd = ground_truth["scenarios"]["range_drop"]
    bw = rd["baseline_window"]
    tw = rd["target_window"]

    # Baseline
    hz_b_wh_km, _ = _hz_window_eff(
        db_rw, bw["start_date"] + "T00:00:00", bw["end_date"] + "T00:00:00"
    )
    tr_b_wh_km, _ = _trips_window_eff(
        db_rw, bw["start_date"] + "T00:00:00", bw["end_date"] + "T00:00:00"
    )
    gt_b = bw["wh_per_km"]

    assert abs(hz_b_wh_km - tr_b_wh_km) / tr_b_wh_km < 0.005, (
        f"Baseline 1Hz vs trips: {hz_b_wh_km:.4f} vs {tr_b_wh_km:.4f}"
    )
    assert abs(hz_b_wh_km - gt_b) / gt_b < 0.005, f"Baseline 1Hz vs GT: {hz_b_wh_km:.4f} vs {gt_b}"

    # Target
    hz_t_wh_km, _ = _hz_window_eff(
        db_rw, tw["start_date"] + "T00:00:00", tw["end_date"] + "T18:00:01"
    )
    tr_t_wh_km, _ = _trips_window_eff(
        db_rw, tw["start_date"] + "T00:00:00", tw["end_date"] + "T18:00:01"
    )
    gt_t = tw["wh_per_km"]

    assert abs(hz_t_wh_km - tr_t_wh_km) / tr_t_wh_km < 0.005, (
        f"Target 1Hz vs trips: {hz_t_wh_km:.4f} vs {tr_t_wh_km:.4f}"
    )
    assert abs(hz_t_wh_km - gt_t) / gt_t < 0.005, f"Target 1Hz vs GT: {hz_t_wh_km:.4f} vs {gt_t}"


# ---------------------------------------------------------------------------
# 9. Total missing rows = 684
# ---------------------------------------------------------------------------


def test_missing_rows_total_is_684(report):
    """quality_report.json trip reconciliation must show exactly 684 missing rows."""
    rec = report["gap_analysis"]["trip_reconciliation"]
    total = rec["total_missing_rows"]
    assert total == 684, f"Expected 684 total missing rows, got {total}"

    # Cross-check components sum correctly
    singles = rec["interior_single_gaps"]
    multi_total = sum(rec["interior_multi_gap_sizes"])
    boundary = rec["boundary_drops"]
    assert singles + multi_total + boundary == total, (
        f"Components {singles}+{multi_total}+{boundary} != {total}"
    )


# ---------------------------------------------------------------------------
# 10. No WAL or SHM files (journal_mode=DELETE)
# ---------------------------------------------------------------------------


def test_no_wal_files(report):
    """journal_mode=DELETE: vehicle.db-wal and vehicle.db-shm must not exist."""
    _ = report  # ensure normalise() has run
    db = Path(config.DB_PATH)
    wal = Path(str(db) + "-wal")
    shm = Path(str(db) + "-shm")
    assert not wal.exists(), f"WAL file should not exist: {wal}"
    assert not shm.exists(), f"SHM file should not exist: {shm}"


# ---------------------------------------------------------------------------
# 11. Energy-weighted samples_1min efficiency vs hand-computed minute
# ---------------------------------------------------------------------------


def test_samples_1min_energy_weighted(db_rw):
    """
    samples_1min.energy_wh_per_km must be energy-weighted:
      1000 * sum(power_kw) / sum(speed_kmh) over the 1Hz samples of that minute.
    It must NOT be the arithmetic mean of 1Hz energy_wh_per_km ratios.
    """
    target_min = "2026-10-27T18:00:00"

    # Query 1min aggregate from DB
    row_1m = db_rw.execute(
        "SELECT energy_wh_per_km FROM samples_1min WHERE ts = ?",
        (target_min,),
    ).fetchone()
    assert row_1m is not None, f"Minute {target_min} not found in samples_1min"
    db_eff = row_1m[0]

    # Query underlying 1Hz samples for the same 60-second window
    row_1hz = db_rw.execute(
        "SELECT COUNT(*), SUM(power_kw), SUM(speed_kmh), AVG(energy_wh_per_km) "
        "FROM samples_1hz WHERE ts >= ? AND ts < '2026-10-27T18:01:00'",
        (target_min,),
    ).fetchone()
    count_1hz, sum_pwr, sum_spd, mean_of_ratios = row_1hz

    assert count_1hz == 60, f"Expected 60 1Hz rows for minute {target_min}, got {count_1hz}"
    assert sum_spd is not None and sum_spd > 0, "Speed sum must be positive"

    # Hand-computed energy-weighted formula
    expected_eff = 1000.0 * sum_pwr / sum_spd

    # DB value matches hand computation to machine precision
    assert abs(db_eff - expected_eff) < 1e-6, (
        f"DB efficiency {db_eff} does not match hand-computed {expected_eff}"
    )

    # Confirm it distinctly differs from arithmetic mean of ratios
    assert abs(db_eff - mean_of_ratios) > 0.05, (
        f"DB efficiency {db_eff} unexpectedly matches mean-of-ratios {mean_of_ratios}"
    )


# ---------------------------------------------------------------------------
# 12. Parked minute produces NULL energy_wh_per_km
# ---------------------------------------------------------------------------


def test_samples_1min_parked_null():
    """
    When all speed_kmh values in a minute are 0 (parked / idle vehicle),
    energy_wh_per_km cannot be divided by zero and must be NULL (NaN in pandas).
    """
    df_parked = pd.DataFrame(
        {
            "ts": [f"2026-10-20T10:00:{s:02d}" for s in range(60)],
            "speed_kmh": [0.0] * 60,
            "power_kw": [1.5] * 60,
            "soc_pct": [80.0] * 60,
            "pack_temp_c": [25.0] * 60,
            "ambient_temp_c": [20.0] * 60,
            "hvac_power_kw": [1.5] * 60,
            "estimated_range_km": [300.0] * 60,
        }
    )

    result = _build_samples_1min(df_parked)
    assert len(result) == 1
    assert result.iloc[0]["ts"] == "2026-10-20T10:00:00"
    assert pd.isna(result.iloc[0]["energy_wh_per_km"]), (
        f"Expected NaN for parked minute, got {result.iloc[0]['energy_wh_per_km']}"
    )
