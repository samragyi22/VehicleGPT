"""
tests/test_ingest.py  –  Phase 1b: Normaliser correctness tests.

Oracle: data/ground_truth.json  (read-only; never passed to normalize.py).
Database under test: data/vehicle.db built by ingest/normalize.normalise().

Tests:
  1. All 7 corrupted cells are repaired to within a sensible tolerance of original_value.
  2. No value outside PLAUSIBLE_BOUNDS remains in samples_1hz after cleaning.
  3. Row counts in the database match expected values.
  4. DB opens read-only via config.get_readonly_db_uri; write attempt raises OperationalError.
  5. Time-range query on samples_1hz uses the idx_samples_1hz_ts index (EXPLAIN QUERY PLAN).
  6. Window-level efficiency and estimated_range_km from DB match ground_truth.json within 1%.
"""

import json
import sqlite3

import pytest

import config
from ingest.normalize import PLAUSIBLE_BOUNDS, normalise

# ---------------------------------------------------------------------------
# Module-level fixture: run the normaliser once for the entire test session.
# We re-use the existing DB if present; callers can force rebuild by deleting it.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report():
    """Return the quality report from a fresh normalise() run."""
    return normalise()


@pytest.fixture(scope="module")
def db_rw(report):
    """Read-write connection for inspection tests (not the query engine path)."""
    con = sqlite3.connect(config.DB_PATH)
    yield con
    con.close()


@pytest.fixture(scope="module")
def ground_truth():
    with open(config.GROUND_TRUTH_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. All 7 corrupted cells are repaired within tolerance
# ---------------------------------------------------------------------------
# Tolerance rationale:
#   - NaN corruptions repaired by linear interpolation; neighbours are ±1 s
#     apart so the interpolated value can differ by up to ~2 Wh/km or ~2 km/h.
#   - Outlier corruptions (999, -99, 150, 0) are nulled first then interpolated;
#     same neighbour gap applies.
# We use 5.0 as a generous but physically meaningful tolerance (≈ 3-5% for the
# most dynamic signals like speed and voltage).
REPAIR_TOLERANCE = 5.0


def test_corrupted_cells_repaired(db_rw, ground_truth):
    """Every corrupted cell exists in the DB and is within REPAIR_TOLERANCE of original_value."""
    corruptions = ground_truth["data_integrity"]["injected_corruptions"]
    assert len(corruptions) == 7, "Expected exactly 7 injected corruptions in ground truth"

    for c in corruptions:
        col = c["column"]
        row = db_rw.execute(
            f"SELECT {col} FROM samples_1hz WHERE ts = ? AND trip_id = ?",
            (c["ts"], c["trip_id"]),
        ).fetchone()

        assert row is not None, f"Row missing from DB: ts={c['ts']} trip={c['trip_id']}"
        db_val = row[0]
        assert db_val is not None, f"Column {col} is still NULL after normalisation at ts={c['ts']}"
        orig = c["original_value"]
        diff = abs(db_val - orig)
        assert diff <= REPAIR_TOLERANCE, (
            f"Repair too far from original: col={col} ts={c['ts']} "
            f"original={orig} db_val={db_val} diff={diff:.3f}"
        )


# ---------------------------------------------------------------------------
# 2. No out-of-range value remains after cleaning
# ---------------------------------------------------------------------------


def test_no_out_of_range_values(db_rw):
    """After normalisation, every numeric column obeys PLAUSIBLE_BOUNDS."""
    for col, (lo, hi) in PLAUSIBLE_BOUNDS.items():
        row = db_rw.execute(
            f"SELECT COUNT(*) FROM samples_1hz WHERE {col} IS NOT NULL AND ({col} < ? OR {col} > ?)",
            (lo, hi),
        ).fetchone()
        count = row[0]
        assert count == 0, (
            f"Column {col} still has {count} out-of-range value(s) outside [{lo}, {hi}]"
        )


# ---------------------------------------------------------------------------
# 3. Row counts match expectations
# ---------------------------------------------------------------------------


def test_row_counts(db_rw, ground_truth):
    """Tables have the expected number of rows."""
    # trips: whatever was in the Parquet (no rows dropped from trips)
    n_trips = db_rw.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    assert n_trips == 69, f"Expected 69 trips, got {n_trips}"

    # events: 9 seeded events
    expected_events = len(ground_truth["all_events"])
    n_events = db_rw.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert n_events == expected_events, f"Expected {expected_events} events, got {n_events}"

    # event_snapshots: 9 events × 601 offsets (-60000 ms to 0 ms in 100 ms steps)
    # 601 = len(range(-60000, 100, 100))
    n_snaps = db_rw.execute("SELECT COUNT(*) FROM event_snapshots").fetchone()[0]
    assert n_snaps == expected_events * 601, (
        f"Expected {expected_events * 601} event_snapshots, got {n_snaps}"
    )

    # samples_1hz: simulator generated 136269 rows (after dropping 684)
    n_1hz = db_rw.execute("SELECT COUNT(*) FROM samples_1hz").fetchone()[0]
    assert n_1hz == 136_269, f"Expected 136269 samples_1hz rows, got {n_1hz}"

    # samples_1min: must have at least 1 row per minute of driving
    n_1min = db_rw.execute("SELECT COUNT(*) FROM samples_1min").fetchone()[0]
    assert n_1min > 2000, f"samples_1min suspiciously small: {n_1min}"


# ---------------------------------------------------------------------------
# 4. Read-only URI: write attempt raises OperationalError
# ---------------------------------------------------------------------------


def test_readonly_db_uri(report):
    """config.get_readonly_db_uri() opens DB read-only; INSERT raises OperationalError."""
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
# 5. Time-range query uses the ts index (EXPLAIN QUERY PLAN)
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
        f"Expected idx_samples_1hz_ts in EXPLAIN QUERY PLAN output, got:\n{plan_text}"
    )


# ---------------------------------------------------------------------------
# 6. Window efficiency and estimated_range_km match ground_truth.json within 1%
# ---------------------------------------------------------------------------
# Formula (per locked decision + simulator comment):
#   wh_per_km   = sum(energy_kwh) * 1000 / sum(distance_km)  from trips table
#   est_range_km = USABLE_BATTERY_KWH * 1000 / wh_per_km
USABLE_BATTERY_WH = 58_000.0  # 58 kWh


def _window_eff(con: sqlite3.Connection, start: str, end: str) -> tuple[float, float]:
    """Return (wh_per_km, estimated_range_km) for trips whose start_ts is in [start, end)."""
    row = con.execute(
        "SELECT SUM(energy_kwh)*1000.0 / SUM(distance_km), SUM(distance_km) "
        "FROM trips WHERE start_ts >= ? AND start_ts < ?",
        (start, end),
    ).fetchone()
    wh_km = row[0]
    est_range = USABLE_BATTERY_WH / wh_km
    return wh_km, est_range


def test_window_efficiency_matches_ground_truth(db_rw, ground_truth):
    """DB-derived efficiency and estimated_range_km are within 1% of ground_truth.json."""
    rd = ground_truth["scenarios"]["range_drop"]
    bw = rd["baseline_window"]
    tw = rd["target_window"]

    # Baseline window
    b_wh_km, b_range = _window_eff(
        db_rw,
        bw["start_date"] + "T00:00:00",
        bw["end_date"] + "T00:00:00",
    )
    gt_b_wh_km = bw["wh_per_km"]
    gt_b_range = bw["estimated_range_km"]

    assert abs(b_wh_km - gt_b_wh_km) / gt_b_wh_km < 0.01, (
        f"Baseline efficiency mismatch: DB={b_wh_km:.2f} GT={gt_b_wh_km:.1f}"
    )
    assert abs(b_range - gt_b_range) / gt_b_range < 0.01, (
        f"Baseline range mismatch: DB={b_range:.1f} GT={gt_b_range:.1f}"
    )

    # Target window (end is open at simulated_now)
    t_wh_km, t_range = _window_eff(
        db_rw,
        tw["start_date"] + "T00:00:00",
        tw["end_date"] + "T18:00:01",  # inclusive of the last trip
    )
    gt_t_wh_km = tw["wh_per_km"]
    gt_t_range = tw["estimated_range_km"]

    assert abs(t_wh_km - gt_t_wh_km) / gt_t_wh_km < 0.01, (
        f"Target efficiency mismatch: DB={t_wh_km:.2f} GT={gt_t_wh_km:.1f}"
    )
    assert abs(t_range - gt_t_range) / gt_t_range < 0.01, (
        f"Target range mismatch: DB={t_range:.1f} GT={gt_t_range:.1f}"
    )
