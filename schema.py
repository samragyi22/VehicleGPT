"""
schema.py - Single Source of Truth for VehicleGPT.

Defines:
1. Closed vocabularies and enums (Metrics, Event Types, Road Segments, Bucket Sizes).
2. SQLite table DDL and column definitions.
3. Symbolic time range models for SLM output.
4. Tool argument schemas (both SLM routing schemas with symbolic time and executable schemas with ISO timestamps).
5. Canonical ToolCall and ToolOutput definitions.

DO NOT add or alter tool names, arguments, or enums without explicit instructions.
"""

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# 1. Closed Vocabularies & Enums (Frozen per PROJECT_CONTEXT.md Section 4)
# ---------------------------------------------------------------------------
class Metric(str, Enum):
    energy_wh_per_km = "energy_wh_per_km"
    estimated_range_km = "estimated_range_km"
    pack_temp_c = "pack_temp_c"
    soc_pct = "soc_pct"
    speed_kmh = "speed_kmh"
    hvac_power_kw = "hvac_power_kw"


class EventType(str, Enum):
    BATTERY_OVERHEAT_WARNING = "BATTERY_OVERHEAT_WARNING"
    ADAS_DISENGAGEMENT = "ADAS_DISENGAGEMENT"
    LOW_SOC_WARNING = "LOW_SOC_WARNING"


class Segment(str, Enum):
    city = "city"
    highway = "highway"


class BucketSize(str, Enum):
    hour = "hour"
    day = "day"
    week = "week"


class ChangeDirection(str, Enum):
    drop = "drop"
    rise = "rise"


class ToolName(str, Enum):
    get_events = "get_events"
    context_before = "context_before"
    compare = "compare"
    find_largest_change = "find_largest_change"
    trend = "trend"
    explain_factors = "explain_factors"
    out_of_scope = "out_of_scope"
    # NOTE: search_logs is optional per PROJECT_CONTEXT.md and disabled until an FTS5 table exists.
    search_logs = "search_logs"


# ---------------------------------------------------------------------------
# 2. Database Schema DDL & Column Specs
# ---------------------------------------------------------------------------
SIGNALS_1HZ_RAW = [
    "ts",
    "trip_id",
    "speed_kmh",
    "power_kw",
    "soc_pct",
    "pack_voltage_v",
    "pack_current_a",
    "pack_temp_c",
    "ambient_temp_c",
    "hvac_power_kw",
    "road_segment",
    "adas_active",
    "odometer_km",
]

DERIVED_SIGNALS = [
    "energy_wh_per_km",
    "estimated_range_km",
]

# Signals available for general feature queries (excluding IDs and timestamps)
ALLOWED_SIGNALS: set[str] = {
    s for s in SIGNALS_1HZ_RAW + DERIVED_SIGNALS if s not in ("ts", "trip_id")
}

# Exactly the signal columns stored in event_snapshots (10 Hz pre-event ring buffer)
SNAPSHOT_SIGNALS: set[str] = {
    "speed_kmh",
    "power_kw",
    "soc_pct",
    "pack_voltage_v",
    "pack_current_a",
    "pack_temp_c",
    "ambient_temp_c",
    "hvac_power_kw",
    "road_segment",
    "adas_active",
}

EVENT_ID_REGEX = re.compile(
    r"^(EV-\d+|latest_(BATTERY_OVERHEAT_WARNING|ADAS_DISENGAGEMENT|LOW_SOC_WARNING))$"
)

TABLES_DDL: dict[str, str] = {
    "trips": """
        CREATE TABLE IF NOT EXISTS trips (
            trip_id TEXT PRIMARY KEY,
            start_ts TEXT NOT NULL,
            end_ts TEXT NOT NULL,
            distance_km REAL NOT NULL,
            energy_kwh REAL NOT NULL,
            avg_ambient_c REAL NOT NULL,
            city_share REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_trips_start_ts ON trips(start_ts);
    """,
    "samples_1hz": """
        CREATE TABLE IF NOT EXISTS samples_1hz (
            ts TEXT NOT NULL,
            trip_id TEXT NOT NULL,
            speed_kmh REAL NOT NULL,
            power_kw REAL NOT NULL,
            soc_pct REAL NOT NULL,
            pack_voltage_v REAL NOT NULL,
            pack_current_a REAL NOT NULL,
            pack_temp_c REAL NOT NULL,
            ambient_temp_c REAL NOT NULL,
            hvac_power_kw REAL NOT NULL,
            road_segment TEXT NOT NULL CHECK(road_segment IN ('city', 'highway')),
            adas_active INTEGER NOT NULL CHECK(adas_active IN (0, 1)),
            odometer_km REAL NOT NULL,
            energy_wh_per_km REAL,
            estimated_range_km REAL,
            PRIMARY KEY (ts, trip_id),
            FOREIGN KEY(trip_id) REFERENCES trips(trip_id)
        );
        CREATE INDEX IF NOT EXISTS idx_samples_1hz_ts ON samples_1hz(ts);
        CREATE INDEX IF NOT EXISTS idx_samples_1hz_trip_id ON samples_1hz(trip_id);
    """,
    # NOTE: samples_1min has no road_segment column, so segment comparisons (e.g. city vs highway) must read samples_1hz.
    "samples_1min": """
        CREATE TABLE IF NOT EXISTS samples_1min (
            ts TEXT PRIMARY KEY,
            avg_speed_kmh REAL NOT NULL,
            avg_power_kw REAL NOT NULL,
            avg_soc_pct REAL NOT NULL,
            avg_pack_temp_c REAL NOT NULL,
            avg_ambient_temp_c REAL NOT NULL,
            avg_hvac_power_kw REAL NOT NULL,
            energy_wh_per_km REAL,
            estimated_range_km REAL
        );
        CREATE INDEX IF NOT EXISTS idx_samples_1min_ts ON samples_1min(ts);
    """,
    "events": """
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('BATTERY_OVERHEAT_WARNING', 'ADAS_DISENGAGEMENT', 'LOW_SOC_WARNING')),
            severity TEXT NOT NULL,
            details_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
    """,
    "event_snapshots": """
        CREATE TABLE IF NOT EXISTS event_snapshots (
            event_id TEXT NOT NULL,
            offset_ms INTEGER NOT NULL,
            speed_kmh REAL NOT NULL,
            power_kw REAL NOT NULL,
            soc_pct REAL NOT NULL,
            pack_voltage_v REAL NOT NULL,
            pack_current_a REAL NOT NULL,
            pack_temp_c REAL NOT NULL,
            ambient_temp_c REAL NOT NULL,
            hvac_power_kw REAL NOT NULL,
            road_segment TEXT NOT NULL,
            adas_active INTEGER NOT NULL,
            PRIMARY KEY (event_id, offset_ms),
            FOREIGN KEY(event_id) REFERENCES events(event_id)
        );
        CREATE INDEX IF NOT EXISTS idx_event_snapshots_event_id ON event_snapshots(event_id);
    """,
}


# ---------------------------------------------------------------------------
# 3. Symbolic Time Range Specifications (SLM Output Representation)
# ---------------------------------------------------------------------------
# last_28_days, last_n_hours, and all_history exist to support edge temporal queries, full-dataset aggregation, and sub-day trend resolution.
class TimeRangeKind(str, Enum):
    yesterday = "yesterday"
    last_7_days = "last_7_days"
    last_14_days = "last_14_days"
    last_28_days = "last_28_days"
    last_n_days = "last_n_days"
    last_n_hours = "last_n_hours"
    all_history = "all_history"


class SymbolicTimeRange(BaseModel):
    """
    Symbolic representation of time spans produced by the SLM.
    These are deterministically translated to concrete ISO timestamps (start, end)
    by engine/dates.py against config.SIMULATED_NOW.
    """

    model_config = ConfigDict(extra="forbid")

    kind: TimeRangeKind = Field(
        ..., description="Named time range window relative to simulated now."
    )
    n: int | None = Field(
        default=None, description="Number of units when kind is 'last_n_days' or 'last_n_hours'."
    )

    @model_validator(mode="after")
    def validate_n_with_kind(self) -> "SymbolicTimeRange":
        if self.kind == TimeRangeKind.last_n_days:
            if self.n is None or not (1 <= self.n <= 28):
                raise ValueError(
                    "n is required and must be between 1 and 28 when kind is 'last_n_days'"
                )
        elif self.kind == TimeRangeKind.last_n_hours:
            if self.n is None or not (1 <= self.n <= 72):
                raise ValueError(
                    "n is required and must be between 1 and 72 when kind is 'last_n_hours'"
                )
        else:
            if self.n is not None:
                raise ValueError(f"n must be None when kind is '{self.kind.value}'")
        return self


# ---------------------------------------------------------------------------
# 4. SLM Tool Call Argument Schemas (Used for Grammar & Model Routing)
# ---------------------------------------------------------------------------
class GetEventsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: EventType = Field(..., description="The event type to search for.")
    range: SymbolicTimeRange = Field(..., description="Symbolic time window.")


class ContextBeforeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(
        ...,
        description="ID of the target event: 'EV-<digits>' (e.g. 'EV-0042') or 'latest_<EVENT_TYPE>'.",
    )
    seconds: int = Field(
        default=300,
        ge=1,
        le=300,
        description="Seconds of telemetry leading up to the event (<= 60 for 10Hz snapshot, <= 300 for 1Hz).",
    )
    signals: list[str] = Field(
        default_factory=list, description="Optional list of specific signals to return."
    )

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, v: str) -> str:
        if not EVENT_ID_REGEX.match(v):
            raise ValueError(
                f"Invalid event_id '{v}'. Must match pattern EV-<digits> (e.g. 'EV-0042') or "
                "'latest_(BATTERY_OVERHEAT_WARNING|ADAS_DISENGAGEMENT|LOW_SOC_WARNING)'."
            )
        return v

    @field_validator("signals")
    @classmethod
    def validate_signals(cls, v: list[str]) -> list[str]:
        for s in v:
            if s not in SNAPSHOT_SIGNALS:
                raise ValueError(
                    f"Invalid signal '{s}'. Must be one of: {sorted(SNAPSHOT_SIGNALS)}"
                )
        return v


class CompareArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Metric = Field(..., description="Metric to compare between segments.")
    range: SymbolicTimeRange = Field(..., description="Symbolic time window.")
    segment_a: Segment = Field(Segment.city, description="First segment.")
    segment_b: Segment = Field(Segment.highway, description="Second segment.")


class FindLargestChangeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Metric = Field(..., description="Metric to analyze for significant shift.")
    range: SymbolicTimeRange = Field(..., description="Symbolic time window.")
    direction: ChangeDirection = Field(
        ChangeDirection.drop, description="Direction of change ('drop' or 'rise')."
    )


class TrendArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Metric = Field(..., description="Metric to track over time.")
    range: SymbolicTimeRange = Field(..., description="Symbolic time window.")
    bucket: BucketSize = Field(
        BucketSize.day, description="Aggregation granularity ('hour', 'day', 'week')."
    )


class ExplainFactorsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Metric = Field(
        ...,
        description="Target metric whose change requires factor ranking (e.g. estimated_range_km).",
    )
    range: SymbolicTimeRange = Field(..., description="Symbolic time window.")


class OutOfScopeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(
        ..., description="Explanation of why the query cannot be answered with vehicle data."
    )


# NOTE: search_logs is optional per PROJECT_CONTEXT.md and disabled until an FTS5 table exists.
class SearchLogsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="Search query string.")
    range: SymbolicTimeRange = Field(..., description="Symbolic time window.")


# ---------------------------------------------------------------------------
# 5. Executable Tool Argument Schemas (After Date Resolution to ISO Timestamps)
# ---------------------------------------------------------------------------
class GetEventsExecArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: EventType
    start: str = Field(..., description="ISO 8601 start timestamp.")
    end: str = Field(..., description="ISO 8601 end timestamp.")


class ContextBeforeExecArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    seconds: int = Field(default=300, ge=1, le=300)
    signals: list[str] = Field(default_factory=list)

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, v: str) -> str:
        if not EVENT_ID_REGEX.match(v):
            raise ValueError(
                f"Invalid event_id '{v}'. Must match pattern EV-<digits> (e.g. 'EV-0042') or "
                "'latest_(BATTERY_OVERHEAT_WARNING|ADAS_DISENGAGEMENT|LOW_SOC_WARNING)'."
            )
        return v

    @field_validator("signals")
    @classmethod
    def validate_signals(cls, v: list[str]) -> list[str]:
        for s in v:
            if s not in SNAPSHOT_SIGNALS:
                raise ValueError(
                    f"Invalid signal '{s}'. Must be one of: {sorted(SNAPSHOT_SIGNALS)}"
                )
        return v


class CompareExecArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Metric
    start: str
    end: str
    segment_a: Segment = Segment.city
    segment_b: Segment = Segment.highway


class FindLargestChangeExecArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Metric
    start: str
    end: str
    direction: ChangeDirection = ChangeDirection.drop


class TrendExecArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Metric
    start: str
    end: str
    bucket: BucketSize = BucketSize.day


class ExplainFactorsExecArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Metric
    start: str
    end: str


class OutOfScopeExecArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str


# NOTE: search_logs is optional per PROJECT_CONTEXT.md and disabled until an FTS5 table exists.
class SearchLogsExecArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    start: str
    end: str


# ---------------------------------------------------------------------------
# 6. Canonical ToolCall & ToolOutput Containers
# ---------------------------------------------------------------------------
class ToolCall(BaseModel):
    """
    Format emitted by the SLM and validated prior to execution.
    Matches: {"tool": "<name>", "args": {...}}
    """

    model_config = ConfigDict(extra="forbid")

    tool: ToolName
    args: dict[str, Any]


class ToolOutputMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_count: int = Field(default=0, description="Total evidence rows evaluated.")
    query_time_ms: float = Field(
        default=0.0, description="Query execution duration in milliseconds."
    )


class ToolOutput(BaseModel):
    """
    Standard return contract for all query engine tools.
    """

    model_config = ConfigDict(extra="forbid")

    result: Any = Field(..., description="Computed metric, summary table, or structured finding.")
    evidence: list[Any] = Field(
        default_factory=list, description="IDs, timestamps, or raw rows backing the result."
    )
    meta: ToolOutputMeta = Field(default_factory=ToolOutputMeta)
