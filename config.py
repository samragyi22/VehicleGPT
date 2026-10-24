"""
config.py - Central configuration for VehicleGPT.

All constants, paths, resource limits, and simulation parameters are defined
here to ensure strictly reproducible behavior across testing, evaluation,
and edge benchmarking.
"""

from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Simulated Time & Determinism
# ---------------------------------------------------------------------------
# Fixed reference timestamp for deterministic date resolution ("now", "yesterday", etc.)
SIMULATED_NOW_STR = "2026-10-28 18:00:00"
SIMULATED_NOW = datetime.fromisoformat(SIMULATED_NOW_STR)
SIMULATED_NOW_ISO = SIMULATED_NOW.isoformat()

# Fixed random seed for synthetic data generation and reproducible fault scenarios
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# History & Simulation Limits
# ---------------------------------------------------------------------------
HISTORY_DAYS = 28
PRE_EVENT_WINDOW_SECONDS = 60
PRE_EVENT_FREQUENCY_HZ = 10
DRIVING_FREQUENCY_HZ = 1

# ---------------------------------------------------------------------------
# Query & Security Limits
# ---------------------------------------------------------------------------
# Maximum rows returned by any query tool to prevent memory exhaustion
ROW_LIMIT = 200

# Strict timeout for deterministic database operations (seconds)
QUERY_TIMEOUT_SECONDS = 2.0

# ---------------------------------------------------------------------------
# Target Edge Environment (Emulated Edge Specs)
# ---------------------------------------------------------------------------
EDGE_MAX_RAM_GB = 2.0
EDGE_CPUS = 2
LATENCY_TARGET_SECONDS = 2.0

# ---------------------------------------------------------------------------
# Filesystem Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "eval"

# SQLite Database & Ground Truth files
DB_PATH = DATA_DIR / "vehicle.db"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"

# Models
DEFAULT_MODEL_PATH = MODELS_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
COMPARISON_MODEL_PATH = MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"


def get_readonly_db_uri(db_path: Path | str = DB_PATH) -> str:
    """
    Constructs an absolute, read-only SQLite URI.
    Ensures the query engine cannot write or mutate the database by construction.
    """
    resolved_path = Path(db_path).resolve()
    return f"file:{resolved_path}?mode=ro"
