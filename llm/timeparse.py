"""
llm/timeparse.py - Deterministic Natural-Language Time Parser for VehicleGPT.

Overrides the model's range argument with deterministic rules:
- "yesterday" -> {"kind": "yesterday"}
- "last 7 days" / "past week" -> {"kind": "last_7_days"}
- "last 14 days" / "last two weeks" -> {"kind": "last_14_days"}
- "last 28 days" / "past month" -> {"kind": "last_28_days"}
- "last N days" -> {"kind": "last_n_days", "n": N}
- "last N hours" -> {"kind": "last_n_hours", "n": N}
- Default when no explicit time expression is found -> {"kind": "all_history"}

Logs any disagreement when model's range differs from the parser's range.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Patterns for deterministic recognition
_YESTERDAY_PATTERN = re.compile(r"\byesterday\b", re.IGNORECASE)

_LAST_14_DAYS_PATTERN = re.compile(
    r"\b(last|past|over\s+the\s+last|over\s+the\s+past)\s+(14\s+days|two\s+weeks|2\s+weeks)\b",
    re.IGNORECASE,
)

_LAST_7_DAYS_PATTERN = re.compile(
    r"\b(last|past|over\s+the\s+last|over\s+the\s+past)\s+(7\s+days|seven\s+days|one\s+week|1\s+week|week)\b",
    re.IGNORECASE,
)

_LAST_28_DAYS_PATTERN = re.compile(
    r"\b(last|past|over\s+the\s+last|over\s+the\s+past)\s+(28\s+days|four\s+weeks|4\s+weeks|month|one\s+month|1\s+month)\b",
    re.IGNORECASE,
)

_LAST_N_DAYS_PATTERN = re.compile(
    r"\b(last|past|over\s+the\s+last|over\s+the\s+past)\s+(\d+)\s+days\b",
    re.IGNORECASE,
)

_LAST_N_HOURS_PATTERN = re.compile(
    r"\b(last|past|over\s+the\s+last|over\s+the\s+past)\s+(\d+)\s+hours?\b",
    re.IGNORECASE,
)

_ALL_HISTORY_PATTERN = re.compile(
    r"\b(all\s+time|all\s+history|entire\s+history|overall|ever)\b",
    re.IGNORECASE,
)


def parse_time_range(
    question: str,
    model_range: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """
    Deterministically parses a time range from natural language question text.

    Args:
        question: The user's query text.
        model_range: Optional range dictionary proposed by the SLM.

    Returns:
        tuple (parsed_range, disagreement_logged):
            parsed_range: The authoritative SymbolicTimeRange dict.
            disagreement_logged: True if model_range differed and was overridden.
    """
    q = question.strip()

    parsed_range: dict[str, Any]

    if _YESTERDAY_PATTERN.search(q):
        parsed_range = {"kind": "yesterday"}
    elif _LAST_14_DAYS_PATTERN.search(q):
        parsed_range = {"kind": "last_14_days"}
    elif _LAST_7_DAYS_PATTERN.search(q):
        parsed_range = {"kind": "last_7_days"}
    elif _LAST_28_DAYS_PATTERN.search(q):
        parsed_range = {"kind": "last_28_days"}
    else:
        m_days = _LAST_N_DAYS_PATTERN.search(q)
        if m_days:
            n = int(m_days.group(2))
            if n == 7:
                parsed_range = {"kind": "last_7_days"}
            elif n == 14:
                parsed_range = {"kind": "last_14_days"}
            elif n == 28:
                parsed_range = {"kind": "last_28_days"}
            else:
                parsed_range = {"kind": "last_n_days", "n": n}
        else:
            m_hours = _LAST_N_HOURS_PATTERN.search(q)
            if m_hours:
                n_h = int(m_hours.group(2))
                parsed_range = {"kind": "last_n_hours", "n": n_h}
            elif _ALL_HISTORY_PATTERN.search(q):
                parsed_range = {"kind": "all_history"}
            else:
                # Default when no explicit time expression is found
                parsed_range = {"kind": "all_history"}

    disagreement = False
    if model_range is not None and model_range != parsed_range:
        disagreement = True
        logger.info(
                "Deterministic time parser override: model range %s disagreed with parsed range %s for query '%s'",
                model_range,
                parsed_range,
                question,
            )

    return parsed_range, disagreement
