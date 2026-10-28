"""
llm/timeparse.py - Deterministic Natural-Language Time Parser for VehicleGPT.

Overrides the model's range argument with deterministic rules:
- "yesterday"                     -> {kind: yesterday}
- "last 7 days" / "past week" / "this week" -> {kind: last_7_days}
- "last 14 days" / "last 2 weeks" -> {kind: last_14_days}
- "last 28 days" / "last month" -> {kind: last_28_days}
- "past fortnight" / "fortnight" (14d, British, well-defined) -> {kind: last_14_days}
- "last N days"                   -> {kind: last_n_days, n: N}
- "last N hours"                  -> {kind: last_n_hours, n: N}
- Explicit "all history / all time / overall / ever" -> {kind: all_history}
- Unresolvable temporal phrase (e.g. "two days ago", "N weeks ago",
  "since last Tuesday", "current week")
  -> returns None with status "unsupported_time"; the pipeline should
     present the CLARIFICATION_MESSAGE to the user.
- No time phrase at all           -> {kind: all_history}

interpretation_text(question, parsed_range) builds the human-readable
statement of the chosen window (e.g. "Interpreted 'this week' as the
last 7 days (21 Oct to 28 Oct).").

Logs any disagreement when model's range differs from the parser's range.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Friendly message shown to users when a time phrase cannot be resolved.
UNSUPPORTED_TIME_MESSAGE = (
    "I recognised a time reference in your question but could not map it to "
    "a supported window (yesterday, last N days/hours, last 7/14/28 days, or all history). "
    "Please rephrase using one of those, e.g. 'last 7 days' or 'yesterday'."
)

# ---------------------------------------------------------------------------
# Compiled patterns — order matters: check longer/more-specific patterns first
# ---------------------------------------------------------------------------

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

# Explicit "all history" phrase
_ALL_HISTORY_PATTERN = re.compile(
    r"\b(all\s+time|all\s+history|entire\s+history|overall|ever)\b",
    re.IGNORECASE,
)

# "this week" is a supported synonym for the last 7 days (now deterministic).
_THIS_WEEK_PATTERN = re.compile(r"\bthis\s+week\b", re.IGNORECASE)

# Unresolvable temporal phrases — present but cannot be mapped to a kind.
# "fortnight" maps to 14 days (British English, well-defined).
_FORTNIGHT_PATTERN = re.compile(r"\bfortnight\b", re.IGNORECASE)

# Phrases that signal a relative time reference we cannot resolve deterministically.
_UNRESOLVABLE_PATTERN = re.compile(
    r"\b("
    r"\d+\s+days?\s+ago"            # "two days ago", "3 days ago"
    r"|\bago\b"                      # any "N ago" form ("N weeks ago")
    r"|since\s+last\s+\w+"          # "since last Tuesday"
    r"|since\s+(yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"  # weekday names
    r"|this\s+(month|year)"         # "this month" / "this year"
    r"|this\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|two\s+days?\s+ago"
    r"|three\s+days?\s+ago"
    r"|current\s+week"
    r")",
    re.IGNORECASE,
)


def parse_time_range(
    question: str,
    model_range: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """
    Deterministically parse a time range from natural language.

    Args:
        question:    The user's query text.
        model_range: Optional range dictionary proposed by the SLM.

    Returns:
        (parsed_range, disagreement_logged):
            parsed_range:        Authoritative SymbolicTimeRange dict, or None when
                                 an unresolvable temporal phrase was found
                                 (status = "unsupported_time").
            disagreement_logged: True when model_range differed and was overridden.

    Callers MUST check for None and return UNSUPPORTED_TIME_MESSAGE to the user
    without calling any tool.
    """
    q = question.strip()

    # --- Tier 1: Exact supported ranges ---

    if _YESTERDAY_PATTERN.search(q):
        parsed_range: dict[str, Any] | None = {"kind": "yesterday"}

    elif _LAST_14_DAYS_PATTERN.search(q):
        parsed_range = {"kind": "last_14_days"}

    elif _LAST_7_DAYS_PATTERN.search(q):
        parsed_range = {"kind": "last_7_days"}

    elif _LAST_28_DAYS_PATTERN.search(q):
        parsed_range = {"kind": "last_28_days"}

    elif _THIS_WEEK_PATTERN.search(q):
        parsed_range = {"kind": "last_7_days"}

    elif _FORTNIGHT_PATTERN.search(q):
        # "fortnight" = 14 days — well-defined, treat as last_14_days
        parsed_range = {"kind": "last_14_days"}

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

            # --- Tier 2: Unresolvable temporal phrase ---
            elif _UNRESOLVABLE_PATTERN.search(q):
                # A time phrase was found but we cannot map it to a supported kind.
                logger.info(
                    "Unsupported temporal phrase in query '%s'; returning unsupported_time.",
                    question,
                )
                parsed_range = None  # signals unsupported_time to pipeline

            else:
                # No time phrase at all → default to all_history
                parsed_range = {"kind": "all_history"}

    disagreement = False
    if model_range is not None and model_range != parsed_range:
        disagreement = True
        logger.info(
            "Deterministic time parser override: model range %s disagreed with parsed range %s "
            "for query '%s'",
            model_range,
            parsed_range,
            question,
        )

    return parsed_range, disagreement


# Kind -> human-readable window description (for interpretation statements).
_KIND_DESCRIPTIONS: dict[str, str] = {
    "yesterday": "yesterday",
    "last_7_days": "the last 7 days",
    "last_14_days": "the last 14 days",
    "last_28_days": "the last 28 days",
    "all_history": "all history",
}


def interpretation_text(question: str, parsed_range: dict[str, Any] | None) -> str | None:
    """
    Builds the statement of the chosen time window shown with every answer, e.g.
    "Interpreted 'this week' as the last 7 days (21 Oct to 28 Oct)."

    Returns None only when parsed_range is None (clarification path).
    """
    if parsed_range is None:
        return None

    q = question.strip()
    phrase: str | None = None
    phrase_pattern: re.Pattern[str] | None = None

    if _YESTERDAY_PATTERN.search(q):
        phrase_pattern = _YESTERDAY_PATTERN
    elif _LAST_14_DAYS_PATTERN.search(q):
        phrase_pattern = _LAST_14_DAYS_PATTERN
    elif _LAST_7_DAYS_PATTERN.search(q):
        phrase_pattern = _LAST_7_DAYS_PATTERN
    elif _LAST_28_DAYS_PATTERN.search(q):
        phrase_pattern = _LAST_28_DAYS_PATTERN
    elif _THIS_WEEK_PATTERN.search(q):
        phrase_pattern = _THIS_WEEK_PATTERN
    elif _FORTNIGHT_PATTERN.search(q):
        phrase_pattern = _FORTNIGHT_PATTERN
    elif _ALL_HISTORY_PATTERN.search(q):
        phrase_pattern = _ALL_HISTORY_PATTERN

    if phrase_pattern is not None:
        phrase = phrase_pattern.search(q).group(0)  # type: ignore[union-attr]

    kind = parsed_range.get("kind", "all_history")
    desc = _KIND_DESCRIPTIONS.get(kind)
    if desc is None and kind == "last_n_days":
        desc = f"the last {parsed_range.get('n', 1)} days"
    if desc is None and kind == "last_n_hours":
        desc = f"the last {parsed_range.get('n', 1)} hours"
    if desc is None:
        desc = kind

    from engine.dates import resolve_symbolic_range
    start_iso, end_iso = resolve_symbolic_range(parsed_range)
    start_dt = datetime.fromisoformat(start_iso)
    end_dt = datetime.fromisoformat(end_iso)

    if phrase is not None:
        return (
            f"Interpreted '{phrase}' as {desc} "
            f"({start_dt.strftime('%d %b')} to {end_dt.strftime('%d %b')})."
        )
    return (
        f"No time phrase found; interpreting as {desc} "
        f"({start_dt.strftime('%d %b')} to {end_dt.strftime('%d %b')})."
    )
