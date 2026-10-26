"""
engine/verify.py - Number-vs-Evidence Verifier for VehicleGPT.

Per PROJECT_CONTEXT.md Section 2, Principle 4:
  "A verifier checks that every number in the answer text appears in the
   tool output; otherwise fall back to the raw template answer."

Design:
- Extract all numeric literals from the rendered answer text using regex.
- For each number, check whether it appears anywhere in the JSON-serialized
  tool output (result + evidence).
- If any number is absent → verification fails; caller should discard the
  model explanation and use the deterministic template answer.
- "Appears in" uses string matching after normalizing both sides to a
  canonical float representation (trailing zeros removed, finite precision)
  so that "367.8" matches 367.8, 367.80, or 3.678e2 in the output.
- Small integers that are likely counts (e.g. "3 events", "2 warnings") are
  also verified unless they appear as part of a longer number in the output.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Matches: integers, decimals, optional sign, optional exponent.
# Examples matched: 42, 3.14, -0.5, 1.23e4, 367.8
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _normalize(raw: str) -> str:
    """Normalize a numeric string to a canonical float representation."""
    try:
        val = float(raw)
        # Represent with up to 6 significant digits, strip trailing zeros
        return f"{val:.6g}"
    except ValueError:
        return raw


def _extract_numbers(text: str) -> list[str]:
    """
    Extract all distinct numeric strings from rendered answer text.
    Returns list of canonical normalized values (may contain duplicates removed).
    """
    raw_matches = _NUMBER_RE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for m in raw_matches:
        norm = _normalize(m)
        if norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def _output_number_set(tool_output_dict: dict[str, Any]) -> set[str]:
    """
    Collect all numeric values that appear anywhere in the tool output dict
    (result + evidence), recursively, normalized to the same canonical form.
    """
    serialized = json.dumps(tool_output_dict, ensure_ascii=False)
    raw_matches = _NUMBER_RE.findall(serialized)
    return {_normalize(m) for m in raw_matches}


def verify(rendered_answer: str, tool_output_dict: dict[str, Any]) -> bool:
    """
    Check that every number in the rendered answer text is present in the
    tool output (result + evidence + meta).

    Args:
        rendered_answer: The fully rendered answer string (from render.py).
        tool_output_dict: The ToolOutput.model_dump() dict.

    Returns:
        True  → all numbers in the answer are backed by the tool output.
        False → at least one number in the answer cannot be found in the output
                (caller must fall back to the template answer).
    """
    answer_numbers = _extract_numbers(rendered_answer)
    if not answer_numbers:
        # No numbers in answer → nothing to verify; pass.
        return True

    output_numbers = _output_number_set(tool_output_dict)

    for num in answer_numbers:
        if num not in output_numbers:
            return False
    return True


def verify_or_fallback(
    rendered_answer: str,
    tool_output_dict: dict[str, Any],
    fallback: str,
) -> tuple[str, bool]:
    """
    Convenience wrapper: returns (chosen_answer, passed_verification).

    If verification passes, returns (rendered_answer, True).
    If it fails, returns (fallback, False) and the caller knows to log/flag it.
    """
    if verify(rendered_answer, tool_output_dict):
        return rendered_answer, True
    return fallback, False
