"""
eval/prompt_overlap.py - Word-overlap check between few-shot prompt examples
and every dev/stress question.

Overlap metric: Jaccard similarity of content-word sets (stopwords removed).
An example with overlap > 0.6 against any question is a contamination risk:
the model can match the example verbatim instead of routing the question.

Usage: python eval/prompt_overlap.py [--threshold 0.6]
"""

from __future__ import annotations

import json
import re
from pathlib import Path

THRESHOLD = 0.6

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "of",
    "in", "on", "to", "for", "and", "or", "my", "your", "it", "its", "this",
    "that", "what", "when", "how", "why", "show", "tell", "me", "i", "you",
    "can", "all", "between", "from", "have", "has", "had", "be", "been",
}

_FEWSHOT_EXAMPLES = [
    ("shots/tool_list/get_events", "get_events: list battery overheating warnings, "
     "ads disengagements, or low-charge warnings."),
    ("shots/example_1", "Show all ADAS disengagement events."),
    ("shots/example_2", "Why has my driving range reduced?"),
    ("shots/example_3", "When did energy efficiency show the largest drop?"),
    ("shots/example_4", "What happened before the battery overheating warning?"),
    ("shots/example_5", "Compare energy consumption between city and highway driving."),
    ("shots/example_6", "What is the capital of France?"),
    ("shots/example_7", "Turn off the air conditioning."),
]
_TOOL_TITLES = [
    ("tools_line_get_events", "get_events"),
    ("tools_line_context_before", "context_before"),
    ("tools_line_compare", "compare"),
    ("tools_line_find_largest_change", "find_largest_change"),
    ("tools_line_trend", "trend"),
    ("tools_line_explain_factors", "explain_factors"),
]


def content_words(text: str) -> set[str]:
    """Lowercased content-word set (words of 3+ chars, stopwords removed)."""
    words = re.findall(r"[a-z_/%-]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def main() -> None:
    eval_dir = Path(__file__).parent
    questions = []
    for fname in ("dev_questions.jsonl", "stress.jsonl"):
        with open(eval_dir / fname, encoding="utf-8") as f:
            questions.extend(
                (json.loads(line)["id"], json.loads(line)["question"])
                for line in f if line.strip()
            )

    contaminated = []
    print(f"{'pair':<58} {'jaccard':>7}")
    print("-" * 68)
    for q_id, q_text in questions:
        for ex_key, ex_text in _FEWSHOT_EXAMPLES:
            ov = jaccard(content_words(ex_text), content_words(q_text))
            flag = " <-- REMOVE/REWORD" if ov > THRESHOLD else ""
            if ov > 0.2 or ov > THRESHOLD:
                print(f"{q_id:12} vs {ex_key:44} {ov:7.3f}{flag}")
            if ov > THRESHOLD:
                contaminated.append((q_id, ex_key, round(ov, 3)))

    print("\n" + "=" * 68)
    print("tool/tool-list lines (informational, they are vocab not examples):")
    for t_key, t_text in _TOOL_TITLES:
        max_ov, max_q = 0.0, ""
        for q_id, q_text in questions:
            ov = jaccard(content_words(t_text), content_words(q_text))
            if ov > max_ov:
                max_ov, max_q = ov, q_id
        print(f"{t_key:52} max={max_ov:.3f} vs {max_q}")

    print("\n" + "=" * 68)
    if contaminated:
        print(f"CONTAMINATED pairs (above {THRESHOLD}): {len(contaminated)}")
        for q_id, ex_key, ov in contaminated:
            print(f"  {q_id} vs {ex_key}: {ov}")
    else:
        print(f"No few-shot example exceeds {THRESHOLD} overlap with any question.")


if __name__ == "__main__":
    main()
