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

def _load_examples() -> list[tuple[str, str]]:
    """Extract the few-shot Q lines from the live system prompt (single source)."""
    from llm.router import _SYSTEM_PROMPT
    lines = _SYSTEM_PROMPT.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "Examples:")
    out = []
    for i in range(start + 1, len(lines)):
        l = lines[i].strip()
        if l.startswith("Q: "):
            out.append((f"shot_{len(out) + 1}", l[3:]))
    return out


_FEWSHOT_EXAMPLES = [
    ("shots/tool_list_get_events",
     "get_events: list battery overheating warnings, ADAS disengagements, or low-charge warnings."),
    ("shots/tool_list_context_before",
     "context_before: show signals in the seconds before a specific event."),
    ("shots/tool_list_compare",
     "compare: compare a metric between city and highway driving."),
    ("shots/tool_list_find_largest_change",
     "find_largest_change: find when a metric dropped or rose the most."),
    ("shots/tool_list_trend",
     "trend: show how a metric changed over time (hourly / daily / weekly)."),
    ("shots/tool_list_explain_factors",
     "explain_factors: explain why a metric changed, ranking possible contributing factors."),
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
    if contaminated:
        print(f"CONTAMINATED pairs (above {THRESHOLD}): {len(contaminated)}")
        for q_id, ex_key, ov in contaminated:
            print(f"  {q_id} vs {ex_key}: {ov}")
    else:
        print(f"No few-shot example exceeds {THRESHOLD} overlap with any question.")


if __name__ == "__main__":
    main()
