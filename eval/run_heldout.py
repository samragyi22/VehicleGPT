"""
eval/run_heldout.py - Scoring runner for the user's held-out question set.

Usage: python eval/run_heldout.py PATH

PATH is a JSONL file with one object per line:
    {"q": "...", "expect_status": "...", "tool": "..." or null, "args": {...} or null}

Runs the FULL pipeline (guard -> gate -> time parser -> grammar-constrained
model -> validation -> grounding -> tools -> render -> verify) exactly once
per question. No tuning against the file: results are reported as they come.

Per question it prints expected vs actual (status/tool/args). Totals separate:
- vehicle questions (expect_status "ok"): tool+args exact, tool-only, wrong,
  false refusal, other;
- refusal questions (any other expect_status): handled by the deterministic
  guard/gate/time-parser layers, pass or fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine.pipeline import run_pipeline


def _args_match(actual: dict | None, expected: dict | None) -> bool:
    """True if every non-null expected key matches the actual tool args."""
    if not expected:
        return True
    actual = actual or {}
    for key, want in expected.items():
        if key not in actual:
            return False
        got = actual[key]
        if isinstance(want, dict):
            if not isinstance(got, dict):
                return False
            for sub_key, sub_want in want.items():
                if sub_key not in got or str(got[sub_key]).lower() != str(sub_want).lower():
                    return False
        elif str(got).lower() != str(want).lower():
            return False
    return True


def run_heldout(path: Path, model_path: Path | None = None) -> None:
    questions: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"\nHeld-out set: {path} ({len(questions)} questions)")
    print("Model pipeline: guard -> gate -> time parser -> grammar-constrained model "
          "-> validation -> tools -> render -> verify\n")

    vehicle = {  # classification counters
        "exact": 0, "tool_only": 0, "wrong": 0,
        "false_refusal": 0, "other": 0,
    }
    ref_pass = ref_fail = 0
    rows = []

    for i, item in enumerate(questions, start=1):
        q = item.get("q", "")
        expect_status = item.get("expect_status", "ok")
        expect_tool = item.get("tool")
        expect_args = item.get("args") or {}

        res = run_pipeline(q, model_path=model_path, with_explanation=False)
        act_tool = res.tool_call.get("tool") if res.tool_call else None
        act_args = res.tool_call.get("args", {}) if res.tool_call else {}

        if expect_status != "ok":
            # Deterministic refusal: handled by guard/gate/time parser, not the model.
            ok = res.status == expect_status
            detail = f"status={res.status}"
            if ok:
                ref_pass += 1
            else:
                ref_fail += 1
            print(f"[{'OK' if ok else 'X'}] #{i} expected={expect_status:18} {detail:28} | {q[:52]}")
            rows.append({"i": i, "ok": ok, "cls": "refusal", "status": res.status})
            continue

        # Vehicle question: full classification.
        if res.status != "ok":
            # Guard/gate/time-parser refused a question the user expected to be answered.
            if res.status in ("control_refused", "off_topic", "unsupported_time", "ambiguous_metric", "out_of_scope"):
                vehicle["false_refusal"] += 1
                cls = "FALSE_REFUSAL"
            else:
                vehicle["other"] += 1
                cls = "OTHER"
            detail = f"status={res.status}"
        elif res.tool_call is None or act_tool != expect_tool:
            vehicle["wrong"] += 1
            cls, detail = "WRONG", f"tool={act_tool}"
        elif _args_match(act_args, expect_args):
            vehicle["exact"] += 1
            cls, detail = "EXACT", ""
        else:
            vehicle["tool_only"] += 1
            cls, detail = "TOOL_ONLY", f"args={act_args}"

        print(f"[{cls}] #{i} expected={expect_tool or '-':18} {detail:28} | {q[:52]}")
        rows.append({"i": i, "cls": cls, "tool": act_tool, "args": act_args,
                     "status": res.status, "answer": res.answer})

    n_vehicle = sum(vehicle.values())
    n_ref = ref_pass + ref_fail
    print("\n" + "=" * 64)
    print(f"Vehicle questions ({n_vehicle}): "
          f"exact={vehicle['exact']}, tool_only={vehicle['tool_only']}, "
          f"wrong={vehicle['wrong']}, false_refusal={vehicle['false_refusal']}, "
          f"other={vehicle['other']}")
    if n_vehicle:
        print(f"Vehicle exact-match accuracy: {vehicle['exact']}/{n_vehicle} "
              f"({100.0 * vehicle['exact'] / n_vehicle:.1f}%)")
    print(f"Refusal questions ({n_ref}): pass={ref_pass}, fail={ref_fail} "
          "(guard/gate/time parser; not model accuracy)")

    # Save machine-readable results next to nothing the model tuned on:
    out = path.with_suffix(".results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"file": str(path), "vehicle": vehicle,
                   "refusals": {"pass": ref_pass, "fail": ref_fail},
                   "rows": rows}, f, indent=2)
    print(f"Raw results saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL file of held-out questions")
    parser.add_argument("--model", default=None, help="Optional GGUF model path override")
    args = parser.parse_args()
    if not args.path.exists():
        print(f"Error: file not found: {args.path}", file=sys.stderr)
        sys.exit(1)
    run_heldout(args.path, model_path=Path(args.model) if args.model else None)


if __name__ == "__main__":
    main()
