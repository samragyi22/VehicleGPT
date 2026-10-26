"""
eval/run_dev.py - Evaluation Runner for VehicleGPT 40-Question Dev Set.

Scores:
1. Tool choice accuracy
2. Key argument accuracy
3. Pipeline status accuracy
4. Per-stage timing statistics (median and p95 for each stage)
5. Prompt token counts
6. Peak RSS memory usage (MB)
7. Full failure report with expected vs actual diffs
"""

from __future__ import annotations

import json
import logging
import platform
import resource
import statistics
import sys
from pathlib import Path
from typing import Any

from engine.pipeline import run_pipeline

logging.basicConfig(level=logging.ERROR)


def get_peak_rss_mb() -> float:
    """Returns peak RSS in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def _match_args(actual_args: dict[str, Any], expected_args: dict[str, Any]) -> bool:
    """Verifies that actual tool arguments satisfy expected key arguments."""
    if not expected_args:
        return True

    for k, exp_v in expected_args.items():
        if k not in actual_args:
            return False
        act_v = actual_args[k]

        if isinstance(exp_v, dict):
            if not isinstance(act_v, dict):
                return False
            for sub_k, sub_exp in exp_v.items():
                if sub_k not in act_v:
                    return False
                if str(act_v[sub_k]).lower() != str(sub_exp).lower():
                    return False
        else:
            if str(act_v).lower() != str(exp_v).lower():
                return False

    return True


def run_evaluation(dev_file: Path | str | None = None) -> dict[str, Any]:
    file_path = Path(dev_file) if dev_file else Path(__file__).parent / "dev_questions.jsonl"
    if not file_path.exists():
        print(f"Error: dev questions file not found at {file_path}")
        sys.exit(1)

    questions: list[dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print("\n========================================================")
    print(f"  VehicleGPT Dev Set Evaluation ({len(questions)} questions)")
    print("========================================================\n")

    results = []
    stage_timings: dict[str, list[float]] = {
        "guard_ms": [],
        "gate_ms": [],
        "model_ms": [],
        "validation_ms": [],
        "timeparse_ms": [],
        "query_ms": [],
        "render_ms": [],
        "verify_ms": [],
        "total_ms": [],
    }
    prompt_tokens_list: list[int] = []
    failures = []

    for idx, item in enumerate(questions, start=1):
        q_id = item["id"]
        q_text = item["question"]
        exp_tool = item["expected_tool"]
        exp_args = item.get("expected_args", {})
        exp_status = item["expected_status"]

        # Run pipeline
        res = run_pipeline(q_text, with_explanation=False)

        act_tool = res.tool_call.get("tool") if res.tool_call else "none"
        act_args = res.tool_call.get("args", {}) if res.tool_call else {}
        act_status = res.status

        tool_correct = (act_tool == exp_tool)
        args_correct = _match_args(act_args, exp_args) if tool_correct else False
        status_correct = (act_status == exp_status)

        # Collect metrics
        for stage, val in res.timings.items():
            if stage in stage_timings:
                stage_timings[stage].append(val)

        p_tok = res.tokens.get("prompt_tokens", 0)
        if p_tok > 0:
            prompt_tokens_list.append(p_tok)

        is_success = tool_correct and args_correct and status_correct
        if not is_success:
            failures.append({
                "id": q_id,
                "question": q_text,
                "expected": {
                    "tool": exp_tool,
                    "args": exp_args,
                    "status": exp_status,
                },
                "actual": {
                    "tool": act_tool,
                    "args": act_args,
                    "status": act_status,
                    "answer_preview": res.answer[:100],
                },
                "failures": {
                    "tool": not tool_correct,
                    "args": not args_correct,
                    "status": not status_correct,
                },
            })

        symbol = "✓" if is_success else "✗"
        print(f"[{symbol}] {q_id}: {q_text[:45]:<45} | tool={act_tool:<19} status={act_status:<15} ({res.timings.get('total_ms', 0):.1f}ms)")

        results.append({
            "id": q_id,
            "tool_correct": tool_correct,
            "args_correct": args_correct,
            "status_correct": status_correct,
            "success": is_success,
        })

    # Summary calculations
    total = len(results)
    tool_acc = (sum(r["tool_correct"] for r in results) / total) * 100.0
    args_acc = (sum(r["args_correct"] for r in results) / total) * 100.0
    status_acc = (sum(r["status_correct"] for r in results) / total) * 100.0
    peak_rss = get_peak_rss_mb()

    # Percentiles helper
    def p95(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        k = round(0.95 * (len(s) - 1))
        return s[k]

    print("\n" + "=" * 60)
    print("                EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Questions Evaluated:  {total}")
    print(f"Tool Choice Accuracy:       {tool_acc:.1f}% ({sum(r['tool_correct'] for r in results)}/{total})")
    print(f"Arguments Accuracy:         {args_acc:.1f}% ({sum(r['args_correct'] for r in results)}/{total})")
    print(f"Status Accuracy:            {status_acc:.1f}% ({sum(r['status_correct'] for r in results)}/{total})")
    print(f"Peak RSS Memory:            {peak_rss:.1f} MB")

    if prompt_tokens_list:
        print(f"Prompt Token Count:         median={int(statistics.median(prompt_tokens_list))}, min={min(prompt_tokens_list)}, max={max(prompt_tokens_list)}")

    print("\n" + "-" * 60)
    print("         STAGE LATENCY BREAKDOWN (ms)")
    print("-" * 60)
    print(f"{'Stage':<18} | {'Median (ms)':>12} | {'p95 (ms)':>10} | {'Max (ms)':>10}")
    print("-" * 60)
    for stage, vals in stage_timings.items():
        if vals:
            med = statistics.median(vals)
            perc95 = p95(vals)
            mx = max(vals)
            print(f"{stage:<18} | {med:>12.2f} | {perc95:>10.2f} | {mx:>10.2f}")
    print("-" * 60)

    if failures:
        print("\n" + "=" * 60)
        print(f"  FAILURES DETECTED ({len(failures)} total):")
        print("=" * 60)
        for f in failures:
            print(f"\n[{f['id']}] Q: \"{f['question']}\"")
            print(f"  Expected: tool={f['expected']['tool']} status={f['expected']['status']} args={f['expected']['args']}")
            print(f"  Actual:   tool={f['actual']['tool']} status={f['actual']['status']} args={f['actual']['args']}")
            print(f"  Answer:   {f['actual']['answer_preview']}")
    else:
        print("\nAll 40 questions passed evaluation successfully! (0 failures)")

    return {
        "total": total,
        "tool_accuracy_pct": round(tool_acc, 2),
        "args_accuracy_pct": round(args_acc, 2),
        "status_accuracy_pct": round(status_acc, 2),
        "peak_rss_mb": round(peak_rss, 2),
        "failures": failures,
    }


if __name__ == "__main__":
    run_evaluation()
