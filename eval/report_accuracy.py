"""
eval/report_accuracy.py - Honest accuracy reporting for dev and stress sets.

Separates model accuracy from deterministic refusals:
- Refusal questions (expected_status != "ok") are handled by the guard, gate,
  or time parser -- they never reach the model, so they are NOT counted as
  model accuracy. They are scored as pass/fail of the safety layers.
- Vehicle questions (expected_status == "ok") are scored as one of:
  tool_correct (tool AND args AND status all correct), wrong (answered with a
  wrong tool/args), false_refusal (deterministic layers refused a valid
  vehicle question), partial (two-part question: first part answered).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from engine.pipeline import run_pipeline
from eval.run_dev import _match_args

MODELS = {
    "0.5b": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "1.5b": "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
}
TIME_REFUSAL_STATUSES = {"unsupported_time", "ambiguous_metric"}
DETERMINISTIC_REFUSALS = {"control_refused", "off_topic"}


def load_questions(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def run_report(paths: list[Path], model_key: str) -> dict:
    model_path = Path(MODELS[model_key])
    questions = load_questions(paths)
    vehicle = [q for q in questions if q["expected_status"] == "ok"]
    refusals = [q for q in questions if q["expected_status"] != "ok"]

    vehicle_rows = []
    refusal_pass = refusal_fail = 0
    latencies = []

    for q in refusals:
        res = run_pipeline(q["question"], model_path=model_path, with_explanation=False)
        if res.status == q["expected_status"]:
            refusal_pass += 1
        else:
            refusal_fail += 1
            print(f"  [X] {q['id']}: expected {q['expected_status']}, got {res.status}")

    for q in vehicle:
        res = run_pipeline(q["question"], model_path=model_path, with_explanation=False)
        latencies.append(res.timings.get("total_ms", 0))
        act_tool = res.tool_call.get("tool") if res.tool_call else "none"
        act_args = res.tool_call.get("args", {}) if res.tool_call else {}
        tool_ok = act_tool == q["expected_tool"]
        args_ok = tool_ok and _match_args(act_args, q.get("expected_args", {}))

        if res.status in DETERMINISTIC_REFUSALS or res.status in TIME_REFUSAL_STATUSES:
            cls, detail = "false_refusal", f"{res.status}"
        elif res.status == "out_of_scope":
            # Gate passed, model refused: false refusal per plan Step 4.
            cls, detail = "false_refusal", "model returned out_of_scope"
        elif res.status == "ok" and tool_ok and args_ok:
            # Two-part question answered fully counts only as partial, per plan:
            # the second part must be answered by a separate question.
            cls = "partial" if q.get("category") == "combined_two_tools" else "tool_correct"
            detail = "two_part_answered_part_1" if cls == "partial" else ""
        elif res.status == "ok" and tool_ok:
            cls = "args_wrong"
            detail = f"args={act_args}"
        elif res.status == "ok":
            cls = "wrong"
            detail = f"tool={act_tool} args={act_args}"
        else:
            cls, detail = "pipeline_error", f"status={res.status} tool={act_tool}"

        symbol = "OK" if cls == "tool_correct" else "X"
        print(f"  [{symbol}] {q['id']}: {cls:14} | {act_tool:<18} | {detail}")
        vehicle_rows.append({"id": q["id"], "cls": cls})

    by_cls: dict[str, int] = {}
    for r in vehicle_rows:
        by_cls[r["cls"]] = by_cls.get(r["cls"], 0) + 1
    n = len(vehicle_rows)
    print(f"\n  Vehicle questions ({n}): "
          f"tool_correct={by_cls.get('tool_correct',0)}, args_wrong={by_cls.get('args_wrong',0)}, "
          f"wrong={by_cls.get('wrong',0)}, false_refusal={by_cls.get('false_refusal',0)}, "
          f"partial={by_cls.get('partial',0)}, pipeline_error={by_cls.get('pipeline_error',0)}")
    if latencies:
        print(f"  Vehicle latency: median={statistics.median(latencies):.0f}ms, max={max(latencies):.0f}ms")
    print(f"  Refusal questions: pass={refusal_pass}, fail={refusal_fail} (deterministic layers, not model)")

    return {
        "model": model_key,
        "vehicle": {"n": n, **{k: v for k, v in by_cls.items()}},
        "refusals_pass": refusal_pass,
        "refusals_fail": refusal_fail,
        "rows": vehicle_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["0.5b"], choices=list(MODELS))
    parser.add_argument("--files", nargs="+", default=["dev", "stress"])
    args = parser.parse_args()

    file_map = {
        "dev": Path(__file__).parent / "dev_questions.jsonl",
        "stress": Path(__file__).parent / "stress.jsonl",
    }
    paths = [file_map[f] for f in args.files]

    for model_key in args.models:
        print(f"\n=== Model {model_key} ===")
        run_report(paths, model_key)


if __name__ == "__main__":
    main()
