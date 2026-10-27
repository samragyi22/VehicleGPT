"""
eval/measure_validity.py - Grammar validity and dev/stress set evaluation.

Measures:
  Part A: Hash check (data/raw/*.parquet) and ruff lint status.
  Part B: Grammar validity - 200 generations at temperature 0.7, count out-of-enum values.
  Part C: Dev set (40 questions) and stress set (15 questions) on both models.
          Reports: wrong, correct_refusal, false_refusal, partial, unsupported_time per model.

Usage:
    .venv/bin/python eval/measure_validity.py [--model 0.5b|1.5b|both] [--skip-validity]

Output: plain text table to stdout + results saved to eval/validity_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from llm.grammar import OUT_OF_SCOPE_REASONS, _enum, build_gbnf
from schema import BucketSize, ChangeDirection, EventType, Metric, Segment, ToolName

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TOOLS = {t.value for t in ToolName if t != ToolName.search_logs}
VALID_METRICS = set(_enum(Metric))
VALID_EVENT_TYPES = set(_enum(EventType))
VALID_DIRECTIONS = set(_enum(ChangeDirection))
VALID_BUCKETS = set(_enum(BucketSize))
VALID_SEGMENTS = set(_enum(Segment))
VALID_REASONS = set(OUT_OF_SCOPE_REASONS)

PARQUET_DIR = config.DATA_DIR / "raw"
DEV_FILE = config.EVAL_DIR / "dev_questions.jsonl"
STRESS_FILE = config.EVAL_DIR / "stress.jsonl"
RESULTS_FILE = config.EVAL_DIR / "validity_results.json"

MODEL_PATHS = {
    "0.5b": config.DEFAULT_MODEL_PATH,
    "1.5b": config.COMPARISON_MODEL_PATH,
}

# ---------------------------------------------------------------------------
# Part A: Hash check
# ---------------------------------------------------------------------------

def hash_parquets() -> dict[str, str]:
    hashes = {}
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        hashes[p.name] = h
    return hashes


def run_ruff() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        check=False, capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


# ---------------------------------------------------------------------------
# Part B: Grammar validity (200 generations at temperature 0.7)
# ---------------------------------------------------------------------------

def _parse_tool_call(raw: str) -> dict | None:
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def _check_enum_violations(call: dict) -> list[str]:
    """Return list of violation descriptions for a parsed tool call."""
    violations = []
    tool = call.get("tool", "")
    args = call.get("args", {})

    if tool not in VALID_TOOLS:
        violations.append(f"invalid tool: {tool!r}")
        return violations  # can't check args if tool is unknown

    if "metric" in args and args["metric"] not in VALID_METRICS:
        violations.append(f"invalid metric: {args['metric']!r}")
    if "type" in args and args["type"] not in VALID_EVENT_TYPES:
        violations.append(f"invalid event type: {args['type']!r}")
    if "direction" in args and args["direction"] not in VALID_DIRECTIONS:
        violations.append(f"invalid direction: {args['direction']!r}")
    if "bucket" in args and args["bucket"] not in VALID_BUCKETS:
        violations.append(f"invalid bucket: {args['bucket']!r}")
    if "segment_a" in args and args["segment_a"] not in VALID_SEGMENTS:
        violations.append(f"invalid segment_a: {args['segment_a']!r}")
    if "segment_b" in args and args["segment_b"] not in VALID_SEGMENTS:
        violations.append(f"invalid segment_b: {args['segment_b']!r}")
    if tool == "out_of_scope" and "reason" in args and args["reason"] not in VALID_REASONS:
        violations.append(f"invalid reason: {args['reason']!r}")
    if "range" in args:
        violations.append("range key present (must be absent from model output)")

    return violations


# Sample questions for validity testing (varied across all tools)
_VALIDITY_QUESTIONS = [
    "Why has my driving range reduced over the last two weeks?",
    "When did energy efficiency show the largest drop?",
    "Show all ADAS disengagement events from yesterday.",
    "What happened before the battery overheating warning?",
    "Compare energy consumption between city and highway driving.",
    "Show the daily trend of battery temperature over the last 7 days.",
    "When did pack temperature rise the most?",
    "List all low battery warnings in the last 14 days.",
    "What caused the reduction in driving range?",
    "Find the biggest drop in fuel economy.",
]


def run_validity_check(model_key: str, n_generations: int = 200) -> dict:
    """Run n_generations at temperature 0.7 and count violations."""
    model_path = MODEL_PATHS[model_key]
    if not model_path.exists():
        return {"error": f"Model not found: {model_path}", "model": model_key}

    from llama_cpp import Llama, LlamaGrammar  # type: ignore[attr-defined]

    from llm.router import _SYSTEM_PROMPT

    gbnf = build_gbnf()
    grammar = LlamaGrammar.from_string(gbnf)

    llm = Llama(
        model_path=str(model_path),
        n_ctx=1024, n_batch=128, n_threads=2, n_gpu_layers=0, verbose=False,
    )

    violations_total = 0
    parse_failures = 0
    violation_examples: list[str] = []
    t0 = time.perf_counter()

    for i in range(n_generations):
        q = _VALIDITY_QUESTIONS[i % len(_VALIDITY_QUESTIONS)]
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            grammar=grammar,
            temperature=0.7,
            max_tokens=150,
        )
        raw = resp["choices"][0]["message"]["content"].strip()
        call = _parse_tool_call(raw)
        if call is None:
            parse_failures += 1
            if len(violation_examples) < 5:
                violation_examples.append(f"[parse_fail] {raw[:80]}")
            continue
        viols = _check_enum_violations(call)
        if viols:
            violations_total += 1
            if len(violation_examples) < 5:
                violation_examples.append(f"{viols} <- {raw[:80]}")

    elapsed = time.perf_counter() - t0
    return {
        "model": model_key,
        "n_generations": n_generations,
        "violations": violations_total,
        "parse_failures": parse_failures,
        "valid_pct": round(100 * (n_generations - violations_total - parse_failures) / n_generations, 1),
        "elapsed_s": round(elapsed, 1),
        "violation_examples": violation_examples,
    }


# ---------------------------------------------------------------------------
# Part C: Dev / stress set evaluation
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _classify_result(item: dict, result) -> str:
    """
    Classify a pipeline result into one of:
      correct_refusal, false_refusal, unsupported_time, partial, wrong, ok
    """
    expected_status = item.get("expected_status", "ok")
    expected_tool = item.get("expected_tool", "")

    actual_status = getattr(result, "status", result.get("status", ""))
    actual_tool = ""
    if hasattr(result, "tool_name") and result.tool_name:
        actual_tool = result.tool_name.value
    elif hasattr(result, "tool_call") and result.tool_call:
        actual_tool = result.tool_call.get("tool", "")

    # Unsupported time — expected or not
    if actual_status == "unsupported_time":
        if expected_status == "unsupported_time":
            return "correct_refusal"
        return "false_refusal"

    # Control refused
    if actual_status == "control_refused":
        if expected_status == "control_refused":
            return "correct_refusal"
        return "false_refusal"

    # Off-topic gate
    if actual_status == "off_topic":
        if expected_status == "off_topic":
            return "correct_refusal"
        return "false_refusal"

    # Model returned out_of_scope but we expected a real tool
    if actual_status == "out_of_scope":
        if expected_status in ("off_topic", "out_of_scope"):
            return "correct_refusal"
        return "false_refusal"

    # Expected a refusal but got a real answer
    if expected_status in ("control_refused", "off_topic", "unsupported_time"):
        return "wrong"

    # Real tool call — check tool name
    if actual_tool != expected_tool:
        return "wrong"

    # Check for error
    error = getattr(result, "error", "") or ""
    if error and actual_status not in ("ok",):
        return "wrong"

    return "ok"


def run_eval_set(questions: list[dict], model_key: str) -> list[dict]:
    """Run the pipeline on a list of questions and return per-question results."""
    model_path = MODEL_PATHS[model_key]
    if not model_path.exists():
        return [{"id": q["id"], "error": "model_not_found"} for q in questions]

    from engine.pipeline import run_pipeline

    rows = []
    for item in questions:
        qid = item["id"]
        question = item["question"]
        t0 = time.perf_counter()
        try:
            result = run_pipeline(question, model_path=model_path, with_explanation=False)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            classification = _classify_result(item, result)
            actual_tool = ""
            if result.tool_call:
                actual_tool = result.tool_call.get("tool", "")
            rows.append({
                "id": qid,
                "question": question,
                "expected_tool": item.get("expected_tool"),
                "expected_status": item.get("expected_status"),
                "actual_tool": actual_tool,
                "actual_status": result.status,
                "classification": classification,
                "elapsed_ms": elapsed_ms,
                "error": result.error,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "id": qid,
                "question": question,
                "expected_tool": item.get("expected_tool"),
                "expected_status": item.get("expected_status"),
                "actual_tool": "",
                "actual_status": "exception",
                "classification": "wrong",
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": str(exc),
            })
    return rows


def summarise(rows: list[dict]) -> dict:
    counts: dict[str, int] = {
        "ok": 0, "correct_refusal": 0, "false_refusal": 0,
        "partial": 0, "wrong": 0, "unsupported_time_correct": 0,
    }
    for r in rows:
        c = r["classification"]
        if c == "ok":
            counts["ok"] += 1
        elif c == "correct_refusal":
            counts["correct_refusal"] += 1
        elif c == "false_refusal":
            counts["false_refusal"] += 1
        elif c == "partial":
            counts["partial"] += 1
        else:
            counts["wrong"] += 1
    counts["total"] = len(rows)
    counts["sound"] = counts["ok"] + counts["correct_refusal"] + counts["partial"]
    return counts


def print_table(title: str, rows: list[dict], counts: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {'ID':<12} {'Expected':<22} {'Got':<22} {'Class':<16} {'ms':>6}")
    print(f"  {'-'*12} {'-'*22} {'-'*22} {'-'*16} {'-'*6}")
    for r in rows:
        exp = f"{r['expected_tool']}({r['expected_status']})"
        got = f"{r['actual_tool']}({r['actual_status']})"
        print(f"  {r['id']:<12} {exp:<22} {got:<22} {r['classification']:<16} {r['elapsed_ms']:>6.0f}")
    print(f"\n  Total: {counts['total']}  Sound: {counts['sound']}  "
          f"OK: {counts['ok']}  CorrectRefusal: {counts['correct_refusal']}  "
          f"FalseRefusal: {counts['false_refusal']}  Partial: {counts['partial']}  "
          f"Wrong: {counts['wrong']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["0.5b", "1.5b", "both"], default="both")
    parser.add_argument("--skip-validity", action="store_true",
                        help="Skip the 200-generation validity check (saves ~10 min)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip dev/stress eval (run validity only)")
    args = parser.parse_args()

    results: dict = {}

    # --- Part A: Hashes and ruff ---
    print("\n=== Part A: Parquet hashes ===")
    hashes = hash_parquets()
    for name, h in hashes.items():
        print(f"  {h}  {name}")
    results["parquet_hashes"] = hashes

    print("\n=== Part A: Ruff ===")
    ruff_ok, ruff_out = run_ruff()
    print(f"  {'PASS' if ruff_ok else 'FAIL'}: {ruff_out or 'no issues'}")
    results["ruff_clean"] = ruff_ok
    results["ruff_output"] = ruff_out

    # --- Part B: Grammar validity ---
    model_keys = ["0.5b", "1.5b"] if args.model == "both" else [args.model]

    if not args.skip_validity:
        print("\n=== Part B: Grammar validity (200 generations @ temp=0.7) ===")
        validity_results = []
        for mk in model_keys:
            print(f"\n  Running {mk}...")
            vr = run_validity_check(mk, n_generations=200)
            validity_results.append(vr)
            if "error" in vr:
                print(f"  SKIP: {vr['error']}")
            else:
                print(f"  Model {mk}: {vr['violations']} violations, "
                      f"{vr['parse_failures']} parse failures, "
                      f"{vr['valid_pct']}% valid, {vr['elapsed_s']}s")
                if vr["violation_examples"]:
                    print("  Examples:")
                    for ex in vr["violation_examples"]:
                        print(f"    {ex}")
        results["validity"] = validity_results

    # --- Part C: Dev and stress eval ---
    if not args.skip_eval:
        dev_questions = _load_jsonl(DEV_FILE)
        stress_questions = _load_jsonl(STRESS_FILE)
        eval_results: dict = {}

        for mk in model_keys:
            print(f"\n=== Part C: Dev set — model {mk} ===")
            dev_rows = run_eval_set(dev_questions, mk)
            dev_counts = summarise(dev_rows)
            print_table(f"Dev set ({mk})", dev_rows, dev_counts)

            print(f"\n=== Part C: Stress set — model {mk} ===")
            stress_rows = run_eval_set(stress_questions, mk)
            stress_counts = summarise(stress_rows)
            print_table(f"Stress set ({mk})", stress_rows, stress_counts)

            eval_results[mk] = {
                "dev": {"rows": dev_rows, "summary": dev_counts},
                "stress": {"rows": stress_rows, "summary": stress_counts},
            }

        results["eval"] = eval_results

    # Save
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
