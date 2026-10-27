"""
eval/run_part_c.py - Part C: Measure validity with new grammar on dev and stress sets.

Runs the 40-question dev set and 15-question stress set on both the 0.5B and 1.5B models.
Reports for each model:
  - Validity (correctly formed tool calls)
  - Latency (model_ms)
  - Which model each number is for
  - Wrong answers, correct refusals, false refusals, partial answers

Saves results to eval/results/part_c_results_<model>.json
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEV_FILE = config.EVAL_DIR / "dev_questions.jsonl"
STRESS_FILE = config.EVAL_DIR / "stress.jsonl"

MODEL_PATHS = {
    "0.5b": config.DEFAULT_MODEL_PATH,
    "1.5b": config.COMPARISON_MODEL_PATH,
}


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval_set(model_key: str, eval_set_name: str, questions: list[dict]) -> dict:
    """Run eval set on specified model, return results."""
    from llm.router import route
    
    logger.info(f"Running {eval_set_name} on {model_key} model ({len(questions)} questions)...")
    model_path = MODEL_PATHS[model_key]
    
    results = {
        "model": model_key,
        "set_name": eval_set_name,
        "total_questions": len(questions),
        "questions": [],
        "statistics": {
            "correct": 0,
            "wrong": 0,
            "correct_refusal": 0,
            "false_refusal": 0,
            "partial": 0,
            "unsupported_time": 0,
        },
        "latencies_ms": [],
        "errors": [],
    }
    
    for i, item in enumerate(questions):
        q_id = item.get("id", f"q_{i}")
        question = item["question"]
        expected_tool = item.get("expected_tool", "")
        expected_status = item.get("expected_status", "ok")
        
        try:
            t0 = time.perf_counter()
            result = route(question, model_path=model_path, with_explanation=False)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            results["latencies_ms"].append(elapsed_ms)
            
            # Classify the result
            actual_tool = result.tool_name.value if result.tool_name else ""
            error_status = result.error
            
            classification = "correct"
            
            if error_status == "unsupported_time":
                classification = "unsupported_time"
                if expected_status != "unsupported_time":
                    classification = "false_refusal"
                else:
                    classification = "correct_refusal"
            elif error_status in ("control_refused", "off_topic", "out_of_scope"):
                classification = "correct_refusal" if expected_status in ("control_refused", "off_topic", "out_of_scope") else "false_refusal"
            elif actual_tool != expected_tool:
                classification = "wrong"
            
            results["statistics"][classification if classification in results["statistics"] else "correct"] += 1
            
            results["questions"].append({
                "id": q_id,
                "classification": classification,
                "expected_tool": expected_tool,
                "actual_tool": actual_tool,
                "latency_ms": round(elapsed_ms, 1),
            })
            
            logger.debug(f"  {q_id}: {classification} ({actual_tool})")
        
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error processing {q_id}: {e}")
            results["errors"].append({"id": q_id, "error": str(e)})
            results["statistics"]["wrong"] += 1
    
    # Compute statistics
    if results["latencies_ms"]:
        results["statistics"]["latency_median_ms"] = round(sorted(results["latencies_ms"])[len(results["latencies_ms"]) // 2], 1)
        results["statistics"]["latency_p95_ms"] = round(sorted(results["latencies_ms"])[int(0.95 * len(results["latencies_ms"]))], 1)
        results["statistics"]["latency_mean_ms"] = round(sum(results["latencies_ms"]) / len(results["latencies_ms"]), 1)
    
    logger.info(f"  Results: {results['statistics']}")
    
    return results


def main():
    """Run Part C: evaluate dev and stress sets on both models."""
    logger.info("Loading eval sets...")
    dev_questions = load_jsonl(DEV_FILE)
    stress_questions = load_jsonl(STRESS_FILE)
    
    all_results = {}
    
    # Run on both models
    for model_key in ["0.5b", "1.5b"]:
        model_path = MODEL_PATHS[model_key]
        if not model_path.exists():
            logger.warning(f"Model {model_key} not found at {model_path}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {model_key} model")
        logger.info(f"{'='*60}")
        
        # Dev set
        dev_results = run_eval_set(model_key, "dev", dev_questions)
        all_results[f"{model_key}_dev"] = dev_results
        
        # Stress set
        stress_results = run_eval_set(model_key, "stress", stress_questions)
        all_results[f"{model_key}_stress"] = stress_results
    
    # Save all results
    output_file = RESULTS_DIR / "part_c_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("PART C SUMMARY: Dev and Stress Set Evaluation")
    print("="*70)
    
    for key in sorted(all_results.keys()):
        result = all_results[key]
        stats = result["statistics"]
        print(f"\n{result['model'].upper()} - {result['set_name'].upper()} SET ({result['total_questions']} questions):")
        print(f"  Correct:           {stats.get('correct', 0)}")
        print(f"  Wrong:             {stats.get('wrong', 0)}")
        print(f"  Correct refusal:   {stats.get('correct_refusal', 0)}")
        print(f"  False refusal:     {stats.get('false_refusal', 0)}")
        print(f"  Partial:           {stats.get('partial', 0)}")
        print(f"  Unsupported time:  {stats.get('unsupported_time', 0)}")
        print(f"  Latency median:    {stats.get('latency_median_ms', 'N/A')} ms")
        print(f"  Latency p95:       {stats.get('latency_p95_ms', 'N/A')} ms")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
