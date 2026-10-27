#!/usr/bin/env python3
"""
eval/run_prototype.py - Run the 15-question prototype list on both models.

These are the 15 foundational vehicle questions that the system must handle correctly.
Reports per model: correct, wrong, correct refusals, false refusals, unsupported_time, latency.

Saves results to eval/results/prototype_results.json
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

# From tests/test_guards.py
FIFTEEN_PROTOTYPE_QUESTIONS = [
    ("proto_01", "Why has my driving range reduced over the last two weeks?", "explain_factors"),
    ("proto_02", "When did fuel efficiency show the largest drop?", "find_largest_change"),
    ("proto_03", "Show all ADAS disengagement events from yesterday.", "get_events"),
    ("proto_04", "What happened before the battery overheating warning?", "context_before"),
    ("proto_05", "Compare energy consumption between city and highway driving.", "compare"),
    ("proto_06", "When did energy efficiency show the largest drop?", "find_largest_change"),
    ("proto_07", "Show ADAS disengagements from yesterday.", "get_events"),
    ("proto_08", "What happened before EV-0002 battery overheating warning?", "context_before"),
    ("proto_09", "Compare energy consumption for city vs highway driving.", "compare"),
    ("proto_10", "Why has vehicle driving range decreased over the past 14 days?", "explain_factors"),
    ("proto_11", "Show the daily trend in energy consumption for the last 7 days.", "trend"),
    ("proto_12", "List all low battery warnings in the last 14 days.", "get_events"),
    ("proto_13", "What happened before the latest ADAS disengagement?", "context_before"),
    ("proto_14", "Compare pack temperature between city and highway driving.", "compare"),
    ("proto_15", "When did pack temperature show the largest rise?", "find_largest_change"),
]

MODEL_PATHS = {
    "0.5b": config.DEFAULT_MODEL_PATH,
    "1.5b": config.COMPARISON_MODEL_PATH,
}


def run_prototype_on_model(model_key: str) -> dict:
    """Run the 15 prototype questions on the specified model."""
    from llm.router import route
    
    logger.info(f"Running 15-question prototype on {model_key} model...")
    model_path = MODEL_PATHS[model_key]
    
    results = {
        "model": model_key,
        "total_questions": len(FIFTEEN_PROTOTYPE_QUESTIONS),
        "questions": [],
        "statistics": {
            "correct": 0,
            "wrong": 0,
            "correct_refusal": 0,
            "false_refusal": 0,
            "unsupported_time": 0,
        },
        "latencies_ms": [],
    }
    
    for q_id, question, expected_tool in FIFTEEN_PROTOTYPE_QUESTIONS:
        try:
            t0 = time.perf_counter()
            result = route(question, model_path=model_path, with_explanation=False)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            results["latencies_ms"].append(elapsed_ms)
            
            # Classify result
            actual_tool = result.tool_name.value if result.tool_name else ""
            error_status = result.error
            
            classification = "correct"
            
            if error_status == "unsupported_time":
                classification = "unsupported_time"
            elif error_status in ("control_refused", "off_topic", "out_of_scope"):
                classification = "false_refusal"  # These shouldn't be refusals in the prototype
            elif actual_tool != expected_tool:
                classification = "wrong"
            
            results["statistics"][classification if classification in results["statistics"] else "correct"] += 1
            
            results["questions"].append({
                "id": q_id,
                "question": question,
                "expected_tool": expected_tool,
                "actual_tool": actual_tool,
                "classification": classification,
                "latency_ms": round(elapsed_ms, 1),
                "error": error_status if error_status else None,
            })
            
            if classification != "correct":
                logger.warning(f"  {q_id}: {classification} (expected {expected_tool}, got {actual_tool})")
            else:
                logger.debug(f"  {q_id}: correct")
        
        except Exception as e:
            logger.error(f"Error processing {q_id}: {e}")
            results["statistics"]["wrong"] += 1
            results["questions"].append({
                "id": q_id,
                "question": question,
                "expected_tool": expected_tool,
                "error": str(e),
                "classification": "wrong",
            })
    
    # Compute latency stats
    if results["latencies_ms"]:
        sorted_latencies = sorted(results["latencies_ms"])
        results["statistics"]["latency_median_ms"] = round(sorted_latencies[len(sorted_latencies) // 2], 1)
        results["statistics"]["latency_p95_ms"] = round(sorted_latencies[int(0.95 * len(sorted_latencies))], 1)
        results["statistics"]["latency_mean_ms"] = round(sum(results["latencies_ms"]) / len(results["latencies_ms"]), 1)
    
    logger.info(f"  Results: {results['statistics']}")
    
    return results


def main():
    """Run prototype on both models."""
    logger.info("Running 15-question prototype test on both models...")
    
    all_results = {}
    
    for model_key in ["0.5b", "1.5b"]:
        model_path = MODEL_PATHS[model_key]
        if not model_path.exists():
            logger.warning(f"Model {model_key} not found at {model_path}")
            continue
        
        results = run_prototype_on_model(model_key)
        all_results[model_key] = results
    
    # Save results
    output_file = RESULTS_DIR / "prototype_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("15-QUESTION PROTOTYPE RESULTS")
    print("="*70)
    
    for model_key in sorted(all_results.keys()):
        result = all_results[model_key]
        stats = result["statistics"]
        print(f"\n{model_key.upper()} Model:")
        print(f"  Correct:           {stats['correct']}/15")
        print(f"  Wrong:             {stats['wrong']}")
        print(f"  False refusals:    {stats['false_refusal']}")
        print(f"  Unsupported time:  {stats['unsupported_time']}")
        print(f"  Latency median:    {stats.get('latency_median_ms', 'N/A')} ms")
        print(f"  Latency p95:       {stats.get('latency_p95_ms', 'N/A')} ms")
        
        # Show wrong/false refusals
        wrong_questions = [q for q in result["questions"] if q["classification"] != "correct"]
        if wrong_questions:
            print(f"\n  Issues:")
            for q in wrong_questions:
                print(f"    {q['id']}: expected {q['expected_tool']}, got {q.get('actual_tool', 'ERROR')}")
    
    return 0


if __name__ == "__main__":
    exit(main())
