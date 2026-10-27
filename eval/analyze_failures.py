"""
Analyze failing questions from Part C evaluation (dev set and 15-question prototype).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_results(results_file: Path) -> dict:
    """Load Part C results."""
    with open(results_file) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze_dev_set():
    """Analyze dev set failures for both models."""
    results = load_results(Path(__file__).parent / "results" / "part_c_results.json")
    dev_questions = load_jsonl(Path(__file__).parent / "dev_questions.jsonl")

    output = []
    output.append("=" * 100)
    output.append("DEV SET ANALYSIS (40 questions)")
    output.append("=" * 100)

    for model_key in ["0.5b", "1.5b"]:
        dev_result = results.get(f"{model_key}_dev")
        if not dev_result:
            continue

        output.append(f"\n{model_key.upper()} MODEL FAILURES:\n")
        
        # Find failures
        failures = [q for q in dev_result["questions"] if q["classification"] != "correct"]
        
        if not failures:
            output.append(f"  NO FAILURES (all {dev_result['total_questions']}/40 correct)\n")
            continue
        
        output.append(f"  Total failures: {len(failures)}/{dev_result['total_questions']}\n")
        
        for fail in failures:
            q_id = fail["id"]
            # Find the original question from dev_questions
            orig = next((q for q in dev_questions if q["id"] == q_id), {})
            
            question = orig.get("question", "unknown")
            expected_tool = fail["expected_tool"]
            actual_tool = fail.get("actual_tool", "ERROR/None")
            category = fail["classification"]
            
            # Diagnose
            diagnosis = "unknown"
            if actual_tool == "":
                diagnosis = "tool call error (empty)"
            elif actual_tool != expected_tool:
                diagnosis = f"wrong tool ({actual_tool} vs {expected_tool})"
            elif category in ("false_refusal", "unsupported_time"):
                diagnosis = f"gate/guard/time parser ({category})"
            else:
                diagnosis = category
            
            output.append(f"  {q_id}:")
            output.append(f"    Question: {question}")
            output.append(f"    Expected: {expected_tool}")
            output.append(f"    Actual: {actual_tool}")
            output.append(f"    Diagnosis: {diagnosis}\n")

    return "\n".join(output)


def analyze_prototype():
    """Analyze 15-question prototype failures."""
    results = load_results(Path(__file__).parent / "results" / "prototype_results.json")

    output = []
    output.append("\n" + "=" * 100)
    output.append("15-QUESTION PROTOTYPE ANALYSIS")
    output.append("=" * 100)

    for model_key in ["0.5b", "1.5b"]:
        proto_result = results.get(model_key)
        if not proto_result:
            continue

        output.append(f"\n{model_key.upper()} MODEL FAILURES:\n")
        
        # Find failures
        failures = [q for q in proto_result["questions"] if q["classification"] != "correct"]
        
        if not failures:
            output.append("  NO FAILURES (all 15/15 correct)\n")
            continue
        
        output.append(f"  Total failures: {len(failures)}/15\n")
        
        for fail in failures:
            q_id = fail["id"]
            question = fail["question"]
            expected_tool = fail["expected_tool"]
            actual_tool = fail.get("actual_tool", "ERROR/None")
            error = fail.get("error", "")
            
            # Diagnose
            diagnosis = "unknown"
            if actual_tool == "":
                diagnosis = "tool call error (empty)"
            elif actual_tool != expected_tool:
                diagnosis = f"wrong tool ({actual_tool} vs {expected_tool})"
            elif error:
                diagnosis = f"error: {error}"
            else:
                diagnosis = "classification error"
            
            output.append(f"  {q_id}:")
            output.append(f"    Question: {question}")
            output.append(f"    Expected: {expected_tool}")
            output.append(f"    Actual: {actual_tool}")
            output.append(f"    Diagnosis: {diagnosis}\n")

    return "\n".join(output)


if __name__ == "__main__":
    analysis = analyze_dev_set() + analyze_prototype()
    
    # Print and save
    print(analysis)
    
    output_file = Path(__file__).parent / "results" / "step1_failures_analysis.txt"
    with open(output_file, "w") as f:
        f.write(analysis)
    
    print(f"\nSaved to {output_file}")
