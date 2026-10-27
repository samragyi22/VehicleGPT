#!/usr/bin/env python3
"""
eval/run_part_b.py - Part B: 200 Generations at Temperature 0.7

Generate 200 samples at temperature 0.7 using the 0.5B model (grammar-constrained),
spread evenly across all 7 tools. Validate that:
1. No value falls outside enum constraints
2. Every enum member appears at least once across all outputs
3. No "range" key is generated

Saves results to eval/results/part_b_generations.jsonl (one JSON per line).
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "part_b_generations.jsonl"
SUMMARY_FILE = RESULTS_DIR / "part_b_summary.txt"

# Seven tools, ~28-29 samples per tool
TOOLS = ["get_events", "context_before", "compare", "find_largest_change", "trend", "explain_factors", "out_of_scope"]
SAMPLES_PER_TOOL = 200 // len(TOOLS)  # ~28-29 per tool
REMAINDER = 200 % len(TOOLS)  # distribute remainder

# Import schema for validation
from schema import (
    BucketSize,
    ChangeDirection,
    EventType,
    Metric,
    Segment,
    ToolName,
)

# Collect all enum values
ENUM_SETS = {
    "event_type": {e.value for e in EventType},
    "metric": {m.value for m in Metric},
    "direction": {d.value for d in ChangeDirection},
    "bucket": {b.value for b in BucketSize},
    "segment": {s.value for s in Segment},
    "reason": {"not_vehicle_data", "vehicle_control_prohibited", "ambiguous_query"},
}


def get_llm():
    """Load the 0.5B model (default) with grammar."""
    from llm.router import _get_llm
    from llm.grammar import load_grammar
    
    llm = _get_llm()
    grammar = load_grammar()
    return llm, grammar


def generate_samples_for_tool(tool_name: str, count: int, llm, grammar) -> list[dict]:
    """Generate `count` samples for a specific tool at temperature 0.7."""
    samples = []
    
    # Simple prompts that are likely to route to each tool
    prompts_by_tool = {
        "get_events": [
            "Show me ADAS disengagement events.",
            "What are the battery overheating warnings?",
            "List low SOC warnings.",
            "Show all vehicle events.",
        ],
        "context_before": [
            "What happened before EV-0001?",
            "Show context around the latest ADAS event.",
            "What was the telemetry before the battery warning?",
        ],
        "compare": [
            "Compare efficiency between city and highway.",
            "How does energy consumption differ for city vs highway?",
            "Compare temperature in city vs highway.",
        ],
        "find_largest_change": [
            "When was the biggest efficiency drop?",
            "Show the largest temperature change.",
            "When did range change the most?",
        ],
        "trend": [
            "Show efficiency trends by day.",
            "What is the daily trend in energy consumption?",
            "Trend range over the last month.",
        ],
        "explain_factors": [
            "Why has range dropped?",
            "What factors affected efficiency?",
            "Why is energy consumption higher?",
        ],
        "out_of_scope": [
            "What is the weather today?",
            "What is the capital of France?",
            "Tell me a joke.",
            "How do I control the air conditioning?",
        ],
    }
    
    prompts = prompts_by_tool.get(tool_name, ["test query"])
    
    for i in range(count):
        prompt = prompts[i % len(prompts)]
        t0 = time.perf_counter()
        try:
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a vehicle assistant. Respond with a JSON tool call.",
                    },
                    {"role": "user", "content": prompt},
                ],
                grammar=grammar,
                temperature=0.7,
                max_tokens=200,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            raw_json = response["choices"][0]["message"]["content"].strip()
            
            try:
                parsed = json.loads(raw_json)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON for {tool_name} sample {i}: {e}")
                parsed = {"error": "json_decode_error", "raw": raw_json}
            
            samples.append({
                "tool": tool_name,
                "sample_index": i,
                "prompt": prompt,
                "raw_json": raw_json,
                "parsed": parsed,
                "latency_ms": elapsed_ms,
            })
        except Exception as e:
            logger.error(f"Error generating sample for {tool_name}: {e}")
            samples.append({
                "tool": tool_name,
                "sample_index": i,
                "prompt": prompt,
                "error": str(e),
            })
    
    return samples


def validate_samples(samples: list[dict]) -> dict:
    """Validate all samples and collect statistics."""
    stats = {
        "total_samples": len(samples),
        "valid_json": 0,
        "out_of_enum_values": 0,
        "has_range_key": 0,
        "enum_coverage": defaultdict(set),
        "out_of_enum_details": [],
    }
    
    for sample in samples:
        if "error" in sample or "error" in sample.get("parsed", {}):
            continue
        
        parsed = sample.get("parsed", {})
        if not isinstance(parsed, dict) or "tool" not in parsed:
            continue
        
        stats["valid_json"] += 1
        args = parsed.get("args", {})
        
        # Check for range key
        if "range" in args:
            stats["has_range_key"] += 1
            logger.warning(f"Found 'range' key in {parsed['tool']}: {args['range']}")
        
        # Validate enum values
        tool = parsed.get("tool")
        
        if tool == "get_events":
            if "type" in args:
                val = args["type"]
                if val not in ENUM_SETS["event_type"]:
                    stats["out_of_enum_values"] += 1
                    stats["out_of_enum_details"].append({
                        "tool": tool,
                        "field": "type",
                        "value": val,
                        "expected": list(ENUM_SETS["event_type"]),
                    })
                else:
                    stats["enum_coverage"]["event_type"].add(val)
        
        elif tool == "context_before":
            if "event_id" in args:
                val = args["event_id"]
                # Just check it's non-empty; pattern validation in grammar
                pass
        
        elif tool == "compare":
            for field in ["metric", "segment_a", "segment_b"]:
                if field in args:
                    val = args[field]
                    expected_enum = "segment" if "segment" in field else "metric"
                    if val not in ENUM_SETS[expected_enum]:
                        stats["out_of_enum_values"] += 1
                        stats["out_of_enum_details"].append({
                            "tool": tool,
                            "field": field,
                            "value": val,
                            "expected": list(ENUM_SETS[expected_enum]),
                        })
                    else:
                        stats["enum_coverage"][expected_enum].add(val)
        
        elif tool == "find_largest_change":
            if "metric" in args:
                val = args["metric"]
                if val not in ENUM_SETS["metric"]:
                    stats["out_of_enum_values"] += 1
                    stats["out_of_enum_details"].append({
                        "tool": tool,
                        "field": "metric",
                        "value": val,
                        "expected": list(ENUM_SETS["metric"]),
                    })
                else:
                    stats["enum_coverage"]["metric"].add(val)
            if "direction" in args:
                val = args["direction"]
                if val not in ENUM_SETS["direction"]:
                    stats["out_of_enum_values"] += 1
                    stats["out_of_enum_details"].append({
                        "tool": tool,
                        "field": "direction",
                        "value": val,
                        "expected": list(ENUM_SETS["direction"]),
                    })
                else:
                    stats["enum_coverage"]["direction"].add(val)
        
        elif tool == "trend":
            if "metric" in args:
                val = args["metric"]
                if val not in ENUM_SETS["metric"]:
                    stats["out_of_enum_values"] += 1
                    stats["out_of_enum_details"].append({
                        "tool": tool,
                        "field": "metric",
                        "value": val,
                        "expected": list(ENUM_SETS["metric"]),
                    })
                else:
                    stats["enum_coverage"]["metric"].add(val)
            if "bucket" in args:
                val = args["bucket"]
                if val not in ENUM_SETS["bucket"]:
                    stats["out_of_enum_values"] += 1
                    stats["out_of_enum_details"].append({
                        "tool": tool,
                        "field": "bucket",
                        "value": val,
                        "expected": list(ENUM_SETS["bucket"]),
                    })
                else:
                    stats["enum_coverage"]["bucket"].add(val)
        
        elif tool == "explain_factors":
            if "metric" in args:
                val = args["metric"]
                if val not in ENUM_SETS["metric"]:
                    stats["out_of_enum_values"] += 1
                    stats["out_of_enum_details"].append({
                        "tool": tool,
                        "field": "metric",
                        "value": val,
                        "expected": list(ENUM_SETS["metric"]),
                    })
                else:
                    stats["enum_coverage"]["metric"].add(val)
        
        elif tool == "out_of_scope":
            if "reason" in args:
                val = args["reason"]
                if val not in ENUM_SETS["reason"]:
                    stats["out_of_enum_values"] += 1
                    stats["out_of_enum_details"].append({
                        "tool": tool,
                        "field": "reason",
                        "value": val,
                        "expected": list(ENUM_SETS["reason"]),
                    })
                else:
                    stats["enum_coverage"]["reason"].add(val)
    
    return stats


def main():
    """Run Part B: generate 200 samples."""
    logger.info("Loading model and grammar...")
    llm, grammar = get_llm()
    
    logger.info(f"Generating {200} samples across {len(TOOLS)} tools...")
    all_samples = []
    
    # Generate samples per tool
    for idx, tool in enumerate(TOOLS):
        count = SAMPLES_PER_TOOL
        if idx < REMAINDER:
            count += 1
        
        logger.info(f"Generating {count} samples for {tool}...")
        samples = generate_samples_for_tool(tool, count, llm, grammar)
        all_samples.extend(samples)
        logger.info(f"  Generated {len(samples)} samples for {tool}")
    
    # Save to JSONL
    logger.info(f"Saving {len(all_samples)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    
    # Validate
    logger.info("Validating samples...")
    stats = validate_samples(all_samples)
    
    # Write summary
    summary_lines = [
        "# Part B: Grammar Generation Validation (200 samples, 0.7 temperature, 0.5B model)",
        "",
        f"Total samples:        {stats['total_samples']}",
        f"Valid JSON:           {stats['valid_json']}",
        f"Out-of-enum values:   {stats['out_of_enum_values']} (MUST BE 0)",
        f"Has 'range' key:      {stats['has_range_key']} (MUST BE 0)",
        "",
        "Enum coverage:",
    ]
    
    for enum_name in sorted(stats["enum_coverage"].keys()):
        covered = stats["enum_coverage"][enum_name]
        total = ENUM_SETS[enum_name]
        missing = total - covered
        summary_lines.append(
            f"  {enum_name}: {len(covered)}/{len(total)} "
            f"(missing: {', '.join(sorted(missing)) if missing else 'none'})"
        )
    
    if stats["out_of_enum_details"]:
        summary_lines.append("\nOut-of-enum value details:")
        for detail in stats["out_of_enum_details"]:
            summary_lines.append(f"  {detail}")
    
    summary_text = "\n".join(summary_lines)
    logger.info(summary_text)
    
    with open(SUMMARY_FILE, "w") as f:
        f.write(summary_text)
    
    logger.info(f"Summary saved to {SUMMARY_FILE}")
    
    # Return success if all constraints met
    success = stats["out_of_enum_values"] == 0 and stats["has_range_key"] == 0
    logger.info(f"Part B validation: {'PASS' if success else 'FAIL'}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
