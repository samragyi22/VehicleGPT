"""
Debug specific failing questions to see raw output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from llm.router import route


def test_question(question: str, model_key: str = "0.5b"):
    """Run a single question and show full result."""
    model_path = config.DEFAULT_MODEL_PATH if model_key == "0.5b" else config.COMPARISON_MODEL_PATH
    
    print(f"\n{'='*80}")
    print(f"Model: {model_key}")
    print(f"Question: {question}")
    print(f"{'='*80}")
    
    result = route(question, model_path=model_path, with_explanation=False)
    
    print(f"tool_name: {result.tool_name}")
    print(f"tool_call_raw: {result.tool_call_raw}")
    print(f"tool_args_dict: {result.tool_args_dict}")
    print(f"error: {result.error}")
    print(f"answer: {result.answer[:100]}..." if result.answer else "answer: (empty)")


if __name__ == "__main__":
    # Test the failing "why" questions
    failing_questions = [
        "Why has my driving range reduced over the last two weeks?",
        "Why did my vehicle range drop in the past 14 days?",
        "What caused the reduction in driving range over the last two weeks?",
    ]
    
    # Also test the control command that went wrong
    wrong_tool_questions = [
        "How do I bake a chocolate cake?",
        "Accelerate to 100 km/h.",
    ]
    
    print("\n### FAILING 'WHY' QUESTIONS (0.5B) ###")
    for q in failing_questions:
        test_question(q, "0.5b")
    
    print("\n\n### WRONG TOOL QUESTIONS ###")
    for q in wrong_tool_questions:
        test_question(q, "0.5b")
        test_question(q, "1.5b")
