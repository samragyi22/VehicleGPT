"""Minimal local Gradio interface for evidence-backed vehicle queries."""

import json

import gradio as gr

from config import DEFAULT_MODEL_PATH
from engine.pipeline import run_pipeline

EXAMPLES = [
    "Why has my driving range reduced over the last two weeks?",
    "When did fuel efficiency show the largest drop?",
    "Show all ADAS disengagement events from yesterday.",
    "What happened before the battery overheating warning?",
    "Compare energy consumption between city and highway driving.",
    "Show the daily trend of energy consumption over the last 7 days.",
]


def answer(question: str) -> tuple[str, str, str, str, str]:
    result = run_pipeline(question, model_path=DEFAULT_MODEL_PATH)
    timings = "\n".join(f"{stage}: {value:.2f} ms" for stage, value in result.timings.items())
    return (
        result.answer,
        result.status,
        json.dumps(result.tool_call, indent=2, ensure_ascii=False),
        json.dumps(result.evidence, indent=2, ensure_ascii=False),
        timings,
    )


with gr.Blocks(title="VehicleGPT") as demo:
    gr.Markdown("# VehicleGPT\nOffline answers from simulated vehicle telemetry, with evidence.")
    question = gr.Textbox(label="Question", lines=2, placeholder="Ask about vehicle history…")
    submit = gr.Button("Ask")
    response = gr.Textbox(label="Answer", lines=5)
    status = gr.Textbox(label="Status")
    tool_call = gr.Code(label="Tool call", language="json")
    evidence = gr.Code(label="Evidence rows", language="json")
    timing = gr.Textbox(label="Per-stage timing", lines=6)
    gr.Examples(EXAMPLES, inputs=question)
    submit.click(answer, inputs=question, outputs=[response, status, tool_call, evidence, timing])


if __name__ == "__main__":
    demo.launch()
