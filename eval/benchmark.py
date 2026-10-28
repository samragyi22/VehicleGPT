"""Run and summarize the fixed 28-question vehicle benchmark in a constrained container."""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
from pathlib import Path
from typing import Any

from engine.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
ENV_LABEL = "emulated constrained environment on Apple Silicon"


def _memory_stat() -> dict[str, int]:
    """Read live cgroup counters; memory.peak includes the cap and is unsuitable."""
    try:
        rows = (Path("/sys/fs/cgroup/memory.stat")).read_text().splitlines()
        values = dict(line.split() for line in rows)
        return {key: int(values.get(key, "0")) for key in ("anon", "file")}
    except (OSError, ValueError):
        return {"anon": 0, "file": 0}


def run(output: Path) -> None:
    questions = [json.loads(line) for line in (ROOT / "eval/dev_questions.jsonl").read_text().splitlines()]
    vehicle_questions = questions[:28]
    model_path = os.environ["VEHICLEGPT_MODEL"]
    label = os.environ.get("VEHICLEGPT_LABEL", Path(model_path).stem)
    peak_split = {"anon": 0, "file": 0}
    peak_rss_mb = 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as out:
        for pass_number in range(1, 4):
            for item in vehicle_questions:
                result = run_pipeline(item["question"], model_path=model_path)
                split = _memory_stat()
                for key, value in peak_split.items():
                    peak_split[key] = max(value, split[key])
                # Linux reports ru_maxrss in KiB; this benchmark runs only in Linux Docker.
                peak_rss_mb = max(peak_rss_mb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
                row = {
                    "environment": ENV_LABEL, "model": label,
                    "question_id": item["id"], "tool": (result.tool_call or {}).get("tool", "none"),
                    "pass": pass_number, "status": result.status,
                    "total_ms": result.timings.get("total_ms", 0),
                    "model_ms": result.timings.get("model_ms", 0),
                    "query_ms": result.timings.get("query_ms", 0),
                    "render_ms": result.timings.get("render_ms", 0),
                    "peak_rss_mb": round(peak_rss_mb, 2),
                    "cgroup_anon_mb": round(peak_split["anon"] / 1048576, 2),
                    "cgroup_file_mb": round(peak_split["file"] / 1048576, 2),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(json.dumps(row), flush=True)


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int((len(ordered) - 1) * p + 0.999999)))]


def summarize(directory: Path) -> None:
    for path in sorted(directory.glob("benchmark_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        lines = [
            f"{rows[0]['model']} — {ENV_LABEL}; 28 vehicle questions × 3 passes",
            "tool | total median/p95 ms | model median/p95 ms | query median/p95 ms | peak RSS MB | cgroup anon/file MB",
        ]
        tools = sorted({row["tool"] for row in rows})
        for tool in tools:
            group = [row for row in rows if row["tool"] == tool]
            fields: list[Any] = []
            for key in ("total_ms", "model_ms", "query_ms"):
                values = [float(row[key]) for row in group]
                fields.extend((round(statistics.median(values), 2), round(_percentile(values, .95), 2)))
            rss = max(row["peak_rss_mb"] for row in group)
            anon = max(row["cgroup_anon_mb"] for row in group)
            file_mb = max(row["cgroup_file_mb"] for row in group)
            lines.append(f"{tool} | {fields[0]}/{fields[1]} | {fields[2]}/{fields[3]} | {fields[4]}/{fields[5]} | {rss} | {anon}/{file_mb}")
        summary = "\n".join(lines) + "\n"
        summary_path = path.with_name(path.stem + "_summary.txt")
        summary_path.write_text(summary)
        print(f"\n{summary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    else:
        if args.output is None:
            parser.error("--output is required")
        run(args.output)


if __name__ == "__main__":
    main()
