# VehicleGPT

VehicleGPT is an offline vehicle-data assistant. A grammar-constrained small language model selects a validated tool; deterministic Python resolves time, queries simulated telemetry in read-only SQLite, renders an answer, and verifies it against returned evidence. The model has no command path to vehicle controls.

## Architecture

`Simulated telemetry → normalizer → read-only SQLite → guarded SLM tool choice → deterministic query → template + evidence → verifier`

## Setup and run

Requires Python, `uv`, and Docker Desktop for the constrained benchmark. Model weights are downloaded separately and are never committed.

```sh
uv sync --extra dev --extra llm
bash scripts/download_models.sh
uv run python -m sim.simulate
uv run python -m ingest.normalize
uv run python -m app.ui
bash docker/run_benchmark.sh
uv run pytest -m "not slow" -q
uv run ruff check .
```

The UI uses the default Qwen2.5-0.5B GGUF. The benchmark runs the 0.5B model first and only starts the 1.5B comparison when the first run finishes within the configured cutoff. The container uses CPU inference and mounts `models/` read-only. Results are written as JSONL and summary text beneath `eval/results/`.

## Results

| Measurement | Result | Source |
|---|---|---|
| Constrained container latency and memory | Not measured: Docker Desktop daemon was unavailable in this session. | [`container_benchmark_blocked.txt`](eval/results/container_benchmark_blocked.txt) |

Benchmark measurements, when available, are labeled **emulated constrained environment on Apple Silicon**. They are not vehicle-hardware timings. This README does not substitute native or prototype measurements for the requested container run.

## Demonstrated / Target to validate / Planned

| Status | Capability |
|---|---|
| **Demonstrated** | Simulated vehicle telemetry and diagnostic events; normalization into SQLite; deterministic, read-only query tools; grammar-constrained tool selection; evidence-backed rendering and verification. |
| **Target to validate** | Response latency, memory use, and history capacity on actual target hardware. The container benchmark is an Apple Silicon emulation. |
| **Planned** | Recorded CAN/CAN-FD input, real ARM board measurements, SOME/IP and Android VHAL sources, QNX and camera events, CARLA integration, and vehicle HPC/NPU scaling. |

## Known limitations

- The data is simulated, not collected from a production vehicle.
- Apple Silicon container timings are not vehicle timings.
- Dev and stress questions are builder-written; they are not held-out accuracy measurements.
- The model selects tools. Deterministic code resolves dates and executes queries.
- Real vehicle integration, target-board validation, encryption, and access controls remain future work.

## Models and licenses

Run `bash scripts/download_models.sh` to fetch the documented GGUF files. The script prints the license for each model. Qwen2.5 model cards list Apache-2.0: [0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and [1.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). The GGUF conversions are hosted by [bartowski](https://huggingface.co/bartowski); see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
