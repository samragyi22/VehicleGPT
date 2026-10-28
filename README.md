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

## Held-out evaluation (first run, unbiased)

The team lead ran the frozen pipeline once, without tuning before or after, against a question file written by the team and kept outside the repository. The run used the pipeline frozen at tag `freeze-v1` (commit `506d995`) and repository commit `f39f634`. The question file SHA-256 is `e4fe4284f519794a71d11cceda73f8f3dce03a7c3aaec3d540ecfe38627d0b34`. Per-question results, without question text, are in [`heldout_first_run.json`](eval/results/heldout_first_run.json); the recorded report is [`heldout_first_run.md`](eval/results/heldout_first_run.md).

| Question category | Count | Result |
|---|---:|---|
| Vehicle questions | 19 | Exact match: 14/19 (73.7%) |
| Vehicle failures | — | Wrong tool: 1; false refusal: 2 (one clarification request and one `out_of_scope`); refused as ungrounded: 2 |
| Non-vehicle questions | 11 | All refused or clarified (5 off-topic, 3 control commands, 1 time clarification, 2 ambiguous); 3 had a different refusal label than expected |

Failure causes, by type: two model errors (one wrong tool and one false `out_of_scope` on an explanatory question); three rule gaps (grounding does not recognize “led up to,” grounding misses “by day,” and a synonym gap for “efficient” with informal road-type words); and one control-command guard vocabulary gap (“cabin”). Two ambiguous questions were refused rather than clarified; this is acceptable behavior but differs from the expected label.

**small sample (19 vehicle questions); dev and stress sets are builder-written and score higher**

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
- The first held-out run exposed two model errors (one wrong tool and one false `out_of_scope` on an explanatory question); grounding gaps for “led up to” and “by day”; a synonym gap for “efficient” with informal road-type words; and a control-command guard vocabulary gap for “cabin.” Two ambiguous questions were refused rather than clarified.

## Future work

Fix the rule gaps and re-measure on new questions, never on this set.

## Models and licenses

Run `bash scripts/download_models.sh` to fetch the documented GGUF files. The script prints the license for each model. Qwen2.5 model cards list Apache-2.0: [0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and [1.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). The GGUF conversions are hosted by [bartowski](https://huggingface.co/bartowski); see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
