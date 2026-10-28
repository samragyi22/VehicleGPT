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

Environment: **emulated constrained environment on Apple Silicon (Docker, 2 GB, 2 CPUs); timings are optimistic compared with automotive cores**. Each model ran the 28 vehicle questions for three passes. Measurements below are from [`benchmark_summary.txt`](eval/results/benchmark_summary.txt) and the per-query records [`benchmark_0_5b.jsonl`](eval/results/benchmark_0_5b.jsonl) and [`benchmark_1_5b.jsonl`](eval/results/benchmark_1_5b.jsonl).

Peak RSS is the in-process peak; anon/file are the highest sampled cgroup `memory.stat` values. The warm per-tool statistics exclude the first record, which is reported separately as cold start. Median and p95 are in milliseconds.

| Model | Peak RSS (MB) | cgroup anon (MB) | cgroup file (MB) |
|---|---:|---:|---:|
| 0.5B | 660.71 | 162.11 | 472.13 |
| 1.5B | 1287.59 | 191.57 | 1065.90 |

### First query (cold start)

The first query in each run was `dev_01` (`explain_factors`). The table reports total / model / query time in milliseconds.

| Model | Total time | Model time | Query time |
|---|---:|---:|---:|
| 0.5B | 12937.51 | 12471.60 | 44.07 |
| 1.5B | 33410.73 | 32458.55 | 50.82 |

### Warm query timings by tool

The first record was omitted from these warm figures. Each timing column gives median / p95 in milliseconds.

**0.5B**

| Tool | Total median / p95 | Model median / p95 | Query median / p95 |
|---|---:|---:|---:|
| compare | 925.68 / 990.26 | 901.74 / 962.57 | 25.64 / 28.59 |
| context_before | 879.64 / 917.06 | 877.72 / 914.97 | 1.67 / 5.72 |
| explain_factors | 748.10 / 768.39 | 706.26 / 722.39 | 43.50 / 45.49 |
| find_largest_change | 801.51 / 867.64 | 768.93 / 833.83 | 33.23 / 36.34 |
| get_events | 697.68 / 731.04 | 697.17 / 730.50 | 0.22 / 0.25 |
| trend | 798.71 / 899.24 | 778.16 / 865.68 | 20.80 / 33.43 |

**1.5B**

| Tool | Total median / p95 | Model median / p95 | Query median / p95 |
|---|---:|---:|---:|
| compare | 2060.24 / 2136.99 | 2032.46 / 2108.29 | 27.59 / 35.40 |
| context_before | 2022.86 / 2144.95 | 2020.47 / 2142.85 | 1.66 / 9.51 |
| explain_factors | 1771.70 / 1819.53 | 1732.67 / 1774.11 | 37.55 / 46.24 |
| find_largest_change | 1817.95 / 1956.30 | 1783.01 / 1921.69 | 34.19 / 37.57 |
| get_events | 1628.39 / 1763.64 | 1627.78 / 1763.01 | 0.23 / 0.30 |
| trend | 1886.92 / 2038.27 | 1876.48 / 2005.74 | 20.30 / 35.48 |

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
- Apple Silicon container timings are optimistic compared with automotive cores and are not vehicle timings.
- Cold-start latency is much higher than warm-query latency.
- The 1.5B model misses the 2 s target on some tools.
- Dev and stress questions are builder-written; they are not held-out accuracy measurements.
- The model selects tools. Deterministic code resolves dates and executes queries.
- Real vehicle integration, target-board validation, encryption, and access controls remain future work.
- The first held-out run exposed two model errors (one wrong tool and one false `out_of_scope` on an explanatory question); grounding gaps for “led up to” and “by day”; a synonym gap for “efficient” with informal road-type words; and a control-command guard vocabulary gap for “cabin.” Two ambiguous questions were refused rather than clarified.

## Future work

Fix the rule gaps and re-measure on new questions, never on this set.

## Models and licenses

Run `bash scripts/download_models.sh` to fetch the documented GGUF files. The script prints the license for each model. Qwen2.5 model cards list Apache-2.0: [0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and [1.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). The GGUF conversions are hosted by [bartowski](https://huggingface.co/bartowski); see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
