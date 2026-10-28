#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -f docker/Dockerfile -t vehiclegpt-benchmark .
mkdir -p eval/results
run_one() {
  local model="$1" label="$2" output="$3"
  docker run --rm --memory=2g --cpus=2 \
    -e VEHICLEGPT_MODEL="/models/$model" -e VEHICLEGPT_LABEL="$label" \
    -v "$PWD/models:/models:ro" -v "$PWD/eval/results:/results" \
    vehiclegpt-benchmark --output "/results/$output"
}
started=$SECONDS
run_one qwen2.5-0.5b-instruct-q4_k_m.gguf 0.5B benchmark_0_5b.jsonl
elapsed=$((SECONDS - started))
if (( elapsed < 900 )); then
  run_one qwen2.5-1.5b-instruct-q4_k_m.gguf 1.5B benchmark_1_5b.jsonl
else
  printf 'Skipping 1.5B: 0.5B benchmark took %s seconds (15 minute cutoff).\n' "$elapsed"
fi
.venv/bin/python -m eval.benchmark --summarize eval/results
