# CODING_AGENT_HANDOFF.md: VehicleGPT (Team Prayas, TELIPORT S4, Case Study 5)

You are taking over an in-progress project. Read this file and `PROJECT_CONTEXT.md` fully before touching anything. Where they differ, the newer decisions in this file win, and you must update `PROJECT_CONTEXT.md` to match.

## 0. Working rules (non-negotiable)
1. **Verify before trusting.** Do not assume earlier reports are correct. First run `pytest -m "not slow" -q` and `ruff check .`, then report the real results before changing anything.
2. **Show real output** for every claim: numbers, test results, timings. Never write "verified" without the command output.
3. **Report honestly.** If something fails, is slower, or is worse than hoped, say so. Separate wrong answers, correct refusals, false refusals and partial answers. Never loosen a test or tolerance to make it pass without explaining why.
4. **Ask before changing** `schema.py`, `sim/`, tool names or signatures, enums, or the ground truth. Keep `data/raw/*.parquet` byte-identical (hash them before and after if a change might touch data).
5. **Held-out test set:** the user will write 30 questions themselves. You must never see, tune against, or create this set. Do not tune prompts or rules against any question in it.
6. One phase at a time. Do not start a later phase unasked. No cloud services in the runtime path. Never commit model weights, secrets or `data/vehicle.db*`.
7. Keep claims tied to measurements: everything on the submission slide is labeled **Demonstrated**, **Target to validate**, or **Planned extension**.

## 1. What the project is
An offline, on-device vehicle assistant. A small language model (SLM) turns a natural-language question into one validated tool call; deterministic code resolves dates, queries a read-only SQLite database of simulated EV telemetry, renders the answer from templates, and shows the evidence. Targets (to validate, not yet proven on target hardware): under 2 s time to first evidence-backed answer for common queries, under 2 GB memory, at least 4 weeks of searchable history, safety-critical functions isolated (AI is read-only with no command path back).

Deadline: Round 1 one-slide submission closes **1 Nov 2026, 23:59 IST**; the user aims to submit by 31 Oct. The slide links to this repo and a 2-3 minute demo video, so the repo must be reproducible by a stranger with one command.

## 2. User and environment
- User (Team Prayas) is guided step by step, wants concise answers with a one-line reason for each step.
- MacBook (Apple Silicon, ARM64), `uv` environment at `.venv`, Python 3.11, Docker Desktop (ARM64 Linux VM ~8 GB). No Raspberry Pi or ARM board, so edge numbers come from a Docker container limited to `--memory=2g --cpus=2` and must be labeled "emulated constrained environment on Apple Silicon". Laptop latency is optimistic and must not be presented as vehicle-hardware latency.
- Models in `models/` (git-ignored): `qwen2.5-0.5b-instruct-q4_k_m.gguf` (default, 469 MB) and `qwen2.5-1.5b-instruct-q4_k_m.gguf` (1.0 GB, comparison). Runtime: `llama-cpp-python==0.3.36`. Note: this version's grammar helpers expect a JSON **string**, not a dict.
- Memory measurement rule: use in-process peak RSS plus the `anon`/`file` split from `/sys/fs/cgroup/memory.stat`. Docker's `memory.peak` always equals the cap (page cache) and is not usable.

## 3. Repository map
`config.py` (fixed simulated now `2026-10-28 18:00:00`, seed 42, `ROW_LIMIT` 200, 2 s query timeout, read-only DB URI helper) / `schema.py` (single source of truth: enums, DDL, strict Pydantic models) / `sim/simulate.py` (EV simulator, ground truth) / `ingest/normalize.py` (Parquet to SQLite, cleaning, `quality_report.json`) / `engine/` (`dates.py`, `validate.py`, `tools.py`, `render.py`, `verify.py`, `pipeline.py`) / `llm/` (`grammar.py`, `router.py`, `guards.py`, `timeparse.py`; `grammar_test.py` is a throwaway prototype excluded from ruff, kept only because `docker/Dockerfile` copies it) / `eval/` (`dev_questions.jsonl`, `stress.jsonl`, `run_dev.py`) / `docker/Dockerfile` (python:3.11-slim, build tools, llama-cpp-python 0.3.36; model mounted at run time, never baked in) / `tests/` (about 154 tests; slow simulator tests carry `@pytest.mark.slow`).

## 4. Data facts and locked decisions
- Data: 28 days from `2026-09-30T08:01` to `2026-10-27`; 69 trips, 136,269 `samples_1hz` rows (684 samples were dropped on purpose: 677 single gaps, three 2-row gaps, 1 boundary drop), 9 events (EV-0001 to EV-0009), 5,409 snapshot rows (10 Hz, 60 s before each event), 7 injected corruptions that the normalizer repairs. `today` was removed from the time vocabulary because no data exists on 28 Oct.
- **Efficiency (Wh/km) = 1000 x sum(power_kw) / sum(speed_kmh)** over `samples_1hz`, **net power (regeneration included)**, everywhere: segments, windows, trends, `explain_factors`. Never average per-row `energy_wh_per_km` or `estimated_range_km`; no tool may return them raw.
- **Range = "estimated full-charge range" = 58,000 / window Wh/km** (58 kWh usable). Always use that wording.
- Ground truth to reproduce: baseline 157.7 Wh/km and 367.8 km, target 192.7 Wh/km and 301.0 km (a 22.2% efficiency change, 18.2% range drop); city about 133.4 and highway about 188.6 Wh/km net; largest efficiency rise on 14 Oct (143.7 to 190.2 Wh/km); 3 ADAS disengagements yesterday (EV-0007 to EV-0009); overheating events EV-0001 and EV-0002 with pack temperature rising before.
- `explain_factors` compares the requested window with the equal-length window just before it, reports two groups (cold weather = ambient plus heating, about 29.0 Wh/km; highway share, about 6.4 Wh/km), and sets `baseline_truncated` with a note when the baseline starts before the first sample. Wording: "possible contributing factors".
- Safety layers in order: control-command guard, vehicle-vocabulary gate, grammar-constrained model, schema validation, grounding check. Model-written free text (`reason`) is never shown; fixed refusal messages are used. The time parser, not the model, decides all time ranges.

## 5. Measured status so far (all native on the Mac unless stated)
| Item | Result |
|---|---|
| Engine vs ground truth, 5 case questions | all match (tools only, no model) |
| Tool latency (query only) | under about 31 ms median |
| 0.5B routing, 40 dev questions | 39/40 sound behavior; the miss is dev_06 ("drop" routed to `find_largest_change` instead of `explain_factors`); 28 of the 40 reach the model |
| Stress set, 15 questions | 8/15 sound behavior; includes real false refusals (stress_01, 02, 11), a silently ignored time phrase (stress_03), invalid enum values (stress_04, 07, 10) and a partial answer to a two-part question (stress_09) |
| Raw model output validity, 41 vehicle questions | **78% before retry, 78% after retry**; retries at temperature 0 repeat the identical bad output 9 of 9 times and cost about 0.9 s each. It was reported as measured on "the 1.5B model", which needs confirming because the default is 0.5B |
| Prompt | about 736 tokens |
| Model stage latency | median about 585 ms, p95 about 828 ms (native) |
| Peak RSS | 858 MB native (0.5B). Container figures from the earlier prototype only: 0.5B 0.75 GB max RSS; 1.5B 2.01 GB (borderline) |
| Earlier prototype | with 3 tools and enums, the model routed 15/15 once the guard, gate and grounding layers existed |

**The dev set is not accuracy.** It was written by the system's builder, six questions are near-copies of prompt examples, and 12 are refusals handled by guards without the model. Real accuracy comes only from the user's held-out set.

## 6. Open work, in order (this is what you do now)
1. **Confirm state:** run the tests and ruff, hash the Parquet files, and report.
2. **Replace the loose output grammar** (it constrains only the tool name; argument values are free-form). Design: the model chooses tool and semantic arguments only; **remove `range` from model output** and fill it from the time parser (default `all_history`). Generate flat, non-recursive GBNF per tool from `schema.py` (a recursive converter caused a stack overflow): either two stages (tool name, then that tool's argument grammar) or a single grammar with enums for metric, event type, segment, direction and bucket, a pattern for `event_id`, fixed key order, and no free-text fields. Test that no out-of-enum value can be generated (200 generations at temperature 0.7) and that every enum member is accepted. Remove the pointless temperature-0 retry or give it a different prompt, and show the latency before and after.
3. **Time phrases first:** run the time parser before the model. An unresolved time phrase must always give a clarification (status `unsupported_time`) whichever tool the model would pick. Support "this week" (last 7 days), "last month" (28 days), "past fortnight" and "two days ago" (or ask), and document each rule.
4. **Early exits:** if the gate passed a question but the model returned `out_of_scope`, answer "I couldn't match that to vehicle data" and count it as a false refusal, not a correct refusal.
5. **Multi-intent questions:** answer the first part and say that the second must be asked separately (count as partial).
6. **dev_06:** deterministic rule: "why" goes to `explain_factors`; "when / biggest / largest" goes to `find_largest_change`.
7. **Reword prompt examples** that overlap any dev question by more than 0.6; show before and after.
8. **Re-run** the 15-question prototype list, dev set and stress set on **both models**; report validity before and after retry, retry latency, and wrong, correct-refusal, false-refusal and partial counts separately. Show `pytest` and `ruff`; list every assertion changed.

Then:
- **Phase 4:** wait for the user's held-out file and score it **once**, reporting the result as it is. Run the container benchmark (`--memory=2g --cpus=2`, CPU-only, model mounted read-only; update `docker/Dockerfile` to include the real code, not the prototype) and report peak RSS (in-process plus the anon/file split), time to first evidence-backed answer (median and p95, split into model, query and render), and a 0.5B versus 1.5B comparison on accuracy and memory. Use one command with a reproducible script.
- **Phase 5:** a Gradio UI that shows the answer, the tool call and the evidence rows; a README with one-command setup, results table and a Demonstrated / Target / Planned table; `THIRD_PARTY_LICENSES.md` checked; a 2-3 minute demo video script.
- **Phase 6:** support the user in updating the slide with measured numbers and status labels, and test the repo link.

Cut lines if time runs short: drop `search_logs` (disabled), the second-model comparison, then the Gradio UI in favour of a command-line demo. Always keep `explain_factors`.

## 7. Style of reports back to the user
Short, plain, table-based. For each step: what you did, the real output, anything that looks wrong, and what you need from them. If you are unsure, say so and propose a check; do not fill gaps with assumptions.
