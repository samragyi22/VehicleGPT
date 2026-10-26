# PROJECT_CONTEXT.md: VehicleGPT (Team Prayas, TELIPORT Season 4, Case Study 5)

Read this file fully before writing any code. It is the source of truth for scope, design, and working rules.

---

## 1. What we are building

**VehicleGPT**: an offline, on-device assistant that answers natural-language questions about a vehicle's data using a small language model (SLM). The prototype runs on simulated EV data on a MacBook (Apple Silicon, ARM64) and is benchmarked inside a Docker container limited to **2 GB RAM and 2 CPUs** as an emulated edge environment.

This prototype supports a **Round 1 one-slide idea submission** (deadline: 1 Nov 2026, 23:59 IST; internal target: submit by 31 Oct). The slide links to this repo and a short demo video. Judges evaluate design soundness, so honesty and clarity matter more than breadth.

### The case study's required capabilities (from the problem statement)
- Runs an SLM/LLM directly in the vehicle, **no cloud dependency**.
- Consumes vehicle data streams, events, and logs and makes them searchable in natural language.
- Finds historical events, trends, and anomalies, and explains causes behind observed behavior.
- Target hardware: ~2 GB RAM, ~2 ARM A5x cores. Response under 2 s for common queries. Several weeks of searchable history.
- Safety-critical functions must remain isolated from the AI system.

### The case study's example questions (our tools must cover all five)
1. "Why has my driving range reduced over the last two weeks?"
2. "When did fuel efficiency show the largest drop?" (we treat this as the efficiency metric, `energy_wh_per_km`, for an EV)
3. "Show all ADAS disengagement events from yesterday."
4. "What happened before the battery overheating warning?"
5. "Compare energy consumption between city and highway driving."

---

## 2. Core design principles (non-negotiable)

1. **The model never writes SQL and never touches raw vehicle systems.** It outputs exactly one JSON tool call chosen from a fixed list. Deterministic code validates and executes it.
2. **Structured retrieval, not raw data in the prompt.** The model sees the question and tool outputs only, never bulk logs.
3. **Read-only by construction.** Read-only SQLite connection, parameterized queries, row limits, query timeout.
4. **Evidence-backed answers.** Every answer shows the tool call and the evidence rows (IDs) used. A verifier checks that every number in the answer text appears in the tool output; otherwise fall back to the raw template answer.
5. **Computed "why", not invented "why".** Causal-style answers come from a deterministic `explain_factors` tool that ranks candidate factors. The model only phrases the result as *possible contributing factors*, never as proven causes.
6. **Deterministic time handling.** Phrases like "yesterday" or "last two weeks" are resolved by code against a fixed simulated "now", not by the model.
7. **Honest claims.** Everything we state is labeled **Demonstrated**, **Target to validate**, or **Planned extension**. Never present laptop numbers as automotive-hardware numbers.
8. **Safety boundary wording (use exactly this idea):** the AI receives approved, read-only telemetry through a one-way gateway and has **no command path back** to safety-critical ECUs.

---

## 3. Architecture (what the code implements)

```
Simulated EV logs --> Normalizer --> SQLite (events, 1 Hz samples, pre-event snapshots)
                                         ^
User question --> SLM (tool selection, JSON-constrained)
              --> Argument validator + date resolver
              --> Query tools (read-only) --> evidence rows
              --> Template renderer --> (optional) SLM short explanation
              --> Verifier (numbers must match evidence)
              --> Answer + tool call + evidence shown in UI
```

### Demonstrated vs planned (must stay consistent in README and slide)
| Item | Status |
|---|---|
| Simulated EV drive logs + diagnostic-style event log | **Demonstrated** |
| Normalization into a common event schema | **Demonstrated** |
| SQLite storage, 4 weeks of history | **Demonstrated** |
| Validated query tools, JSON-constrained SLM tool selection | **Demonstrated** |
| Evidence-backed answers + verifier | **Demonstrated** |
| 2 GB / 2 CPU container benchmark (emulated edge) | **Demonstrated (emulated)** |
| CAN/CAN-FD adapter (via recorded or simulated frames) | Planned (stretch if time) |
| SOME/IP, Android VHAL, QNX logs, camera events | **Planned extension** (design only) |
| CARLA integration | **Planned extension** |
| Real ARM automotive board / Raspberry Pi benchmark | **Planned: validate on target hardware** |
| Vehicle HPC / NPU scaling | **Planned extension** |
| FTS5 log text search (`search_logs`) | Optional, only if time allows |

---

## 4. Simulated vehicle and data

**Vehicle:** a passenger EV. Simulated "now" is a fixed constant in config (e.g. `2026-10-28 18:00:00`) so tests are reproducible. Generate **28 days** of history with a fixed random seed.

### Signals (1 Hz while driving)
`ts, trip_id, speed_kmh, power_kw, soc_pct, pack_voltage_v, pack_current_a, pack_temp_c, ambient_temp_c, hvac_power_kw, road_segment ('city'|'highway'), adas_active (0/1), odometer_km`

Derived: `energy_wh_per_km` (rolling or per-trip), `estimated_range_km` (from SOC and recent efficiency).

### Tables
- `trips(trip_id, start_ts, end_ts, distance_km, energy_kwh, avg_ambient_c, city_share)`
- `samples_1hz(...signals above...)`: only while driving
- `samples_1min(...)`: aggregates for long-window queries
- `events(event_id, ts, type, severity, details_json)`
- `event_snapshots(event_id, offset_ms, signals...)`: high-rate (10 Hz) window of 60 s **before** each event, written when the event fires (pre-event ring buffer concept)
- `ground_truth.json` (separate file, NOT in the DB the model queries): every seeded fault with exact timestamps and causes, used only for scoring tests

### Closed vocabularies (enums, define once in `schema.py`, reuse everywhere, including the grammar)
- **Metrics:** `energy_wh_per_km`, `estimated_range_km`, `pack_temp_c`, `soc_pct`, `speed_kmh`, `hvac_power_kw`
- **Event types:** `BATTERY_OVERHEAT_WARNING`, `ADAS_DISENGAGEMENT`, `LOW_SOC_WARNING`
- **Segments:** `city`, `highway`
- **Bucket sizes:** `hour`, `day`, `week`

### Seeded scenarios (with known ground truth)
1. **Range drop over ~2 weeks:** caused by a cold-weather period plus increased highway share plus higher HVAC use. `explain_factors` should rank these.
2. **Battery overheating warnings:** preceded by sustained high power, high ambient temperature, and rising pack temperature in the minutes before.
3. **ADAS disengagements:** clustered on specific days/road segments, including "yesterday" relative to simulated now.
4. **City vs highway efficiency gap:** realistic and measurable.
5. **Largest efficiency drop on a known date.**

Keep scenarios plausible, not cartoonish. Add mild noise and a few missing samples so the normalizer has something to handle.

---

## 5. Query tools (FROZEN. Do not add or change signatures without asking the user)

All tools return JSON with `result`, `evidence` (row IDs / event IDs / time ranges), and `meta` (row count, query time ms).

| Tool | Arguments | Purpose |
|---|---|---|
| `get_events` | `type`, `start`, `end` | List events in a window |
| `context_before` | `event_id`, `seconds` (<=60 or <=300 using 1 Hz), `signals[]` | Signals leading up to an event |
| `compare` | `metric`, `start`, `end`, `segment_a`, `segment_b` | Compare a metric across segments |
| `find_largest_change` | `metric`, `start`, `end`, `direction` ('drop'|'rise') | Biggest change by bucket |
| `trend` | `metric`, `start`, `end`, `bucket` | Time series + slope |
| `explain_factors` | `metric`, `start`, `end` | Deterministic ranking of candidate factors (speed mix/highway share, ambient temperature, HVAC use, trip length) with effect sizes |
| `out_of_scope` | `reason` | Polite decline when no tool fits |
| `search_logs` (optional) | `text`, `start`, `end` | SQLite FTS5 search |

`start` and `end` given to tools are ISO timestamps produced by the **date resolver**; the model outputs symbolic time ranges (e.g. `{"kind":"yesterday"}`, `{"kind":"last_n_days","n":14}`), which code converts.

### SLM output format
Exactly one JSON object: `{"tool": "<name>", "args": {...}}`, enforced with **llama.cpp grammar-constrained decoding** (GBNF / JSON schema). Validate again in code after decoding (never trust the grammar alone). Reject unknown tools, unknown enums, out-of-range values.

### Answer rendering
- Numbers and tables are **rendered by code from templates**.
- The SLM may add a short (<= 2 sentences) explanation; this step is optional and skippable for latency.
- `explain_factors` results must be phrased as "possible contributing factors" with the evidence shown.
- If the verifier finds a number not present in the evidence, discard the model text and show the template answer.

---

## 6. Tech stack

- **Python 3.11+**, `uv` or `venv`, `pytest`, `ruff`.
- **SQLite** (stdlib `sqlite3`), read-only URI connection for the query engine (`file:...?mode=ro`).
- **llama.cpp** (use `llama-cpp-python` or the `llama-server` binary) with a 4-bit GGUF model.
  - Candidate models: one ~1B and one ~1.5B instruct model. Check each model's license before use and record it in `THIRD_PARTY_LICENSES.md`. Do not commit model weights; provide a download script.
- **Gradio** (or Streamlit) for the demo UI: shows question, answer, tool call JSON, evidence rows.
- **Docker** (ARM64) for the constrained benchmark: `--memory=2g --cpus=2`, CPU-only.
- No cloud APIs anywhere in the runtime path. No telemetry. No secrets in the repo.

---

## 7. Repository structure

```
vehiclegpt/
  PROJECT_CONTEXT.md        # this file
  README.md                 # setup, one-command run, results table, demonstrated-vs-planned
  THIRD_PARTY_LICENSES.md
  config.py                 # fixed "now", seed, paths, limits
  schema.py                 # enums, table DDL, tool arg schemas (single source of truth)
  sim/
    simulate.py             # EV simulator + fault seeding -> data/*.parquet|csv + ground_truth.json
  ingest/
    normalize.py            # clean + load into SQLite, build 1-min aggregates, snapshots
  engine/
    dates.py                # symbolic time -> ISO ranges
    tools.py                # the tools above (read-only, validated)
    validate.py             # arg validation against schema.py
    render.py               # templates for answers
    verify.py               # number-vs-evidence verifier
  llm/
    grammar.py              # builds GBNF/JSON schema from schema.py
    router.py               # question -> tool call via llama.cpp
  app/
    ui.py                   # Gradio app
  eval/
    questions.jsonl         # 30 questions + paraphrases + expected tool/args/answer
    run_eval.py             # scores tool choice, args, final answer separately
    benchmark.py            # latency split + peak RAM, run inside container
  docker/
    Dockerfile
    run_benchmark.sh
  tests/
```

---

## 8. Phases (work one at a time; do not skip ahead)

| Phase | Goal | Done when |
|---|---|---|
| 0 | Setup: repo, env, model download script, freeze `schema.py` and tool list | Model runs locally; signatures written |
| 1 | Simulator + 4 weeks of data + `ground_truth.json` | SQLite populated; ground truth lists every seeded fault |
| 2 | Deterministic engine (tools, validator, dates, templates) with unit tests | **All 5 case questions answered correctly with no LLM** |
| 3 | SLM router with grammar-constrained JSON + verifier + `out_of_scope` | End-to-end works (target date: 27 Oct) |
| 4 | Eval set (30 questions + paraphrases), scoring, container benchmark, compare 2 models | Results table with tool-choice accuracy, arg accuracy, answer correctness, peak RAM, latency split |
| 5 | Gradio UI, demo video script, README with one-command reproduction | A stranger can clone and reproduce numbers |
| 6 | Support slide updates (numbers, status labels) | Repo and link tested in incognito and on a phone |

**Cut lines if behind schedule (in order):** drop `search_logs` and second model -> CLI demo instead of Gradio -> reduce eval set to 20 -> keep `explain_factors` always.

---

## 9. Evaluation and benchmarking requirements

- **Eval set:** 30 questions with paraphrases, covering all tools plus a few out-of-scope and ambiguous ones. Expected answers come from `ground_truth.json` or deterministic computation, so scoring is automatic.
- **Report three scores separately:** tool-choice accuracy, argument accuracy, final-answer correctness.
- **Benchmark in container (2 GB, 2 CPUs, CPU-only):** peak RSS, time to first evidence-backed answer, total latency, split into model time vs query time vs render time. Run each query type multiple times; report median and p95.
- **Latency target definition:** "time to first evidence-backed answer for common queries" (target: under 2 s). The optional model explanation may stream after it.
- **Targets (all labeled "to validate" until measured):** under 2 s for common queries; under 2 GB total runtime memory; at least 4 weeks of searchable history.
- If a target is missed, report the real number and the plan to close the gap. Do not tune the numbers or hide failures.
- State clearly in the README: the container runs on Apple Silicon, so latency is likely optimistic compared with automotive-grade A5x cores.

---

## 10. Locked decisions (do not reopen without explicit user instruction)

These were resolved during Phase 1a and must not be changed silently.

| Decision | Detail |
|---|---|
| **`explain_factors` comparison window & design** | Always compares the requested window with the equal-length window immediately before it (e.g. last 14 days vs. the 14 days before that). Reports two groups: **highway_share** ($\Delta \text{highway share} \times \text{target}(Wh/km_{hwy} - Wh/km_{city})$) and **cold_weather** (the remainder, with changes in ambient temperature and HVAC power shown as evidence). Must be phrased as possible contributing factors. If no earlier window exists, returns a limited-evidence result instead of a guess. |
| **Efficiency formula** | Exactly one formula everywhere: **total energy (Wh) ÷ total distance (km)** across the window. Evaluated as `1000 * sum(power_kw) / sum(speed_kmh)` over `samples_1hz` for every efficiency number, including per-bucket values. Never use a per-sample rolling mean or arithmetic average of ratios. |
| **Range definition** | One documented definition everywhere: **"range at full charge" = 58,000 / window Wh/km**. Never average per-row `estimated_range_km` or `energy_wh_per_km` across windows (which conflates instantaneous SOC and rolling filters). |
| **Date resolution rules** | Symbolic time ranges are resolved day-aligned against `config.SIMULATED_NOW` (`2026-10-28 18:00:00`):<br>- `last_n_days` (`last_7_days`, `last_14_days`, `last_28_days`): start at midnight (00:00:00) $n$ days before now.date() and end at now (`2026-10-28T18:00:00`).<br>- `yesterday`: 27 Oct 00:00:00 to 28 Oct 00:00:00.<br>- `last_n_hours`: `[now - n h, now]`.<br>- `all_history`: data start (28 days before now at 00:00:00) to now. |
| **`today` removed from schema** | `TimeRangeKind` has no `today` entry. Simulation history runs through 27 Oct (the last complete day before `SIMULATED_NOW = 2026-10-28 18:00:00`). Adding morning trips for 28 Oct to fill a `today` bucket is explicitly ruled out. |
| **Default model** | The default inference model is the **0.5 B GGUF** (qwen2.5-0.5b-instruct-q4_k_m.gguf). The **1.5 B** variant (qwen2.5-1.5b-instruct-q4_k_m.gguf) is the comparison candidate used in the Phase 4 benchmark. |

---

## 11. Working rules for the coding agent

1. Read this file first; re-read the relevant section before each phase.
2. **Ask before** changing the tool list, enums, table schemas, or scope. Do not add features not listed here.
3. Work phase by phase. Finish and test each phase before starting the next. Show test output.
4. `schema.py` is the single source of truth for enums and tool argument schemas. Generate the grammar and validators from it; never duplicate enum lists.
5. Everything must be reproducible: fixed seeds, fixed "now", pinned dependency versions, one-command setup and run.
6. Write tests first for the engine (Phase 2). Use `ground_truth.json` as the oracle.
7. Never let model output reach SQL or a shell. Never open the DB read-write from the query path.
8. No secrets, keys, tokens, or model weights in git. Document third-party licenses.
9. Keep code simple and readable; prefer the standard library. Add short docstrings explaining *why*, not just *what*.
10. When something fails or a number looks too good, say so and investigate rather than working around it.
11. Keep README claims aligned with the Demonstrated / Target / Planned labels in section 3.

---

## 12. Context for the slide (for reference, not for coding)

- Differentiator: **auditable answers from local vehicle data**, not just a chatbot in a car.
- Lead users: service engineers and validation teams; also drivers and fleet operators (one compact line).
- Scalability story: same tool interface from a laptop container, to a Raspberry Pi/ARM board, to a vehicle HPC with an NPU and a larger model (planned).
- Privacy: local-only processing, encrypted storage (planned), role-based access (planned), camera data kept as extracted events rather than stored frames (planned).
