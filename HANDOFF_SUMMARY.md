# HANDOFF SUMMARY: VehicleGPT (Team Prayas, TELIPORT S4, Case Study 5)

Paste this into a new chat first, together with PROJECT_CONTEXT.md. Last updated: 24 Oct 2026.

---

## 1. How the user wants to be helped
- Guide **one step at a time**, in simple, concise language.
- Every step gets a one-line **"Why"** so the user knows what they are doing.
- Give deeper explanations only when asked.
- The user pastes terminal output or agent output back; review it critically and say plainly what is wrong or unproven.
- User wants the best possible Round 1 submission. Do not promise selection. Never let claims outrun measurements.

## 2. User and team
- Team name: **Prayas**. Member names not yet added to the cover slide.
- Skills: GenAI/CV/RAG builder, good storyteller, embedded/automotive + LLM experience. No ML-training, mechanical/electrical/EV, or networking/Linux depth.
- Hardware: **MacBook M5 (Apple Silicon, ARM64)**. No Raspberry Pi or ARM board. Docker Desktop works (ARM64 Linux, VM has ~8 GB).
- A coding agent writes the code. The user pastes its summaries here for review.

## 3. Competition facts
- Tata Elxsi TELIPORT Season 4. **Round 1 = one-slide idea submission**, open 23 Sep to **1 Nov 2026, 23:59 IST**. Top teams go to a detailed submission round.
- Plan: submit by **31 Oct**, keeping 1 Nov as buffer.
- Template: `Teliport_S4_Submission_Template.pptx`. Slide 1 = cover (Case Study No., Team Details). Slide 2 = blank content slide with TELIPORT and Tata Elxsi banner. The one-slider goes on slide 2.
- The user plans to add a **prototype link** (public GitHub repo + 2-3 min demo video, plus QR code on the slide). Check the submission form accepts a link.

## 4. Problem choice: Case Study 5 (VehicleGPT)
Natural-language vehicle information using an on-device SLM/LLM. Requirements: no cloud, ~2 GB RAM / ~2 ARM A5x cores, under 2 s for common queries, several weeks of searchable history, safety-critical functions isolated. The case's five example questions: range reduction over two weeks, largest fuel-efficiency drop, ADAS disengagement events from yesterday, what happened before the battery overheating warning, city vs highway energy comparison.
Evaluation criteria: system architecture, AI capability, data integration, search and analytics, edge performance, privacy and security, innovation and practicality.
Other cases were rejected (1 needs ML, 2 battery/EV domain, 3 networking, 4 is open-ended). Case 4 was the backup.

## 5. Design decisions (frozen unless the user agrees to change)
- **The model never writes SQL.** It emits one JSON tool call, constrained by a grammar, then code validates it. Read-only SQLite, row limit 200, 2 s timeout.
- **Tools:** `get_events`, `context_before`, `compare`, `find_largest_change`, `trend`, `explain_factors`, `out_of_scope`; optional `search_logs` (disabled until an FTS5 table exists).
- **Vehicle:** simulated EV, 28 days of history ending at fixed simulated now `2026-10-28 18:00:00`, seed 42.
- **No vector DB.** SQLite only; FTS5 optional.
- **"Why" answers are computed** by `explain_factors` (deterministic) and phrased as "possible contributing factors". `explain_factors` must compare the requested window against the **equal-length window immediately before it**, because the simulated range drop is a step change.
- **Safety wording:** the AI receives approved, read-only telemetry through a one-way gateway and has no command path back to safety-critical ECUs.
- **Layered routing guard (prototype-validated):** control-command guard (regex) -> vehicle-vocabulary gate -> grammar-constrained SLM -> grounding check (event type and time range must be supported by the question's words). Heuristics: measure their false-reject rate in Phase 4.
- **Latency target** is defined as time to first evidence-backed answer for common queries; model explanation may stream afterwards. Everything is labeled Demonstrated / Target to validate / Planned extension.
- Lead users: service engineers and validation teams; drivers and fleet operators in one compact line.
- Schema notes: `samples_1min` has no road_segment column, so segment comparisons read `samples_1hz`. `context_before` signals are validated against `SNAPSHOT_SIGNALS` (columns of `event_snapshots`).

## 6. What exists

### Slide (needs revision)
`Prayas_Teliport_S4_Round1.pptx` was built from the template (cover: Case Study 5 / Team Prayas / VehicleGPT; slide 2: architecture diagram with data path and query path, read-only gateway, 5 design choices, targets, evidence plan, who benefits). **It must be updated** before submission:
- Remove the vector index mention; say SQLite.
- Pitch strip must say "targets to validate", not facts.
- Mark each item Demonstrated vs Planned.
- Replace "1-3B model" with the real candidate(s) and measured numbers.
- Lead with service engineers and validation teams.
- Add validated-tool panel mapped to the case's five questions, pre-event buffer, scalability strip (laptop container -> Pi/ARM board -> vehicle HPC), privacy details.
- Add repo link / QR code. Add member names on cover.

### Repo (`/Users/samragyisharma/Desktop/VehicleGPT`)
`.venv` (uv, Python 3.11), `pyproject.toml`, `.gitignore`, `README.md`, `THIRD_PARTY_LICENSES.md` (agent says it verified Apache 2.0 for both Qwen models; user should double-check), `PROJECT_CONTEXT.md`, `config.py`, `schema.py` (strict Pydantic models, `extra="forbid"`, validated event IDs, signals and time ranges), `tests/test_schema.py` (5 tests), `sim/simulate.py` and `tests/test_sim.py` (10 tests total passing), `docker/Dockerfile`, `.dockerignore`, `llm/grammar_test.py` (throwaway prototype), folders `app/ engine/ eval/ ingest/ data/ models/`.
Models in `models/` (git-ignored): `qwen2.5-0.5b-instruct-q4_k_m.gguf` (469 MB, working default) and `qwen2.5-1.5b-instruct-q4_k_m.gguf` (1.0 GB, comparison candidate). Keep both until Phase 4.
Docker image `vehiclegpt-bench` (python:3.11-slim + llama-cpp-python 0.3.36; model mounted at run time).
**Check `git log` and `git status`:** confirm Phase 0 and Phase 1a work is committed and pushed. Repo should be private until just before submission.

## 7. Measured results so far (Docker on Apple Silicon, CPU-only, --memory=2g --cpus=2)
| Model | Max RSS | Private (anon) memory | Routing result |
|---|---|---|---|
| 0.5B Q4_K_M | **0.75 GB** | ~0.33 GB | 15/15 correct on 15 hand-written questions once guard, gate and grounding check were added (vehicle questions 5/5 even without them; off-topic 2/4 without them) |
| 1.5B Q4_K_M | 2.01 GB | 1.11 GB | borderline on memory; off-topic 3/4 without layers |
- Routing-step latency about 0.5-1.0 s on 2 CPUs (first call 3-4 s while loading). Fast Apple cores, so **optimistic; never claim for automotive hardware**.
- Cgroup `memory.peak` always equals the cap (page cache), so it is not usable. Measure in-process RSS plus anon/file split.
- 15 questions is a tiny sample. Real accuracy comes from the Phase 4 eval set.

## 8. Simulator status (Phase 1a)
Generated: 69 trips, 136,269 samples at 1 Hz, 9 events (EV-0001..EV-0009), 5,409 snapshot rows (10 Hz, 60 s before each event), `data/ground_truth.json`. City about 153.9 Wh/km, highway about 192.8 Wh/km. Cold snap begins 14 Oct (ambient ~26 C -> ~8 C, HVAC ~0.7 -> ~2.3 kW, highway share ~62% -> ~84%). Three ADAS events "yesterday" (27 Oct), earlier ADAS cluster on 10 Oct, two overheating events (4-5 Oct heatwave), two low-SOC events.

**Problems found, fix pending (Step 24, sent to the coding agent, result not yet reviewed):**
1. `range_drop` ground truth text contradicts the data (wrong temperatures, shares, window).
2. Two inconsistent efficiency figures in use (164.5/163.8 vs 158.6/151.6 weekly; +22.5% vs +29.4%). Must use one formula: total energy / total distance.
3. The factor ranking was asserted, not computed. Replace with counterfactual simulator runs (same seed, one factor held at baseline). The "+1.65 kW adds ~45 Wh/km" claim looked too high. Note HVAC load and cold are linked.
4. `injected_corruptions` block missing from ground truth (needs `ts`, `trip_id`, `column`, real `original_value`, and count of dropped samples).
5. Replace the ad hoc `voltage > 0` test filter with exclusion of the listed corrupted rows; add a test recomputing every ground-truth number from the Parquet files.

## 9. Remaining roadmap
| Phase | Work | Target |
|---|---|---|
| 1a fix | Step 24 above; review the three printed tables; commit and push | now |
| 1b | `ingest/normalize.py`: clean and load into SQLite, build `samples_1min` and event snapshots; test that every injected corruption is repaired | 25 Oct |
| 2 | Deterministic engine (`engine/`: dates, tools, validate, render, verify) with tests; **all 5 case questions correct with no LLM** | by 26 Oct |
| 3 | SLM router generated from `schema.py` (grammar), guards, verifier, `out_of_scope` | **end-to-end by 27 Oct** |
| 4 | Eval set (30 questions + paraphrases), scoring of tool choice / arguments / final answer, container benchmark (RSS, p50/p95 latency split into model/query/render), compare 0.5B vs 1.5B | 28-29 Oct |
| 5 | Gradio UI (answer + tool call + evidence rows), 2-3 min demo video, README with one-command reproduction | 29-30 Oct |
| 6 | Update slide with measured numbers and status labels; test repo link in incognito and on a phone; make repo public; submit | 30-31 Oct |

Cut lines if behind (in order): drop `search_logs` and the second model -> CLI demo instead of Gradio -> eval set of 20 -> always keep `explain_factors`.

## 10. Open items and cautions
- Add member names to the cover slide. Verify both model licenses on Hugging Face. Check `THIRD_PARTY_LICENSES.md`.
- Update `PROJECT_CONTEXT.md` with the decisions in section 5 and the schema notes, so the agent works from current truth.
- Ask the agent to generate the grammar, guard and gate vocabularies from `schema.py` so nothing is duplicated.
- A rejected or out-of-scope question needs a friendly message in the app.
- If possible, borrow a Raspberry Pi or ARM board for one real benchmark run; otherwise word results as "emulated constrained environment on Apple Silicon".
- Keep claims honest: targets (under 2 s, under 2 GB, 4+ weeks) stay "to validate" unless measured. Do not let prototype scope leak into slide claims.
- Differentiator to emphasize: **auditable, evidence-backed answers from local vehicle data**, with an explicit safety boundary and validated tools.

## 11. Suggested first message in the new chat
"I'm continuing the TELIPORT Case Study 5 (VehicleGPT) project. Here is HANDOFF_SUMMARY.md and PROJECT_CONTEXT.md. Please keep guiding me one step at a time, with a one-line 'why' for each step. Next I will paste the output of the Step 24 simulator fixes for review."
