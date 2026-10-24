# VehicleGPT

> Offline, on-device assistant that answers natural-language questions about vehicle telemetry and diagnostics using an SLM.

Developed for TELIPORT Season 4 (Case Study 5) by Team Prayas.

## Demonstrated vs Planned

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

## Project Architecture

1. **Simulated EV Telemetry**: 28 days of reproducible 1 Hz driving signals & pre-event 10 Hz snapshots.
2. **Ingest & Normalization**: Cleaned data ingested into read-only SQLite database.
3. **Grammar-Constrained SLM**: llama.cpp GBNF grammar forces JSON tool calls matching predefined query schemas.
4. **Deterministic Query Engine**: Read-only parameterized tools compute metrics, trends, and factor explanations.
5. **Auditable Verifier**: Answers are generated from structured templates and verified against tool evidence before output.
