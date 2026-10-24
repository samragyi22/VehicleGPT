# Third-Party Licenses

This document tracks all external open-source models, libraries, and datasets used in VehicleGPT.

## Dependencies

- **Python**: PSF License
- **llama.cpp / llama-cpp-python**: MIT License
- **Gradio**: Apache 2.0 License
- **Pydantic**: MIT License
- **Pandas**: BSD 3-Clause License
- **NumPy**: BSD 3-Clause License
- **Pytest**: MIT License
- **Ruff**: MIT / Apache 2.0 License

## Candidate SLM Models

| Model | License | Notes |
|---|---|---|
| Qwen2.5-0.5B-Instruct (`qwen2.5-0.5b-instruct-q4_k_m.gguf`) | **Apache 2.0** (Verified via Hugging Face model card `Qwen/Qwen2.5-0.5B-Instruct`) | Working default edge SLM. Permissive open source. |
| Qwen2.5-1.5B-Instruct (`qwen2.5-1.5b-instruct-q4_k_m.gguf`) | **Apache 2.0** (Verified via Hugging Face model card `Qwen/Qwen2.5-1.5B-Instruct`) | Candidate comparison SLM for Phase 4 benchmarking. |
