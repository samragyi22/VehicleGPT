# Third-Party Licenses

This document tracks all external open-source models, libraries, and datasets used in VehicleGPT.

## Dependencies

- **Python**: PSF License
- **llama.cpp / llama-cpp-python**: MIT License
- **Gradio**: Apache 2.0 License
- **Pydantic**: MIT License
- **Pandas**: BSD 3-Clause License
- **NumPy**: BSD 3-Clause License
- **PyArrow**: Apache 2.0 License
- **Hugging Face Hub client**: Apache 2.0 License
- **Pytest**: MIT License
- **Ruff**: MIT / Apache 2.0 License

## Candidate SLM Models

| Model | License | Notes |
|---|---|---|
| Qwen2.5-0.5B-Instruct (`qwen2.5-0.5b-instruct-q4_k_m.gguf`) | **Apache 2.0** ([official model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)) | Default model. GGUF conversion hosted by [bartowski](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF). |
| Qwen2.5-1.5B-Instruct (`qwen2.5-1.5b-instruct-q4_k_m.gguf`) | **Apache 2.0** ([official model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)) | Comparison model. GGUF conversion hosted by [bartowski](https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF). |

The GGUF files are quantized conversions of the Qwen2.5 Instruct weights; the model license is Apache-2.0. The download helper prints this license before each download. Review the upstream model and conversion repositories for their current terms before redistribution.
