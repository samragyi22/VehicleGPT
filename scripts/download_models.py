"""Download the documented GGUF variants into the git-ignored models directory."""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = (
    ("bartowski/Qwen2.5-0.5B-Instruct-GGUF", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
     "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ("bartowski/Qwen2.5-1.5B-Instruct-GGUF", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
     "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
)


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "models"
    target.mkdir(exist_ok=True)
    for repo, filename, output_name in MODELS:
        print(f"Model: {repo}/{filename}\nLicense: Apache-2.0 (Qwen2.5 model license)", flush=True)
        downloaded = Path(hf_hub_download(repo_id=repo, filename=filename))
        destination = target / output_name
        if downloaded.resolve() != destination.resolve():
            shutil.copy2(downloaded, destination)
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
